"""Create local, traceable RAG text evidence from Word sales playbooks.

This uses only the Python standard library and never sends documents to a
cloud parser. DOCX files are selected by the ASCII extension, avoiding Windows
console code-page issues with Chinese document names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "sales" / "raw" / "sales_playbooks"
DEFAULT_ASSETS_ROOT = ROOT / "data" / "sales" / "processed" / "rag_assets"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]}"


def paragraph_text(paragraph: ET.Element) -> str:
    fragments: list[str] = []
    for child in paragraph.iter():
        if child.tag == f"{{{WORD_NS}}}t":
            fragments.append(child.text or "")
        elif child.tag == f"{{{WORD_NS}}}tab":
            fragments.append("\t")
        elif child.tag == f"{{{WORD_NS}}}br":
            fragments.append("\n")
    return re.sub(r"\s+", " ", "".join(fragments)).strip()


def paragraph_style(paragraph: ET.Element) -> str:
    node = paragraph.find("w:pPr/w:pStyle", NS)
    return str(node.get(f"{{{WORD_NS}}}val") or "") if node is not None else ""


def read_docx_paragraphs(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    output: list[tuple[str, str]] = []
    for paragraph in root.findall(".//w:body/w:p", NS):
        text = paragraph_text(paragraph)
        if text:
            output.append((text, paragraph_style(paragraph)))
    return output


def heading_level(text: str, style: str) -> int | None:
    match = re.search(r"heading\s*([1-9])", style.lower())
    if match:
        return int(match.group(1))
    chinese = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341"
    if re.match(rf"^(?:\u7b2c?[{chinese}]+[\u3001\uff1a:]|[{chinese}]+\u3001)", text):
        return 1
    if re.match(rf"^[\uff08(][{chinese}]+[\uff09)]", text):
        return 2
    if re.match(r"^\d+[\.\u3001\uff1a:]", text) and len(text) <= 100:
        return 3
    return None


def iter_chunks(
    paragraphs: Iterable[tuple[str, str]], max_characters: int = 900
) -> Iterable[tuple[str, str, int, int]]:
    hierarchy: dict[int, str] = {}
    current_parts: list[str] = []
    current_heading = "sales playbook"
    start_position = 1
    position = 0

    def flush() -> tuple[str, str, int, int] | None:
        nonlocal current_parts, start_position
        if not current_parts:
            return None
        content = "\n".join(current_parts).strip()
        output = (current_heading, content, start_position, position)
        current_parts = []
        return output

    for position, (text, style) in enumerate(paragraphs, start=1):
        level = heading_level(text, style)
        if level is not None:
            item = flush()
            if item is not None:
                yield item
            hierarchy = {key: value for key, value in hierarchy.items() if key < level}
            hierarchy[level] = text
            current_heading = " / ".join(hierarchy[key] for key in sorted(hierarchy))
            current_parts = [text]
            start_position = position
            continue

        candidate_length = sum(len(part) + 1 for part in current_parts) + len(text)
        if current_parts and candidate_length > max_characters:
            item = flush()
            if item is not None:
                yield item
            start_position = position
        current_parts.append(text)

    item = flush()
    if item is not None:
        yield item


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_sources(explicit_source: Path | None) -> list[Path]:
    if explicit_source is not None:
        source = explicit_source.resolve()
        if not source.exists() or source.suffix.lower() != ".docx":
            raise SystemExit(f"DOCX source was not found: {source}")
        return [source]

    sources = sorted(
        (path.resolve() for path in DEFAULT_SOURCE_DIR.glob("*.docx") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not sources:
        raise SystemExit(f"No DOCX sales playbooks were found in: {DEFAULT_SOURCE_DIR}")
    return sources


def ingest_one(source: Path, assets_root: Path) -> dict:
    document_name = source.stem
    document_id = stable_id("docx", str(source))
    asset_directory = assets_root / document_name
    asset_directory.mkdir(parents=True, exist_ok=True)
    paragraphs = read_docx_paragraphs(source)
    records: list[dict] = []
    for heading, content, start, end in iter_chunks(paragraphs):
        text = content if content.startswith(heading) else f"{heading}\n{content}"
        records.append(
            {
                "record_type": "text_evidence",
                "document_id": document_id,
                "document_name": document_name,
                "source_pdf": str(source),
                "source_page": None,
                "bbox": None,
                "section_heading": heading,
                "text": text,
                "source_format": "docx",
                "source_paragraph_range": [start, end],
                "review_status": "unreviewed",
                "customer_shareable": False,
            }
        )

    write_jsonl(asset_directory / "text_evidence.jsonl", records)
    manifest = {
        "document_id": document_id,
        "document_name": document_name,
        "source_file": str(source),
        "source_format": "docx",
        "paragraph_count": len(paragraphs),
        "text_record_count": len(records),
        "visual_asset_count": 0,
        "customer_shareable_default": False,
        "ingestion_policy": "internal_sales_playbook_source_preserved_locally",
    }
    (asset_directory / "manifest_metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local Word sales playbooks into RAG assets.")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    args = parser.parse_args()
    manifests = [ingest_one(source, args.assets_root.resolve()) for source in resolve_sources(args.source)]
    print(json.dumps({"ingested_documents": manifests}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
