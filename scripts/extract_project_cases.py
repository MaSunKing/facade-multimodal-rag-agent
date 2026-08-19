"""Extract traceable project-case records from the comprehensive product catalogue.

The catalogue stores one real-world application case on most case pages.  This
script reads only the local source PDF and writes a compact, queryable record
for every page that exposes a project name plus structured project fields.
Records preserve the original page number and are intentionally labelled as
enterprise-catalogue material rather than third-party verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "data" / "sales" / "raw" / "product_pdfs" / "product_catalogue.pdf"
DEFAULT_OUTPUT = ROOT / "data" / "sales" / "processed" / "rag_ready" / "project_cases.jsonl"

FIELD_LABELS = (
    "项目类型",
    "应用部位",
    "使用产品",
    "施工工艺",
    "使用面积",
    "建成时间",
)
HEADER_LINES = {"应用案例", "APPLICATION CASE", "案例", "企业产品", "产品手册"}


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("\u00a0", " ")).strip()


def stable_id(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"case_{digest}"


def normalise_case_identity(value: str) -> str:
    """Catalogue spreads one case across pages and varies a few title glyphs."""

    return re.sub(r"[·.。\s]", "", value)


def field_value(lines: list[str], label: str) -> str | None:
    """Read a one-line catalogue field, tolerating label/value on one line."""

    for index, line in enumerate(lines):
        if not line.startswith(label):
            continue
        remainder = compact(line.removeprefix(label).lstrip("：:"))
        if remainder:
            return remainder
        for candidate in lines[index + 1 : index + 3]:
            candidate = compact(candidate)
            if candidate and not any(candidate.startswith(other) for other in FIELD_LABELS):
                return candidate
    return None


def project_name(lines: list[str]) -> str | None:
    """Use the title lines immediately before the first structured field."""

    first_field = next((index for index, line in enumerate(lines) if line.startswith(("项目类型", "应用部位"))), None)
    if first_field is None:
        return None
    candidates = [compact(line) for line in lines[:first_field]]
    candidates = [line for line in candidates if line and line not in HEADER_LINES]
    # Case pages usually end their title area with the city/region followed by
    # the project name.  Keep up to the final two meaningful lines.
    if not candidates:
        return None
    return "".join(candidates[-2:])


def project_region(name: str) -> str | None:
    """A conservative convenience value; the full title remains canonical."""

    province_or_city = re.match(
        r"(.{2,8}?(?:省|市|自治区|特别行政区|县|区))",
        name,
    )
    return province_or_city.group(1) if province_or_city else None


def case_record(pdf: Path, page_number: int, text: str) -> dict[str, Any] | None:
    lines = [compact(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    fields = {label: field_value(lines, label) for label in FIELD_LABELS}
    name = project_name(lines)
    if not name or not fields["使用产品"] or not fields["施工工艺"]:
        return None

    case_key = "|".join(
        str(value or "")
        for value in (
            pdf.name,
            normalise_case_identity(name),
            fields["项目类型"],
            fields["使用产品"],
            fields["施工工艺"],
            fields["使用面积"],
            fields["建成时间"],
        )
    )
    return {
        "record_type": "project_case",
        "case_id": stable_id(case_key),
        "project_name": name,
        "region": project_region(name),
        "project_type": fields["项目类型"],
        "application_area": fields["应用部位"],
        "product": fields["使用产品"],
        "installation_method": fields["施工工艺"],
        "area_m2": fields["使用面积"],
        "completion_year": fields["建成时间"],
        "source_document": pdf.stem,
        "source_pdf": str(pdf),
        "source_page": page_number,
        "source_pages": [page_number],
        "source_type": "企业综合产品画册",
        "verification_note": "项目字段来自企业产品画册，用于案例检索与展示，不构成第三方验收或性能证明。",
        "customer_shareable": True,
        "visual_asset_ids": [],
        "source_text": "\n".join(lines),
    }


def extract_cases(pdf: Path) -> list[dict[str, Any]]:
    unique_records: dict[str, dict[str, Any]] = {}
    with fitz.open(pdf) as document:
        for page_index in range(document.page_count):
            text = document.load_page(page_index).get_text("text")
            record = case_record(pdf, page_index + 1, text)
            if record:
                existing = unique_records.get(record["case_id"])
                if existing is None:
                    unique_records[record["case_id"]] = record
                else:
                    existing["source_pages"].append(record["source_page"])
                    existing["source_text"] += "\n" + record["source_text"]
    return sorted(unique_records.values(), key=lambda record: int(record["source_page"]))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract project cases from a local product catalogue PDF.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.exists():
        raise SystemExit(f"Source PDF not found: {pdf}")
    records = extract_cases(pdf)
    if not records:
        raise SystemExit("No structured project cases were found in the source PDF.")
    write_jsonl(args.output.resolve(), records)
    print(f"Extracted {len(records)} local project-case records: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
