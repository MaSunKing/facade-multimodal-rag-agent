"""Safe structural intake for archives, HTML/iXBRL, XML, and SEC submissions.

These formats are not reduced to a blind text blob.  The parsers preserve the
repeatable structure that makes the source useful: archive members, HTML
tables and inline-XBRL facts, XML records, or SEC submission attachments.
They deliberately stop short of inventing financial mappings; downstream
logic can only promote source-backed, confirmed values.
"""

from __future__ import annotations

import codecs
import hashlib
import re
import zipfile
from collections import Counter, defaultdict
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel, Field

from backend.document_parsing.ingestion import (
    EvidenceBlock,
    FinanceIntakeError,
    IntakeResult,
    IntermediateCell,
    IntermediateDocument,
    IntermediateTable,
    SourcePointer,
    StandardFinancialDocument,
    ValidationIssue,
    _headers_for_rows,
    evidence_block,
    finalize_intermediate_evidence,
    numeric_candidates_for_text,
    parse_excel_to_intermediate,
    standardize_tables,
)


MAX_MARKUP_BYTES = 20 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_SEC_TEXT_BYTES = 25 * 1024 * 1024
MAX_ZIP_BYTES = 20 * 1024 * 1024
MAX_ZIP_MEMBERS = 24
MAX_ZIP_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 250
MAX_MARKUP_TABLES = 250
MAX_MARKUP_TABLE_CELLS = 100_000
MAX_INLINE_XBRL_FACTS = 25_000
MAX_MARKUP_TEXT_BLOCKS = 12_000
MAX_XML_RECORDS = 20_000
MAX_SEC_ATTACHMENTS = 2_000
_TABULAR_MEMBER_SUFFIXES = {".csv", ".xlsx", ".xls"}
_BLOCK_BOUNDARIES = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}
_SKIP_TEXT_TAGS = {"script", "style", "template"}
_INLINE_XBRL_FACT_TAGS = {"ix:nonfraction", "ix:nonnumeric", "ix:fraction"}


class StructuredFileSummary(BaseModel):
    kind: str
    record_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    member_count: int = Field(default=0, ge=0)
    inline_xbrl_fact_count: int = Field(default=0, ge=0)


class StructuredFileIntakeResult(IntakeResult):
    structured_file: StructuredFileSummary


def _issue(code: str, message: str, source: SourcePointer | None = None) -> ValidationIssue:
    return ValidationIssue(severity="warning", code=code, message=message, source=source)


def _decode_markup(content: bytes) -> tuple[str, str]:
    """Decode a markup document by its declared charset before fallbacks."""

    head = content[:16_384].decode("latin-1", errors="ignore")
    match = re.search(r"charset\s*=\s*['\"]?([\w.\-]+)", head, flags=re.IGNORECASE)
    candidates = [match.group(1)] if match else []
    candidates.extend(["utf-8-sig", "utf-8", "gb18030", "windows-1252"])
    for candidate in candidates:
        try:
            encoding = codecs.lookup(candidate).name
            return content.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            continue
    raise FinanceIntakeError("无法识别文件编码；请转换为 UTF-8、GB18030 或带 charset 声明的 HTML/XML 后重试。")


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _html_cell(table_index: int, row_index: int, column_index: int, value: str) -> IntermediateCell:
    display = _normalise_text(value) or None
    return IntermediateCell(
        coordinate=f"t{table_index}r{row_index}c{column_index}",
        row_index=row_index,
        column_index=column_index,
        value=display,
        display_text=display,
        cached_value=display,
        numeric_candidates=numeric_candidates_for_text(display),
    )


class _HtmlTableBuilder:
    def __init__(self, index: int) -> None:
        self.index = index
        self.rows: list[list[list[str]]] = []
        self.current_row: list[list[str]] | None = None
        self.current_cell: list[str] | None = None

    def start_row(self) -> None:
        if self.current_row is not None:
            self.finish_row()
        self.current_row = []

    def start_cell(self) -> None:
        if self.current_row is None:
            self.start_row()
        if self.current_cell is not None:
            self.finish_cell()
        self.current_cell = []

    def append_text(self, value: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(value)

    def finish_cell(self) -> None:
        if self.current_cell is None:
            return
        if self.current_row is None:
            self.current_row = []
        self.current_row.append(self.current_cell)
        self.current_cell = None

    def finish_row(self) -> None:
        self.finish_cell()
        if self.current_row is not None and any(_normalise_text("".join(cell)) for cell in self.current_row):
            self.rows.append(self.current_row)
        self.current_row = None

    def finish(self) -> None:
        self.finish_row()


class _HtmlEvidenceParser(HTMLParser):
    """HTML parser that retains visible text, tables, images and iXBRL facts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[_HtmlTableBuilder] = []
        self.table_stack: list[_HtmlTableBuilder] = []
        self.text_blocks: list[str] = []
        self._current_text: list[str] = []
        self._skip_depth = 0
        self._hidden_depth = 0
        self._fact_stack: list[dict[str, Any]] = []
        self.inline_xbrl_facts: list[dict[str, str | None]] = []
        self.images: list[dict[str, str]] = []

    def _flush_text(self) -> None:
        text = _normalise_text("".join(self._current_text))
        self._current_text = []
        if text:
            self.text_blocks.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value or "" for key, value in attrs}
        if tag in _SKIP_TEXT_TAGS:
            self._skip_depth += 1
            return
        if tag == "ix:hidden":
            self._hidden_depth += 1
        if tag in _INLINE_XBRL_FACT_TAGS:
            self._fact_stack.append({"tag": tag, "attrs": attributes, "text": []})
        if tag == "img":
            self.images.append({key: value for key, value in attributes.items() if key in {"src", "alt", "title"} and value})
        if tag == "table":
            if len(self.tables) + len(self.table_stack) >= MAX_MARKUP_TABLES:
                raise FinanceIntakeError(f"HTML 表格超过 {MAX_MARKUP_TABLES} 张限制，请拆分后上传。")
            self._flush_text()
            self.table_stack.append(_HtmlTableBuilder(len(self.tables) + len(self.table_stack) + 1))
        elif tag == "tr" and self.table_stack:
            self.table_stack[-1].start_row()
        elif tag in {"td", "th"} and self.table_stack:
            self.table_stack[-1].start_cell()
        elif tag == "br" and not self.table_stack:
            self._flush_text()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._fact_stack:
            self._fact_stack[-1]["text"].append(data)
        if self._skip_depth:
            return
        if self.table_stack:
            self.table_stack[-1].append_text(data)
        elif not self._hidden_depth:
            self._current_text.append(data)
            if sum(len(item) for item in self._current_text) >= 1_400:
                self._flush_text()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _INLINE_XBRL_FACT_TAGS and self._fact_stack:
            frame = self._fact_stack.pop()
            if frame["tag"] == tag and len(self.inline_xbrl_facts) < MAX_INLINE_XBRL_FACTS:
                attrs = frame["attrs"]
                value = _normalise_text("".join(frame["text"]))
                if value:
                    self.inline_xbrl_facts.append(
                        {
                            "tag": tag,
                            "name": attrs.get("name") or None,
                            "context_ref": attrs.get("contextref") or None,
                            "unit_ref": attrs.get("unitref") or None,
                            "decimals": attrs.get("decimals") or None,
                            "format": attrs.get("format") or None,
                            "value": value,
                        }
                    )
        if tag in {"td", "th"} and self.table_stack:
            self.table_stack[-1].finish_cell()
        elif tag == "tr" and self.table_stack:
            self.table_stack[-1].finish_row()
        elif tag == "table" and self.table_stack:
            table = self.table_stack.pop()
            table.finish()
            self.tables.append(table)
        if tag in _BLOCK_BOUNDARIES and not self.table_stack:
            self._flush_text()
        if tag == "ix:hidden" and self._hidden_depth:
            self._hidden_depth -= 1
        if tag in _SKIP_TEXT_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def close(self) -> None:
        super().close()
        self._flush_text()
        while self.table_stack:
            table = self.table_stack.pop()
            table.finish()
            self.tables.append(table)


def _html_tables_to_intermediate(file_name: str, parsed: _HtmlEvidenceParser) -> list[IntermediateTable]:
    tables: list[IntermediateTable] = []
    cell_count = 0
    for table_index, builder in enumerate(parsed.tables, start=1):
        rows: list[list[IntermediateCell]] = []
        for row_index, raw_row in enumerate(builder.rows, start=1):
            row = [_html_cell(table_index, row_index, column_index, "".join(value)) for column_index, value in enumerate(raw_row, start=1)]
            if any(cell.display_text for cell in row):
                rows.append(row)
                cell_count += len(row)
        if cell_count > MAX_MARKUP_TABLE_CELLS:
            raise FinanceIntakeError(f"HTML 表格单元格超过 {MAX_MARKUP_TABLE_CELLS} 个限制，请拆分后上传。")
        if not rows:
            continue
        tables.append(
            IntermediateTable(
                table_id=f"html:table:{table_index}",
                title=f"HTML table {table_index}",
                sheet_name=f"HTML table {table_index}",
                range=f"table:{table_index}",
                parser="html.parser",
                extraction_confidence=1.0,
                headers=_headers_for_rows(rows),
                rows=rows,
            )
        )
    return tables


def ingest_html(file_name: str, content: bytes) -> StructuredFileIntakeResult:
    """Parse HTML/Inline XBRL as source-backed tables, text and fact candidates."""

    if len(content) > MAX_MARKUP_BYTES:
        raise FinanceIntakeError(f"HTML 文件超过 {MAX_MARKUP_BYTES // (1024 * 1024)} MB 限制，请拆分后上传。")
    text, encoding = _decode_markup(content)
    parser = _HtmlEvidenceParser()
    try:
        parser.feed(text)
        parser.close()
    except FinanceIntakeError:
        raise
    except Exception as exc:
        raise FinanceIntakeError("HTML 结构无法安全解析，请确认文件未损坏。") from exc
    if len(parser.text_blocks) > MAX_MARKUP_TEXT_BLOCKS:
        raise FinanceIntakeError(f"HTML 文字块超过 {MAX_MARKUP_TEXT_BLOCKS} 个限制，请拆分后上传。")
    tables = _html_tables_to_intermediate(file_name, parser)
    blocks: list[EvidenceBlock] = []
    for index, item in enumerate(parser.text_blocks, start=1):
        source = SourcePointer(
            source_type="html",
            file_name=Path(file_name).name,
            section_id=f"html:text:{index}",
            parser="html.parser",
            extraction_confidence=1.0,
        )
        blocks.append(evidence_block(block_id=f"html:text:{index}", kind="paragraph", source=source, text=item, metadata={"sequence": index}))
    for table in tables:
        source = SourcePointer(
            source_type="html",
            file_name=Path(file_name).name,
            sheet_name=table.sheet_name,
            section_id=table.table_id,
            parser=table.parser,
            extraction_confidence=table.extraction_confidence,
        )
        blocks.append(evidence_block(block_id=f"evidence:{table.table_id}", kind="table", source=source, table_id=table.table_id, metadata={"headers": table.headers, "row_count": len(table.rows)}))
    for group_index, start in enumerate(range(0, len(parser.inline_xbrl_facts), 200), start=1):
        facts = parser.inline_xbrl_facts[start : start + 200]
        source = SourcePointer(source_type="html", file_name=Path(file_name).name, section_id=f"html:inline-xbrl:{group_index}", parser="html.parser", extraction_confidence=1.0)
        blocks.append(
            evidence_block(
                block_id=f"html:inline-xbrl:{group_index}",
                kind="paragraph",
                source=source,
                text="\n".join(
                    f"name={fact['name'] or ''}; context={fact['context_ref'] or ''}; unit={fact['unit_ref'] or ''}; decimals={fact['decimals'] or ''}; value={fact['value']}"
                    for fact in facts
                ),
                metadata={"record_type": "inline_xbrl_fact", "fact_count": len(facts), "facts": facts},
            )
        )
    for index, image in enumerate(parser.images, start=1):
        source = SourcePointer(source_type="html", file_name=Path(file_name).name, section_id=f"html:image:{index}", parser="html.parser", extraction_confidence=1.0)
        blocks.append(evidence_block(block_id=f"html:image:{index}", kind="embedded_visual", source=source, metadata={"delivery_status": "metadata_only", **image}))
    document = finalize_intermediate_evidence(
        IntermediateDocument(
            source_type="html",
            file_name=Path(file_name).name,
            parser="html.parser+inline-xbrl",
            tables=tables,
            raw_text="\n".join(parser.text_blocks) or None,
            blocks=blocks,
        ),
        content,
    )
    return StructuredFileIntakeResult(
        intermediate=document,
        standard=StandardFinancialDocument(),
        validation=[_issue("markup_structure_only", "HTML/iXBRL 的文字、表格与原始 context/unit 候选已保留；尚未自动映射为公司财务事实。")],
        structured_file=StructuredFileSummary(kind="html", record_count=len(parser.text_blocks), table_count=len(tables), inline_xbrl_fact_count=len(parser.inline_xbrl_facts)),
    )


def _local_tag(tag: str) -> str:
    return tag.split("}")[-1]


def _xml_record_value(element: ElementTree.Element) -> str:
    return _normalise_text("".join(element.itertext()))


def ingest_xml(file_name: str, content: bytes) -> StructuredFileIntakeResult:
    """Parse repeating XML records into a generic, source-addressable grid."""

    if len(content) > MAX_XML_BYTES:
        raise FinanceIntakeError(f"XML 文件超过 {MAX_XML_BYTES // (1024 * 1024)} MB 限制，请拆分后上传。")
    text, encoding = _decode_markup(content)
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise FinanceIntakeError("XML 结构无法安全解析，请确认文件未损坏。") from exc
    root_children_by_tag: dict[str, list[ElementTree.Element]] = defaultdict(list)
    for child in list(root):
        root_children_by_tag[_local_tag(child.tag)].append(child)
    repeated_groups: list[tuple[str, str, list[ElementTree.Element]]] = []
    for parent in root.iter():
        children_by_tag: dict[str, list[ElementTree.Element]] = defaultdict(list)
        for child in list(parent):
            children_by_tag[_local_tag(child.tag)].append(child)
        for tag, elements in children_by_tag.items():
            if len(elements) >= 2:
                repeated_groups.append((_local_tag(parent.tag), tag, elements))
    tables: list[IntermediateTable] = []
    blocks: list[EvidenceBlock] = []
    root_source = SourcePointer(source_type="xml", file_name=Path(file_name).name, section_id=f"xml:{_local_tag(root.tag)}", parser="xml.etree", extraction_confidence=1.0)
    blocks.append(evidence_block(block_id="xml:root", kind="paragraph", source=root_source, text=f"root={_local_tag(root.tag)}; encoding={encoding}", metadata={"root": _local_tag(root.tag), "encoding": encoding}))
    record_count = 0
    for tag, elements in root_children_by_tag.items():
        if len(elements) < 2:
            value = _xml_record_value(elements[0])
            if value:
                source = root_source.model_copy(update={"section_id": f"xml:{tag}:1"})
                blocks.append(evidence_block(block_id=f"xml:value:{tag}", kind="paragraph", source=source, text=f"{tag}={value}", metadata={"field": tag}))
    for parent_tag, tag, elements in repeated_groups:
        if record_count + len(elements) > MAX_XML_RECORDS:
            raise FinanceIntakeError(f"XML 重复记录超过 {MAX_XML_RECORDS} 条限制，请拆分后上传。")
        headers: list[str] = []
        records: list[dict[str, str]] = []
        for element in elements:
            record: dict[str, str] = {f"@{key}": value for key, value in element.attrib.items()}
            for child in list(element):
                key = _local_tag(child.tag)
                value = _xml_record_value(child)
                if key in record and value:
                    record[key] = f"{record[key]} | {value}"
                elif value:
                    record[key] = value
            if not record:
                value = _xml_record_value(element)
                if value:
                    record["value"] = value
            for key in record:
                if key not in headers:
                    headers.append(key)
            records.append(record)
        rows: list[list[IntermediateCell]] = []
        for row_index, record in enumerate(records, start=1):
            row = [
                IntermediateCell(
                    coordinate=f"{tag}:r{row_index}:c{column_index}",
                    row_index=row_index,
                    column_index=column_index,
                    value=record.get(header) or None,
                    display_text=record.get(header) or None,
                    cached_value=record.get(header) or None,
                    numeric_candidates=numeric_candidates_for_text(record.get(header)),
                )
                for column_index, header in enumerate(headers, start=1)
            ]
            rows.append(row)
        table_id = f"xml:{parent_tag}:{tag}"
        table_title = f"{parent_tag}/{tag}"
        table = IntermediateTable(table_id=table_id, title=table_title, sheet_name=table_title, range=f"records:{len(rows)}", parser="xml.etree", extraction_confidence=1.0, headers=headers, rows=rows)
        tables.append(table)
        source = root_source.model_copy(update={"sheet_name": table_title, "section_id": table_id})
        blocks.append(evidence_block(block_id=f"evidence:{table_id}", kind="table", source=source, table_id=table_id, metadata={"headers": headers, "row_count": len(rows)}))
        record_count += len(records)
    document = finalize_intermediate_evidence(
        IntermediateDocument(source_type="xml", file_name=Path(file_name).name, parser="xml.etree", tables=tables, raw_text=None, blocks=blocks),
        content,
    )
    return StructuredFileIntakeResult(
        intermediate=document,
        standard=StandardFinancialDocument(),
        validation=[_issue("xml_structure_only", "XML 的根节点、字段与重复记录已保留；尚未自动映射为公司财务事实。", root_source)],
        structured_file=StructuredFileSummary(kind="xml", record_count=record_count, table_count=len(tables)),
    )


def looks_like_sec_submission(content: bytes) -> bool:
    return content.lstrip(b"\xef\xbb\xbf\t\r\n ").startswith(b"<SEC-DOCUMENT>")


def _sec_field(part: str, name: str) -> str | None:
    match = re.search(rf"(?im)^<{name}>\s*(.+?)\s*$", part)
    return match.group(1).strip() if match else None


def ingest_sec_submission(file_name: str, content: bytes) -> StructuredFileIntakeResult:
    """Index a long SEC submission without sending its entire attachment bundle to a model."""

    if len(content) > MAX_SEC_TEXT_BYTES:
        raise FinanceIntakeError(f"SEC 提交包超过 {MAX_SEC_TEXT_BYTES // (1024 * 1024)} MB 限制，请拆分后上传。")
    text, encoding = _decode_markup(content)
    if not text.lstrip().startswith("<SEC-DOCUMENT>"):
        raise FinanceIntakeError("该文本不是可识别的 SEC 提交包。")
    first_document = re.search(r"(?is)<DOCUMENT>(.*?)(?:</DOCUMENT>|\Z)", text)
    header = text[: first_document.start()] if first_document else text[:20_000]
    parts = re.findall(r"(?is)<DOCUMENT>(.*?)(?:</DOCUMENT>|\Z)", text)
    if len(parts) > MAX_SEC_ATTACHMENTS:
        raise FinanceIntakeError(f"SEC 附件超过 {MAX_SEC_ATTACHMENTS} 个限制，请拆分后上传。")
    rows: list[list[IntermediateCell]] = []
    attachment_lines: list[str] = []
    for row_index, part in enumerate(parts, start=1):
        type_name = _sec_field(part, "TYPE") or "unknown"
        sequence = _sec_field(part, "SEQUENCE") or ""
        filename = _sec_field(part, "FILENAME") or ""
        description = _sec_field(part, "DESCRIPTION") or ""
        length = len(part)
        digest = hashlib.sha256(part.encode("utf-8", errors="replace")).hexdigest()
        values = [type_name, sequence, filename, description, str(length), digest]
        rows.append(
            [
                IntermediateCell(
                    coordinate=f"sec:r{row_index}:c{column_index}",
                    row_index=row_index,
                    column_index=column_index,
                    value=value or None,
                    display_text=value or None,
                    cached_value=value or None,
                    numeric_candidates=numeric_candidates_for_text(value),
                )
                for column_index, value in enumerate(values, start=1)
            ]
        )
        attachment_lines.append(f"attachment={row_index}; type={type_name}; sequence={sequence}; filename={filename}; characters={length}; sha256={digest}")
    table = IntermediateTable(
        table_id="sec:attachment-manifest",
        title="SEC attachment manifest",
        sheet_name="SEC attachment manifest",
        range=f"records:{len(rows)}",
        parser="sec-sgml",
        extraction_confidence=1.0,
        headers=["type", "sequence", "filename", "description", "characters", "sha256"],
        rows=rows,
    )
    root_source = SourcePointer(source_type="text", file_name=Path(file_name).name, section_id="sec:header", parser="sec-sgml", extraction_confidence=1.0)
    manifest_source = root_source.model_copy(update={"sheet_name": table.sheet_name, "section_id": table.table_id})
    blocks: list[EvidenceBlock] = [
        evidence_block(block_id="sec:header", kind="paragraph", source=root_source, text=header.strip(), metadata={"encoding": encoding, "attachment_count": len(parts)}),
        evidence_block(block_id="evidence:sec:attachment-manifest", kind="table", source=manifest_source, table_id=table.table_id, metadata={"headers": table.headers, "row_count": len(rows)}),
    ]
    for group_index, start in enumerate(range(0, len(attachment_lines), 100), start=1):
        source = manifest_source.model_copy(update={"section_id": f"sec:attachment-manifest:{group_index}"})
        blocks.append(evidence_block(block_id=f"sec:manifest:{group_index}", kind="paragraph", source=source, text="\n".join(attachment_lines[start : start + 100]), metadata={"attachment_start": start + 1, "attachment_count": min(100, len(attachment_lines) - start)}))
    document = finalize_intermediate_evidence(
        IntermediateDocument(source_type="text", file_name=Path(file_name).name, parser="sec-sgml", tables=[table], raw_text=header.strip() or None, blocks=blocks),
        content,
    )
    return StructuredFileIntakeResult(
        intermediate=document,
        standard=StandardFinancialDocument(),
        validation=[_issue("sec_submission_manifest_only", "SEC 提交包已解析为附件清单；附件正文未被一次性送入模型。请上传或选择具体 HTML/XML/报表附件继续解析。", root_source)],
        structured_file=StructuredFileSummary(kind="sec_submission", record_count=len(parts), table_count=1),
    )


def _member_name(archive_name: str, member_name: str) -> str:
    return f"{Path(archive_name).name}!{member_name}"


def _safe_zip_members(file_name: str, content: bytes) -> list[tuple[zipfile.ZipInfo, bytes]]:
    if len(content) > MAX_ZIP_BYTES:
        raise FinanceIntakeError(f"ZIP 文件超过 {MAX_ZIP_BYTES // (1024 * 1024)} MB 限制，请拆分后上传。")
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise FinanceIntakeError("ZIP 文件已损坏或不是有效压缩包。") from exc
    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise FinanceIntakeError("ZIP 中没有可读取文件。")
        if len(members) > MAX_ZIP_MEMBERS:
            raise FinanceIntakeError(f"ZIP 内文件超过 {MAX_ZIP_MEMBERS} 个限制，请拆分后上传。")
        total = sum(item.file_size for item in members)
        if total > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise FinanceIntakeError(f"ZIP 解压后超过 {MAX_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)} MB 限制，请拆分后上传。")
        selected = [item for item in members if Path(item.filename).suffix.lower() in _TABULAR_MEMBER_SUFFIXES]
        if not selected:
            raise FinanceIntakeError("ZIP 中未找到可读取的 CSV、XLSX 或 XLS 文件。")
        content_by_member: list[tuple[zipfile.ZipInfo, bytes]] = []
        for item in selected:
            if item.flag_bits & 0x1:
                raise FinanceIntakeError("暂不支持加密 ZIP，请解压后上传文件。")
            if item.compress_size and item.file_size / item.compress_size > MAX_ZIP_COMPRESSION_RATIO:
                raise FinanceIntakeError("ZIP 的压缩比例异常，已拒绝解压以保护服务。")
            content_by_member.append((item, archive.read(item)))
    return content_by_member


def ingest_zip(file_name: str, content: bytes) -> IntakeResult:
    """Safely merge tabular archive members while retaining member provenance."""

    members = _safe_zip_members(file_name, content)
    tables: list[IntermediateTable] = []
    blocks: list[EvidenceBlock] = []
    visual_assets = []
    member_lines: list[str] = []
    for member_index, (info, member_content) in enumerate(members, start=1):
        member_file = _member_name(file_name, info.filename)
        parsed = parse_excel_to_intermediate(member_file, member_content)
        id_map: dict[str, str] = {}
        for table in parsed.tables:
            table_id = f"zip:{member_index}:{table.table_id}"
            id_map[table.table_id] = table_id
            tables.append(
                table.model_copy(
                    update={
                        "table_id": table_id,
                        "parent_table_id": f"zip:{member_index}:{table.parent_table_id}" if table.parent_table_id else None,
                        "section_id": f"zip:{member_index}:{table.section_id}" if table.section_id else None,
                        "sheet_name": f"{info.filename}::{table.sheet_name}" if table.sheet_name else info.filename,
                        "title": table.title or info.filename,
                    }
                )
            )
        for block in parsed.blocks:
            if block.kind == "derived_text":
                continue
            blocks.append(
                block.model_copy(
                    update={
                        "block_id": f"zip:{member_index}:{block.block_id}",
                        "table_id": id_map.get(block.table_id or "", block.table_id),
                        "source": block.source.model_copy(update={"file_name": member_file}),
                    }
                )
            )
        visual_assets.extend(asset.model_copy(update={"visual_id": f"zip:{member_index}:{asset.visual_id}", "source": asset.source.model_copy(update={"file_name": member_file})}) for asset in parsed.visual_assets)
        member_lines.append(f"member={info.filename}; compressed_bytes={info.compress_size}; uncompressed_bytes={info.file_size}; parsed_tables={len(parsed.tables)}")
    manifest_source = SourcePointer(source_type="excel", file_name=Path(file_name).name, section_id="zip:manifest", parser="zip+tabular", extraction_confidence=1.0)
    blocks.insert(0, evidence_block(block_id="zip:manifest", kind="paragraph", source=manifest_source, text="\n".join(member_lines), metadata={"member_count": len(members)}))
    document = finalize_intermediate_evidence(
        IntermediateDocument(source_type="excel", file_name=Path(file_name).name, parser="zip+tabular", tables=tables, blocks=blocks, visual_assets=visual_assets),
        content,
    )
    result = standardize_tables(document)
    return result.model_copy(
        update={
            "validation": [
                *result.validation,
                _issue("archive_members_combined", f"已安全解析 ZIP 中 {len(members)} 个表格文件；每张表的来源包含 archive!member。", manifest_source),
            ]
        }
    )
