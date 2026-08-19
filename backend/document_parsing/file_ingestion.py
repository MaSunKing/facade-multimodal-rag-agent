"""Single dispatch point for customer-uploaded finance files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.document_parsing.document_ingestion import ingest_image, ingest_word, is_supported_image_name
from backend.document_parsing.ingestion import FinanceIntakeError, ingest_excel, ingest_text_file
from backend.document_parsing.pdf_ingestion import ingest_pdf
from backend.document_parsing.structured_file_ingestion import ingest_html, ingest_sec_submission, ingest_xml, ingest_zip, looks_like_sec_submission


def ingest_uploaded_file(file_name: str, content: bytes) -> Any:
    """Dispatch one in-memory customer upload by its safe, explicit suffix."""

    suffix = Path(file_name).suffix.lower()
    if suffix in {".xlsx", ".xls", ".csv"}:
        return ingest_excel(file_name, content)
    if suffix == ".pdf":
        return ingest_pdf(file_name, content)
    if suffix == ".docx":
        return ingest_word(file_name, content)
    if suffix == ".txt":
        if looks_like_sec_submission(content):
            return ingest_sec_submission(file_name, content)
        return ingest_text_file(file_name, content)
    if suffix == ".zip":
        return ingest_zip(file_name, content)
    if suffix in {".html", ".htm"}:
        return ingest_html(file_name, content)
    if suffix == ".xml":
        return ingest_xml(file_name, content)
    if is_supported_image_name(file_name):
        return ingest_image(file_name, content)
    raise FinanceIntakeError(
        "当前支持 TXT、Excel（.xlsx/.xls）、CSV、ZIP 表格包、PDF、Word（.docx）、"
        "HTML/Inline XBRL、XML 和 JPG/PNG/WebP/GIF/BMP/TIFF 图片。"
    )
