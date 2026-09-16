"""Read-only structured retrieval for the sales product/project catalogue."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
CATALOG_DB_PATH = ROOT / "data" / "sales" / "processed" / "catalog" / "sales_catalog.sqlite3"


_STRUCTURED_FIELD_MARKERS = (
    "品牌",
    "标准规格",
    "规格",
    "厚度",
    "重量",
    "饰面层效果",
    "装饰层品种",
    "应用领域",
)


def clean_catalog_introduction(value: Any, *, product_name: str = "") -> str:
    """Return the descriptive part of a public product page without page chrome.

    The website crawler keeps the original page text for auditability.  The
    structured catalogue already stores specifications in dedicated columns,
    so repeating the field tail (and phone/QR controls) only pollutes a product
    answer.  Cleaning happens at read time and never mutates the source record.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if product_name and text.startswith(product_name):
        text = text[len(product_name) :].lstrip(" ：:，,;-_")
    text = re.sub(r"(?<![A-Za-z])QQ(?![A-Za-z])", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b0\d{2,3}\s*[-－—]\s*\d{7,8}\b", " ", text)
    text = re.sub(r"(?:扫码|微信)?二维码(?:查看|咨询|联系)?", " ", text)
    marker_positions = [text.find(marker) for marker in _STRUCTURED_FIELD_MARKERS if text.find(marker) >= 0]
    if marker_positions:
        text = text[: min(marker_positions)]
    text = re.sub(r"\s+", " ", text).strip(" ：:，,;-_")
    return text


def _values(value: Any) -> list[str]:
    if isinstance(value, dict):
        source: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple, set)):
        source = value
    else:
        source = [value]
    output: list[str] = []
    for item in source:
        if isinstance(item, (list, tuple, set)):
            output.extend(_values(item))
        else:
            text = str(item or "").strip()
            if text and text not in output:
                output.append(text)
    return output


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=3.0)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _citation(row: sqlite3.Row, section: str) -> list[dict[str, Any]]:
    return [
        {
            "document_name": str(row["name"] if "name" in row.keys() else row["project_name"]),
            "source_page": None,
            "section_heading": section,
            "source_url": str(row["source_url"] or ""),
            "source_type": "official_website_structured_catalog",
        }
    ]


def retrieve_catalog_evidence(
    query: str,
    *,
    target_terms: Iterable[str] = (),
    case_filters: dict[str, Any] | None = None,
    include_products: bool = True,
    include_project_cases: bool = True,
    limit: int = 6,
    database_path: Path = CATALOG_DB_PATH,
) -> list[dict[str, Any]]:
    """Return precise candidates whose citations point to original sources.

    The function intentionally does not attempt general semantic intent
    classification.  It consumes entities already supplied by the model plan;
    when no bounded term/filter exists, ordinary hybrid RAG remains the right
    retrieval path.
    """

    if not database_path.is_file():
        return []
    filters = case_filters or {}
    terms = _values(list(target_terms))
    product_terms = _values(filters.get("products"))
    locations = _values(filters.get("locations"))
    project_types = _values(filters.get("project_types"))
    methods = _values(filters.get("installation_methods"))
    bounded_terms = list(dict.fromkeys([*terms, *product_terms, *locations, *project_types, *methods]))
    if not bounded_terms:
        return []

    output: list[dict[str, Any]] = []
    with _connect(database_path) as connection:
        product_search_terms = list(dict.fromkeys([*product_terms, *terms]))[:5] if include_products else []
        for term in product_search_terms:
            rows = connection.execute(
                """SELECT * FROM products
                   WHERE access_scope='public'
                     AND (name LIKE ? OR family LIKE ? OR surface_finish LIKE ? OR insulation_type LIKE ?)
                   ORDER BY CASE WHEN name=? THEN 0 WHEN name LIKE ? THEN 1 ELSE 2 END, name
                   LIMIT ?""",
                (f"%{term}%", f"%{term}%", f"%{term}%", f"%{term}%", term, f"%{term}%", limit),
            ).fetchall()
            for row in rows:
                evidence_id = "S" + hashlib.sha1(f"product|{row['product_id']}".encode()).hexdigest()[:12]
                introduction = clean_catalog_introduction(
                    row["introduction"], product_name=str(row["name"])
                ).rstrip("。；; ")
                text = "；".join(
                    part
                    for part in (
                        f"产品：{row['name']}",
                        f"系列：{row['family']}",
                        f"产品介绍：{introduction}" if introduction else "",
                        f"饰面效果：{row['surface_finish']}" if row["surface_finish"] else "",
                        f"标准规格：{row['standard_specification']}" if row["standard_specification"] else "",
                        f"板材厚度：{row['panel_thickness']}" if row["panel_thickness"] else "",
                        f"重量：{row['weight_kg_m2']}kg/m²" if row["weight_kg_m2"] else "",
                        f"保温层：{row['insulation_type']} {row['insulation_thickness'] or ''}".strip()
                        if row["insulation_type"]
                        else "",
                        f"适用范围：{row['application_scope']}" if row["application_scope"] else "",
                    )
                    if part
                )
                output.append(
                    {
                        "evidence_id": evidence_id,
                        "text": text,
                        "source_type": "catalog_sql",
                        "catalog_record_type": "product",
                        "record_id": row["product_id"],
                        "main_visual_asset_id": row["main_visual_asset_id"],
                        "facts_eligible": True,
                        "citations": _citation(row, "官方网站产品目录"),
                    }
                )

        clauses = ["access_scope='public'"]
        parameters: list[Any] = []
        for column, values in (
            ("product_name", product_terms),
            ("project_category", project_types),
            ("installation_method", methods),
        ):
            if values:
                clauses.append("(" + " OR ".join(f"{column} LIKE ?" for _ in values) + ")")
                parameters.extend(f"%{value}%" for value in values)
        if locations:
            clauses.append(
                "(" + " OR ".join("(province LIKE ? OR city LIKE ? OR project_name LIKE ?)" for _ in locations) + ")"
            )
            for value in locations:
                parameters.extend((f"%{value}%", f"%{value}%", f"%{value}%"))
        if not any((product_terms, locations, project_types, methods)) and terms:
            clauses.append(
                "(" + " OR ".join(
                    "(project_name LIKE ? OR product_name LIKE ? OR province LIKE ? OR city LIKE ? OR project_category LIKE ?)"
                    for _ in terms[:5]
                ) + ")"
            )
            for value in terms[:5]:
                parameters.extend((f"%{value}%",) * 5)
        case_rows = []
        if include_project_cases:
            case_rows = connection.execute(
                f"SELECT * FROM project_cases WHERE {' AND '.join(clauses)} ORDER BY project_name LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        for row in case_rows:
            evidence_id = "S" + hashlib.sha1(f"case|{row['case_id']}".encode()).hexdigest()[:12]
            text = "；".join(
                part
                for part in (
                    f"项目：{row['project_name']}",
                    f"地区：{row['province'] or ''}{row['city'] or ''}" if row["province"] or row["city"] else "",
                    f"项目类别：{row['project_category']}" if row["project_category"] else "",
                    f"使用产品：{row['product_name']}",
                    f"安装方式：{row['installation_method']}" if row["installation_method"] else "",
                    (
                        f"面积：{row['area_m2']}"
                        if row["area_m2"] and any(unit in str(row["area_m2"]) for unit in ("㎡", "m²", "m2"))
                        else f"面积：{row['area_m2']}㎡"
                        if row["area_m2"]
                        else ""
                    ),
                    f"完成年份：{row['completion_year']}" if row["completion_year"] else "",
                    f"关联原图：{row['image_count']}张",
                )
                if part
            )
            output.append(
                {
                    "evidence_id": evidence_id,
                    "text": text,
                    "source_type": "catalog_sql",
                    "catalog_record_type": "project_case",
                    "record_id": row["case_id"],
                    "facts_eligible": True,
                    "citations": _citation(row, "官方网站项目案例"),
                }
            )

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in output:
        evidence_id = str(item["evidence_id"])
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique
