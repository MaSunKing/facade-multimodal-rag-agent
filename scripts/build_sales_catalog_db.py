"""Build a regenerable SQLite product/project catalogue beside the RAG index.

SQLite stores deterministic filters and relationships.  Narrative evidence and
visual semantics remain in the existing JSONL/BM25/dense RAG pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "data" / "sales" / "processed" / "rag_ready"
DEFAULT_OUTPUT = ROOT / "data" / "sales" / "processed" / "catalog" / "sales_catalog.sqlite3"
DEFAULT_AUDIT = ROOT / "output" / "sales_catalog" / "audit.json"

CATEGORY_MARKERS = (
    "办公类-写字楼 · 产业园",
    "教育-医疗-金融外装",
    "住宿类-酒店 · 公寓",
    "零售类-商铺 · 商场",
    "商住综合体类",
    "自建别墅类",
    "售楼部类",
    "住宅类",
    "办公楼",
    "视频介绍",
)

# Ordered from specific locality to broad province.  Unknown cities remain
# NULL rather than being inferred from a building or developer name.
LOCATION_RULES = (
    ("武汉", "湖北省", "武汉市"), ("随县", "湖北省", "随县"),
    ("济南", "山东省", "济南市"), ("德州", "山东省", "德州市"), ("东营", "山东省", "东营市"),
    ("保定", "河北省", "保定市"), ("邢台", "河北省", "邢台市"), ("唐山", "河北省", "唐山市"),
    ("石家庄", "河北省", "石家庄市"), ("沧州", "河北省", "沧州市"), ("廊坊", "河北省", "廊坊市"),
    ("张家口", "河北省", "张家口市"), ("涿州", "河北省", "涿州市"), ("定州", "河北省", "定州市"),
    ("安国", "河北省", "安国市"),
    ("太原", "山西省", "太原市"), ("临猗", "山西省", "临猗县"), ("万荣", "山西省", "万荣县"),
    ("平遥", "山西省", "平遥县"),
    ("常州", "江苏省", "常州市"), ("焦作", "河南省", "焦作市"), ("洛阳", "河南省", "洛阳市"),
    ("北京", "北京市", "北京市"), ("重庆", "重庆市", "重庆市"),
    ("河北", "河北省", None), ("山西", "山西省", None), ("山东", "山东省", None),
    ("湖北", "湖北省", None), ("江苏", "江苏省", None), ("河南", "河南省", None),
    ("安徽", "安徽省", None), ("黑龙江", "黑龙江省", None), ("内蒙古", "内蒙古自治区", None),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalise_name(value: str) -> str:
    value = re.sub(r"真岩®?无机仿石材案例", "", value)
    value = re.sub(r"项目$", "", value)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value).lower()


def extract_between(text: str, label: str, following: tuple[str, ...]) -> str | None:
    pattern = re.escape(label) + r"\s*[:：]?\s*(.+?)\s*(?=" + "|".join(re.escape(item) for item in following) + r"|$)"
    match = re.search(pattern, text)
    return match.group(1).strip(" ：") if match else None


def product_family(name: str) -> tuple[str, str | None, str | None]:
    if "保温装饰一体板" in name:
        insulation = "XPS挤塑板" if "挤塑" in name else "石墨聚苯板" if "石墨聚苯" in name else "岩棉板" if "岩棉" in name else None
        return "真岩®保温装饰一体板", None, insulation
    finish = "荔枝面" if "荔枝面" in name else "光面" if "光面" in name else "哑光面" if "哑光面" in name else None
    return "真岩®无机仿石装饰板", finish, None


def location_for(name: str) -> tuple[str | None, str | None]:
    for marker, province, city in LOCATION_RULES:
        if marker in name:
            return province, city
    return None, None


def project_category(source_text: str) -> str | None:
    return next((marker for marker in CATEGORY_MARKERS if marker in source_text), None)


def catalogue_match(case_name: str, catalogue_cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = normalise_name(case_name)
    exact = [item for item in catalogue_cases if normalise_name(str(item.get("project_name") or "")) == target]
    if exact:
        return exact[0]
    candidates = []
    for item in catalogue_cases:
        other = normalise_name(str(item.get("project_name") or ""))
        if min(len(target), len(other)) >= 6 and (target in other or other in target):
            candidates.append(item)
    return candidates[0] if len(candidates) == 1 else None


def build(output: Path, audit_path: Path) -> dict[str, Any]:
    website_text = read_jsonl(READY / "website_text_evidence_tagged.jsonl")
    website_visuals = read_jsonl(READY / "website_visual_assets_tagged.jsonl")
    website_catalog = read_jsonl(READY / "website_visual_catalog.jsonl")
    website_cases = read_jsonl(READY / "website_project_cases_tagged.jsonl")
    catalogue_cases = read_jsonl(READY / "project_cases_tagged.jsonl")
    visual_by_url = {
        str(item.get("source_url")): item
        for item in website_catalog
        if item.get("gallery_type") == "product_sample" and item.get("source_url")
    }

    product_rows: list[dict[str, Any]] = []
    for item in website_text:
        if "official_website_product_page" not in set(item.get("document_categories") or []):
            continue
        source = (item.get("source_refs") or [{}])[0]
        name = str(source.get("document_name") or "").strip()
        url = str(source.get("source_url") or "").strip()
        text = str(item.get("text") or "").strip()
        family, finish, insulation = product_family(name)
        product_rows.append(
            {
                "product_id": "product_" + hashlib.sha1(url.encode()).hexdigest()[:16],
                "name": name,
                "family": family,
                "surface_finish": finish,
                "insulation_type": insulation,
                "introduction": text,
                "standard_specification": extract_between(text, "标准规格 (mm)", ("厚度 (mm)", "保温层厚度 (cm)")),
                "panel_thickness": extract_between(text, "厚度 (mm)", ("重量(kg/m2)", "饰面层效果")),
                "weight_kg_m2": extract_between(text, "重量(kg/m2)", ("饰面层效果",)),
                "insulation_thickness": extract_between(text, "保温层厚度 (cm)", ("保温层类型",)),
                "application_scope": extract_between(text, "应用领域", ("0312", "二维码", "产品详情")),
                "source_url": url,
                "main_visual_asset_id": (visual_by_url.get(url) or {}).get("asset_id"),
                "access_scope": "public",
            }
        )

    case_rows: list[dict[str, Any]] = []
    case_visual_rows: list[tuple[str, str, int]] = []
    for item in website_cases:
        case_id = str(item["case_id"])
        name = str(item.get("project_name") or "")
        source_text = str(item.get("source_text") or "")
        province, city = location_for(name)
        matched = catalogue_match(name, catalogue_cases)
        title_product = "真岩®岩棉保温装饰一体板" if "岩棉保温装饰一体板" in name else None
        product = str((matched or {}).get("product") or title_product or item.get("product") or "真岩®无机仿石材")
        case_rows.append(
            {
                "case_id": case_id,
                "project_name": name,
                "province": province,
                "city": city,
                "project_category": project_category(source_text) or (matched or {}).get("project_type"),
                "product_name": product,
                "product_specificity": "catalogue_matched" if matched and matched.get("product") else "title_explicit" if title_product else "generic_website_product",
                "installation_method": (matched or {}).get("installation_method"),
                "area_m2": (matched or {}).get("area_m2"),
                "completion_year": (matched or {}).get("completion_year"),
                "image_count": len(item.get("visual_asset_ids") or []),
                "source_url": item.get("source_url"),
                "source_text": source_text,
                "access_scope": "public",
            }
        )
        for position, asset_id in enumerate(item.get("visual_asset_ids") or [], start=1):
            case_visual_rows.append((case_id, str(asset_id), position))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.sqlite3")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE products(
                product_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, family TEXT NOT NULL,
                surface_finish TEXT, insulation_type TEXT, introduction TEXT NOT NULL,
                standard_specification TEXT, panel_thickness TEXT, weight_kg_m2 TEXT,
                insulation_thickness TEXT, application_scope TEXT, source_url TEXT NOT NULL,
                main_visual_asset_id TEXT, access_scope TEXT NOT NULL CHECK(access_scope IN ('public','internal'))
            );
            CREATE TABLE project_cases(
                case_id TEXT PRIMARY KEY, project_name TEXT NOT NULL, province TEXT, city TEXT,
                project_category TEXT, product_name TEXT NOT NULL, product_specificity TEXT NOT NULL,
                installation_method TEXT, area_m2 TEXT, completion_year TEXT, image_count INTEGER NOT NULL,
                source_url TEXT NOT NULL, source_text TEXT NOT NULL, access_scope TEXT NOT NULL
            );
            CREATE TABLE project_case_visuals(
                case_id TEXT NOT NULL REFERENCES project_cases(case_id) ON DELETE CASCADE,
                asset_id TEXT NOT NULL, display_order INTEGER NOT NULL,
                PRIMARY KEY(case_id, asset_id)
            );
            CREATE INDEX idx_products_family ON products(family);
            CREATE INDEX idx_cases_location ON project_cases(province, city);
            CREATE INDEX idx_cases_category ON project_cases(project_category);
            CREATE INDEX idx_cases_product ON project_cases(product_name);
            CREATE VIRTUAL TABLE product_search USING fts5(product_id UNINDEXED, name, family, introduction, application_scope, tokenize='unicode61');
            CREATE VIRTUAL TABLE case_search USING fts5(case_id UNINDEXED, project_name, province, city, project_category, product_name, source_text, tokenize='unicode61');
            """
        )
        for row in product_rows:
            connection.execute(
                "INSERT INTO products VALUES(:product_id,:name,:family,:surface_finish,:insulation_type,:introduction,:standard_specification,:panel_thickness,:weight_kg_m2,:insulation_thickness,:application_scope,:source_url,:main_visual_asset_id,:access_scope)",
                row,
            )
            connection.execute(
                "INSERT INTO product_search VALUES(?,?,?,?,?)",
                (row["product_id"], row["name"], row["family"], row["introduction"], row["application_scope"]),
            )
        for row in case_rows:
            connection.execute(
                "INSERT INTO project_cases VALUES(:case_id,:project_name,:province,:city,:project_category,:product_name,:product_specificity,:installation_method,:area_m2,:completion_year,:image_count,:source_url,:source_text,:access_scope)",
                row,
            )
            connection.execute(
                "INSERT INTO case_search VALUES(?,?,?,?,?,?,?)",
                (row["case_id"], row["project_name"], row["province"], row["city"], row["project_category"], row["product_name"], row["source_text"]),
            )
        connection.executemany("INSERT INTO project_case_visuals VALUES(?,?,?)", case_visual_rows)
        generated = datetime.now(timezone.utc).isoformat()
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            (("schema_version", "sales_catalog_v1"), ("generated_at_utc", generated), ("source", "official_website_plus_enterprise_catalogue")),
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    os.replace(temporary, output)

    audit = {
        "schema_version": "sales_catalog_v1",
        "database": str(output),
        "integrity_check": integrity,
        "product_count": len(product_rows),
        "product_with_main_image_count": sum(bool(item["main_visual_asset_id"]) for item in product_rows),
        "project_case_count": len(case_rows),
        "case_visual_link_count": len(case_visual_rows),
        "case_with_city_count": sum(bool(item["city"]) for item in case_rows),
        "case_with_category_count": sum(bool(item["project_category"]) for item in case_rows),
        "case_with_specific_product_count": sum(item["product_specificity"] != "generic_website_product" for item in case_rows),
        "generic_product_case_count": sum(item["product_specificity"] == "generic_website_product" for item in case_rows),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the structured sales catalogue SQLite database.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve(), args.audit.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
