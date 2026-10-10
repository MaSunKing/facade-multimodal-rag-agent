"""Read-only structured retrieval for the sales product/project catalogue."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
CATALOG_DB_PATH = ROOT / "data" / "sales" / "processed" / "catalog" / "sales_catalog.sqlite3"


def choose_gallery_scope_ids(explicit_ids, resolved_ids, *, scope, target, resolution):
    """Reuse an already resolved referent, never broaden an unmatched SKU.

    Registered names/aliases are backend data. An empty explicit result alone
    does not authorize substituting a broader family or other products.
    """
    from backend.sales.relation_scope import normalized
    matches = [r for r in resolution.get('entity_targets', [])
               if normalized(target) and normalized(target) in
               {normalized(n) for n in [r.get('name', ''), *r.get('aliases', [])]}]
    if resolved_ids is not None and (explicit_ids is None or (scope == 'exact' and matches)):
        return resolved_ids
    return explicit_ids


def query_product_gallery(*, family='unknown', scope='unknown', target='',
                          allowed_access_scopes=('public',), database_path=None, entity_ids=None):
    """Source-labelled IDs only; model values are parameters, never SQL/paths.

    Category browsing is not name matching. Unknown/descriptive requests keep
    the existing retrieval path. Return the full small candidate pool before Top-K.
    """
    resolved = entity_ids is not None
    if resolved:
        entity_ids = sorted(set(entity_ids))
        if not entity_ids:return []
    if not resolved and (scope not in {'category', 'exact'} or family not in {'unknown','all','decorative_panel','insulated_panel'}):
        return None
    if not resolved and ((scope == 'category' and family == 'unknown') or (scope == 'exact' and not target.strip())):
        return None
    path=Path(database_path or CATALOG_DB_PATH)
    if not path.is_file():
        return None
    scopes=sorted(set(allowed_access_scopes))
    if not scopes:
        return []
    clauses=["a.category='product_sample'", 'a.access_scope IN ('+','.join('?' for _ in scopes)+')',
             "json_extract(l.labels_json,'$.product_gallery_eligible')=1",
             "json_extract(l.labels_json,'$.display_ready')=1",
             "json_array_length(json_extract(l.labels_json,'$.label_issues'))=0"]
    params=list(scopes)
    if resolved:
        # Directory family IDs represent a set of ACL-checked child products,
        # not literal product primary keys. Resolve their existing SQL edges.
        product_ids = [key for key in entity_ids if not key.startswith('family:')]
        families = [key.removeprefix('family:') for key in entity_ids if key.startswith('family:')]
        membership = []
        if product_ids:
            membership.append('p.product_id IN ('+','.join('?' for _ in product_ids)+')')
        if families:
            membership.append('p.family IN ('+','.join('?' for _ in families)+')')
        clauses.append('EXISTS (SELECT 1 FROM product_visuals pv JOIN products p USING(product_id) '
                       'WHERE pv.asset_id=a.asset_id AND p.access_scope IN ('+','.join('?' for _ in scopes)+') '
                       'AND ('+' OR '.join(membership)+'))')
        params.extend([*scopes,*product_ids,*families])
    elif family in {'decorative_panel','insulated_panel'}:
        clauses.append("json_extract(l.labels_json,'$.product_family_key')=?")
        params.append(family)
    if not resolved and scope == 'exact':
        # A literal named entity, never wildcard expansion or generic query words.
        clauses.append("(instr(lower(replace(coalesce(json_extract(l.labels_json,'$.variant_or_code'),''),' ','')),?)>0 "
                       "OR instr(lower(replace(coalesce(json_extract(l.labels_json,'$.product_name'),''),' ','')),?)>0)")
        value=re.sub(r'\s+', '', target).casefold()
        params.extend([value,value])
    with _connect(path) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='visual_asset_labels'").fetchone():
            return None
        if resolved and not all(connection.execute("SELECT 1 FROM sqlite_master WHERE name=?",(t,)).fetchone()
                                for t in ('product_visuals','products')):
            return None
        rows=connection.execute('SELECT a.asset_id,l.labels_json FROM visual_assets a '
            'JOIN visual_asset_labels l ON l.asset_id=a.asset_id WHERE '+' AND '.join(clauses)
            +' ORDER BY a.asset_id',params).fetchall()
    # One representative per variant/product before repeated source images.
    unique,repeated,groups,hashes=[],[],set(),set()
    for row in rows:
        from backend.sales.catalog_repairs import policy
        if row['asset_id'] in policy().get('excluded_visual_assets',{}):
            continue
        labels=json.loads(row['labels_json'])
        digest=labels.get('sha256') or row['asset_id']
        if digest in hashes:
            continue
        hashes.add(digest)
        group=(labels.get('variant_or_code') or labels.get('product_name'),labels.get('visual_role'))
        (repeated if group in groups else unique).append(row['asset_id'])
        groups.add(group)
    return unique+repeated


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


def _case_citations(connection: sqlite3.Connection, row: sqlite3.Row) -> list[dict[str, Any]]:
    import json
    refs = _citation(row, "官方网站项目案例") if row["source_url"] else []
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='case_provenance'").fetchone()
    if exists:
        records = connection.execute("SELECT source_json FROM case_provenance WHERE case_id=?", (row["case_id"],)).fetchall()
        for record in records:
            source = json.loads(record[0])
            if not source.get("source_pages") and not source.get("source_page"):
                continue
            pages = source.get("source_pages") or [source.get("source_page")]
            for page in pages:
                refs.append({"document_name":source.get("source_document") or row["project_name"],
                    "source_page":page,"section_heading":row["project_name"],
                    "source_type":"enterprise_catalogue_case_reference"})
        if refs:
            return refs
    return _citation(row, "企业画册项目案例")


def _product_citations(connection, row):
    if row['source_url']:
        return _citation(row,'官方网站产品目录')
    tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if {'product_evidence','knowledge_evidence_sources','knowledge_text_evidence'} <= tables:
        refs=[]
        for record in connection.execute('''SELECT DISTINCT s.reference_json FROM product_evidence p
            JOIN knowledge_evidence_sources s USING(evidence_id)
            JOIN knowledge_text_evidence e USING(evidence_id)
            WHERE p.product_id=? AND p.relation='catalog_color_definition' AND e.access_scope='public' ''',(row['product_id'],)):
            refs.append({**json.loads(record[0]),'source_type':'enterprise_source_catalogue'})
        if refs:
            return refs
    return _citation(row,'企业产品目录')


def retrieve_catalog_evidence(
    query: str,
    *,
    target_terms: Iterable[str] = (),
    case_filters: dict[str, Any] | None = None,
    include_products: bool = True,
    include_project_cases: bool = True,
    limit: int = 6,
    database_path: Path = CATALOG_DB_PATH,
    directory_scope: dict[str, Any] | None = None,
    operation: str = 'lookup',
    knowledge_categories: Iterable[str] = (),
    collection: dict[str, Any] | None = None,
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
    output: list[dict[str, Any]] = []
    with _connect(database_path) as connection:
        if operation == 'summary' and directory_scope is not None:
            from backend.sales.directory_evidence import from_directory
            output.extend(from_directory(connection, directory_scope, knowledge_categories, collection, question=query, operation=operation))
        elif include_products and directory_scope is not None and '02_product_information' in (knowledge_categories or []):
            from backend.sales.directory_evidence import from_directory
            output.extend(from_directory(connection,directory_scope,['02_product_information'],
                question=query,operation=operation))
        # Trusted directory aliases bind a product-form collection even when a
        # compact Planner omits target_terms. No keyword-specific task router.
        from backend.sales.relation_scope import normalized
        has_aliases = connection.execute("SELECT 1 FROM sqlite_master WHERE name='technical_family_aliases'").fetchone()
        families = sorted({r['family'] for r in connection.execute('SELECT alias,family FROM technical_family_aliases')
                           if normalized(r['alias']) in normalized(query)}) if has_aliases and include_products else []
        if families and directory_scope is None:
            members = [r for family in families for r in connection.execute(
                "SELECT * FROM products WHERE access_scope='public' AND family=? ORDER BY name", (family,))]
            if members:
                refs = [ref for row in members for ref in _product_citations(connection,row)]
                output.append({'evidence_id':'S'+hashlib.sha256(repr([r['product_id'] for r in members]).encode()).hexdigest()[:12],
                    'catalog_record_type':'product_collection','source_type':'catalog_sql','facts_eligible':True,
                    'entity_coverage_ids':[r['product_id'] for r in members],
                    'text':'目录筛选范围：'+ '、'.join(families)+'。已登记成员（不推断未登记组合）：'+
                        '；'.join(r['name'] for r in members)+'。',
                    'citations':list({json.dumps(ref,ensure_ascii=False,sort_keys=True):ref for ref in refs}.values())})
        # Identity is a small cited baseline inside an already selected company
        # tool, not a new route. Older catalogue schemas remain supported.
        has_identity = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='company_profile'").fetchone()
        if has_identity:
            for row in connection.execute("SELECT * FROM company_profile WHERE access_scope='public'"):
                profile = json.loads(row["profile_json"])
                output.append({"evidence_id":"S"+hashlib.sha1(row["company_id"].encode()).hexdigest()[:12],
                    "catalog_record_type":"company_identity", "record_id":row["company_id"],
                    "source_type":"catalog_sql", "facts_eligible":True,
                    "text": f"厂家：{row['legal_name']}；品牌：{'、'.join(profile['brand_names'])}；所在地：{row['province']}{row['city']}{row['county']}。" +
                        ("" if families else f"产品系列：{'、'.join(profile['product_families'])}。"),
                    "citations":[{"document_name":row["legal_name"],"source_url":row["source_url"],"source_page":None,"section_heading":"官网企业身份档案"}]})
        if operation == 'summary' and directory_scope is not None:
            from backend.sales.knowledge_scope_notes import enrich_scope_notes
            return enrich_scope_notes(output, database_path)  # No global Top-K of members.
        if not bounded_terms:
            return output[:limit]
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
                if families and row['family'] not in families:
                    continue  # Bounded aliases cannot be expanded by a generic search term.
                evidence_id = "S" + hashlib.sha1(f"product|{row['product_id']}".encode()).hexdigest()[:12]
                introduction = clean_catalog_introduction(
                    row["introduction"], product_name=str(row["name"])
                ).rstrip("。；; ")
                from backend.sales.finish_options import finish_option_text
                from backend.sales.product_source_profiles import read_product_profile, product_profile_text, product_profile_citations
                profile = read_product_profile(connection, row['product_id'])
                text = "；".join(
                    part
                    for part in (
                        f"产品：{row['name']}",
                        f"系列：{row['family']}",
                        f"产品介绍：{introduction}" if introduction else "",
                        f"饰面效果：{row['surface_finish']}" if row["surface_finish"] else "",
                        finish_option_text(connection, row['product_id']),
                        f"标准规格：{profile.get('standard_specification') or row['standard_specification']}" if profile.get('standard_specification') or row["standard_specification"] else "",
                        f"板材厚度：{profile.get('panel_thickness') or row['panel_thickness']}" if profile.get('panel_thickness') or row["panel_thickness"] else "",
                        f"重量：{profile.get('weight_kg_m2') or row['weight_kg_m2']}kg/m²" if profile.get('weight_kg_m2') or row["weight_kg_m2"] else "",
                        product_profile_text(connection, row['product_id']),
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
                        "citations": [*_product_citations(connection,row), *product_profile_citations(connection,row['product_id'])],
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
            from backend.sales.project_product_links import read_links
            product_links=read_links(connection,row['case_id'])
            field_provenance = None
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='project_field_supplements'").fetchone():
                supplied = connection.execute('SELECT provenance_json FROM project_field_supplements WHERE case_id=?', (row['case_id'],)).fetchone()
                if supplied:
                    field_provenance = json.loads(supplied[0])
            evidence_id = "S" + hashlib.sha1(f"case|{row['case_id']}".encode()).hexdigest()[:12]
            conflict_note = ""
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='case_conflicts'").fetchone():
                conflicts = connection.execute("SELECT field,values_json FROM case_conflicts WHERE case_id=?", (row["case_id"],)).fetchall()
                if conflicts:
                    conflict_note = "原始案例记录存在差异，不能选一个数值作为确定事实：" + "；".join(field+"="+" / ".join(json.loads(values)) for field,values in conflicts)
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
                    conflict_note,
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
                    "catalog_field_provenance": field_provenance,
                    "facts_eligible": True,
                    "associated_products": [
                        {"product_id": link['product_id'], "product_name": link['product_name']}
                        for link in product_links
                    ],
                    "citations": _case_citations(connection, row),
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
    from backend.sales.knowledge_scope_notes import enrich_scope_notes
    return enrich_scope_notes(unique, database_path)
