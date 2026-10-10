"""Lossless index compaction: never summarize, fuzzy-merge or cross ACLs."""
from __future__ import annotations

import copy
import json
import re
from typing import Any


def comparison_text(text: str) -> str:
    # PDF layouts insert spaces between Chinese glyphs. Keep English word
    # boundaries, punctuation, digits, negation and units unchanged.
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])", "", text)


def compact_documents(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    kept: list[dict[str, Any]] = []
    groups: dict[tuple, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for original in documents:
        row = copy.deepcopy(original)
        # Image IDs and case records carry different source/context links even
        # when their bytes or descriptions match. SQL already consolidates
        # files and canonical cases; don't destroy their provenance here.
        if row.get("kind") != "text":
            kept.append(row)
            continue
        text_key = comparison_text(str(row.get("text") or ""))
        navigation_heading = bool(re.fullmatch(r"\d+(?:\.\d+)* ?[\u3400-\u9fff、，]{2,25}", text_key))
        # A repeated short table cell may belong to a different entity/test.
        # Keep its page/heading context unless it is just a numbered heading.
        context = [] if len(text_key) >= 60 or navigation_heading else sorted({
            (str(r.get("source_page")), str(r.get("section_heading")),
             str(r.get("source_sheet")), str(r.get("source_range")))
            for r in row.get("source_refs") or []})
        policy = json.dumps({
            "short_context": context,
            "scope": row.get("access_scope", "public"),
            "categories": sorted(row.get("document_categories") or []),
            "labels": sorted(row.get("content_labels") or []),
            "playbook": row.get("sales_playbook_use"),
            "authority": sorted({str(t.get("source_authority") or "")
                for t in row.get("source_taxonomy") or [] if isinstance(t, dict)}),
            # Different documents can contain identical short table cells,
            # but their entities and applicability are not interchangeable.
            "documents": sorted({str(r.get("document_id") or r.get("source_url") or r.get("document_name") or "")
                for r in row.get("source_refs") or []}),
        }, ensure_ascii=False, sort_keys=True)
        key = (policy, text_key)
        if not key[1] or key not in groups:
            groups[key] = row
            kept.append(row)
            continue
        canonical = groups[key]
        aliases[str(row["id"])] = str(canonical["id"])
        variants = canonical.setdefault("duplicate_variants", [])
        variants.append({"id": row["id"], "text": row["text"], "source_refs": row.get("source_refs") or []})
        for field in ("source_refs", "source_taxonomy", "knowledge_domains", "content_labels"):
            values = canonical.setdefault(field, [])
            for value in row.get(field) or []:
                if value not in values:
                    values.append(value)
    return kept, aliases
