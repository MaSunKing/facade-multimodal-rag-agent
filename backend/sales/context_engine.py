"""Query-aware, semantics-preserving context preparation.

The engine operates only on request-time Evidence views.  Canonical company
and customer-document Evidence remains untouched.  It deliberately uses
extractive, deterministic protection instead of a second compression model so
the local 16 GB GPU remains reserved for Qwen3-VL generation.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


_LATIN_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[./%-][A-Za-z0-9_]+)*")
_HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_NUMBER_UNIT = re.compile(
    r"(?P<value>-?\d+(?:[,.]\d+)?)\s*(?P<unit>mm|cm|m²|m2|㎡|m|kg/m²|kg/m2|kg|%|℃|°c|天|小时|元|万元|亿元)",
    re.IGNORECASE,
)
_FACT_TRIPLE = re.compile(
    r"(?P<metric>[A-Za-z\u3400-\u4dbf\u4e00-\u9fff][A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff（）()_\-/ ]{1,24}?)"
    r"\s*(?:为|是|[:：=])?\s*"
    r"(?P<value>-?\d+(?:[,.]\d+)?)\s*"
    r"(?P<unit>mm|cm|m²|m2|㎡|m|kg/m²|kg/m2|kg|%|℃|°c|天|小时|元|万元|亿元)",
    re.IGNORECASE,
)
_CONDITION_MARKERS = (
    "不得", "禁止", "不应", "不可", "不能", "仅限", "仅适用", "只适用", "除非",
    "条件下", "前提", "应当", "必须", "建议", "宜", "严禁", "not ", "must ",
    "only ", "unless ", "except ",
)


@dataclass(frozen=True)
class ContextBudget:
    """Independent candidate, prompt, visual and output budgets."""

    candidate_text_tokens: int
    max_prompt_tokens: int
    max_images: int
    max_output_tokens: int
    components: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plan_value(plan: Any, name: str, default: Any = None) -> Any:
    if isinstance(plan, dict):
        return plan.get(name, default)
    return getattr(plan, name, default)


def choose_context_budget(
    plan: Any,
    *,
    has_documents: bool,
    has_image: bool,
    prompt_ceiling: int = 5_800,
    source_document_count: int = 1,
) -> ContextBudget:
    """Build a demand-based budget without using file size as importance.

    Candidate retrieval is deliberately wider than the final prompt.  This
    lets the CPU-side selector compare more Evidence while the exact tokenizer
    keeps Qwen3-VL inside the safe local inference ceiling.
    """

    tools = set(_plan_value(plan, "tools", []) or [])
    document_scope = str(_plan_value(plan, "document_scope", "unknown") or "unknown")
    target_terms = list(_plan_value(plan, "target_terms", []) or [])
    wants_visuals = bool(_plan_value(plan, "wants_visuals", False))
    components = {
        "base": 3_500,
        "customer_documents": 2_500 if has_documents else 0,
        "company_rag": 1_500 if "company_rag" in tools else 0,
        "public_web": 750 if "public_web_search" in tools else 0,
        "cross_document": 1_250 if document_scope == "cross_document" else 0,
        "whole_document": 1_500 if document_scope == "whole_document" else 0,
        "target_coverage": min(1_000, 200 * len(target_terms)),
    }
    candidate_tokens = min(14_000, max(4_000, sum(components.values())))

    source_count = sum(
        1
        for present in (
            has_documents,
            "company_rag" in tools,
            "public_web_search" in tools,
        )
        if present
    )
    desired_prompt = 4_200 + max(0, source_count - 1) * 550
    if document_scope in {"cross_document", "whole_document"}:
        desired_prompt += 450
    if len(target_terms) >= 3:
        desired_prompt += 250
    prompt_tokens = min(max(2_048, int(prompt_ceiling)), desired_prompt)
    # Uploaded PDFs/Word/Excel files may contain a scanned page or chart even
    # when the user asks for an answer rather than explicitly asking to see an
    # image.  Keep a small visual-reading allowance for document QA.
    # A direct image must not reduce the retrieved attachment-image allowance.
    # Final merging/dedup still caps total images at four and shares pixels.
    max_images = 2 if has_documents else 1 if has_image else 2 if wants_visuals else 0
    if wants_visuals and document_scope == "cross_document":
        max_images = 3
    max_images = min(4, max_images)
    # The 8B 4-bit model runs on one 16 GB card and responses must finish
    # within the interactive budget.  The prompts already require compact,
    # non-repetitive JSON, so 520/620 tokens preserve a useful analysis while
    # preventing an overly verbose tail from monopolising the GPU.
    output_tokens = 620 if document_scope in {"cross_document", "whole_document"} else 520
    if "company_rag" in tools and not has_documents:
        output_tokens = 620
    if document_scope in {"cross_document", "whole_document"}:
        output_tokens += 90 * max(0, min(4, source_document_count) - 2)
    aspects = list(_plan_value(plan, 'answer_aspects', []) or _plan_value(plan, 'answer_goals', []) or [])
    if len(aspects) > 3:
        # More requested dimensions need answer space, not additional images
        # or larger source prompts. Remain bounded on the 16 GB GPU.
        output_tokens = min(1040, max(output_tokens, 520 + 70 * len(aspects)))
    return ContextBudget(
        candidate_text_tokens=candidate_tokens,
        max_prompt_tokens=prompt_tokens,
        max_images=max_images,
        max_output_tokens=output_tokens,
        components=components,
    )


def _terms(text: str) -> set[str]:
    value = str(text or "").casefold()
    terms = {token for token in _LATIN_TOKEN.findall(value) if len(token) >= 2}
    for run in _HAN_RUN.findall(value):
        if len(run) <= 4:
            terms.add(run)
        terms.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return terms


def _normalised_text(text: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", "", text.casefold())


def _source_type(item: dict[str, Any]) -> str:
    explicit = str(item.get("source_type") or "").strip()
    if explicit:
        return explicit
    evidence_id = str(item.get("evidence_id") or "")
    return {
        "U": "customer_document",
        "T": "company_rag",
        "V": "visual",
        "W": "public_web",
        "S": "catalog_sql",
    }.get(evidence_id[:1], "unknown")


def _protected_relations(item: dict[str, Any], query_terms: set[str]) -> list[str]:
    text = str(item.get("text") or "")
    relations: list[str] = []
    if _NUMBER_UNIT.search(text) and query_terms & _terms(text):
        relations.append("entity_value")
    if (
        item.get("source_group")
        or item.get("original_chunk_id")
    ) and any(marker in text for marker in ("[ROW", "[COLUMNS]", "[TABLE", "Sheet", "sheet=")):
        relations.append("table_row")
    lowered = text.casefold()
    if any(marker in lowered for marker in _CONDITION_MARKERS):
        relations.append("condition")
    if _source_type(item) == "visual" or item.get("visual_id"):
        relations.append("visual_binding")
    return relations


def _fact_triples(text: str) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for match in _FACT_TRIPLE.finditer(text):
        metric = re.sub(r"\s+", "", match.group("metric"))[-16:].casefold()
        value = match.group("value").replace(",", "")
        unit = match.group("unit").casefold().replace("m2", "m²").replace("㎡", "m²")
        triples.append((metric, value, unit))
    return triples


def _authority_bonus(source_type: str) -> float:
    # Uploaded evidence and approved local RAG are facts; web and SQL are
    # useful candidates but must retain links to their original source.
    return {
        "customer_document": 0.30,
        "company_rag": 0.25,
        "catalog_sql": 0.12,
        "public_web": 0.06,
        "visual": 0.08,
    }.get(source_type, 0.0)


def _structured_authority_bonus(item: dict[str, Any]) -> float:
    """Prefer reviewed government/standards metadata within one RAG source."""

    authority = str(item.get("source_authority") or "")
    if authority.startswith("T1_"):
        return 0.16
    if authority in {"reviewed_company_catalog", "company_reviewed_material"}:
        return 0.08
    return 0.0


def _target_anchor_present(target: str, text: str) -> bool:
    """Normalised lexical anchor coverage; never used as semantic gold."""
    from backend.sales.recovery_policy import normalise_anchor
    if normalise_anchor(target) in normalise_anchor(text):
        return True
    terms = _terms(target)
    # Word-separated English labels can appear across PDF line breaks. Han
    # synonyms require semantic planning; do not equate arbitrary characters.
    hits = terms & _terms(text)
    return bool(terms and re.fullmatch(r'[\x00-\x7f]+', target)
                and len(hits) >= max(1, math.ceil(len(terms)*2/3)))


def optimise_evidence_context(
    question: str,
    evidence: Iterable[dict[str, Any]],
    *,
    target_terms: Iterable[str] = (),
    wants_visuals: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse, deduplicate, protect and rank already-retrieved Evidence.

    Scores from BM25, dense retrieval, SQL and visual similarity are never
    compared directly.  Source-local rank is converted to an RRF-like base,
    followed by transparent query/authority/relation bonuses.
    """

    query_terms = _terms(question)
    from backend.sales.recovery_policy import content_search_anchor
    explicit_targets = list(dict.fromkeys(content_search_anchor(str(term)) for term in target_terms
                                         if content_search_anchor(str(term))))
    source_ranks: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    navigation_ids: list[str] = []
    for ordinal, raw in enumerate(evidence, start=1):
        item = dict(raw)
        evidence_id = str(item.get("evidence_id") or f"E{ordinal}")
        if item.get('evidence_role') == 'navigation' or item.get('evidence_scope') == 'document_index':
            navigation_ids.append(evidence_id)
            continue
        item["evidence_id"] = evidence_id
        source = _source_type(item)
        source_ranks[source] = source_ranks.get(source, 0) + 1
        local_rank = source_ranks[source]
        text = str(item.get("text") or "")
        # Window construction separates factual row/section relevance from
        # repeated native headers. Keep those headers in text, but do not let
        # their duplicated keywords dominate the second-stage score.
        text_terms = _terms(str(item.get('ranking_text') or text))
        overlap = query_terms & text_terms
        relation_types = _protected_relations(item, query_terms)
        exact_target_hits = sum(1 for term in explicit_targets if _target_anchor_present(term, text))
        retrieval_aspect = str(item.get("retrieval_aspect") or "").strip()
        # Scaled reciprocal rank stays source-comparable without pretending
        # that heterogeneous raw retrieval scores share one probability scale.
        rrf = 60.0 / (60.0 + local_rank)
        semantic_bonus = min(0.90, sum(0.08 + math.log1p(len(term)) * 0.04 for term in overlap))
        # A field name in a header plus an actual matching entity/row value is
        # stronger support than a header alone. This feature is format-based,
        # not tied to particular metrics, products or benchmark questions.
        binding_bonus = 0.0
        if '[TABLE_CONTEXT' in text:
            body, headers = text.split('[TABLE_CONTEXT', 1)
            row_values = re.findall(r"='([^']*)'", '\n'.join(line for line in body.splitlines() if line.startswith('[ROW')))
            has_number = any(re.fullmatch(r'-?\d+(?:,\d{3})*(?:\.\d+)?', value.strip()) for value in row_values)
            row_terms = _terms(' '.join(value for value in row_values if not re.fullmatch(r'[\d., -]+', value)))
            if has_number and query_terms & _terms(headers):
                binding_bonus = min(1.2, 0.4 * len(query_terms & row_terms))
        target_bonus = min(
            0.90,
            exact_target_hits * 0.30
            + (0.55 if retrieval_aspect and retrieval_aspect in explicit_targets else 0.0),
        )
        relation_bonus = min(0.35, len(relation_types) * 0.08)
        semantic_goals = dict(item.get('goal_support_scores') or {})
        semantic_goal_bonus = 4.0 * max(semantic_goals.values(), default=0.0)
        # Preserve explicit table/period constraints through the SECOND ranker;
        # otherwise lexical header noise can evict correctly retrieved rows.
        period_bonus = min(1.0, sum(0.5 for period in re.findall(r'\b\d{4}[/\-]\d{2,4}\b', question) if period in text))
        group = str(item.get('source_group') or '')
        group_bonus = 0.7 if group and re.search(r'(?<!\w)'+re.escape(group)+r'(?!\w)', question, re.IGNORECASE) else 0.0
        item.update(
            {
                "source_type": source,
                "source_local_rank": local_rank,
                "protected_relation_types": relation_types,
                "semantic_context_score": round(
                    rrf
                    + semantic_bonus
                    + binding_bonus
                    + semantic_goal_bonus
                    + target_bonus
                    + relation_bonus
                    + period_bonus
                    + group_bonus
                    + _authority_bonus(source)
                    + _structured_authority_bonus(item),
                    6,
                ),
                "packing_group_id": evidence_id,
                "original_ordinal": ordinal,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "retrieval_aspect": retrieval_aspect or None,
            }
        )
        candidates.append(item)

    candidates.sort(
        key=lambda item: (-float(item["semantic_context_score"]), int(item["original_ordinal"]))
    )
    deduplicated: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    removed_duplicates: list[str] = []
    fingerprint_items: dict[str, dict] = {}
    for item in candidates:
        fingerprint = _normalised_text(str(item.get("text") or ""))
        if len(fingerprint) >= 32 and fingerprint in seen_text:
            previous = fingerprint_items[fingerprint]
            previous['retrieval_aspects'] = list(dict.fromkeys([
                *previous.get('retrieval_aspects', []), *item.get('retrieval_aspects', []),
                *([item['retrieval_aspect']] if item.get('retrieval_aspect') else [])]))
            removed_duplicates.append(str(item["evidence_id"]))
            continue
        if fingerprint:
            seen_text.add(fingerprint)
            fingerprint_items[fingerprint] = item
        deduplicated.append(item)

    # Aspect recovery deliberately retrieves one small candidate set per
    # semantic sub-question (for example anchoring, fire and acceptance). Keep
    # the best representative of each aspect ahead of generic high-frequency
    # evidence so prompt packing cannot silently drop one requested dimension.
    # This is query-plan coverage, not a product-keyword route.
    aspect_representatives: list[dict[str, Any]] = []
    represented_aspects: set[str] = set()
    for item in deduplicated:
        aspects = set(item.get('retrieval_aspects') or []) | {str(item.get("retrieval_aspect") or "").strip()}
        aspects.discard('')
        if not aspects - represented_aspects:
            continue
        represented_aspects.update(aspects)
        aspect_representatives.append(item)
    if aspect_representatives:
        representative_ids = {str(item["evidence_id"]) for item in aspect_representatives}
        deduplicated = [
            *aspect_representatives,
            *(item for item in deduplicated if str(item["evidence_id"]) not in representative_ids),
        ]

    # Reserve a compact representative of each planned fact goal per source
    # document. This is retrieval coverage, NOT proof that the answer is there.
    # Publisher/navigation matches must not evict a metric-bearing window.
    goal_representatives = []
    seen_goal_ids = set()
    semantic_goal_winners = {}
    for item in deduplicated:
        for goal, score in (item.get('goal_support_scores') or {}).items():
            key = (str(item.get('document_name') or ''), goal)
            previous = semantic_goal_winners.get(key)
            if float(score) >= 0.1 and (previous is None or float(score) > previous[0]):
                semantic_goal_winners[key] = (float(score), str(item['evidence_id']))
    semantic_mode = any(item.get('goal_rerank_applied') for item in deduplicated)
    for item in deduplicated:
        document = str(item.get('document_name') or '')
        if not document or item.get('evidence_scope') == 'document_index':
            continue
        terms_for_item = ([goal for (doc, goal), (_, winner) in semantic_goal_winners.items()
                           if doc == document and winner == str(item['evidence_id'])]
                          if semantic_mode else [target for target in explicit_targets
                          if _target_anchor_present(target, str(item.get('text') or ''))])
        goal_ids = [f'{document}:{target}' for target in terms_for_item]
        new_ids = set(goal_ids)-seen_goal_ids
        if not new_ids:
            continue
        item['protected_goal_ids'] = goal_ids
        item['protected_goal_terms'] = terms_for_item
        item['goal_coverage_method'] = 'semantic_relevance_not_verified_fact' if semantic_mode else 'lexical'
        seen_goal_ids.update(new_ids)
        goal_representatives.append(item)
    if goal_representatives:
        reserved_ids = {str(item['evidence_id']) for item in goal_representatives}
        deduplicated = [*goal_representatives,
                        *(item for item in deduplicated if str(item['evidence_id']) not in reserved_ids)]

    # Bind entity + canonical metric + scope/version before comparing
    # normalised values. Preserve a potential disagreement atomically;
    # this does not assert which source is correct or applicable.
    from backend.sales.fact_normalization import disagreement_groups
    by_id = {str(item["evidence_id"]): item for item in deduplicated}
    conflict_groups: list[dict[str, Any]] = []
    for group in disagreement_groups(deduplicated):
        unique_members = group['evidence_ids']
        group_id = 'conflict_' + hashlib.sha1(repr(group).encode()).hexdigest()[:12]
        for evidence_id in unique_members:
            item = by_id[evidence_id]
            relations = list(item.get("protected_relation_types") or [])
            if "conflict_group" not in relations:
                relations.append("conflict_group")
            item["protected_relation_types"] = relations
            item["packing_group_id"] = group_id
        conflict_groups.append(
            {**group, "group_id": group_id}
        )

    # An evidence block may support several fact keys. Merge overlapping
    # groups for atomic packing instead of overwriting its last group ID.
    components: list[set[str]] = []
    for group in conflict_groups:
        members = set(group['evidence_ids'])
        touching = [part for part in components if part & members]
        for part in touching:
            members |= part
            components.remove(part)
        components.append(members)
    for members in components:
        packing_id = 'conflict_pack_' + hashlib.sha1('|'.join(sorted(members)).encode()).hexdigest()[:12]
        for evidence_id in members:
            by_id[evidence_id]['packing_group_id'] = packing_id

    selected_text = "\n".join(str(item.get("text") or "") for item in deduplicated)
    missing_targets = [term for term in explicit_targets if not _target_anchor_present(term, selected_text)]
    scored_goals = {goal for item in deduplicated for goal in (item.get('goal_support_scores') or {})}
    missing_answer_goals = [goal for goal in sorted(scored_goals)
                            if not any(g == goal for _, g in semantic_goal_winners)]
    visual_present = any(_source_type(item) == "visual" for item in deduplicated)
    audit = {
        "engine": "query_aware_multimodal_evidence_context_v1",
        "candidate_count": len(candidates),
        "excluded_navigation_evidence_ids": navigation_ids,
        "after_dedup_count": len(deduplicated),
        "removed_duplicate_evidence_ids": removed_duplicates,
        "source_candidate_counts": {
            source: sum(1 for item in deduplicated if _source_type(item) == source)
            for source in sorted({_source_type(item) for item in deduplicated})
        },
        "protected_relation_counts": {
            relation: sum(
                1 for item in deduplicated if relation in set(item.get("protected_relation_types") or [])
            )
            for relation in ("entity_value", "table_row", "condition", "visual_binding", "conflict_group")
        },
        "conflict_groups": conflict_groups,
        "target_terms": explicit_targets,
        "reserved_retrieval_aspects": sorted(represented_aspects),
        "reserved_goal_ids": sorted(seen_goal_ids),
        "goal_coverage_method": "semantic_relevance_not_verified_fact" if semantic_mode else "lexical",
        "missing_target_terms": missing_targets,
        "missing_answer_goals": missing_answer_goals,
        "visual_requested": wants_visuals,
        "visual_candidate_present": visual_present,
        "coverage_sufficient_before_generation": (not missing_answer_goals if semantic_mode else not missing_targets) and (not wants_visuals or visual_present),
        "ranking_policy": "source_local_rrf_plus_query_authority_relation_boost",
        "raw_scores_compared_across_sources": False,
    }
    return deduplicated, audit


def validate_packed_evidence(
    candidates: Iterable[dict[str, Any]],
    packed: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate IDs, content hashes and all-or-none protected groups."""

    candidate_list = list(candidates)
    packed_list = list(packed)
    candidate_by_id = {str(item.get("evidence_id") or ""): item for item in candidate_list}
    packed_ids = [str(item.get("evidence_id") or "") for item in packed_list]
    violations: list[dict[str, Any]] = []
    if len(packed_ids) != len(set(packed_ids)):
        violations.append({"type": "duplicate_evidence_id"})
    for item in packed_list:
        evidence_id = str(item.get("evidence_id") or "")
        original = candidate_by_id.get(evidence_id)
        if original is None:
            violations.append({"type": "unknown_evidence_id", "evidence_id": evidence_id})
            continue
        expected_hash = str(original.get("content_sha256") or "")
        actual_hash = hashlib.sha256(str(item.get("text") or "").encode("utf-8")).hexdigest()
        # Provenance-prefix compaction is allowed to change row text, so hash
        # checks apply only when the candidate text was copied verbatim.
        if expected_hash and str(item.get("text") or "") == str(original.get("text") or "") and actual_hash != expected_hash:
            violations.append({"type": "content_hash_mismatch", "evidence_id": evidence_id})

    groups: dict[str, set[str]] = {}
    for item in candidate_list:
        if "conflict_group" not in set(item.get("protected_relation_types") or []):
            continue
        group = str(item.get("packing_group_id") or "")
        groups.setdefault(group, set()).add(str(item.get("evidence_id") or ""))
    packed_set = set(packed_ids)
    for group, members in groups.items():
        present = members & packed_set
        if present and present != members:
            violations.append(
                {"type": "partial_conflict_group", "group_id": group, "expected": sorted(members), "present": sorted(present)}
            )
    return {
        "valid": not violations,
        "violation_count": len(violations),
        "violations": violations,
        "packed_evidence_count": len(packed_list),
        "protected_conflict_group_count": len(groups),
    }
