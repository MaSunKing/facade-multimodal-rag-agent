"""Local, traceable retrieval over prepared construction RAG evidence.

The first retrieval layer is intentionally lexical BM25.  It needs no cloud
service, no GPU and no embedding model at request time.  Every returned item
preserves the document and page needed for a customer-visible citation.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAG_INDEX_PATH = ROOT / "data" / "sales" / "processed" / "rag_index" / "lexical_index.json"
DENSE_INDEX_PATH = ROOT / "data" / "sales" / "processed" / "rag_index" / "dense_index.npz"
DENSE_INDEX_METADATA_PATH = ROOT / "data" / "sales" / "processed" / "rag_index" / "dense_index_metadata.json"
PRODUCT_ALIASES_PATH = ROOT / "data" / "sales" / "config" / "product_aliases_v1.json"
RRF_K = 60
HYBRID_RULE_SCALE = 900.0
HYBRID_RULE_MIN = -0.25
HYBRID_RULE_MAX = 0.4
GENERIC_VISUAL_BIGRAMS = {
    "安装", "施工", "示意", "意图", "节点", "构造", "工艺", "流程", "做法", "方法", "步骤",
    "系统", "材料", "板材", "装饰", "保温", "图片", "图纸", "什么", "怎么", "是否",
}
REQUIRED_VISUAL_PHRASES = (
    "阳角", "阴角", "岩棉", "粘锚", "穿透", "龙骨", "女儿墙", "门窗", "开槽", "锚固", "干挂",
)
CONSTRUCTION_QUERY_EXPANSION = "粘锚 干挂 穿透 保温装饰一体板 基层 锚固 施工方法"
RETRIEVAL_MODES = {
    "factual_lookup",
    "procedure",
    "node_detail",
    "case_reference",
    "comparison",
    "project_fit",
    "commercial",
    "unknown",
}
VISUAL_SCOPES = {"product", "case", "node", "process", "mixed"}
PRODUCT_OVERVIEW_INVENTORY_MARKERS = (
    "有什么产品",
    "有哪些产品",
    "产品有哪些",
    "产品种类",
    "产品体系",
    "产品总览",
    "产品介绍",
    "介绍产品",
    "介绍一下产品",
    "介绍一下你们的产品",
    "介绍一下你的产品",
)
PRODUCT_OVERVIEW_DETAIL_MARKERS = ("详细介绍", "具体介绍", "展开介绍", "详细说说")
PRODUCT_OVERVIEW_SUBJECT_MARKERS = ("产品", "真岩", "无机仿石", "一体板")
PRODUCT_OVERVIEW_SECTION_MARKERS = ("产品体系", "产品总览", "产品线", "产品分类", "产品目录")
PRODUCT_OVERVIEW_DISCOVERY_ACTIONS = (
    "想了解", "了解一下", "介绍", "介绍一下", "看看", "想看看",
)
PRODUCT_OVERVIEW_COMPANY_SUBJECTS = (
    "你们的产品", "你们公司的产品", "公司的产品", "公司产品", "企业产品", "产品体系",
)
NODE_ATLAS_QUERY_MARKERS = (
    "节点", "图集", "门窗", "洞口", "窗口", "阴角", "阳角", "勒脚", "女儿墙", "檐口", "收口",
)
STANDARD_QUERY_MARKERS = ("规范", "标准", "jgj", "jgt", "jg/t", "验收", "性能指标")
SOURCE_NAME_GENERIC_BIGRAMS = {
    "真岩", "无机", "仿石", "石材", "保温", "装饰", "一体", "施工", "方案",
    "应用", "技术", "工程", "产品", "材料", "外墙",
}
RARE_QUERY_GENERIC_BIGRAMS = {
    "什么", "怎么", "哪些", "多少", "分别", "提出", "原则", "要求", "应如", "如何",
    "规范", "标准", "方案", "材料", "施工", "项目",
}
PRODUCT_VISUAL_GENERIC_BIGRAMS = {
    "真岩", "岩石", "产品", "图片", "照片", "展示", "查看", "看看", "样板", "板材",
    "饰面", "装饰", "保温", "一体", "外墙", "相关", "有没有", "有图",
}


def reciprocal_rank_fusion(
    lexical: list[tuple[float, dict[str, Any]]],
    dense_ids: list[str],
    document_by_id: dict[str, dict[str, Any]],
    *,
    rank_constant: int = RRF_K,
) -> list[tuple[float, dict[str, Any]]]:
    """Fuse lexical and dense ranks without favouring list insertion order."""

    fused: dict[str, float] = {}
    for rank, (_, document) in enumerate(lexical, start=1):
        document_id = str(document.get("id") or "")
        if document_id:
            fused[document_id] = fused.get(document_id, 0.0) + 1.0 / (rank_constant + rank)
    for rank, document_id in enumerate(dense_ids, start=1):
        document_id = str(document_id)
        if document_id in document_by_id:
            fused[document_id] = fused.get(document_id, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(
        (
            (score, document_by_id[document_id])
            for document_id, score in fused.items()
            if document_id in document_by_id
        ),
        key=lambda item: (item[0], str(item[1].get("id") or "")),
        reverse=True,
    )


def select_rerank_candidates(
    fused: list[tuple[float, dict[str, Any]]],
    lexical_ids: list[str],
    dense_ids: list[str],
    document_by_id: dict[str, dict[str, Any]],
    *,
    non_visual_limit: int = 24,
    visual_limit: int = 8,
    lexical_reserve: int = 8,
    dense_only_reserve: int = 4,
) -> list[tuple[float, dict[str, Any]]]:
    """Select a balanced rerank pool from explicit lexical and dense ranks.

    RRF scores do not encode which retriever contributed a result.  Inferring
    source membership from a score threshold can therefore drop a strong
    lexical-only result when many documents occur in both rank lists.  Keep
    small, explicit reserves for the lexical head and for dense-only recall,
    then fill the rest from the fused ranking.
    """

    fusion_by_id = {str(document.get("id")): score for score, document in fused}
    lexical_ranked_ids = [str(document_id) for document_id in lexical_ids]
    lexical_id_set = set(lexical_ranked_ids)
    dense_only_ids = [
        str(document_id)
        for document_id in dense_ids
        if str(document_id) in document_by_id and str(document_id) not in lexical_id_set
    ]

    def choose(
        kind_visual: bool,
        limit: int,
        *,
        lexical_quota: int,
        dense_only_quota: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        ranked = [item for item in fused if (item[1].get("kind") == "visual") == kind_visual]
        selected: list[tuple[float, dict[str, Any]]] = []
        selected_ids: set[str] = set()

        def add_reserved(ranked_ids: list[str], quota: int) -> None:
            added = 0
            for document_id in ranked_ids:
                if len(selected) >= limit or added >= quota:
                    break
                document = document_by_id.get(document_id)
                if document is None or (document.get("kind") == "visual") != kind_visual:
                    continue
                if document_id in selected_ids:
                    continue
                selected.append((fusion_by_id.get(document_id, 0.0), document))
                selected_ids.add(document_id)
                added += 1

        add_reserved(lexical_ranked_ids, max(0, lexical_quota))
        add_reserved(dense_only_ids, max(0, dense_only_quota))
        for item in ranked:
            if len(selected) >= limit:
                break
            if str(item[1].get("id")) not in selected_ids:
                selected.append(item)
                selected_ids.add(str(item[1].get("id")))

        # Preserve fused order for stable reranker input while retaining the
        # reserved membership chosen above.
        fused_position = {
            str(document.get("id")): position
            for position, (_, document) in enumerate(ranked)
        }
        selected.sort(
            key=lambda item: fused_position.get(str(item[1].get("id")), len(ranked))
        )
        return selected

    return [
        *choose(
            False,
            non_visual_limit,
            lexical_quota=lexical_reserve,
            dense_only_quota=dense_only_reserve,
        ),
        *choose(
            True,
            visual_limit,
            lexical_quota=min(2, lexical_reserve),
            dense_only_quota=min(2, dense_only_reserve),
        ),
    ]


def tokenize(text: str) -> list[str]:
    """Tokenise mixed Chinese / dimensions for a small local BM25 index."""

    # Product catalogues use registered marks inside product names (for
    # example 真岩®石).  Treat them as formatting, not token boundaries, so a
    # customer's “真岩石” query matches the source document.
    text = (
        text.lower()
        .replace("®", "")
        .replace("™", "")
        .replace("©", "")
        .replace("㎜", "mm")
    )
    tokens = re.findall(r"[a-z0-9]+(?:[×*./-][a-z0-9]+)*", text)
    for group in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(group)
        tokens.extend(group[index : index + 2] for index in range(len(group) - 1))
    return tokens


def load_product_alias_groups(path: Path = PRODUCT_ALIASES_PATH) -> list[list[str]]:
    """Load audited product-name equivalence groups used only for query recall."""

    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups: list[list[str]] = []
    for item in payload.get("products", []):
        if not isinstance(item, dict):
            continue
        names = [item.get("canonical_name"), *(item.get("aliases") or [])]
        cleaned = [str(name).strip() for name in names if isinstance(name, str) and name.strip()]
        if cleaned:
            groups.append(list(dict.fromkeys(cleaned)))
    return groups


def expand_product_aliases(query: str, groups: list[list[str]]) -> str:
    """Append only the canonical name when the query names an audited alias.

    Adding every alias would let product-name tokens overwhelm the customer's
    actual intent terms (for example ``售后保障``), causing comparison rows to
    crowd out the specifically requested evidence.
    """

    compact_query = re.sub(r"\s+", "", query).replace("®", "")
    matched_canonical_names: list[str] = []
    for group in groups:
        canonical_compact = re.sub(r"\s+", "", group[0]).replace("®", "")
        matched_names = [
            re.sub(r"\s+", "", name).replace("®", "")
            for name in group
            if re.sub(r"\s+", "", name).replace("®", "") in compact_query
        ]
        if matched_names and not any(name in canonical_compact for name in matched_names):
            matched_canonical_names.append(group[0])
    return " ".join([query, *dict.fromkeys(matched_canonical_names)])


def specific_chinese_bigrams(text: str) -> set[str]:
    """Return the subject-bearing query bigrams used to avoid wrong diagrams."""

    bigrams: set[str] = set()
    for group in re.findall(r"[\u4e00-\u9fff]+", text):
        for index in range(len(group) - 1):
            bigram = group[index : index + 2]
            if bigram not in GENERIC_VISUAL_BIGRAMS:
                bigrams.add(bigram)
    return bigrams


def required_visual_phrases(text: str) -> set[str]:
    """Domain terms that must occur in a returned visual's own label."""

    return {phrase for phrase in REQUIRED_VISUAL_PHRASES if phrase in text}


def is_product_overview_query(text: str) -> bool:
    """Recognise broad catalogue questions, including contextual follow-ups.

    ``text`` may contain recent customer turns followed by the current short
    question.  A bare “详细介绍” is therefore treated as a product overview
    only when the combined context also contains a product subject.
    """

    compact = re.sub(r"\s+", "", text).lower()
    has_model_code = bool(re.search(r"[a-z]{1,8}[-_]?[a-z0-9]*\d{2,}[a-z0-9-]*", compact))
    if has_model_code:
        return False
    if any(marker in compact for marker in PRODUCT_OVERVIEW_INVENTORY_MARKERS):
        return True
    # Broad discovery phrasing is an overview only when the action directly
    # targets the company's product inventory/system.  A model code or a named
    # product between “公司的” and “产品” therefore remains a specific lookup.
    # Explicit alphanumeric model codes are a final fail-closed guard.
    if any(action in compact for action in PRODUCT_OVERVIEW_DISCOVERY_ACTIONS):
        if any(subject in compact for subject in PRODUCT_OVERVIEW_COMPANY_SUBJECTS):
            return True
    for marker in PRODUCT_OVERVIEW_DETAIL_MARKERS:
        if marker not in compact:
            continue
        residual = compact.replace(marker, "")
        for filler in ("请", "一下", "给我", "你们的", "你们公司的", "公司的"):
            residual = residual.replace(filler, "")
        return residual in {
            "产品", "产品体系", "产品总览", "产品线", "产品分类", "产品目录",
            "真岩", "真岩产品", "无机仿石", "一体板",
        }
    return False


class LocalRagRetriever:
    """Read-only RAG index with text evidence and original visual assets."""

    def __init__(
        self,
        index_path: Path = RAG_INDEX_PATH,
        *,
        allowed_access_scopes: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.index_path = index_path
        if not index_path.exists():
            raise FileNotFoundError(
                "未找到本地 RAG 索引。请先运行 scripts\\build_rag_index.ps1。"
            )
        self.payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.allowed_access_scopes = frozenset(allowed_access_scopes or {"public"})
        raw_documents: list[dict[str, Any]] = self.payload.get("documents", [])
        # Semantic taxonomy (for example ``internal_sales_playbook``) is not
        # an access-control decision. Existing records predate this field and
        # are public by the business owner's explicit decision; future private
        # material must opt in with ``access_scope=internal``.
        self.documents = [
            document
            for document in raw_documents
            if str(document.get("access_scope") or document.get("visibility") or "public")
            in self.allowed_access_scopes
        ]
        # The repository was reorganised from data/processed to
        # data/sales/processed.  Older, otherwise valid indexes may still hold
        # absolute paths from before that move.  Resolve those paths at load
        # time so an index rebuild is not required merely to serve the original
        # evidence crop.
        for document in self.documents:
            if document.get("kind") == "visual" and document.get("image_path"):
                document["image_path"] = str(self._resolve_visual_path(str(document["image_path"])))
        self.document_frequency = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document.get("tokens") or []))
        self.avg_doc_length = (
            sum(len(document.get("tokens") or []) for document in self.documents)
            / max(len(self.documents), 1)
        ) or 1.0
        self.document_count = len(self.documents)
        self.product_alias_groups = load_product_alias_groups()
        self.visual_by_asset_id = {
            str(document["asset_id"]): document
            for document in self.documents
            if document.get("kind") == "visual" and document.get("asset_id")
        }
        self.project_case_by_id = {
            str(document["id"]): document
            for document in self.documents
            if document.get("kind") == "project_case"
        }
        self.document_position_by_id = {
            str(document.get("id")): position
            for position, document in enumerate(self.documents)
            if document.get("id")
        }
        self._dense_ids: list[str] = []
        self._dense_vectors: np.ndarray | None = None
        self._hybrid_validation = self._validate_dense_index()

    def _validate_dense_index(
        self,
        dense_path: Path = DENSE_INDEX_PATH,
        metadata_path: Path = DENSE_INDEX_METADATA_PATH,
    ) -> dict[str, Any]:
        """Fail closed when dense vectors do not describe this lexical index.

        A dense index is an optional acceleration artifact, not a source of
        truth.  Fingerprints and the complete document-ID set must therefore
        match the loaded lexical index before hybrid retrieval is enabled.
        """

        requested = os.getenv("RAG_HYBRID_ENABLED", "0").strip() == "1"
        lexical_fingerprint = str(
            self.metadata.get("index_fingerprint")
            or self.metadata.get("lexical_index_fingerprint")
            or ""
        )
        status: dict[str, Any] = {
            "requested": requested,
            "ready": False,
            "reason": "disabled_by_environment" if not requested else None,
            "lexical_fingerprint": lexical_fingerprint or None,
            "dense_source_fingerprint": None,
            "lexical_document_count": len(self.documents),
            "dense_document_count": 0,
            "missing_document_count": 0,
            "extra_document_count": 0,
        }
        if not requested:
            return status
        if not dense_path.exists():
            status["reason"] = "dense_index_missing"
            return status
        if not metadata_path.exists():
            status["reason"] = "dense_metadata_missing"
            return status

        try:
            dense_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dense_source_fingerprint = str(
                dense_metadata.get("source_index_fingerprint")
                or dense_metadata.get("source_lexical_fingerprint")
                or ""
            )
            status["dense_source_fingerprint"] = dense_source_fingerprint or None
            if not lexical_fingerprint or not dense_source_fingerprint:
                status["reason"] = "fingerprint_missing"
                return status
            if dense_source_fingerprint != lexical_fingerprint:
                status["reason"] = "fingerprint_mismatch"
                return status

            dense = np.load(dense_path, allow_pickle=False)
            ids = [str(item) for item in dense["ids"].tolist()]
            vectors = np.asarray(dense["vectors"], dtype=np.float32)
            status["dense_document_count"] = len(ids)
            if len(ids) != len(vectors) or vectors.ndim != 2 or len(set(ids)) != len(ids):
                status["reason"] = "dense_shape_or_duplicate_id_error"
                return status

            lexical_ids = {
                str(document.get("id"))
                for document in self.documents
                if document.get("id") and self._document_text(document)
            }
            dense_ids = set(ids)
            missing = sorted(lexical_ids - dense_ids)
            extra = sorted(dense_ids - lexical_ids)
            status["missing_document_count"] = len(missing)
            status["extra_document_count"] = len(extra)
            if missing:
                status["reason"] = "document_id_mismatch"
                status["missing_document_ids_sample"] = missing[:5]
                status["extra_document_ids_sample"] = extra[:5]
                return status

            # Extra vectors are expected for a public-only view over a mixed
            # public/internal index. Keep only vectors visible to this view.
            allowed_positions = [position for position, document_id in enumerate(ids) if document_id in lexical_ids]
            self._dense_ids = [ids[position] for position in allowed_positions]
            self._dense_vectors = vectors[allowed_positions]
            status["ready"] = True
            status["reason"] = None
            return status
        except Exception as exc:
            status["reason"] = f"dense_validation_error:{type(exc).__name__}"
            return status

    @staticmethod
    def _resolve_visual_path(raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_file():
            return path

        legacy_processed = ROOT / "data" / "processed"
        current_processed = ROOT / "data" / "sales" / "processed"
        try:
            relative = path.relative_to(legacy_processed)
        except ValueError:
            return path
        migrated = current_processed / relative
        return migrated if migrated.is_file() else path

    @property
    def metadata(self) -> dict[str, Any]:
        return self.payload.get("metadata", {})

    def _score(self, query_tokens: list[str], document: dict[str, Any]) -> float:
        frequencies = Counter(document.get("tokens") or [])
        doc_length = max(len(document.get("tokens") or []), 1)
        k1, b = 1.5, 0.75
        score = 0.0
        for token in set(query_tokens):
            term_frequency = frequencies.get(token, 0)
            if not term_frequency:
                continue
            # A single Chinese character (for example “法” or “图”) is too
            # ambiguous for visual retrieval.  Bigrams and numeric dimensions
            # carry the main signal, while single characters remain a weak tie
            # breaker for short queries.
            token_weight = 0.12 if len(token) == 1 and "\u4e00" <= token <= "\u9fff" else 1.0
            frequency = self.document_frequency.get(token, 0)
            inverse_document_frequency = math.log(1 + (self.document_count - frequency + 0.5) / (frequency + 0.5))
            denominator = term_frequency + k1 * (1 - b + b * doc_length / self.avg_doc_length)
            score += token_weight * inverse_document_frequency * term_frequency * (k1 + 1) / denominator
        return score

    @staticmethod
    def _source_name_affinity(query: str, document: dict[str, Any]) -> float:
        """Prefer the enterprise document explicitly named by the question.

        This is deliberately derived from the source name and the current
        question, not from benchmark IDs or expected answers.  It separates a
        named company scheme (for example an 岩棉 or 穿透法方案) from a generic
        standard that happens to contain similar numeric terms.
        """

        query_terms = specific_chinese_bigrams(query) - SOURCE_NAME_GENERIC_BIGRAMS
        if not query_terms:
            return 0.0
        source_refs = document.get("source_refs") or []
        source_label = " ".join(
            f"{source.get('document_name') or ''} {source.get('section_heading') or ''}"
            for source in source_refs
        )
        source_terms = specific_chinese_bigrams(source_label)
        overlap = query_terms & source_terms
        # Cap the contribution so source affinity chooses the correct source
        # family but BM25 still ranks passages inside that family.
        return min(80.0, 16.0 * len(overlap))

    def _rare_query_affinity(self, query: str, document: dict[str, Any]) -> float:
        """Reward rare subject phrases without maintaining a domain lexicon."""

        document_tokens = set(document.get("tokens") or [])
        bonus = 0.0
        rare_cutoff = max(12, int(self.document_count * 0.02))
        for term in specific_chinese_bigrams(query) - RARE_QUERY_GENERIC_BIGRAMS:
            frequency = int(self.document_frequency.get(term, self.document_count))
            if term in document_tokens and frequency <= rare_cutoff:
                bonus += 24.0
        return min(120.0, bonus)

    def _rare_query_character_coverage(self, query: str, document: dict[str, Any]) -> float:
        """Reward co-occurrence of multiple rare query characters.

        Chinese bigrams can cross word boundaries: for example a passage about
        “石材防护剂” accidentally matches fragments from “板材防护”.  A single
        character is too weak on its own, but two or more uncommon characters
        from the question occurring in one passage provide a useful generic
        coverage signal without a hand-maintained domain phrase list.
        """

        ignored = set("什么怎么哪些多少要求如何有何提出可以是否的了和与及或对中时后前内外")
        document_tokens = set(document.get("tokens") or [])
        rare_cutoff = max(12, int(self.document_count * 0.02))
        clauses = [part.strip() for part in re.split(r"[，,；;：:]", query) if part.strip()]
        focus_query = clauses[-1] if len(clauses) > 1 and len(clauses[-1]) >= 4 else query
        matches = {
            character
            for character in re.findall(r"[\u4e00-\u9fff]", focus_query)
            if character not in ignored
            and character in document_tokens
            and int(self.document_frequency.get(character, self.document_count)) <= rare_cutoff
        }
        if len(matches) < 2:
            return 0.0
        return min(100.0, 40.0 * len(matches))

    @staticmethod
    def _exact_query_phrase_affinity(query: str, document: dict[str, Any]) -> float:
        """Reward exact 3-4 character subject phrases such as 幕墙材料/耐候性."""

        text = str(document.get("text") or "")
        bonus = 0.0
        ignored = ("什么", "怎么", "哪些", "多少", "提出", "原则", "如何", "要求")
        seen: set[str] = set()
        # In a long question the final clause normally carries the requested
        # field/condition, while the leading clause only identifies the broad
        # product.  Scoring every n-gram from the full product name lets many
        # generic passages hit the cap before terms such as “抹灰层/养护” are
        # considered.
        clauses = [part.strip() for part in re.split(r"[，,；;：:]", query) if part.strip()]
        focus_query = clauses[-1] if len(clauses) > 1 and len(clauses[-1]) >= 4 else query
        for run in re.findall(r"[\u4e00-\u9fff]+", focus_query):
            for width in (4, 3):
                for index in range(max(0, len(run) - width + 1)):
                    phrase = run[index:index + width]
                    if phrase in seen or any(marker in phrase for marker in ignored):
                        continue
                    seen.add(phrase)
                    if phrase in text:
                        bonus += 12.0 if width == 4 else 7.0
        return min(100.0, bonus)

    @staticmethod
    def _question_subject_affinity(query: str, document: dict[str, Any]) -> float:
        """Boost passages that contain every explicit subject of the question."""

        text = str(document.get("text") or "")
        candidate_groups: list[str] = []
        for pattern in (
            r"对(.{2,24}?)(?:提出|有什么|有何|如何)",
            r"(.{2,16}?)的(?:具体)?(?:流程|步骤|工序顺序)",
        ):
            match = re.search(pattern, query)
            if match:
                candidate_groups.append(match.group(1))
        bonus = 0.0
        for group in candidate_groups:
            terms = [
                term.strip()
                for term in re.split(r"的|和|与|、|及", group)
                if len(term.strip()) >= 2
            ]
            def supported(term: str) -> bool:
                if term in text:
                    return True
                if len(term) >= 4:
                    midpoint = len(term) // 2
                    return term[:midpoint] in text and term[midpoint:] in text
                return False

            if terms and all(supported(term) for term in terms):
                bonus = max(bonus, 180.0)
            elif any(supported(term) for term in terms):
                bonus = max(bonus, 35.0)
        return bonus

    @staticmethod
    def temporal_precondition_affinity(query: str, document: dict[str, Any]) -> float:
        """Reward evidence that explains what must happen before an action.

        Queries such as ``注胶前如何处理`` need an ordered preparatory
        passage, not a short definition that merely repeats ``密封胶``.  The
        target action and ordering cues are derived from query grammar, so the
        rule also applies to preparation before coating, installation or
        acceptance without maintaining product-specific answers.
        """

        compact_query = re.sub(r"\s+", "", query)
        match = re.search(r"在([^\n，,。；;？?]{2,24}?)(?:之前|前)(?=应|需|要|如何|怎么|怎样|处理|准备)", compact_query)
        if match is None:
            match = re.search(
                r"([^\n，,。；;？?]{2,18}?)(?:之前|前)(?=应|需|要|如何|怎么|怎样|处理|准备)",
                compact_query,
            )
        if match is None:
            return 0.0

        action = match.group(1)
        action_terms = specific_chinese_bigrams(action)
        text = re.sub(r"\s+", "", str(document.get("text") or ""))
        text_terms = specific_chinese_bigrams(text)
        if not action_terms or not (action_terms & text_terms):
            return 0.0

        ordering_markers = ("之前", "前进行", "待", "先", "再", "然后", "步骤", "清理后")
        ordering_hits = sum(marker in text for marker in ordering_markers)
        if ordering_hits >= 2:
            return 220.0
        if ordering_hits == 1:
            return 90.0
        return 0.0

    @staticmethod
    def _product_overview_section_priority(document: dict[str, Any]) -> int:
        """Identify a catalogue's hierarchy/overview section from metadata."""

        headings = " ".join(
            str(source.get("section_heading") or "")
            for source in (document.get("source_refs") or [])
            if isinstance(source, dict)
        )
        return 1 if any(marker in headings for marker in PRODUCT_OVERVIEW_SECTION_MARKERS) else 0

    @staticmethod
    def _choice_condition_affinity(query: str, document: dict[str, Any]) -> float:
        """Prefer a concrete rule that covers both choices and their conditions.

        A planner may append broad phrases such as ``施工方案 选择条件`` to the
        customer's question. Generic summaries then contain many query words,
        while the useful clause is the one that mentions both alternatives and
        says when each applies. The alternatives are derived from question
        grammar rather than product names or benchmark labels.
        """

        condition_markers = (
            "依据", "根据", "按照", "什么条件", "哪些条件", "选择条件",
            "如何选择", "怎么选择", "何时采用", "何时选用",
        )
        lines = [re.sub(r"\s+", "", line) for line in query.splitlines() if line.strip()]
        choice_line = next(
            (
                line
                for line in lines
                if any(connector in line for connector in ("还是", "或者", "或"))
                and any(marker in line for marker in condition_markers)
            ),
            "",
        )
        if not choice_line:
            return 0.0

        connector = next((item for item in ("还是", "或者", "或") if item in choice_line), "")
        if not connector:
            return 0.0
        left, right = choice_line.split(connector, 1)

        # The left side normally contains the document subject followed by an
        # action verb (for example “一体板采用点粘法”). Keep only the choice.
        for marker in ("采用", "选用", "选择", "使用"):
            if marker in left:
                left = left.rsplit(marker, 1)[-1]
        left = re.split(r"[，,。；;：:]", left)[-1].strip("？?：:，,。；;")

        # The right side may be followed immediately by the condition question.
        right = re.split(r"[，,。；;：:\n]", right, maxsplit=1)[0]
        trailing_question_markers = (
            "应依据", "需依据", "依据", "根据", "按照", "按什么条件",
            "如何选择", "怎么选择", "什么条件", "哪些条件", "选择条件",
        )
        cut_positions = [right.find(marker) for marker in trailing_question_markers if marker in right]
        if cut_positions:
            right = right[: min(cut_positions)]
        right = right.strip("？?：:，,。；;")
        for marker in ("采用", "选用", "选择", "使用"):
            if right.startswith(marker):
                right = right[len(marker):]

        alternatives = [term for term in (left, right) if 2 <= len(term) <= 18]
        if len(alternatives) != 2:
            return 0.0
        text = re.sub(r"\s+", "", str(document.get("text") or ""))
        if not all(term in text for term in alternatives):
            return 0.0

        has_numeric_threshold = bool(
            re.search(
                r"(?:大于|小于|高于|低于|不少于|不大于|不小于|不低于|不超过|至少|至多|≥|≤|>|<)"
                r"[^，,。；;]{0,12}\d",
                text,
            )
        )
        has_explicit_condition = any(
            marker in text
            for marker in ("条件", "适用", "当", "时", "依据", "根据", "取决于")
        )
        if has_numeric_threshold:
            return 360.0
        if has_explicit_condition:
            return 220.0
        return 0.0

    @staticmethod
    def _answer_form_type(query: str) -> str | None:
        compact = re.sub(r"\s+", "", query)
        if re.search(r"(?:如何|怎么|怎样)(?:检查|检测|测量|测定)", compact):
            return "measurement_method"
        if re.search(
            r"(?:由(?:哪些|什么).{0,12}(?:组成|构成)|(?:哪些|什么).{0,12}(?:组成|构成)|包括哪些)",
            compact,
        ):
            return "composition_list"
        return None

    @classmethod
    def _answer_form_affinity(cls, query: str, document: dict[str, Any]) -> float:
        """Small, explainable match between question predicate and answer form.

        The signal is deliberately capped below a clear reranker margin.  It
        distinguishes an actionable measurement sentence from a requirement
        heading, and an actual ``由...组成`` list from a passage that merely
        mentions the words ``组成材料``.
        """

        answer_form = cls._answer_form_type(query)
        if answer_form is None:
            return 0.0
        text = re.sub(r"\s+", "", str(document.get("text") or ""))
        compact_query = re.sub(r"\s+", "", query)
        if answer_form == "measurement_method":
            match = re.search(r"(?:检查|检测|测量|测定)([^？?。；;，,]{2,28})", compact_query)
            target = match.group(1) if match else ""
            target_terms = specific_chinese_bigrams(target) - RARE_QUERY_GENERIC_BIGRAMS
            text_terms = specific_chinese_bigrams(text)
            target_supported = bool(target and target in text)
            if not target_supported and target_terms:
                target_supported = len(target_terms & text_terms) / len(target_terms) >= 0.6
            has_measurement_action = bool(
                re.search(r"(?:采用|使用|用|以).{0,16}(?:测量|测定|检查|检测|靠尺|塞尺|仪)", text)
                or re.search(r"(?:测量|测定|检查|检测).{0,16}(?:方法|靠尺|塞尺|仪)", text)
            )
            return 0.18 if target_supported and has_measurement_action else 0.0

        # Exact list predicates are stronger than descriptive "复合而成".
        # Both remain useful; neither is inferred from a product dictionary.
        if re.search(r"由[^。；;\n]{4,180}组成", text) and len(re.findall(r"[、，,及和]", text)) >= 2:
            return 0.18
        if re.search(r"由[^。；;\n]{4,180}构成", text) and len(re.findall(r"[、，,及和]", text)) >= 2:
            return 0.15
        if re.search(r"由[^。；;\n]{4,180}复合而成", text) and len(re.findall(r"[、，,及和]", text)) >= 2:
            return 0.12
        if re.search(r"(?:包括|主要有)[^。；;\n]{4,180}[、，,]", text):
            return 0.1
        return 0.0

    @classmethod
    def _rerank_text_limit(cls, query: str) -> int:
        """Spend eight extra cross-encoder slots only on predicate-heavy QA."""

        return 32 if cls._answer_form_type(query) is not None else 24

    @staticmethod
    def _domain_priority(
        document: dict[str, Any], *, node_atlas_request: bool, standard_request: bool,
        procedure_request: bool, product_overview_request: bool
    ) -> int:
        """Route specialised questions to their authoritative knowledge domain.

        BM25 still ranks the evidence inside a domain.  The priority only
        prevents a generic construction-plan phrase from crowding out the
        requested node atlas or standard before it is considered.
        """

        domains = set(document.get("knowledge_domains") or [])
        categories = set(document.get("document_categories") or [])
        labels = set(document.get("content_labels") or [])
        if product_overview_request:
            if "canonical_product_profile" in labels or "enterprise_product_master_profile" in categories:
                # A broad “有哪些产品” question needs the product hierarchy
                # before finish samples, materials or maintenance details.
                return 7 if LocalRagRetriever._product_overview_section_priority(document) else 6
            if "enterprise_product_catalogue" in categories:
                return 2
            if "internal_product_comparison" in categories:
                return 1
            return 0
        if procedure_request:
            if "03_construction_method" in domains:
                return 3
            if "04_node_atlas" in domains:
                return 1
            return 0
        if node_atlas_request:
            if "04_node_atlas" in domains:
                return 3
            if "03_construction_method" in domains:
                return 2
            return 0
        if standard_request:
            if "01_standard_specification" in domains:
                return 3
            if "03_construction_method" in domains:
                return 1
        return 0

    def _lexical_scored(
        self, query_tokens: list[str], *, node_atlas_request: bool, standard_request: bool,
        procedure_request: bool, product_overview_request: bool
    ) -> list[tuple[float, dict[str, Any]]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for document in self.documents:
            score = self._score(query_tokens, document)
            if score <= 0:
                continue
            score += self._domain_priority(
                document,
                node_atlas_request=node_atlas_request,
                standard_request=standard_request,
                procedure_request=procedure_request,
                product_overview_request=product_overview_request,
            ) * 100.0
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        return str(document.get("text") or document.get("search_text") or "").strip()

    @staticmethod
    def _primary_document_id(document: dict[str, Any]) -> str:
        source_refs = document.get("source_refs") or []
        return str(source_refs[0].get("document_id") or "") if source_refs else ""

    @staticmethod
    def _primary_source_page(document: dict[str, Any]) -> int | None:
        source_refs = document.get("source_refs") or []
        if not source_refs:
            return None
        page = source_refs[0].get("source_page")
        return int(page) if isinstance(page, int) else None

    @staticmethod
    def _is_construction_step(document: dict[str, Any]) -> bool:
        labels = set(document.get("content_labels") or [])
        domains = set(document.get("knowledge_domains") or [])
        taxonomy = document.get("source_taxonomy") or []
        if isinstance(taxonomy, dict):
            taxonomy = [taxonomy]
        primary_category = (
            str(taxonomy[0].get("document_category") or "")
            if taxonomy and isinstance(taxonomy[0], dict)
            else ""
        )
        return primary_category == "enterprise_construction_method" and "03_construction_method" in domains and bool(
            {"construction_method", "construction_step"} & labels
        )

    def _procedure_sequence(
        self, query: str, scored: list[tuple[float, dict[str, Any]],], top_k: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return adjacent evidence from one authoritative procedure section.

        A process question is different from a comparison question: the useful
        answer is normally distributed across neighbouring source chunks.  The
        most relevant heading is used as an anchor and the following labelled
        steps from the same document are kept in source order.  This works for
        any procedure document and never depends on a product-specific rule.
        """

        eligible = [
            item
            for item in scored
            if item[1].get("kind") == "text"
            and self._is_construction_step(item[1])
            and self._document_text(item[1])
        ]
        # Prefer a substantive instruction over a bare chapter heading.  This
        # keeps the sequence anchored on the actual steps even when the short
        # heading receives a slightly higher BM25 score.
        substantive = [
            item for item in eligible
            if len(self._document_text(item[1])) >= 28
            and not self._document_text(item[1]).strip().endswith("施工方案")
        ]
        asks_compact_sequence = any(term in query for term in ("主要工序", "工序顺序", "步骤顺序"))
        if asks_compact_sequence and substantive:
            action_markers = ("预", "安装", "固定", "拧", "修补", "嵌缝", "打胶", "清理", "处理")
            anchor = max(
                substantive,
                key=lambda item: (
                    sum(marker in self._document_text(item[1]) for marker in action_markers),
                    self._document_text(item[1]).count("—"),
                    item[0],
                ),
            )
        else:
            subject_matched = [
                item for item in eligible
                if self._question_subject_affinity(query, item[1]) >= 180.0
            ]
            anchor = (
                subject_matched[0]
                if subject_matched
                else (substantive[0] if substantive else (eligible[0] if eligible else None))
            )
        if anchor is None:
            return []

        anchor_score, anchor_document = anchor
        document_id = self._primary_document_id(anchor_document)
        anchor_position = self.document_position_by_id.get(str(anchor_document.get("id")))
        anchor_page = self._primary_source_page(anchor_document)
        if not document_id or anchor_position is None:
            return [anchor]

        score_by_id = {str(document.get("id")): score for score, document in scored}
        selected: list[tuple[float, dict[str, Any]]] = []
        # Three source pages usually cover a complete short process while
        # avoiding unrelated later chapters such as special node details.
        maximum_page = anchor_page + 2 if anchor_page is not None else None
        for document in self.documents[anchor_position:]:
            if len(selected) >= top_k:
                break
            if document.get("kind") != "text" or not self._is_construction_step(document):
                continue
            if self._primary_document_id(document) != document_id:
                continue
            page = self._primary_source_page(document)
            if maximum_page is not None and page is not None and page > maximum_page:
                break
            text = self._document_text(document)
            if not text:
                continue
            selected.append((score_by_id.get(str(document.get("id")), anchor_score * 0.01), document))

        return selected or [anchor]

    def _hybrid_scored(
        self,
        query: str,
        query_tokens: list[str],
        *,
        node_atlas_request: bool,
        standard_request: bool,
        procedure_request: bool,
        product_overview_request: bool,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Dense-recall plus local reranking, with a lexical fallback.

        The two 0.6B retrieval models are cached for the process lifetime and
        accessed serially.  Repeated construction/destruction caused native
        CUDA allocator instability during long sequential evaluations.  Set
        ``RAG_HYBRID_ENABLED=0`` to force BM25-only retrieval.
        """

        lexical = self._lexical_scored(
            query_tokens,
            node_atlas_request=node_atlas_request,
            standard_request=standard_request,
            procedure_request=procedure_request,
            product_overview_request=product_overview_request,
        )
        # Query-aware affinities must participate before the reranker candidate
        # cutoff.  Applying them only after reranking is too late: an exact
        # normative clause can otherwise be discarded by many generic chunks
        # from the same long standard.
        lexical_for_candidates = []
        for score, document in lexical:
            text = self._document_text(document)
            adjusted = (
                score
                + self._source_name_affinity(query, document)
                + self._rare_query_affinity(query, document)
                + self._rare_query_character_coverage(query, document)
                + self._exact_query_phrase_affinity(query, document)
                + self._question_subject_affinity(query, document)
                + self._choice_condition_affinity(query, document)
                + self.temporal_precondition_affinity(query, document)
                + self._answer_form_affinity(query, document) * HYBRID_RULE_SCALE
            )
            if standard_request and "原则" in query and "应选用" in text:
                adjusted += 140.0
            if standard_request and any(
                marker in text for marker in ("本条", "本规范条文", "鉴于", "因此，根据")
            ):
                adjusted -= 90.0
            lexical_for_candidates.append((adjusted, document))
        lexical_for_candidates.sort(key=lambda item: item[0], reverse=True)
        if not self._hybrid_validation.get("ready"):
            return lexical

        try:
            from backend.sales.dense_retrieval import (
                RETRIEVAL_INFERENCE_LOCK,
                get_embedding_model,
                get_reranker_model,
            )

            ids = self._dense_ids
            vectors = self._dense_vectors
            if vectors is None:
                return lexical
            with RETRIEVAL_INFERENCE_LOCK:
                embedding = get_embedding_model()
                query_vector = embedding.encode([query], query=True, batch_size=1, max_length=1024)[0]

            dense_scores = vectors @ query_vector.astype(np.float32)
            dense_order = np.argsort(-dense_scores)[:48]
            document_by_id = {str(document.get("id")): document for document in self.documents}
            dense_ranked_ids = [ids[int(index)] for index in dense_order]
            fused = reciprocal_rank_fusion(
                lexical_for_candidates[:48], dense_ranked_ids, document_by_id
            )
            fused = [item for item in fused if self._document_text(item[1])]
            # Keep original diagrams/photos in the reranking pool.  Without a
            # quota, many short text chunks can crowd out a relevant node
            # drawing before the visual-selection stage ever sees it.
            candidate_pairs = select_rerank_candidates(
                fused,
                [str(document.get("id")) for _, document in lexical_for_candidates[:48]],
                dense_ranked_ids,
                document_by_id,
                non_visual_limit=self._rerank_text_limit(query),
            )
            candidates = [document for _, document in candidate_pairs]
            if not candidates:
                return lexical

            with RETRIEVAL_INFERENCE_LOCK:
                reranker = get_reranker_model()
                rerank_scores = reranker.score(query, [self._document_text(item) for item in candidates])

            max_fusion = max((score for score, _ in candidate_pairs), default=1.0) or 1.0
            fusion_by_id = {
                str(document.get("id")): score / max_fusion
                for score, document in candidate_pairs
            }
            reranked: list[tuple[float, dict[str, Any]]] = []
            for reranker_score, document in zip(rerank_scores, candidates):
                fusion_score = fusion_by_id.get(str(document.get("id")), 0.0)
                combined_score = 0.85 * float(reranker_score) + 0.15 * fusion_score
                tagged_document = dict(document)
                tagged_document["_retrieval_score_mode"] = "hybrid"
                tagged_document["_reranker_score"] = float(reranker_score)
                tagged_document["_fusion_score"] = float(fusion_score)
                reranked.append((combined_score, tagged_document))
            reranked.sort(
                key=lambda item: (
                    self._domain_priority(
                        item[1],
                        node_atlas_request=node_atlas_request,
                        standard_request=standard_request,
                        procedure_request=procedure_request,
                        product_overview_request=product_overview_request,
                    ),
                    item[0],
                ),
                reverse=True,
            )
            return reranked
        except Exception:
            # The sales assistant remains available if an optional local
            # retrieval model is updating or temporarily lacks GPU memory.
            return lexical

    @staticmethod
    def _public_sources(source_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "document_name": source.get("document_name"),
                "source_page": source.get("source_page"),
                "section_heading": source.get("section_heading"),
                "bbox": source.get("bbox"),
                "source_format": source.get("source_format"),
                "source_sheet": source.get("source_sheet"),
                "source_range": source.get("source_range"),
                "source_paragraph_range": source.get("source_paragraph_range"),
            }
            for source in source_refs
        ]

    @staticmethod
    def _is_page_fallback_visual(document: dict[str, Any]) -> bool:
        """Return true for a rendered whole page used only as a last resort."""

        return (
            str(document.get("asset_type") or "") == "original_pdf_page_render"
            or "page_fallback" in str(document.get("asset_id") or "")
        )

    @classmethod
    def _is_product_gallery_visual(cls, document: dict[str, Any]) -> bool:
        """Return only catalogue-approved product overview/variant crops.

        Product retrieval is deliberately fail-closed.  Project photos,
        standards, components, accessories and whole-page renders must not be
        used merely to fill a customer gallery.
        """

        return bool(
            document.get("kind") == "visual"
            and document.get("gallery_type") == "product_sample"
            and document.get("product_gallery_eligible") is True
            and document.get("visual_role") in {"product_overview", "product_variant"}
            and document.get("review_status") != "needs_review"
            and not cls._is_page_fallback_visual(document)
        )

    @staticmethod
    def _visual_explanation(
        gallery_type: str,
        *,
        case: dict[str, Any] | None = None,
        page_fallback: bool = False,
        visual_role: str | None = None,
        product_name: str | None = None,
        variant_or_code: str | None = None,
    ) -> str:
        if gallery_type == "case" and case:
            name = str(case.get("project_name") or "该项目").strip()
            suffix = "（原页兜底）" if page_fallback else ""
            return f"该图片与项目案例“{name}”的结构化记录直接关联{suffix}。"
        if gallery_type == "product":
            product_label = str(product_name or "产品").strip()
            if visual_role == "product_overview":
                return f"企业产品目录中的{product_label}产品样板总览；图片来源页与位置可追溯核对。"
            if visual_role == "product_variant":
                variant_label = str(variant_or_code or "具体饰面型号").strip()
                return f"企业产品目录中的{product_label} {variant_label}样板图；图片来源页与位置可追溯核对。"
            suffix = "；当前使用相关原页作为兜底" if page_fallback else ""
            return f"该图片来自企业产品目录{suffix}，来源页与位置可追溯核对。"
        if gallery_type == "node":
            return "该图片来自与当前节点关键词匹配的节点资料。"
        if gallery_type == "process":
            return "该图片来自与当前施工工艺或流程匹配的资料。"
        return "该图片来自本次问题召回的本地企业资料。"

    def _visual_payload(
        self,
        document: dict[str, Any],
        *,
        score: float | int | None,
        gallery_type: str,
        case: dict[str, Any] | None = None,
        selection_reason: str = "semantic_match",
        matched_product: str | None = None,
        matched_variant_or_code: str | None = None,
    ) -> dict[str, Any]:
        """Create the stable customer-facing visual contract.

        Case metadata deliberately travels with the asset so a gallery cannot
        silently pair a project caption with a picture from another record.
        Non-case images keep the same keys with null values, which makes the
        API shape predictable for the frontend and evaluation scripts.
        """

        asset_id = str(document.get("asset_id") or "")
        page_fallback = self._is_page_fallback_visual(document)
        related_case = case or {}
        product_name = (
            document.get("product_name")
            or document.get("canonical_product")
            or related_case.get("product")
        )
        variant_or_code = document.get("variant_or_code") or matched_variant_or_code
        return {
            "asset_id": asset_id,
            "customer_title": document.get("customer_title")
            or related_case.get("project_name"),
            "asset_type": document.get("asset_type"),
            "effective_image_kind": document.get("effective_image_kind"),
            "score": round(float(score or 0.0), 4),
            "citation": document.get("citation"),
            "visual_endpoint": f"/api/copilot/visual/{asset_id}",
            "facts_eligible": False,
            "linked_text_evidence": document.get("linked_text_evidence") or [],
            "multimodal_bundle_id": document.get("multimodal_bundle_id"),
            "related_case_id": related_case.get("case_id"),
            "project_name": related_case.get("project_name"),
            "product": related_case.get("product"),
            "product_name": product_name,
            "variant_or_code": variant_or_code,
            "visual_role": document.get("visual_role"),
            "installation_method": related_case.get("installation_method"),
            "area_m2": related_case.get("area_m2"),
            "completion_year": related_case.get("completion_year"),
            "gallery_type": gallery_type,
            "selection_reason": selection_reason,
            "matched_product": matched_product or product_name,
            "matched_variant_or_code": matched_variant_or_code or document.get("variant_or_code"),
            "is_page_fallback": page_fallback,
            "explanation": self._visual_explanation(
                gallery_type,
                case=case,
                page_fallback=page_fallback,
                visual_role=str(document.get("visual_role") or "") or None,
                product_name=str(product_name or "") or None,
                variant_or_code=str(variant_or_code or "") or None,
            ),
        }

    def visuals_for_project_cases(
        self,
        project_cases: list[dict[str, Any]],
        *,
        visual_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return at most one directly linked primary image per selected case.

        The selected case order is authoritative.  An explicitly linked whole
        page may serve as a fallback for that same case, but an unrelated
        generic visual is never substituted merely to fill the gallery.
        """

        visuals: list[dict[str, Any]] = []
        selected_asset_ids: set[str] = set()
        for case in project_cases:
            candidates: list[dict[str, Any]] = []
            for raw_asset_id in case.get("visual_asset_ids") or []:
                asset_id = str(raw_asset_id)
                asset = self.visual_by_asset_id.get(asset_id)
                if asset is not None and asset_id not in selected_asset_ids:
                    candidates.append(asset)
            if not candidates:
                continue
            # ``visual_asset_ids`` is authored hero-first by the catalogue
            # builder.  Preserve that audited order; only skip a missing or
            # duplicate ID rather than reinterpreting which picture is primary.
            selected = candidates[0]
            selected_asset_ids.add(str(selected.get("asset_id") or ""))
            visuals.append(
                self._visual_payload(
                    selected,
                    score=case.get("score"),
                    gallery_type="case",
                    case=case,
                    selection_reason="case_primary",
                )
            )
            if len(visuals) >= visual_k:
                break
        return visuals

    @staticmethod
    def _product_visual_match(query: str, document: dict[str, Any]) -> tuple[bool, str | None]:
        compact_query = re.sub(r"\s+", "", query.lower()).replace("®", "")
        for marker in (
            "有没有", "有", "请", "给我", "发", "展示", "查看", "看看", "一下", "相关",
            "产品", "图片", "照片", "样板图", "产品图", "吗", "呢", "的",
        ):
            compact_query = compact_query.replace(marker, "")
        query_codes = set(re.findall(r"[a-z]+[a-z0-9-]*\d+[a-z0-9-]*", compact_query))
        identity_label = "\n".join(
            str(value).strip()
            for value in (
                document.get("customer_title"),
                document.get("customer_caption"),
                document.get("product_name"),
                document.get("canonical_product"),
                document.get("variant_or_code"),
            )
            if isinstance(value, str) and value.strip()
        )
        primary_label = re.sub(
            r"\s+",
            "",
            identity_label.lower(),
        ).replace("®", "")
        # A cropped product photo can be surrounded by labels for neighbouring
        # variants on the same catalogue page.  Match it against its own title,
        # while a whole-page fallback legitimately uses the complete page text.
        match_text = primary_label
        if LocalRagRetriever._is_page_fallback_visual(document):
            match_text = re.sub(
                r"\s+",
                "",
                str(document.get("search_text") or "").lower(),
            ).replace("®", "")
        matching_codes = sorted(code for code in query_codes if code in match_text)
        if matching_codes:
            return True, matching_codes[0]

        variant_label = re.sub(
            r"\s+", "", str(document.get("variant_or_code") or "").lower()
        ).replace("®", "")
        variant_terms = specific_chinese_bigrams(compact_query) - PRODUCT_VISUAL_GENERIC_BIGRAMS
        if variant_label and variant_terms and variant_terms & set(tokenize(variant_label)):
            return True, str(document.get("variant_or_code") or "")

        product_name = re.sub(
            r"\s+",
            "",
            str(document.get("product_name") or document.get("canonical_product") or "").lower(),
        ).replace("®", "")
        if product_name and product_name in compact_query:
            return True, None

        subject_terms = variant_terms
        document_terms = set(tokenize(match_text))
        matching_terms = sorted(subject_terms & document_terms, key=len, reverse=True)
        if matching_terms:
            return True, matching_terms[0]
        return False, None

    def _prioritise_product_visuals(
        self,
        query: str,
        candidates: list[tuple[float, dict[str, Any]]],
        *,
        product_overview_request: bool,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return only catalogue-approved product crops.

        A broad product overview receives the reviewed overview crop followed
        by reviewed variant samples.  A specific model/finish request is more
        conservative: if no catalogue identity explicitly matches, no image is
        returned instead of substituting an unrelated sample or page render.
        """

        eligible = [item for item in candidates if self._is_product_gallery_visual(item[1])]

        def sort_key(item: tuple[float, dict[str, Any]]) -> tuple[int, int, float, str]:
            score, document = item
            role_priority = 2 if document.get("visual_role") == "product_overview" else 1
            return (
                role_priority,
                int(document.get("display_priority") or 0),
                score,
                str(document.get("asset_id") or ""),
            )

        # The semantic Planner owns the distinction between a broad catalogue
        # overview and a named-product lookup.  Reclassifying the query here by
        # keywords can override a correct model plan after conversation context
        # or a query rewrite has been appended.  Callers must therefore pass
        # the resolved semantic flag explicitly.
        if product_overview_request:
            return sorted(eligible, key=sort_key, reverse=True)

        exact = [
            item for item in eligible if self._product_visual_match(query, item[1])[0]
        ]
        return sorted(exact, key=sort_key, reverse=True)

    def _product_gallery_candidates(
        self,
        query: str,
        retrieved_candidates: list[tuple[float, dict[str, Any]]] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Score every reviewed product-gallery asset, independent of RRF.

        The global hybrid rerank pool is intentionally small because it must
        balance text, cases and several visual domains.  It is therefore not a
        safe source of truth for a gallery the semantic Planner has already
        selected: an approved product crop can be absent from that pool even
        though it is the exact asset the customer requested.  Build the product
        pool from the complete canonical index, while reusing a stronger global
        retrieval score when one is available.

        Eligibility remains fail-closed in ``_is_product_gallery_visual`` and
        named-product matching remains fail-closed in
        ``_prioritise_product_visuals``.  This method broadens recall only; it
        does not make an unreviewed or unrelated image returnable.
        """

        retrieved_score_by_id = {
            str(document.get("id") or document.get("asset_id") or ""): float(score)
            for score, document in (retrieved_candidates or [])
        }
        expanded_query = expand_product_aliases(query, self.product_alias_groups)
        query_tokens = tokenize(expanded_query)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for document in self.documents:
            if not self._is_product_gallery_visual(document):
                continue
            document_id = str(document.get("id") or document.get("asset_id") or "")
            local_score = self._score(query_tokens, document) if query_tokens else 0.0
            candidates.append(
                (max(local_score, retrieved_score_by_id.get(document_id, 0.0)), document)
            )
        return candidates

    def _rank_with_rules(
        self,
        query: str,
        scored: list[tuple[float, dict[str, Any]]],
        *,
        node_atlas_request: bool,
        standard_request: bool,
        procedure_request: bool,
        product_overview_request: bool,
        reviewed_product_fact_request: bool,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Apply bounded rule evidence while retaining domain routing.

        Lexical scores retain their historical rule scale.  Hybrid scores are
        probability-like, so the same raw 16--360 point bonuses are compressed
        to a bounded adjustment instead of overwhelming the reranker.
        """

        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for score, document in scored:
            text = self._document_text(document)
            rule_score = (
                self._source_name_affinity(query, document)
                + self._rare_query_affinity(query, document)
                + self._rare_query_character_coverage(query, document)
                + self._exact_query_phrase_affinity(query, document)
                + self._question_subject_affinity(query, document)
                + self._choice_condition_affinity(query, document)
                + self.temporal_precondition_affinity(query, document)
                + self._answer_form_affinity(query, document) * HYBRID_RULE_SCALE
            )
            if standard_request and "原则" in query and "应选用" in text:
                rule_score += 140.0
            if reviewed_product_fact_request and document.get("sales_playbook_use") == "approved_product_master_profile":
                rule_score += 180.0
            if (
                document.get("sales_playbook_use") == "approved_product_master_profile"
                and not product_overview_request
                and not reviewed_product_fact_request
            ):
                rule_score -= 160.0
            if standard_request and any(
                marker in text for marker in ("本条", "本规范条文", "鉴于", "因此，根据")
            ):
                rule_score -= 90.0

            if document.get("_retrieval_score_mode") == "hybrid":
                rule_score = max(
                    HYBRID_RULE_MIN,
                    min(HYBRID_RULE_MAX, rule_score / HYBRID_RULE_SCALE),
                )
            priority = self._domain_priority(
                document,
                node_atlas_request=node_atlas_request,
                standard_request=standard_request,
                procedure_request=procedure_request,
                product_overview_request=product_overview_request,
            )
            # Lexical scores already contain the historic 100-point domain
            # boost.  Hybrid scores do not, so retain routing as a calibrated
            # preference rather than an absolute tier that can suppress a
            # clearly stronger reranker result.
            domain_adjustment = (
                0.1 * priority
                if document.get("_retrieval_score_mode") == "hybrid"
                else 0.0
            )
            adjusted = float(score) + rule_score + domain_adjustment
            ranked.append((adjusted, priority, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [(score, document) for score, _, document in ranked]

    @staticmethod
    def _has_substantive_query_overlap(
        query_tokens: list[str], document: dict[str, Any]
    ) -> bool:
        """Reject hits supported only by an ambiguous Chinese singleton."""

        meaningful = {
            token
            for token in query_tokens
            if len(token) >= 2 or bool(re.search(r"[a-z0-9]", token, re.IGNORECASE))
        }
        return bool(meaningful & set(document.get("tokens") or []))

    @staticmethod
    def _weak_match_threshold(
        scored: list[tuple[float, dict[str, Any]]], kind: str
    ) -> float:
        relevant = [
            float(score)
            for score, document in scored
            if document.get("kind") == kind
        ]
        if not relevant:
            return math.inf
        best = max(relevant)
        hybrid = any(
            document.get("_retrieval_score_mode") == "hybrid"
            for _, document in scored
        )
        return max(0.05, best * 0.12) if hybrid else max(0.2, best * 0.06)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        visual_k: int = 3,
        case_k: int = 5,
        retrieval_mode: str = "factual_lookup",
        *,
        wants_visuals: bool = False,
        visual_scope: str = "mixed",
        visual_query: str | None = None,
        product_overview_request: bool | None = None,
    ) -> dict[str, Any]:
        retrieval_mode = retrieval_mode if retrieval_mode in RETRIEVAL_MODES else "factual_lookup"
        visual_scope = visual_scope if visual_scope in VISUAL_SCOPES else "mixed"
        visual_query = visual_query or query
        expanded_query = expand_product_aliases(query, self.product_alias_groups)
        query_tokens = tokenize(expanded_query)
        procedure_request = retrieval_mode == "procedure"
        project_fit_request = retrieval_mode == "project_fit"
        normal_query = query.lower()
        node_atlas_request = retrieval_mode == "node_detail" or any(marker in query for marker in NODE_ATLAS_QUERY_MARKERS)
        standard_request = any(marker in normal_query for marker in STANDARD_QUERY_MARKERS)
        # Model-planned calls pass this flag explicitly.  ``None`` preserves the
        # legacy direct-retrieval API, whose callers have no semantic plan.
        if product_overview_request is None:
            product_overview_request = is_product_overview_query(query)
        else:
            product_overview_request = bool(product_overview_request)
        text_evidence_limit = top_k
        if product_overview_request:
            # A product overview is a reviewed multi-section dossier.  Do not
            # silently omit a section merely because a generic top-k is lower
            # than the number of approved master-profile rows.
            approved_profile_count = sum(
                1
                for document in self.documents
                if document.get("kind") == "text"
                and document.get("sales_playbook_use") == "approved_product_master_profile"
            )
            text_evidence_limit = max(top_k, approved_profile_count)
        reviewed_product_fact_request = (
            not procedure_request
            and not standard_request
            and any(term in query for term in ("真岩", "饰面板", "仿石材"))
            and any(term in query for term in ("原材料", "成型", "构成", "交付形态", "产品体系"))
        )
        if procedure_request:
            # A named process should stay inside its procedure document.  The
            # neutral terms also cover paraphrases such as “怎么做/工序”.
            query_tokens.extend(tokenize("施工流程 施工步骤 工序"))
        elif project_fit_request:
            # A project-fit question can legitimately compare several document
            # backed installation paths before it asks for missing conditions.
            query_tokens.extend(tokenize(CONSTRUCTION_QUERY_EXPANSION))
        required_visual_terms = specific_chinese_bigrams(visual_query)
        required_named_visual_terms = required_visual_phrases(visual_query)
        if not query_tokens:
            return {
                "text_evidence": [],
                "visual_assets": [],
                "project_cases": [],
                "meta": self._meta(
                    query_tokens,
                    retrieval_mode=retrieval_mode,
                    wants_visuals=wants_visuals,
                    visual_scope=visual_scope,
                ),
            }

        scored = self._hybrid_scored(
            expanded_query,
            query_tokens,
            node_atlas_request=node_atlas_request,
            standard_request=standard_request,
            procedure_request=procedure_request,
            product_overview_request=product_overview_request,
        )
        scored = self._rank_with_rules(
            query,
            scored,
            node_atlas_request=node_atlas_request,
            standard_request=standard_request,
            procedure_request=procedure_request,
            product_overview_request=product_overview_request,
            reviewed_product_fact_request=reviewed_product_fact_request,
        )

        text_evidence: list[dict[str, Any]] = []
        selected_text_chunk_ids: set[str] = set()
        text_threshold = self._weak_match_threshold(scored, "text")
        case_threshold = self._weak_match_threshold(scored, "project_case")
        selected_procedure_document_ids: set[str] = set()

        def append_text_evidence(score: float, document: dict[str, Any]) -> None:
            """Append one text row once while preserving its public provenance."""

            chunk_id = str(document.get("id") or "")
            if not chunk_id or chunk_id in selected_text_chunk_ids:
                return
            text_evidence.append(
                {
                    "chunk_id": chunk_id,
                    "text": document["text"],
                    "score": round(score, 4),
                    "facts_eligible": True,
                    "citations": self._public_sources(document.get("source_refs") or []),
                    "source_taxonomy": document.get("source_taxonomy") or [],
                    "sales_playbook_use": document.get("sales_playbook_use"),
                }
            )
            selected_text_chunk_ids.add(chunk_id)

        if procedure_request:
            for score, document in self._procedure_sequence(query, scored, top_k):
                append_text_evidence(score, document)
                document_id = self._primary_document_id(document)
                if document_id:
                    selected_procedure_document_ids.add(document_id)

        if product_overview_request and not procedure_request:
            # The reviewed product dossier is a five-part canonical record.
            # Hybrid reranking may score a generic catalogue heading above one
            # of those parts; allowing that heading to consume ``top_k`` made
            # broad product answers silently omit the finish/style section.
            # Reserve every approved profile row first, then let ordinary
            # evidence compete only for any remaining slots.
            for score, document in scored:
                if (
                    document.get("kind") == "text"
                    and document.get("sales_playbook_use")
                    == "approved_product_master_profile"
                ):
                    append_text_evidence(score, document)

        project_cases: list[dict[str, Any]] = []
        selected_case_names: set[str] = set()
        selected_source_documents: set[str] = set()
        visual_candidates: list[tuple[float, dict[str, Any]]] = []
        for score, document in scored:
            if document.get("kind") == "text" and not procedure_request and len(text_evidence) < text_evidence_limit:
                if str(document.get("id") or "") in selected_text_chunk_ids:
                    continue
                approved_overview = bool(
                    product_overview_request
                    and document.get("sales_playbook_use") == "approved_product_master_profile"
                )
                if not approved_overview and (
                    score < text_threshold
                    or not self._has_substantive_query_overlap(query_tokens, document)
                ):
                    continue
                source_refs = document.get("source_refs") or []
                primary_document_id = str(source_refs[0].get("document_id") or "") if source_refs else ""
                # Only project-fit questions should cover distinct schemes.
                # A procedure request needs adjacent evidence from one scheme.
                if (
                    project_fit_request
                    and primary_document_id
                    and primary_document_id in selected_source_documents
                    and len(selected_source_documents) < 3
                ):
                    continue
                append_text_evidence(score, document)
                if primary_document_id:
                    selected_source_documents.add(primary_document_id)
            elif document.get("kind") == "project_case" and len(project_cases) < case_k:
                if (
                    score < case_threshold
                    or not self._has_substantive_query_overlap(query_tokens, document)
                ):
                    continue
                case = document.get("case") if isinstance(document.get("case"), dict) else {}
                # A catalogue can repeat the same named project on a later
                # page while presenting a different photo, edition or area.
                # Preserve every record in the index for auditability, but do
                # not show the customer the same project twice in one compact
                # result list.  The highest-ranked matching record remains.
                case_name_key = re.sub(r"\s+", "", str(case.get("project_name") or ""))
                if case_name_key and case_name_key in selected_case_names:
                    continue
                project_cases.append(
                    {
                        "case_id": case.get("case_id"),
                        "project_name": case.get("project_name"),
                        "region": case.get("region"),
                        "project_type": case.get("project_type"),
                        "application_area": case.get("application_area"),
                        "product": case.get("product"),
                        "installation_method": case.get("installation_method"),
                        "area_m2": case.get("area_m2"),
                        "completion_year": case.get("completion_year"),
                        "source_type": case.get("source_type"),
                        "verification_note": case.get("verification_note"),
                        "score": round(score, 4),
                        "citation": {
                            "document_name": case.get("source_document"),
                            "source_page": case.get("source_page"),
                            "section_heading": "应用案例",
                            "bbox": None,
                        },
                        "visual_asset_ids": case.get("visual_asset_ids") or [],
                    }
                )
                if case_name_key:
                    selected_case_names.add(case_name_key)
            elif document.get("kind") == "visual":
                if visual_scope == "product" and not self._is_product_gallery_visual(document):
                    continue
                if procedure_request:
                    source_refs = document.get("source_refs") or []
                    visual_document_id = str(source_refs[0].get("document_id") or "") if source_refs else ""
                    if visual_document_id not in selected_procedure_document_ids:
                        continue
                if any(marker in visual_query for marker in ("图", "示意", "节点")) and document.get("effective_image_kind") == "table_or_parameter_sheet":
                    continue
                is_requested_node_visual = (
                    node_atlas_request
                    and "04_node_atlas" in set(document.get("knowledge_domains") or [])
                )
                if (
                    visual_scope != "product"
                    and
                    required_visual_terms
                    and not (required_visual_terms & set(document.get("tokens") or []))
                    and not is_requested_node_visual
                ):
                    continue
                source_label = str(document.get("search_text") or "")
                if required_named_visual_terms and not all(term in source_label for term in required_named_visual_terms):
                    continue
                visual_candidates.append((score, document))

        visual_assets: list[dict[str, Any]] = []
        effective_case_scope = visual_scope == "case" or retrieval_mode == "case_reference"
        if effective_case_scope and visual_k and wants_visuals:
            # Case records are the selection source of truth.  Generic visual
            # recall must not consume the budget before their linked photos.
            visual_assets = self.visuals_for_project_cases(project_cases, visual_k=visual_k)
        elif visual_k and wants_visuals and (visual_candidates or visual_scope == "product"):
            # The cross-encoder reranker emits probability-like values between
            # zero and one, while the original BM25 branch used larger scores.
            # A fixed minimum of 1.0 therefore discarded *every* node drawing
            # after hybrid retrieval, even when the node-atlas domain had been
            # selected deliberately.  For a node request, prefer pictures
            # whose own local context names the requested node, then return the
            # best remaining atlas visuals as evidence.  The UI still presents
            # their source/page and linked text instead of treating an image as
            # an independent technical fact.
            if node_atlas_request:
                node_terms = (
                    "门窗", "窗口", "洞口", "阴角", "阳角", "勒脚", "女儿墙", "檐口", "收口",
                )
                requested_terms = tuple(term for term in node_terms if term in visual_query)

                def node_visual_sort_key(item: tuple[float, dict[str, Any]]) -> tuple[int, float]:
                    score, document = item
                    label = str(document.get("search_text") or "")
                    direct_match = bool(requested_terms) and any(term in label for term in requested_terms)
                    return (1 if direct_match else 0, score)

                visual_candidates.sort(key=node_visual_sort_key, reverse=True)
                candidate_iterable = visual_candidates
            elif visual_scope == "product":
                product_gallery_candidates = self._product_gallery_candidates(
                    visual_query,
                    visual_candidates,
                )
                candidate_iterable = self._prioritise_product_visuals(
                    visual_query,
                    product_gallery_candidates,
                    product_overview_request=product_overview_request,
                )
            else:
                best_visual_score = visual_candidates[0][0]
                minimum_visual_score = max(1.0, best_visual_score * 0.5)
                candidate_iterable = [
                    item for item in visual_candidates if item[0] >= minimum_visual_score
                ]

            for score, document in candidate_iterable:
                if len(visual_assets) >= visual_k:
                    break
                gallery_type = visual_scope if visual_scope != "mixed" else (
                    "node" if node_atlas_request else "process" if procedure_request else "mixed"
                )
                selection_reason = "semantic_match"
                matched_variant_or_code = None
                if gallery_type == "product":
                    exact_product, matched_variant_or_code = self._product_visual_match(
                        visual_query,
                        document,
                    )
                    if exact_product:
                        selection_reason = "exact_product_catalog_match"
                    elif product_overview_request:
                        selection_reason = "product_catalog_overview"
                visual_assets.append(
                    self._visual_payload(
                        document,
                        score=score,
                        gallery_type=gallery_type,
                        selection_reason=selection_reason,
                        matched_variant_or_code=matched_variant_or_code,
                    )
                )

        return {
            "text_evidence": text_evidence,
            "visual_assets": visual_assets,
            "project_cases": project_cases,
            "meta": self._meta(
                query_tokens,
                matched_document_count=len(scored),
                node_atlas_request=node_atlas_request,
                standard_request=standard_request,
                product_overview_request=product_overview_request,
                retrieval_mode=retrieval_mode,
                wants_visuals=wants_visuals,
                visual_scope="case" if effective_case_scope else visual_scope,
            ),
        }

    def _meta(
        self,
        query_tokens: list[str],
        matched_document_count: int = 0,
        node_atlas_request: bool = False,
        standard_request: bool = False,
        product_overview_request: bool = False,
        retrieval_mode: str = "factual_lookup",
        wants_visuals: bool = False,
        visual_scope: str = "mixed",
    ) -> dict[str, Any]:
        hybrid_validation = getattr(
            self,
            "_hybrid_validation",
            {"requested": False, "ready": False, "reason": "not_initialised"},
        )
        return {
            "strategy": (
                "local_hybrid_bm25_dense_rerank"
                if hybrid_validation.get("ready")
                else "local_bm25_lexical"
            ),
            "query_token_count": len(query_tokens),
            "matched_document_count": matched_document_count,
            "knowledge_domain_routing": {
                "node_atlas_priority": node_atlas_request,
                "standard_priority": standard_request,
                "product_overview_priority": product_overview_request,
                "retrieval_mode": retrieval_mode,
            },
            "index_metadata": self.metadata,
            "hybrid_validation": hybrid_validation,
            "weak_match_policy": "fail_closed_substantive_overlap_and_relative_threshold",
            "privacy": "local_index_no_cloud_upload",
            "wants_visuals": bool(wants_visuals),
            "visual_scope": visual_scope if visual_scope in VISUAL_SCOPES else "mixed",
        }

    def visual_asset(self, asset_id: str) -> dict[str, Any] | None:
        return self.visual_by_asset_id.get(asset_id)
