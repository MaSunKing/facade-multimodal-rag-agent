"""Local, traceable retrieval over prepared construction RAG evidence.

The first retrieval layer is intentionally lexical BM25.  It needs no cloud
service, no GPU and no embedding model at request time.  Every returned item
preserves the document and page needed for a customer-visible citation.
"""

from __future__ import annotations

import json
import math
import gc
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
RAG_INDEX_PATH = ROOT / "data" / "sales" / "processed" / "rag_index" / "lexical_index.json"
DENSE_INDEX_PATH = ROOT / "data" / "sales" / "processed" / "rag_index" / "dense_index.npz"
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
NODE_ATLAS_QUERY_MARKERS = (
    "节点", "图集", "门窗", "洞口", "窗口", "阴角", "阳角", "勒脚", "女儿墙", "檐口", "收口",
)
STANDARD_QUERY_MARKERS = ("规范", "标准", "jgj", "jgt", "jg/t", "验收", "性能指标")


def tokenize(text: str) -> list[str]:
    """Tokenise mixed Chinese / dimensions for a small local BM25 index."""

    # Product catalogues use registered marks inside product names (for
    # example 企业产品).  Treat them as formatting, not token boundaries, so a
    # customer's “保温装饰一体板” query matches the source document.
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


class LocalRagRetriever:
    """Read-only RAG index with text evidence and original visual assets."""

    def __init__(self, index_path: Path = RAG_INDEX_PATH) -> None:
        self.index_path = index_path
        if not index_path.exists():
            raise FileNotFoundError(
                "未找到本地 RAG 索引。请先运行 scripts\\build_rag_index.ps1。"
            )
        self.payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.documents: list[dict[str, Any]] = self.payload.get("documents", [])
        # The repository was reorganised from data/processed to
        # data/sales/processed.  Older, otherwise valid indexes may still hold
        # absolute paths from before that move.  Resolve those paths at load
        # time so an index rebuild is not required merely to serve the original
        # evidence crop.
        for document in self.documents:
            if document.get("kind") == "visual" and document.get("image_path"):
                document["image_path"] = str(self._resolve_visual_path(str(document["image_path"])))
        self.document_frequency: dict[str, int] = self.payload.get("document_frequency", {})
        self.avg_doc_length = float(self.payload.get("average_document_length", 1.0)) or 1.0
        self.document_count = len(self.documents)
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
    def _domain_priority(
        document: dict[str, Any], *, node_atlas_request: bool, standard_request: bool, procedure_request: bool
    ) -> int:
        """Route specialised questions to their authoritative knowledge domain.

        BM25 still ranks the evidence inside a domain.  The priority only
        prevents a generic construction-plan phrase from crowding out the
        requested node atlas or standard before it is considered.
        """

        domains = set(document.get("knowledge_domains") or [])
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
        self, query_tokens: list[str], *, node_atlas_request: bool, standard_request: bool, procedure_request: bool
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
        return "03_construction_method" in domains and bool(
            {"construction_method", "construction_step"} & labels
        )

    def _procedure_sequence(
        self, scored: list[tuple[float, dict[str, Any]],], top_k: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return adjacent evidence from one authoritative procedure section.

        A process question is different from a comparison question: the useful
        answer is normally distributed across neighbouring source chunks.  The
        most relevant heading is used as an anchor and the following labelled
        steps from the same document are kept in source order.  This works for
        any procedure document and never depends on a product-specific rule.
        """

        anchor: tuple[float, dict[str, Any]] | None = next(
            (
                item
                for item in scored
                if item[1].get("kind") == "text"
                and self._is_construction_step(item[1])
                and self._document_text(item[1])
            ),
            None,
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
    ) -> list[tuple[float, dict[str, Any]]]:
        """Dense-recall plus local reranking, with a lexical fallback.

        The two 0.6B retrieval models are deliberately loaded one after the
        other and released before the answer model generates text.  Set
        ``RAG_HYBRID_ENABLED=0`` to force BM25-only retrieval.
        """

        lexical = self._lexical_scored(
            query_tokens,
            node_atlas_request=node_atlas_request,
            standard_request=standard_request,
            procedure_request=procedure_request,
        )
        if os.getenv("RAG_HYBRID_ENABLED", "0").strip() != "1" or not DENSE_INDEX_PATH.exists():
            return lexical

        try:
            from backend.sales.dense_retrieval import LocalQwenEmbedding, LocalQwenReranker

            dense = np.load(DENSE_INDEX_PATH, allow_pickle=False)
            ids = [str(item) for item in dense["ids"].tolist()]
            vectors = np.asarray(dense["vectors"], dtype=np.float32)
            if len(ids) != len(vectors) or vectors.ndim != 2:
                return lexical
            embedding = LocalQwenEmbedding()
            try:
                query_vector = embedding.encode([query], query=True, batch_size=1, max_length=1024)[0]
            finally:
                del embedding
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            dense_scores = vectors @ query_vector.astype(np.float32)
            dense_order = np.argsort(-dense_scores)[:48]
            document_by_id = {str(document.get("id")): document for document in self.documents}
            candidate_ids = [str(document.get("id")) for _, document in lexical[:48]]
            candidate_ids.extend(ids[int(index)] for index in dense_order)
            candidate_pool = [document_by_id[item] for item in dict.fromkeys(candidate_ids) if item in document_by_id]
            candidate_pool = [item for item in candidate_pool if self._document_text(item)]
            # Keep original diagrams/photos in the reranking pool.  Without a
            # quota, many short text chunks can crowd out a relevant node
            # drawing before the visual-selection stage ever sees it.
            non_visual = [item for item in candidate_pool if item.get("kind") != "visual"][:24]
            visual = [item for item in candidate_pool if item.get("kind") == "visual"][:8]
            candidates = [*non_visual, *visual]
            if not candidates:
                return lexical

            reranker = LocalQwenReranker()
            try:
                rerank_scores = reranker.score(query, [self._document_text(item) for item in candidates])
            finally:
                del reranker
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            reranked = list(zip(rerank_scores, candidates))
            reranked.sort(
                key=lambda item: (
                    self._domain_priority(
                        item[1],
                        node_atlas_request=node_atlas_request,
                        standard_request=standard_request,
                        procedure_request=procedure_request,
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
            }
            for source in source_refs
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        visual_k: int = 3,
        case_k: int = 5,
        retrieval_mode: str = "factual_lookup",
    ) -> dict[str, Any]:
        retrieval_mode = retrieval_mode if retrieval_mode in RETRIEVAL_MODES else "factual_lookup"
        query_tokens = tokenize(query)
        procedure_request = retrieval_mode == "procedure"
        project_fit_request = retrieval_mode == "project_fit"
        normal_query = query.lower()
        node_atlas_request = retrieval_mode == "node_detail" or any(marker in query for marker in NODE_ATLAS_QUERY_MARKERS)
        standard_request = any(marker in normal_query for marker in STANDARD_QUERY_MARKERS)
        if procedure_request:
            # A named process should stay inside its procedure document.  The
            # neutral terms also cover paraphrases such as “怎么做/工序”.
            query_tokens.extend(tokenize("施工流程 施工步骤 工序"))
        elif project_fit_request:
            # A project-fit question can legitimately compare several document
            # backed installation paths before it asks for missing conditions.
            query_tokens.extend(tokenize(CONSTRUCTION_QUERY_EXPANSION))
        required_visual_terms = specific_chinese_bigrams(query)
        required_named_visual_terms = required_visual_phrases(query)
        if not query_tokens:
            return {
                "text_evidence": [],
                "visual_assets": [],
                "project_cases": [],
                "meta": self._meta(query_tokens, retrieval_mode=retrieval_mode),
            }

        scored = self._hybrid_scored(
            query,
            query_tokens,
            node_atlas_request=node_atlas_request,
            standard_request=standard_request,
            procedure_request=procedure_request,
        )

        text_evidence: list[dict[str, Any]] = []
        selected_procedure_document_ids: set[str] = set()
        if procedure_request:
            for score, document in self._procedure_sequence(scored, top_k):
                text_evidence.append(
                    {
                        "chunk_id": document["id"],
                        "text": document["text"],
                        "score": round(score, 4),
                        "facts_eligible": True,
                        "citations": self._public_sources(document.get("source_refs") or []),
                        "source_taxonomy": document.get("source_taxonomy") or [],
                        "sales_playbook_use": document.get("sales_playbook_use"),
                    }
                )
                document_id = self._primary_document_id(document)
                if document_id:
                    selected_procedure_document_ids.add(document_id)
        project_cases: list[dict[str, Any]] = []
        selected_case_names: set[str] = set()
        selected_source_documents: set[str] = set()
        visual_candidates: list[tuple[float, dict[str, Any]]] = []
        for score, document in scored:
            if document.get("kind") == "text" and not procedure_request and len(text_evidence) < top_k:
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
                text_evidence.append(
                    {
                        "chunk_id": document["id"],
                        "text": document["text"],
                        "score": round(score, 4),
                        "facts_eligible": True,
                        "citations": self._public_sources(document.get("source_refs") or []),
                        "source_taxonomy": document.get("source_taxonomy") or [],
                        "sales_playbook_use": document.get("sales_playbook_use"),
                    }
                )
                if primary_document_id:
                    selected_source_documents.add(primary_document_id)
            elif document.get("kind") == "project_case" and len(project_cases) < case_k:
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
                if procedure_request:
                    source_refs = document.get("source_refs") or []
                    visual_document_id = str(source_refs[0].get("document_id") or "") if source_refs else ""
                    if visual_document_id not in selected_procedure_document_ids:
                        continue
                if any(marker in query for marker in ("图", "示意", "节点")) and document.get("effective_image_kind") == "table_or_parameter_sheet":
                    continue
                is_requested_node_visual = (
                    node_atlas_request
                    and "04_node_atlas" in set(document.get("knowledge_domains") or [])
                )
                if (
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
        if visual_candidates and visual_k:
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
                requested_terms = tuple(term for term in node_terms if term in query)

                def node_visual_sort_key(item: tuple[float, dict[str, Any]]) -> tuple[int, float]:
                    score, document = item
                    label = str(document.get("search_text") or "")
                    direct_match = bool(requested_terms) and any(term in label for term in requested_terms)
                    return (1 if direct_match else 0, score)

                visual_candidates.sort(key=node_visual_sort_key, reverse=True)
                candidate_iterable = visual_candidates
            else:
                best_visual_score = visual_candidates[0][0]
                minimum_visual_score = max(1.0, best_visual_score * 0.5)
                candidate_iterable = [
                    item for item in visual_candidates if item[0] >= minimum_visual_score
                ]

            for score, document in candidate_iterable:
                if len(visual_assets) >= visual_k:
                    break
                visual_assets.append(
                    {
                        "asset_id": document["asset_id"],
                        "customer_title": document.get("customer_title"),
                        "asset_type": document.get("asset_type"),
                        "effective_image_kind": document.get("effective_image_kind"),
                        "score": round(score, 4),
                        "citation": document.get("citation"),
                        "visual_endpoint": f"/api/copilot/visual/{document['asset_id']}",
                        "facts_eligible": False,
                        "linked_text_evidence": document.get("linked_text_evidence") or [],
                        "multimodal_bundle_id": document.get("multimodal_bundle_id"),
                    }
                )

        # A catalogue-case request should surface the page's original project
        # photo even if the wording did not separately match its image label.
        selected_asset_ids = {str(asset["asset_id"]) for asset in visual_assets}
        for case in project_cases:
            for asset_id in case.get("visual_asset_ids") or []:
                asset = self.visual_by_asset_id.get(str(asset_id))
                if asset is None or str(asset_id) in selected_asset_ids or len(visual_assets) >= visual_k:
                    continue
                visual_assets.append(
                    {
                        "asset_id": asset_id,
                        "customer_title": asset.get("customer_title") or case.get("project_name"),
                        "asset_type": asset.get("asset_type"),
                        "effective_image_kind": asset.get("effective_image_kind"),
                        "score": case.get("score"),
                        "citation": asset.get("citation"),
                        "visual_endpoint": f"/api/copilot/visual/{asset_id}",
                        "facts_eligible": False,
                        "linked_text_evidence": asset.get("linked_text_evidence") or [],
                        "multimodal_bundle_id": asset.get("multimodal_bundle_id"),
                    }
                )
                selected_asset_ids.add(str(asset_id))

        return {
            "text_evidence": text_evidence,
            "visual_assets": visual_assets,
            "project_cases": project_cases,
            "meta": self._meta(
                query_tokens,
                matched_document_count=len(scored),
                node_atlas_request=node_atlas_request,
                standard_request=standard_request,
                retrieval_mode=retrieval_mode,
            ),
        }

    def _meta(
        self,
        query_tokens: list[str],
        matched_document_count: int = 0,
        node_atlas_request: bool = False,
        standard_request: bool = False,
        retrieval_mode: str = "factual_lookup",
    ) -> dict[str, Any]:
        return {
            "strategy": (
                "local_hybrid_bm25_dense_rerank"
                if os.getenv("RAG_HYBRID_ENABLED", "0").strip() == "1" and DENSE_INDEX_PATH.exists()
                else "local_bm25_lexical"
            ),
            "query_token_count": len(query_tokens),
            "matched_document_count": matched_document_count,
            "knowledge_domain_routing": {
                "node_atlas_priority": node_atlas_request,
                "standard_priority": standard_request,
                "retrieval_mode": retrieval_mode,
            },
            "index_metadata": self.metadata,
            "privacy": "local_index_no_cloud_upload",
        }

    def visual_asset(self, asset_id: str) -> dict[str, Any] | None:
        return self.visual_by_asset_id.get(asset_id)
