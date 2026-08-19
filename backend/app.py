"""Local Qwen3-VL service for the Facade Multimodal RAG Agent."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import re
import threading
import time
from contextlib import contextmanager
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Literal
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.sales.answer_graph import build_customer_answer_graph
from backend.sales.baidu_search import BaiduSearchError, is_baidu_search_configured, search_baidu_web
from backend.documents.router import router as customer_documents_router
from backend.documents.customer_sessions import (
    get_session,
    retrieve as retrieve_customer_documents,
    temporary_visual_files,
)
from backend.sales.retriever import LocalRagRetriever, RAG_INDEX_PATH
from backend.sales.tool_planner import ToolPlan, fallback_plan, guard_plan


# Customer questions, product PDFs and uploaded images are private.  Do not
# allow a developer-machine LangSmith setting to export LangGraph run data.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

ROOT = Path(__file__).resolve().parents[1]
# Qwen3-VL is the single local generation model for both text-only and
# customer-image questions.  It is loaded with 4-bit NF4 quantization below;
# the original BF16 files remain on disk and are never uploaded anywhere.
MODEL_PATH = Path(os.getenv("FACADE_MODEL_PATH", ROOT / "models" / "Qwen3-VL-8B-Instruct"))
PUBLIC_FRONTEND = os.getenv("FACADE_PUBLIC_FRONTEND", "*")
ALLOWED_ORIGINS = [origin.strip() for origin in PUBLIC_FRONTEND.split(",") if origin.strip()]
MAX_UPLOADED_IMAGE_BYTES = 8 * 1024 * 1024
MAX_UPLOADED_IMAGE_PIXELS = 20_000_000
IMAGE_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
VISUAL_IDENTITY_INDEX_PATH = ROOT / "data" / "sales" / "processed" / "visual_identity_index" / "appearance_index.npz"
VISUAL_IDENTITY_MANIFEST_PATH = ROOT / "data" / "sales" / "processed" / "visual_identity_index" / "appearance_manifest.json"
# Appearance matching is only a candidate-selection signal.  It intentionally
# uses a high threshold and a margin over the next result, because the local
# catalogue currently has only positive reference images and therefore cannot
# prove product authenticity from an arbitrary internet image.
VISUAL_APPEARANCE_MATCH_THRESHOLD = float(os.getenv("FACADE_VISUAL_MATCH_THRESHOLD", "0.82"))
VISUAL_APPEARANCE_MATCH_MARGIN = float(os.getenv("FACADE_VISUAL_MATCH_MARGIN", "0.03"))
EXPLICIT_PRODUCT_TERMS = (
    "企业产品",
    "保温装饰一体板",
    "无机仿石",
    "保温装饰一体板",
    "岩棉一体板",
)
FACADE_DOMAIN_TERMS = (
    "企业产品",
    "无机仿石",
    "外墙",
    "幕墙",
    "饰面",
    "保温",
    "干挂",
    "粘锚",
    "锚固",
    "挂件",
    "龙骨",
    "一体板",
    "岩棉",
    "石材",
    "节点",
    "窗洞",
    "阴角",
    "阳角",
    "勒脚",
    "女儿墙",
    "施工方案",
    "建筑立面",
    "项目案例",
)

app = FastAPI(title="Facade Copilot Local Model Service", version="0.1.0")
app.include_router(customer_documents_router)
app.add_middleware(
    CORSMiddleware,
    # The public front end is deliberately static and may move from the
    # prototype host to a China-friendly static host. The API has no cookies
    # or user credentials, while request rate limiting protects GPU use.
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def allow_private_network_request(request, call_next):
    """Allow the public static prototype to call the owner's loopback API.

    This does not expose the API to the internet; it only helps a browser on
    the same computer reach 127.0.0.1 while the FastAPI process is running.
    """
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.middleware("http")
async def limit_anonymous_generation(request: Request, call_next):
    """Keep a public tunnel from allowing unlimited GPU requests per IP."""
    if request.method == "POST" and request.url.path in {
        "/api/copilot/draft",
        "/api/copilot/answer",
        "/api/copilot/documents",
    }:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _rate_limit_lock:
            history = _request_history[client_ip]
            while history and now - history[0] >= RATE_LIMIT_WINDOW_SECONDS:
                history.popleft()
            if len(history) >= RATE_LIMIT_MAX_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "模型演示当前繁忙：每个 IP 每分钟最多 4 次请求，请稍后再试。"},
                )
            history.append(now)
    return await call_next(request)

_model_lock = threading.Lock()
_generation_lock = threading.Lock()
_model_activity_lock = threading.Lock()
_tokenizer: Any | None = None
_processor: Any | None = None
_model: Any | None = None
_model_last_used_monotonic: float | None = None
_model_idle_shutdown = threading.Event()
_model_idle_reaper_started = False
_answer_graph_lock = threading.Lock()
_answer_graph: Any | None = None
_retriever: LocalRagRetriever | None = None
_retriever_index_mtime_ns: int | None = None
_retriever_lock = threading.Lock()
_visual_identity_index: tuple[Any, list[dict[str, Any]], int] | None = None
_visual_identity_index_lock = threading.Lock()
_request_history: defaultdict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 4
# Keep the local GPU available for normal desktop work when the Copilot is not
# serving a request.  Set this to 0 only when a permanently warm model is
# explicitly desired; the default releases Qwen3-VL after ten idle minutes.
MODEL_IDLE_UNLOAD_SECONDS = max(0, int(os.getenv("FACADE_MODEL_IDLE_UNLOAD_SECONDS", "600")))
MODEL_IDLE_REAPER_INTERVAL_SECONDS = 15

INTENTS = {
    "product_parameter",
    "technical_performance",
    "application_condition",
    "construction",
    "quote_delivery",
    "warranty",
    "case_reference",
    "comparison",
    "complaint_after_sales",
    "unknown",
}
TASK_TYPES = {
    "factual_lookup",
    "procedure",
    "node_detail",
    "case_reference",
    "comparison",
    "project_fit",
    "commercial",
    "unknown",
}
TASK_TYPE_DEFAULT_INTENT = {
    "factual_lookup": "product_parameter",
    "procedure": "construction",
    "node_detail": "construction",
    "case_reference": "case_reference",
    "comparison": "comparison",
    "project_fit": "application_condition",
    "commercial": "quote_delivery",
    "unknown": "unknown",
}
COMMERCIAL_EVIDENCE_TERMS = ("报价", "价格", "交期", "到货", "折扣", "付款", "质保", "保修", "合同")
MANUAL_HANDOFF_TERMS = ("转人工", "人工复核", "人工确认", "联系售前", "售前技术", "技术人员", "提交售前")


class ProjectContext(BaseModel):
    product_label: str | None = Field(default=None, max_length=100)
    customer_type: str | None = Field(default=None, max_length=50)
    region: str | None = Field(default=None, max_length=100)
    area_sqm: str | None = Field(default=None, max_length=50)


class ConversationTurn(BaseModel):
    """Short, client-held context for resolving follow-up questions.

    This is intentionally limited to text.  Images and full RAG evidence are
    never persisted as conversation memory, and the backend never writes these
    turns to a database or debug file by default.
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=800)


class DraftRequest(BaseModel):
    customer_question: str = Field(min_length=1, max_length=3000)
    project_context: ProjectContext = Field(default_factory=ProjectContext)
    conversation_context: list[ConversationTurn] = Field(default_factory=list, max_length=6)
    # The public-search request sends the full current question only.  Local
    # PDFs, images and browser history always remain on this workstation.
    use_online_search: bool = False
    # The browser sends a compressed image data URL directly to the local
    # backend.  It is decoded into a temporary local file only for this one
    # inference and is deleted in ``temporary_uploaded_image``.
    image_data_url: str | None = Field(default=None, max_length=12_000_000)
    # Up to four heterogeneous files parsed into process-local canonical
    # evidence.  The ID expires and is never written into the company RAG.
    document_session_id: str | None = Field(default=None, max_length=100)


class DraftResponse(BaseModel):
    intent: str
    normalized_terms: list[dict[str, Any]]
    answerable: bool
    customer_reply: str
    key_points: list[str]
    citations: list[dict[str, Any]]
    missing_information: list[str]
    risk_warnings: list[str]
    next_action: str
    meta: dict[str, Any]
    # These are deliberately separated from RAG-backed key points.  They are
    # limited to directly visible, non-engineering observations from a
    # customer-uploaded image and never replace a cited technical conclusion.
    image_observations: list[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    customer_question: str = Field(min_length=1, max_length=3000)
    top_k: int = Field(default=5, ge=1, le=8)
    visual_k: int = Field(default=5, ge=0, le=5)


class AnswerResponse(DraftResponse):
    visual_assets: list[dict[str, Any]]
    retrieval: dict[str, Any]
    online_sources: list[dict[str, Any]] = Field(default_factory=list)


def compact_conversation_context(request: DraftRequest) -> list[dict[str, str]]:
    """Return only bounded text context that the browser voluntarily supplied."""

    turns: list[dict[str, str]] = []
    for turn in request.conversation_context[-6:]:
        content = re.sub(r"\s+", " ", turn.content).strip()
        if content:
            turns.append({"role": turn.role, "content": content[:800]})
    return turns


def conversation_context_text(request: DraftRequest) -> str:
    """Create a compact local-only context block for routing and generation."""

    labels = {"user": "Customer", "assistant": "Assistant"}
    return "\n".join(
        f"{labels.get(turn['role'], 'Message')}: {turn['content']}"
        for turn in compact_conversation_context(request)
    )


def conversation_retrieval_hint(request: DraftRequest) -> str:
    """Keep prior customer constraints available to the local retriever.

    Assistant replies are deliberately excluded: a prior reply can be an
    incomplete explanation, whereas customer turns are the source of project
    constraints such as the building type, region or selected installation
    method.
    """

    prior_customer_turns = [
        turn["content"] for turn in compact_conversation_context(request) if turn["role"] == "user"
    ]
    return "\n".join(prior_customer_turns[-3:])[:1600]


PROJECT_NAME_MARKERS = (
    "项目",
    "工程",
    "大厦",
    "广场",
    "园区",
    "学校",
    "医院",
    "中心",
    "小区",
    "酒店",
    "厂房",
    "基地",
    "场馆",
    "车站",
    "剧院",
)
GENERIC_CASE_ENQUIRIES = (
    "有哪些项目",
    "有什么项目",
    "项目有哪些",
    "有项目吗",
    "项目案例",
    "案例有哪些",
    "哪些案例",
    "有没有案例",
)


def is_named_project_web_query(question: str, *, case_reference: bool = False) -> bool:
    """Identify a project-specific lookup that benefits from public search.

    This is routing only: it does not decide whether any claim is true.  Broad
    catalogue requests remain local, while a named/identifiable project can
    automatically look for public information even when the customer has not
    manually selected the online-search option.
    """

    value = re.sub(r"\s+", "", question).strip()
    if len(value) < 6 or any(pattern in value for pattern in GENERIC_CASE_ENQUIRIES):
        return False
    if any(mark in value for mark in ("《", "》", "\"", "“", "”", "'")):
        return True
    for marker in PROJECT_NAME_MARKERS:
        if marker not in value:
            continue
        prefix = value.split(marker, 1)[0].rstrip("的")
        # A broad "这个项目" / "我的项目" question should stay local.  A
        # longer, non-generic prefix such as "济南万象城" is a concrete
        # public-project signal, even when it is outside our façade domain.
        generic_prefixes = ("这个", "本", "该", "我的", "我们的", "你们的", "当前", "客户", "哪些", "什么", "有没有", "有", "如何", "怎么")
        if case_reference or (len(prefix) >= 3 and not any(token in prefix for token in generic_prefixes)):
            return True
    return False


def maybe_search_online(
    request: DraftRequest, query: str, *, automatic_named_project_lookup: bool = False
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Search public web pages with the full current question.

    The browser option requests search for any question.  A named-project
    lookup can additionally trigger it automatically.  In both cases only the
    current customer question is sent; local evidence, PDFs, images and
    browser history are not included in the outbound request.
    """

    if not request.use_online_search and not automatic_named_project_lookup:
        return [], {"status": "not_requested"}
    if not is_baidu_search_configured():
        return [], {"status": "not_configured", "message": "Baidu AI Search API key is not configured locally."}
    trigger = "customer_selected" if request.use_online_search else "named_project_auto"
    try:
        result = search_baidu_web(query)
        return list(result.get("sources") or []), {
            "status": "ok",
            "trigger": trigger,
            "query": str(result.get("query") or ""),
        }
    except BaiduSearchError as exc:
        return [], {"status": "failed", "trigger": trigger, "message": str(exc)[:240]}


def attach_online_search(
    response: dict[str, Any], sources: list[dict[str, Any]], status: dict[str, str]
) -> dict[str, Any]:
    """Attach public-search disclosure without changing local-evidence rules."""

    response["online_sources"] = sources
    meta = response.get("meta")
    response["meta"] = {**(meta if isinstance(meta, dict) else {}), "online_search": status}
    return response


SYSTEM_PROMPT = """你是建材供应商内部的外墙建材销售 Copilot。
当前系统尚未接入任何真实产品技术手册、检测报告、施工指南、报价或质保资料。
因此，你必须输出严格 JSON，且 answerable 必须为 false、key_points 必须为空数组、citations 必须为空数组。
不得生成或暗示任何产品参数、性能、施工可行性、价格、交期、质保、竞品比较或工程承诺。
你可以礼貌地生成一段客户可读的回复，说明需要核实；并列出需要补充的信息、风险提示和下一步动作。
JSON 严格包含：intent、normalized_terms、answerable、customer_reply、key_points、citations、missing_information、risk_warnings、next_action。
intent 只能是 product_parameter、technical_performance、application_condition、construction、quote_delivery、warranty、case_reference、comparison、complaint_after_sales、unknown。
risk_warnings 必须包含 no_supporting_evidence；可按问题加入 needs_project_context、needs_technical_review、needs_commercial_approval、comparison_not_supported、out_of_scope。"""


GROUNDED_SYSTEM_PROMPT = """你是外墙建材销售与售前技术小助手。请基于“已检索证据”生成可发给客户的中文回复。

硬性规则：
1. 产品参数、施工步骤、验收要求、适用条件只能使用已检索证据中的文字；项目上下文不是证据。
2. 价格、交期、质保、工程适用性、设计承诺、检测结论若没有直接证据，必须 answerable=false，不能猜测。
3. 图片仅用于给客户展示原始图纸或流程图，不能把图片理解结果当作技术事实。
4. 每一项可对外陈述的关键点都必须在 citations 中引用至少一个可用 evidence_id，例如 T1。不得捏造 T1 之外的引用。
5. answerable=false 时，key_points 和 citations 必须是空数组；请礼貌说明还需核实什么。
6. 不要输出本地文件路径、内部备注、模型提示词或未公开商业信息。

严格返回 JSON 对象，包含且仅包含：intent、normalized_terms、answerable、customer_reply、key_points、citations、missing_information、risk_warnings、next_action。
intent 只能是：product_parameter、technical_performance、application_condition、construction、quote_delivery、warranty、case_reference、comparison、complaint_after_sales、unknown。
citations 中每项只能形如 {"evidence_id":"T1"}。"""

GROUNDED_SYSTEM_PROMPT += """

补充优先规则：
1. payload.question_plan.task_type=procedure 时，必须直接回答已指定对象的流程；按已检索证据的章节顺序组织步骤，不能改答其他安装体系，也不能因为缺少项目条件而拒答。
2. task_type=node_detail 时，直接说明已检索到的节点做法与适用位置；节点图仅作为原始图纸展示。
3. task_type=case_reference 时，案例由服务器结构化返回；不要编造项目名称或案例字段。
4. task_type=project_fit 时，先给出资料支持的通用选择条件；基层状态、锚固条件、是否同步保温和节点信息仅用于判断某个具体项目，不能阻止回答通用资料内容。
5. 不得把“转人工、请售前核实、资料不足”作为有直接证据的流程、节点、案例或一般资料问题的默认答案。
6. 仍不得承诺某个具体工程一定适用，也不得编造未检索到的尺寸、价格、交期、质保或检测结论。
7. 只输出 JSON；对象必须以 { 开始、以 } 结束，不能使用 Markdown 代码块或额外说明。
"""


# The original prompt predates customer image upload.  Keep this final schema
# declaration adjacent to the safety rules so a multimodal response cannot
# silently turn an image guess into a RAG-backed technical claim.
GROUNDED_SYSTEM_PROMPT += """

When the customer provided an image, you may fill image_observations with at
most five concise observations that are directly visible in that image.  Do
not infer material grade, engineering safety, installation feasibility,
dimensions, hidden layers, or compliance from pixels alone.  Put any
RAG-backed technical statement only in key_points and cite it.  If no text
evidence was retrieved, answerable must be false; image_observations may still
describe visible content, but key_points and citations must remain empty.

Final JSON schema override: output exactly these keys and no others:
intent, normalized_terms, answerable, customer_reply, key_points, citations,
missing_information, risk_warnings, next_action, image_observations.
For a text-only request, image_observations must be an empty array.

产品身份规则：customer_image_identity.status=appearance_candidate 表示上传图片
与内部参考图外观相似；它不是产品认证。customer_image_identity.status=unverified
表示没有匹配到足够可信的外观参考。两种状态下，都不得把图片中的任何饰面、板材、
节点或工程称为“企业产品”或“保温装饰一体板产品”，也不得因为图片外观相似而关联企业产品资料。
只有用户在问题中明确点名某个产品时，才可以回答该“被点名产品”的资料内容；仍要
明确说明上传图片本身尚未完成产品身份确认。不要根据图片推断性能、真伪、工程适用性
或合格性。
"""


GROUNDED_SYSTEM_PROMPT += """

Conversation-context policy: `conversation_context` is short-lived context supplied by the browser. Use it only to resolve references and retain customer constraints. It is not technical evidence. Technical facts, construction steps and product claims must still be supported by `retrieved_text_evidence` and cited with its evidence IDs.
"""


GROUNDED_SYSTEM_PROMPT += """

Online-source policy: evidence IDs starting with `W` are public web results retrieved from the current customer question. You may cite a `W` source only for the exact information in its excerpt. For a named external project, you may use `W` only to describe publicly stated project facts, and must call it "公开网页资料" rather than a company case or product application. Never use online results to verify private company product specifications, whether the project used this company's product, prices, delivery, warranty, engineering applicability, or an uploaded image's identity. Structured entries in `structured_project_cases` are local company catalogue cases and may be summarised only using their provided fields. Cite every factual statement with its matching `T` or `W` evidence ID.

Internal sales-playbook policy: an evidence item whose source_taxonomy has document_category=internal_sales_playbook may be used only when sales_playbook_use=supplementary_product_information. Present it as supplementary enterprise product information, not as a standard, test conclusion, engineering guarantee, price commitment, lifetime commitment, safety conclusion or competitor comparison. Other sales-playbook chunks are deliberately excluded from customer factual retrieval and must never be reconstructed from model memory.
"""


GENERAL_CHAT_SYSTEM_PROMPT = """你是一个乐于交流的中文助手，可以回答日常知识、写作、学习、代码和图片中直接可见的内容。

如果 payload 中提供了 online_sources，可基于其中的公开网页摘要回答与当前问题直接相关的公开事实；不得扩写摘要之外的细节，并在回复中明确说明这是“公开网页资料”。没有 online_sources 时，不要假装已经联网或引用实时信息。
不要把建材资料库或企业产品信息带入与其无关的问题。

严格只返回 JSON：
{
  "customer_reply": "自然、简洁的中文回复",
  "image_observations": ["仅当用户上传图片时，列出直接可见的内容；不要推断品牌、材质性能或安全结论"]
}
"""


def mark_model_activity() -> None:
    """Record active or completed local-model work for the idle reaper."""

    global _model_last_used_monotonic
    with _model_activity_lock:
        _model_last_used_monotonic = time.monotonic()


def model_idle_seconds() -> float | None:
    """Return the current warm-model idle time, or ``None`` when cold."""

    if _model is None:
        return None
    with _model_activity_lock:
        if _model_last_used_monotonic is None:
            return None
        return max(0.0, time.monotonic() - _model_last_used_monotonic)


@contextmanager
def generation_session() -> Iterator[None]:
    """Serialise GPU inference and make it ineligible for idle unloading."""

    with _generation_lock:
        mark_model_activity()
        try:
            yield
        finally:
            mark_model_activity()


def unload_model_if_idle() -> bool:
    """Release Qwen3-VL after the configured idle window without stopping API/RAG."""

    global _tokenizer, _processor, _model, _model_last_used_monotonic
    if MODEL_IDLE_UNLOAD_SECONDS <= 0:
        return False

    # A request takes this same lock while it generates or extracts visual
    # features, so a loaded model is never deleted during GPU work.
    with _generation_lock:
        with _model_lock:
            if _model is None:
                return False
            with _model_activity_lock:
                last_used = _model_last_used_monotonic
                if last_used is None or time.monotonic() - last_used < MODEL_IDLE_UNLOAD_SECONDS:
                    return False

            released_model = _model
            released_processor = _processor
            released_tokenizer = _tokenizer
            _model = None
            _processor = None
            _tokenizer = None
            with _model_activity_lock:
                _model_last_used_monotonic = None

    # Drop Python references before returning cached GPU blocks to CUDA.
    del released_model, released_processor, released_tokenizer
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            # ipc_collect is not implemented by every CUDA/Windows runtime.
            pass
    return True


def _idle_model_reaper() -> None:
    """Run independently of requests so the backend can remain API-online."""

    while not _model_idle_shutdown.wait(MODEL_IDLE_REAPER_INTERVAL_SECONDS):
        try:
            if unload_model_if_idle():
                print("Released idle Qwen3-VL model from GPU memory.", flush=True)
        except Exception as exc:
            # A later request can still load the model; cleanup must never
            # take down the local service.
            print(f"Idle model cleanup skipped: {type(exc).__name__}", flush=True)


@app.on_event("startup")
def start_idle_model_reaper() -> None:
    """Start one lightweight local thread for idle GPU cleanup."""

    global _model_idle_reaper_started
    if _model_idle_reaper_started:
        return
    _model_idle_shutdown.clear()
    thread = threading.Thread(target=_idle_model_reaper, name="facade-model-idle-reaper", daemon=True)
    thread.start()
    _model_idle_reaper_started = True


@app.on_event("shutdown")
def stop_idle_model_reaper() -> None:
    _model_idle_shutdown.set()


def load_model() -> tuple[Any, Any]:
    """Lazily load Qwen3-VL once so starting FastAPI is instant."""
    global _tokenizer, _processor, _model
    with _model_lock:
        if _model is not None and _tokenizer is not None:
            mark_model_activity()
            return _tokenizer, _model

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            mark_model_activity()
            return _tokenizer, _model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到本地模型目录：{MODEL_PATH}")

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA，无法加载本地 Qwen3-VL-8B。")

        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
        _tokenizer = _processor.tokenizer
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        _model.eval()
    mark_model_activity()
    return _tokenizer, _model


def load_processor() -> Any:
    """Return the Qwen3-VL processor after ensuring the local model is ready."""

    load_model()
    if _processor is None:
        raise RuntimeError("Qwen3-VL processor was not initialized.")
    return _processor


def generate_finance_semantic_mapping(prompt_text: str) -> str:
    """Run one local, JSON-only table-label mapping request.

    This helper is intentionally narrow: the finance intake service supplies
    table labels and validates the JSON before it can affect an upload.  No
    customer file is sent to an external API and the model is never asked to
    produce amounts or financial conclusions.
    """

    tokenizer, model = load_model()
    import torch

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "你只输出合法 JSON。不要解释、不要 Markdown、不要计算或生成客户财务金额。",
            },
            {"role": "user", "content": prompt_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with generation_session(), torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=420,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    mark_model_activity()
    return tokenizer.decode(generated[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)


def generate_finance_chat_response(prompt_text: str) -> str:
    """Generate a natural-language finance reply from a bounded local context."""

    tokenizer, model = load_model()
    import torch

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "你是企业财务与经营分析助手，用自然、专业的中文对话。"
                    "只能依据用户问题和提供的已确认资料回答；没有资料时明确说明并给出下一步。"
                    "不得编造公司事实、金额、期间、原因或数据来源，不得自行计算未提供的数字。"
                    "可给出有条件的管理建议，但必须标注为建议而非事实结论。"
                    "回答控制在 5 个短段以内，不要输出 Markdown 表格。"
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with generation_session(), torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=520,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    mark_model_activity()
    return tokenizer.decode(generated[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()


def generate_finance_clarification_action(prompt_text: str) -> str:
    """Classify a customer metric-definition clarification into guarded JSON."""

    tokenizer, model = load_model()
    import torch

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "只输出合法 JSON，不要 Markdown。不得计算、生成或修改任何金额。",
            },
            {"role": "user", "content": prompt_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with generation_session(), torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    mark_model_activity()
    return tokenizer.decode(generated[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()


def _decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    """Validate one browser image without retaining customer input on disk."""

    match = re.fullmatch(r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=\s]+)", data_url.strip())
    if not match:
        raise ValueError("图片格式仅支持 JPG、PNG 或 WebP。")
    mime_type, encoded = match.groups()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片内容无法读取。") from exc
    if not raw or len(raw) > MAX_UPLOADED_IMAGE_BYTES:
        raise ValueError("单张图片请控制在 8 MB 以内。")

    try:
        from PIL import Image

        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width * height > MAX_UPLOADED_IMAGE_PIXELS:
                raise ValueError("图片像素过大，请压缩后重试。")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("图片文件损坏或无法解析。") from exc
    return raw, mime_type


@contextmanager
def temporary_uploaded_image(data_url: str | None) -> Iterator[Path | None]:
    """Yield a short-lived local file required by Qwen3-VL image processing.

    The API does not create an upload directory.  The operating system's
    temporary file is removed in ``finally`` even if model inference fails.
    """

    if not data_url:
        yield None
        return

    raw, mime_type = _decode_image_data_url(data_url)
    path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="facade-customer-image-", suffix=IMAGE_MIME_TYPES[mime_type], delete=False) as file:
            file.write(raw)
            path = Path(file.name)
        yield path
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def generate_visual_response(
    *,
    system_prompt: str,
    payload_text: str,
    image_path: Path,
    max_new_tokens: int,
) -> str:
    """Run one Qwen3-VL request with a temporary customer image.

    Qwen3-VL receives the image and the same structured RAG payload used for a
    text question.  The image is therefore additional context, not a detached
    second answer path.
    """

    return generate_multi_visual_response(
        system_prompt=system_prompt,
        payload_text=payload_text,
        image_paths=[image_path],
        max_new_tokens=max_new_tokens,
    )


def generate_multi_visual_response(
    *,
    system_prompt: str,
    payload_text: str,
    image_paths: list[Path],
    max_new_tokens: int,
) -> str:
    """Run one bounded Qwen3-VL request with one relevant visual.

    The deployed 8B 4-bit model shares a 16 GB GPU with its visual tokens and
    KV cache. Visual assets remain preserved, but inference consumes one page
    at a time to retain a stable safety margin.
    """

    if not image_paths:
        raise ValueError("at_least_one_visual_required")
    paths = image_paths[:1]
    _, model = load_model()
    processor = load_processor()
    import torch
    from qwen_vl_utils import process_vision_info

    visual_content = [
        {
            "type": "image",
            "image": str(path.resolve()),
            # Bound visual tokens for the 16 GB demonstration GPU.
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        }
        for path in paths
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [*visual_content, {"type": "text", "text": payload_text}],
        },
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with generation_session(), torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def generate_finance_visual_observation(asset: Any, payload_text: str) -> str:
    """Inspect one in-memory Excel visual beside its Evidence JSON.

    The temporary image is removed immediately after local Qwen3-VL inference;
    chart observations remain supplementary and never become financial facts.
    """

    raw = getattr(asset, "image_bytes", None)
    media_type = getattr(asset, "media_type", None)
    suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    if not isinstance(raw, bytes) or not raw or media_type not in suffixes:
        raise ValueError("Excel 视觉对象没有可供本地视觉模型读取的图片内容。")
    path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="finance-visual-", suffix=suffixes[media_type], delete=False) as file:
            file.write(raw)
            path = Path(file.name)
        return generate_visual_response(
            system_prompt=(
                "你是企业文件的视觉证据核对器。只输出合法 JSON，不输出 Markdown。"
                "图表图片仅能提供趋势、图例和说明性信息；金额、期间和财务事实必须以并列的 Evidence JSON 原始单元格为准。"
                "不得从图片像素估读或编造精确金额。"
            ),
            payload_text=payload_text,
            image_path=path,
            max_new_tokens=360,
        )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def generate_document_vision_candidate(asset: Any, payload_text: str) -> str:
    """Transcribe one image or scanned PDF page as a generic candidate.

    Unlike chart observations, this path may transcribe visible text and table
    cells.  Its output is still marked as a customer-review candidate by the
    finance document layer and never creates a business fact on its own.
    """

    raw = getattr(asset, "image_bytes", None)
    media_type = getattr(asset, "media_type", None)
    suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    if not isinstance(raw, bytes) or not raw or media_type not in suffixes:
        raise ValueError("文档页面没有可供本地视觉模型读取的图片内容。")
    path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="document-vision-", suffix=suffixes[media_type], delete=False) as file:
            file.write(raw)
            path = Path(file.name)
        return generate_visual_response(
            system_prompt=(
                "你是本地企业文档识别器。只输出合法 JSON，不输出 Markdown。"
                "只抄录图片中可见的文字、表格和表单结构；不预设业务领域，不计算，不补全，不下结论。"
                "任何文字或数字都只是待客户确认的候选转录，不能当作已确认业务事实。"
            ),
            payload_text=payload_text,
            image_path=path,
            max_new_tokens=1_600,
        )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def default_image_identity() -> dict[str, Any]:
    """Return the safe default whenever no customer image is available."""
    return {
        "status": "not_provided",
        "visible_subject": "",
        "message": "未上传图片。",
        "matches": [],
    }


def load_visual_identity_index() -> tuple[Any, list[dict[str, Any]]] | None:
    """Load the local-only appearance reference index when it is available."""
    global _visual_identity_index
    if not VISUAL_IDENTITY_INDEX_PATH.exists() or not VISUAL_IDENTITY_MANIFEST_PATH.exists():
        return None
    mtime = max(VISUAL_IDENTITY_INDEX_PATH.stat().st_mtime_ns, VISUAL_IDENTITY_MANIFEST_PATH.stat().st_mtime_ns)
    if _visual_identity_index is not None and _visual_identity_index[2] == mtime:
        return _visual_identity_index[0], _visual_identity_index[1]

    with _visual_identity_index_lock:
        if _visual_identity_index is not None and _visual_identity_index[2] == mtime:
            return _visual_identity_index[0], _visual_identity_index[1]
        import numpy as np

        loaded = np.load(VISUAL_IDENTITY_INDEX_PATH)
        vectors = loaded["vectors"].astype("float32", copy=False)
        manifest = json.loads(VISUAL_IDENTITY_MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(manifest, list) or vectors.ndim != 2 or len(manifest) != vectors.shape[0]:
            raise ValueError("外观相似度索引格式无效，请重新运行 scripts\\build_visual_identity_index.py")
        _visual_identity_index = (vectors, manifest, mtime)
    return _visual_identity_index[0], _visual_identity_index[1]


def image_appearance_embedding(image_path: Path) -> Any:
    """Mean-pool Qwen3-VL's local visual encoder features for similarity search."""
    _, model = load_model()
    processor = load_processor()
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device)
    image_grid_thw = inputs["image_grid_thw"].to(model.device)
    with generation_session(), torch.inference_mode():
        output = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    # Qwen3-VL returns one variable-length sequence per input image.  A
    # normalized mean vector is sufficient for candidate retrieval; it is not
    # used as a factual product classifier.
    token_features = output.pooler_output[0].float()
    vector = token_features.mean(dim=0)
    vector = vector / vector.norm().clamp_min(1e-12)
    return vector.detach().cpu().numpy().astype(np.float32, copy=False)


def inspect_customer_image_identity(image_path: Path) -> dict[str, Any]:
    """Return appearance candidates from local internal reference images.

    This is deliberately not a brand/authenticity claim: an arbitrary web
    image can resemble a known façade.  The result only surfaces references
    for a user to confirm before any product RAG answer is enabled.
    """
    loaded = load_visual_identity_index()
    if loaded is None:
        return {
            "status": "unverified",
            "visible_subject": "",
            "message": "本地产品外观样本库尚未建立；系统不会根据图片外观把它归为企业产品产品。",
            "matches": [],
        }
    vectors, manifest = loaded
    vector = image_appearance_embedding(image_path)
    import numpy as np

    scores = vectors @ vector
    ranked = np.argsort(-scores)[:3]
    matches: list[dict[str, Any]] = []
    for position in ranked:
        item = manifest[int(position)]
        source = item.get("citation") or {}
        matches.append(
            {
                "asset_id": str(item["asset_id"]),
                "customer_title": str(item.get("customer_title") or "内部外观参考图"),
                "visual_endpoint": f"/api/copilot/visual/{item['asset_id']}",
                "citation": {
                    "document_name": source.get("document_name"),
                    "source_page": source.get("source_page"),
                },
                "similarity": round(float(scores[int(position)]), 3),
            }
        )
    best_score = float(scores[int(ranked[0])]) if len(ranked) else -1.0
    second_score = float(scores[int(ranked[1])]) if len(ranked) > 1 else -1.0
    confident_candidate = best_score >= VISUAL_APPEARANCE_MATCH_THRESHOLD and (
        len(ranked) == 1 or best_score - second_score >= VISUAL_APPEARANCE_MATCH_MARGIN
    )
    if confident_candidate:
        return {
            "status": "appearance_candidate",
            "visible_subject": "",
            "message": "图片外观与本地参考图存在较高相似度。以下仅为外观候选，不代表已确认产品品牌、真伪、性能或工程适用性；请由用户选择或补充产品型号后再检索资料。",
            "matches": matches,
        }
    return {
        "status": "unverified",
        "visible_subject": "",
        "message": "未匹配到足够可信的本地产品外观参考图；系统不会仅凭图片外观把它归为企业产品产品。",
        "matches": matches,
    }


def request_explicitly_names_product(request: DraftRequest) -> bool:
    """Whether RAG may answer about a named product despite an unverified image."""
    product_label = request.project_context.product_label or ""
    combined = f"{request.customer_question}\n{product_label}"
    return any(term in combined for term in EXPLICIT_PRODUCT_TERMS)


def is_facade_domain_request(request: DraftRequest) -> bool:
    """Keep the private product RAG out of ordinary conversation."""
    product_label = request.project_context.product_label or ""
    combined = f"{request.customer_question}\n{product_label}\n{conversation_context_text(request)}"
    return any(term in combined for term in FACADE_DOMAIN_TERMS)


TOOL_PLANNER_SYSTEM_PROMPT = """You are a tool router for a construction-material assistant.
Choose only the minimum useful tools and return JSON only:
{"tools":["general_chat|customer_documents|company_rag|visual_inspection|public_web_search"],"reason":"short reason"}
Rules: customer_documents reads files uploaded in this session; company_rag reads the private product and
construction knowledge base; visual_inspection reads the current uploaded image; public_web_search is only
for current public information; general_chat is for ordinary conversation. Never invent tool names."""


def plan_customer_tools(request: DraftRequest) -> ToolPlan:
    """Let the local model propose tools, then enforce availability and privacy policy."""

    has_documents = bool(request.document_session_id and get_session(request.document_session_id))
    has_image = bool(request.image_data_url)
    facade_related = is_facade_domain_request(request)
    named_public_project = is_named_project_web_query(request.customer_question)
    web_allowed = bool(request.use_online_search or named_public_project)
    # Ordinary text/image chat should not pay for a separate planning
    # generation.  The model planner is used when there is a real choice among
    # customer evidence, company knowledge and public search.
    if not has_documents and not facade_related and not web_allowed:
        return fallback_plan(
            has_documents=False,
            has_image=has_image,
            facade_related=False,
            web_requested=False,
        )
    raw_plan: dict[str, Any] | ToolPlan
    try:
        tokenizer, model = load_model()
        import torch

        planner_input = {
            "question": request.customer_question,
            "available": {
                "customer_documents": has_documents,
                "company_rag": True,
                "visual_inspection": has_image,
                "public_web_search": web_allowed,
            },
            "uploaded_document_names": (
                [document.file_name for document in get_session(request.document_session_id).documents]
                if has_documents and request.document_session_id
                else []
            ),
        }
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": TOOL_PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(planner_input, ensure_ascii=False)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with generation_session(), torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        raw_plan = parse_json(raw)
    except Exception:
        raw_plan = fallback_plan(
            has_documents=has_documents,
            has_image=has_image,
            facade_related=facade_related,
            web_requested=web_allowed,
        )
    return guard_plan(
        raw_plan,
        has_documents=has_documents,
        has_image=has_image,
        facade_related=facade_related,
        web_allowed=web_allowed,
    )


def general_local_chat_answer(
    request: DraftRequest,
    online_sources: list[dict[str, Any]] | None = None,
    online_search_meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Use the local model normally when a request is outside the façade domain."""
    payload_text = json.dumps(
        {
            "conversation_context": compact_conversation_context(request),
            "customer_question": request.customer_question,
            "customer_image_present": bool(request.image_data_url),
            "online_sources": online_sources or [],
            "online_source_policy": (
                "Online sources are public references, not private product evidence. "
                "If you use them, distinguish their source and do not invent details beyond their excerpts."
            ),
        },
        ensure_ascii=False,
    )
    try:
        with temporary_uploaded_image(request.image_data_url) as uploaded_image_path:
            if uploaded_image_path is not None:
                raw = generate_visual_response(
                    system_prompt=GENERAL_CHAT_SYSTEM_PROMPT,
                    payload_text=payload_text,
                    image_path=uploaded_image_path,
                    max_new_tokens=420,
                )
            else:
                tokenizer, model = load_model()
                import torch

                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": payload_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with generation_session(), torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=420,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(generated[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        parsed = parse_json(raw)
        reply = parsed.get("customer_reply") if isinstance(parsed, dict) else None
        observations = parsed.get("image_observations", []) if isinstance(parsed, dict) else []
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("general_chat_output_invalid")
        if not isinstance(observations, list):
            observations = []
        return {
            "intent": "unknown",
            "normalized_terms": [],
            "answerable": True,
            "customer_reply": reply.strip(),
            "key_points": [],
            "citations": [],
            "missing_information": [],
            "risk_warnings": [],
            "next_action": "如需查询企业产品产品、施工方案、节点或项目案例，可直接说明具体问题。",
            "image_observations": [
                item.strip() for item in observations if isinstance(item, str) and item.strip()
            ][:5],
            "visual_assets": [],
            "online_sources": online_sources or [],
            "retrieval": {"result_count": 0, "supporting_results": [], "visual_count": 0, "strategy": "not_used"},
            "meta": {
                "model_used": True,
                "mode": "local_general_chat",
                "customer_image_processed_locally": bool(request.image_data_url),
                "online_search": online_search_meta or {"status": "not_requested"},
            },
        }
    except Exception:
        return {
            "intent": "unknown",
            "normalized_terms": [],
            "answerable": True,
            "customer_reply": "我可以正常协助处理这个问题；当前本地模型暂时没有生成出可用回复，请换一种说法再试。",
            "key_points": [],
            "citations": [],
            "missing_information": [],
            "risk_warnings": [],
            "next_action": "如需查询企业产品产品、施工方案、节点或项目案例，可直接说明具体问题。",
            "image_observations": [],
            "visual_assets": [],
            "online_sources": online_sources or [],
            "retrieval": {"result_count": 0, "supporting_results": [], "visual_count": 0, "strategy": "not_used"},
            "meta": {
                "model_used": False,
                "mode": "local_general_chat_fallback",
                "customer_image_processed_locally": bool(request.image_data_url),
                "online_search": online_search_meta or {"status": "not_requested"},
            },
        }


def public_project_search_unavailable_response(
    online_search_meta: dict[str, str],
) -> dict[str, Any]:
    """Avoid answering a named public-project query from model memory alone."""

    return {
        "intent": "unknown",
        "normalized_terms": [],
        "answerable": False,
        "customer_reply": "当前未能取得该具体项目的公开网页资料，因此暂不根据模型记忆给出项目事实。请稍后重试联网查询，或换用项目全称、所在地、建设单位等信息重新搜索。",
        "key_points": [],
        "citations": [],
        "missing_information": ["可用的公开网页检索结果"],
        "risk_warnings": ["public_project_search_unavailable"],
        "next_action": "可补充项目全称、所在地或建设单位后重新联网查询。",
        "image_observations": [],
        "visual_assets": [],
        "online_sources": [],
        "retrieval": {"result_count": 0, "supporting_results": [], "visual_count": 0, "strategy": "public_web_search"},
        "meta": {
            "model_used": False,
            "mode": "public_project_search_unavailable",
            "online_search": online_search_meta,
        },
    }


def apply_image_identity_guard(result: dict[str, Any], image_identity: dict[str, Any]) -> dict[str, Any]:
    """Make the identity boundary visible even if a model ignores the prompt."""
    if image_identity.get("status") not in {"unverified", "appearance_candidate"}:
        return result

    notice = str(image_identity["message"])
    observations = result.get("image_observations", [])
    if not isinstance(observations, list):
        observations = []
    result["image_observations"] = [notice] + [
        observation
        for observation in observations
        if isinstance(observation, str) and observation.strip() and notice not in observation
    ][:4]

    reply = result.get("customer_reply", "")
    if isinstance(reply, str) and notice not in reply:
        result["customer_reply"] = f"{notice}\n\n{reply}".strip()
    return result


def load_retriever() -> LocalRagRetriever:
    """Reload the local index after a completed ingestion batch when needed."""
    global _retriever, _retriever_index_mtime_ns
    current_mtime_ns = RAG_INDEX_PATH.stat().st_mtime_ns if RAG_INDEX_PATH.exists() else None
    if _retriever is not None and current_mtime_ns == _retriever_index_mtime_ns:
        return _retriever
    with _retriever_lock:
        current_mtime_ns = RAG_INDEX_PATH.stat().st_mtime_ns if RAG_INDEX_PATH.exists() else None
        if _retriever is None or current_mtime_ns != _retriever_index_mtime_ns:
            _retriever = LocalRagRetriever()
            _retriever_index_mtime_ns = current_mtime_ns
    return _retriever


def fallback_draft(request: DraftRequest, reason: str) -> dict[str, Any]:
    question = request.customer_question
    context = request.project_context
    missing = ["具体产品型号", "基层类型及当前状态"]
    if context.region is None:
        missing.append("项目所在地区")
    if any(word in question for word in ("施工", "冬季", "夏季", "温度", "雨天")):
        missing.append("施工期间的环境条件")
    if any(word in question for word in ("报价", "价格", "交期", "到货", "折扣")):
        missing.extend(["具体配置与数量", "期望到货时间和收货地点"])

    return {
        "intent": "unknown",
        "normalized_terms": [],
        "answerable": False,
        "customer_reply": "为了准确回复您，需要先核实具体产品型号、项目条件及对应技术资料。当前系统尚未接入可用于对外承诺的产品资料，因此暂不直接给出产品参数、施工可行性、价格、交期或质保结论；我们核实后会尽快回复您。",
        "key_points": [],
        "citations": [],
        "missing_information": list(dict.fromkeys(missing)),
        "risk_warnings": ["no_supporting_evidence", "needs_project_context", "needs_technical_review"],
        "next_action": "补充产品型号与现场信息，待技术资料和商务资料接入后由系统检索并复核。",
        "meta": {"model_used": False, "fallback_reason": reason},
    }


def parse_json(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return parsed


def is_safe_pre_rag_draft(value: dict[str, Any]) -> bool:
    required = {
        "intent",
        "normalized_terms",
        "answerable",
        "customer_reply",
        "key_points",
        "citations",
        "missing_information",
        "risk_warnings",
        "next_action",
    }
    return (
        set(value) == required
        and value.get("intent") in INTENTS
        and value.get("answerable") is False
        and isinstance(value.get("customer_reply"), str)
        and isinstance(value.get("key_points"), list)
        and not value["key_points"]
        and isinstance(value.get("citations"), list)
        and not value["citations"]
        and isinstance(value.get("risk_warnings"), list)
        and "no_supporting_evidence" in value["risk_warnings"]
    )


def _materialize_retrieval_citations(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for index, evidence in enumerate(retrieval.get("text_evidence", []), start=1):
        evidence_id = f"T{index}"
        for source in evidence.get("citations", []):
            key = (evidence_id, str(source.get("document_name")), source.get("source_page"))
            if key in seen:
                continue
            seen.add(key)
            output.append({"evidence_id": evidence_id, **source})
    return output


def customer_visible_retrieval(retrieval: dict[str, Any], limit: int = 5) -> dict[str, Any]:
    """Expose a small, source-safe result list for an optional UI disclosure.

    It deliberately includes excerpts, document names and page numbers only;
    paths, internal taxonomy notes and raw scores remain server-side.
    """

    items: list[dict[str, Any]] = []
    project_cases = retrieval.get("project_cases") or []
    if project_cases:
        for index, case in enumerate(project_cases[:limit], start=1):
            source = case.get("citation") if isinstance(case.get("citation"), dict) else {}
            facts = [
                str(case.get("project_type") or ""),
                str(case.get("product") or ""),
                str(case.get("installation_method") or ""),
                str(case.get("area_m2") or ""),
                str(case.get("completion_year") or ""),
            ]
            excerpt = "；".join(value for value in facts if value) or "项目案例资料"
            items.append(
                {
                    "result_id": f"R{index}",
                    "excerpt": excerpt,
                    "document_name": source.get("document_name"),
                    "source_page": source.get("source_page"),
                    "section_heading": str(case.get("project_name") or source.get("section_heading") or "项目案例"),
                }
            )
    else:
        for index, evidence in enumerate((retrieval.get("text_evidence") or [])[:limit], start=1):
            source = (evidence.get("citations") or [{}])[0]
            excerpt = str(evidence.get("text") or "").strip()
            if len(excerpt) > 220:
                excerpt = excerpt[:220].rsplit("。", 1)[0].strip() + "。"
            items.append(
                {
                    "result_id": f"R{index}",
                    "excerpt": excerpt,
                    "document_name": source.get("document_name"),
                    "source_page": source.get("source_page"),
                    "section_heading": source.get("section_heading"),
                }
            )
    return {
        "result_count": len(items),
        "supporting_results": items,
        "visual_count": min(len(retrieval.get("visual_assets") or []), limit),
        "strategy": str((retrieval.get("meta") or {}).get("strategy") or "local_retrieval"),
    }


def unverified_image_identity_response(
    request: DraftRequest, retrieval: dict[str, Any], image_identity: dict[str, Any]
) -> dict[str, Any]:
    """Stop an arbitrary image from being silently mapped to the product RAG."""
    subject = str(image_identity.get("visible_subject") or "").strip()
    observations = [str(image_identity["message"])]
    if subject:
        observations.append(f"图片中可见内容：{subject}")
    return {
        "intent": "unknown",
        "normalized_terms": [],
        "answerable": False,
        "customer_reply": (
            f"{image_identity['message']}\n\n"
            "请补充产品型号、包装/板背标签、品牌标识，或在问题中明确要咨询的产品名称；"
            "确认前，系统不会把这张图片关联到企业产品产品资料或施工方案。"
        ),
        "key_points": [],
        "citations": [],
        "missing_information": ["产品型号或品牌标识", "板背/包装标签或清晰近照"],
        "risk_warnings": ["unverified_customer_image_identity"],
        "next_action": "可上传含品牌、型号或包装标签的清晰近照；也可直接在问题中填写要咨询的产品名称。",
        "image_observations": observations,
        "visual_assets": [],
        "retrieval": {
            **customer_visible_retrieval(retrieval),
            "result_count": 0,
            "supporting_results": [],
            "visual_count": 0,
        },
        "meta": {
            "model_used": True,
            "mode": "customer_image_identity_unverified",
            "customer_image_processed_locally": True,
            "image_identity": image_identity,
        },
    }


def _insufficient_evidence_answer(
    request: DraftRequest, retrieval: dict[str, Any], reason: str, task_type: str
) -> dict[str, Any]:
    """State an evidence gap without pretending that a human handoff is an answer."""

    missing = ["可用于直接支持该问题的产品、工艺或节点资料"]
    if task_type == "project_fit":
        missing.extend(["基层状态", "锚固条件", "门窗及阴阳角等节点条件"])
    if task_type == "commercial":
        missing.extend(["已授权的报价、交期或质保资料"])
    return {
        "intent": TASK_TYPE_DEFAULT_INTENT.get(task_type, "unknown"),
        "normalized_terms": [],
        "answerable": False,
        "customer_reply": "当前本地资料中没有检索到能够直接支持该问题的证据，因此不对外编造结论。补充对应资料或更具体的筛选条件后，可以重新检索并生成基于资料的回答。",
        "key_points": [],
        "citations": [],
        "missing_information": list(dict.fromkeys(missing)),
        "risk_warnings": ["insufficient_grounded_evidence"],
        "next_action": "补充资料或明确产品、工艺、节点、地区等筛选条件后重新提问。",
        "visual_assets": [],
        "retrieval": customer_visible_retrieval(retrieval),
        "meta": {"model_used": False, "fallback_reason": reason, "mode": "insufficient_local_evidence"},
    }


def _source_derived_answer(
    request: DraftRequest, retrieval: dict[str, Any], reason: str, task_type: str
) -> dict[str, Any]:
    """Keep the product useful when JSON generation fails after successful retrieval.

    The fallback deliberately uses source text verbatim rather than trying to
    infer a new technical answer.  Procedure evidence is already retrieved in
    source order, so it remains a meaningful customer-facing sequence.
    """

    evidence = retrieval.get("text_evidence") or []
    if not evidence or task_type == "commercial":
        return _insufficient_evidence_answer(request, retrieval, reason, task_type)

    snippets = [str(item.get("text") or "").strip() for item in evidence]
    snippets = [snippet for snippet in snippets if snippet]
    if not snippets:
        return _insufficient_evidence_answer(request, retrieval, reason, task_type)

    if task_type == "procedure":
        reply = "根据已检索到的施工方案，相关步骤可按资料章节顺序查看：\n" + "\n".join(
            f"{index}. {snippet}" for index, snippet in enumerate(snippets, start=1)
        )
        next_action = "如需针对某个基层或节点细化流程，可补充该项目条件后继续检索。"
    else:
        primary = snippets[0]
        if len(primary) > 420:
            primary = primary[:420].rsplit("。", 1)[0].strip() + "。"
        reply = "根据已检索到的资料：\n" + primary
        next_action = "可继续补充产品、工艺、节点、地区或项目类型，缩小资料检索范围。"

    missing = ["基层状态、锚固条件和节点条件"] if task_type == "project_fit" else []
    return {
        "intent": TASK_TYPE_DEFAULT_INTENT.get(task_type, "unknown"),
        "normalized_terms": [],
        "answerable": True,
        "customer_reply": reply,
        "key_points": snippets[:6],
        "citations": _materialize_retrieval_citations(retrieval),
        "missing_information": missing,
        "risk_warnings": ["source_derived_summary"],
        "next_action": next_action,
        "visual_assets": retrieval.get("visual_assets", []),
        "retrieval": customer_visible_retrieval(retrieval),
        "meta": {"model_used": False, "fallback_reason": reason, "mode": "source_derived_evidence_fallback"},
    }


def fallback_grounded_answer(
    request: DraftRequest,
    retrieval: dict[str, Any],
    reason: str,
    question_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_type = str((question_plan or {}).get("task_type") or "unknown")
    if task_type not in TASK_TYPES:
        task_type = "unknown"
    return _source_derived_answer(request, retrieval, reason, task_type)


def _is_project_case_question(question: str) -> bool:
    """Route catalogue-case queries to a deterministic, source-backed list."""

    case_terms = ("项目", "案例", "医院", "学校", "办公楼", "商业", "产业园")
    request_terms = ("哪些", "有什么", "有哪", "有没有", "查看", "展示", "参考", "列出", "多少")
    has_case_scope = any(term in question for term in case_terms)
    has_inventory_request = any(term in question for term in request_terms)
    # Natural customer wording commonly uses “有山东的项目吗？” rather than
    # “有没有山东项目？”. Both must reach the same structured case search.
    has_yes_no_case_request = "有" in question and ("吗" in question or "没" in question)
    return has_case_scope and (has_inventory_request or has_yes_no_case_request)


QUESTION_UNDERSTANDING_PROMPT = """你是本地外墙建材知识库的“问题理解层”。
阅读客户的中文问题，但不要回答问题，也不要陈述任何产品事实。
只返回一个 JSON 对象，且只能包含以下键：
{"intent":"case_reference|product_parameter|technical_performance|application_condition|construction|quote_delivery|warranty|comparison|complaint_after_sales|unknown","task_type":"factual_lookup|procedure|node_detail|case_reference|comparison|project_fit|commercial|unknown","requires_project_conditions":true|false,"target_terms":["..."],"case_reference":true|false,"retrieval_query":"简短中文检索词","case_filters":{"locations":["..."],"project_types":["..."],"installation_methods":["..."],"products":["..."]}}

task_type 的定义：
- factual_lookup：问产品定义、规格、材料、性能或一般资料事实。
- procedure：问“怎么做、流程、步骤、工序、安装顺序”等已指定对象的流程。
- node_detail：问窗洞口、阴阳角、勒脚、女儿墙等节点做法或图集。
- case_reference：问公司已有项目、案例、地区、项目类型或参考项目。
- comparison：问两个或多个产品/工艺的区别、比较或选择差异。
- project_fit：问某个具体项目、现场或旧楼是否适用、应如何选择；可以提供资料支持的通用条件，但不能给未证实的工程结论。
- commercial：问报价、价格、交期、付款、质保或合同。
- unknown：其余无法明确理解的问题。

requires_project_conditions 只能在 task_type=project_fit 时为 true；“流程、节点、案例、一般产品资料”不能因为没有项目条件就被标记为 true。
只有询问已完成项目/案例时 case_reference=true。像“干挂流程怎么做”“旧楼如何安装”不是案例查询。
target_terms 只放问题中明确出现的产品、工艺、节点、地区或项目类型名词；不要编造同义词或事实。
retrieval_query 必须保留用户的核心名词，只可补充中性的“施工方案、流程、节点图集、项目案例、产品资料”等检索词。
case_filters 只填写用户明确提出的案例筛选条件；没有则用空数组。"""


def _fallback_task_type(question: str) -> str:
    """Grammar-level fallback used only if local model planning fails.

    This is deliberately about what the user is asking for, not about any
    particular product name or construction method.  It keeps routing useful
    during a cold model failure without turning the service into a keyword
    collection maintained one question at a time.
    """

    if any(term in question for term in COMMERCIAL_EVIDENCE_TERMS):
        return "commercial"
    if _is_project_case_question(question):
        return "case_reference"
    if any(term in question for term in ("节点", "图集", "洞口", "窗口", "门窗", "阴角", "阳角", "勒脚", "女儿墙", "檐口", "收口")):
        return "node_detail"
    if any(term in question for term in ("流程", "步骤", "工序", "顺序", "怎么施工", "怎么安装", "怎么做")):
        return "procedure"
    if any(term in question for term in ("区别", "对比", "比较", "哪个好")):
        return "comparison"
    if any(term in question for term in ("适合", "能不能", "可不可以", "本项目", "我的项目", "现场", "旧楼", "旧墙")):
        return "project_fit"
    return "factual_lookup"


def _fallback_question_plan(question: str) -> dict[str, Any]:
    task_type = _fallback_task_type(question)
    return {
        "intent": TASK_TYPE_DEFAULT_INTENT[task_type],
        "task_type": task_type,
        "requires_project_conditions": task_type == "project_fit",
        "target_terms": [],
        "case_reference": task_type == "case_reference",
        "retrieval_query": question,
        "case_filters": {"locations": [], "project_types": [], "installation_methods": [], "products": []},
        "model_used": False,
    }


def save_question_router_debug(kind: str, question: str, raw: str) -> None:
    """Persist a local-only router sample only when explicitly enabled."""

    if os.getenv("FACADE_RAG_DEBUG") != "1":
        return
    path = ROOT / "runtime" / f"last_{kind}_router_output.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"question": question, "raw": raw}, ensure_ascii=False), encoding="utf-8")


def understand_customer_question(question: str, conversation_context: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Let the local model interpret phrasing, while keeping facts in the RAG layer."""

    fallback = _fallback_question_plan(question)
    try:
        tokenizer, model = load_model()
        import torch

        messages = [
            {"role": "system", "content": QUESTION_UNDERSTANDING_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "conversation_context": conversation_context or [],
                        "current_question": question,
                        "instruction": (
                            "Use prior turns only to resolve references such as 'that method' or 'the previous project'. "
                            "Do not treat prior assistant replies as factual evidence."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with generation_session(), torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        save_question_router_debug("question", question, raw)
        value = parse_json(raw)
        intent = str(value.get("intent") or "unknown")
        task_type = str(value.get("task_type") or "unknown")
        retrieval_query = str(value.get("retrieval_query") or "").strip()
        if intent not in INTENTS or task_type not in TASK_TYPES or not retrieval_query or len(retrieval_query) > 300:
            return fallback
        target_terms = value.get("target_terms") if isinstance(value.get("target_terms"), list) else []
        target_terms = [
            str(term).strip()
            for term in target_terms
            if isinstance(term, str) and str(term).strip()
        ][:8]
        raw_filters = value.get("case_filters") if isinstance(value.get("case_filters"), dict) else {}
        case_filters: dict[str, list[str]] = {}
        for key in ("locations", "project_types", "installation_methods", "products"):
            candidates = raw_filters.get(key) if isinstance(raw_filters.get(key), list) else []
            case_filters[key] = [
                str(candidate).strip()
                for candidate in candidates
                if isinstance(candidate, str) and str(candidate).strip()
            ][:5]
        return {
            "intent": intent,
            "task_type": task_type,
            # A task type is authoritative for routing.  The old boolean is
            # retained only for compatibility with the structured case path.
            "requires_project_conditions": task_type == "project_fit",
            "target_terms": target_terms,
            "case_reference": task_type == "case_reference" or bool(value.get("case_reference")),
            "retrieval_query": retrieval_query,
            "case_filters": case_filters,
            "model_used": True,
        }
    except Exception:
        # A model failure falls back to the original question and the narrow
        # deterministic route; it never creates facts or a fabricated case.
        return fallback


CASE_INTENT_PROMPT = """Classify the customer's Chinese question for a facade-materials assistant.
Return exactly one line and no other text.
Return CASE_REFERENCE|<location names from the question, comma separated> if the customer asks whether the company has done projects in a place, asks for completed cases, reference projects, or project examples.
Return NOT_CASE_REFERENCE| if the question is about product information, installation, the customer's own project, price, delivery, warranty, or anything other than past company cases.
Examples:
你们在山东做过吗 -> CASE_REFERENCE|山东
有山东的项目吗 -> CASE_REFERENCE|山东
保温装饰一体板是什么 -> NOT_CASE_REFERENCE|
旧楼改造适合怎么安装 -> NOT_CASE_REFERENCE|"""


def classify_case_reference(question: str) -> dict[str, Any]:
    """Use an intentionally tiny local-model contract for robust case routing."""

    fallback = {"case_reference": False, "locations": [], "model_used": False}
    try:
        tokenizer, model = load_model()
        import torch

        messages = [
            {"role": "system", "content": CASE_INTENT_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with generation_session(), torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=36,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()
        save_question_router_debug("case", question, raw)
        upper = raw.upper()
        if "NOT_CASE_REFERENCE" in upper:
            return {"case_reference": False, "locations": [], "model_used": True}
        match = re.search(r"CASE_REFERENCE\s*\|?\s*([^\n]*)", raw, flags=re.IGNORECASE)
        if not match:
            return fallback
        requested_locations = [
            item.strip(" ，,；;。.!！")
            for item in re.split(r"[,，;；]", match.group(1))
            if item.strip(" ，,；;。.!！")
        ]
        normal_question = _normalise_location(question)
        requested_locations = [
            location
            for location in requested_locations
            if _normalise_location(location) in normal_question
        ][:5]
        return {"case_reference": True, "locations": requested_locations, "model_used": True}
    except Exception:
        if os.getenv("FACADE_RAG_DEBUG") == "1":
            import traceback

            (ROOT / "runtime" / "last_case_router_exception.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        return fallback


def _normalise_location(value: str) -> str:
    return value.strip().removesuffix("省").removesuffix("市").removesuffix("自治区")


def apply_case_filters(retrieval: dict[str, Any], case_filters: dict[str, list[str]]) -> dict[str, Any]:
    """Apply model-extracted customer filters to factual case records only."""

    active_filters = {key: values for key, values in case_filters.items() if values}
    if not active_filters:
        return retrieval

    def matches(case: dict[str, Any]) -> bool:
        fields = {
            "locations": " ".join(str(case.get(key) or "") for key in ("project_name", "region")),
            "project_types": str(case.get("project_type") or ""),
            "installation_methods": str(case.get("installation_method") or ""),
            "products": str(case.get("product") or ""),
        }
        for key, requested_values in active_filters.items():
            source_value = fields[key]
            if key == "locations":
                if not all(_normalise_location(value) in source_value for value in requested_values):
                    return False
            elif not all(value in source_value for value in requested_values):
                return False
        return True

    filtered_cases = [case for case in retrieval.get("project_cases", []) if matches(case)]
    filtered_asset_ids = {
        str(asset_id)
        for case in filtered_cases
        for asset_id in (case.get("visual_asset_ids") or [])
    }
    filtered_visuals = [
        asset for asset in retrieval.get("visual_assets", []) if str(asset.get("asset_id")) in filtered_asset_ids
    ]
    return {**retrieval, "project_cases": filtered_cases[:5], "visual_assets": filtered_visuals[:5]}


def catalogue_case_answer(
    request: DraftRequest, retrieval: dict[str, Any], *, model_planned: bool = False
) -> dict[str, Any]:
    """Reply from structured case records without asking the text model to invent a list."""

    cases = retrieval.get("project_cases") or []
    total = int((retrieval.get("meta") or {}).get("index_metadata", {}).get("project_case_count", len(cases)))
    if not cases:
        return {
            "intent": "case_reference",
            "normalized_terms": [],
            "answerable": False,
            "customer_reply": "当前已导入的项目画册中，暂未检索到与该条件匹配的项目案例。可以换用地区、项目类型、产品或施工工艺继续检索。",
            "key_points": [],
            "citations": [],
            "missing_information": ["可用于筛选的地区、项目类型、产品或施工工艺条件"],
            "risk_warnings": ["no_matching_catalogue_case"],
            "next_action": "补充筛选条件后，系统将继续在本地项目案例库中检索。",
            "visual_assets": [],
            "retrieval": customer_visible_retrieval(retrieval),
            "meta": {"model_used": model_planned, "mode": "structured_catalogue_case_retrieval"},
        }

    lines: list[str] = []
    key_points: list[str] = []
    citations: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        fields = [
            str(case.get("project_type") or ""),
            str(case.get("product") or ""),
            str(case.get("installation_method") or ""),
            str(case.get("area_m2") or ""),
            str(case.get("completion_year") or ""),
        ]
        summary = "；".join(field for field in fields if field)
        name = str(case.get("project_name") or "未命名项目")
        lines.append(f"{index}. {name}：{summary}")
        key_points.append(f"{name} - {summary}")
        citation = case.get("citation") if isinstance(case.get("citation"), dict) else {}
        citations.append({"evidence_id": f"C{index}", **citation})

    reply = (
        f"目前已从企业产品综合产品画册中结构化收录 {total} 个项目案例。"
        "以下先展示与当前问题最相关的部分：\n" + "\n".join(lines)
        + "\n这些字段来自企业产品画册；如需按医院、学校、地区、产品或安装方式继续筛选，可以直接说明条件。"
    )
    return {
        "intent": "case_reference",
        "normalized_terms": [],
        "answerable": True,
        "customer_reply": reply,
        "key_points": key_points,
        "citations": citations,
        "missing_information": [],
        "risk_warnings": ["catalogue_case_information"],
        "next_action": "可继续按项目类型、地区、产品、施工工艺或面积范围筛选案例。",
        "visual_assets": retrieval.get("visual_assets", []),
        "retrieval": customer_visible_retrieval(retrieval),
        "meta": {"model_used": model_planned, "mode": "structured_catalogue_case_retrieval"},
    }


def has_manual_handoff(value: dict[str, Any]) -> bool:
    customer_reply = str(value.get("customer_reply") or "")
    next_action = str(value.get("next_action") or "")
    return any(term in customer_reply or term in next_action for term in MANUAL_HANDOFF_TERMS)


def is_safe_grounded_answer(value: dict[str, Any], evidence_ids: set[str]) -> bool:
    required = {
        "intent",
        "normalized_terms",
        "answerable",
        "customer_reply",
        "key_points",
        "citations",
        "missing_information",
        "risk_warnings",
        "next_action",
        "image_observations",
    }
    if set(value) != required or value.get("intent") not in INTENTS:
        return False
    if not isinstance(value.get("answerable"), bool) or not isinstance(value.get("customer_reply"), str):
        return False
    if not all(
        isinstance(value.get(field), list)
        for field in ("normalized_terms", "key_points", "citations", "missing_information", "risk_warnings", "image_observations")
    ):
        return False
    citations = value["citations"]
    if value["answerable"] is False:
        return not value["key_points"] and not citations
    if not citations:
        return False
    return all(
        isinstance(citation, dict)
        and set(citation) == {"evidence_id"}
        and citation.get("evidence_id") in evidence_ids
        for citation in citations
    )


def materialize_citations(
    citation_requests: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace model-provided IDs with server-controlled document/page citations."""
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for request in citation_requests:
        evidence_id = str(request["evidence_id"])
        evidence = evidence_by_id[evidence_id]
        for source in evidence.get("citations", []):
            key = (
                evidence_id,
                str(source.get("document_name")),
                source.get("source_page"),
                source.get("sheet_name"),
                source.get("source_range"),
                source.get("section_heading"),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append({"evidence_id": evidence_id, **source})
    return output


def evidence_support_audit(
    result: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Conservative post-generation support audit.

    Evidence-ID validation prevents fabricated sources. This second check
    reports whether answer terms and numeric values are actually present in the
    cited evidence. It does not pretend to be a full natural-language theorem
    prover, but blocks unsupported numeric claims.
    """

    cited_ids = [
        str(citation.get("evidence_id"))
        for citation in result.get("citations", [])
        if isinstance(citation, dict) and citation.get("evidence_id") in evidence_by_id
    ]
    cited_text = "\n".join(str(evidence_by_id[item].get("text") or "") for item in cited_ids)
    answer_text = "\n".join(
        [str(result.get("customer_reply") or ""), *[str(item) for item in result.get("key_points", [])]]
    )
    def tokens(value: str) -> set[str]:
        latin = set(re.findall(r"[a-zA-Z]{3,}|\d+(?:\.\d+)?%?", value.lower()))
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", value)
        chinese = {run[index:index + 2] for run in chinese_runs for index in range(max(0, len(run) - 1))}
        return latin | chinese

    answer_terms = tokens(answer_text)
    evidence_terms = tokens(cited_text)
    overlap = answer_terms & evidence_terms
    numeric_claims = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", answer_text))
    evidence_numbers = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", cited_text))
    unsupported_numbers = sorted(numeric_claims - evidence_numbers)
    return {
        "cited_evidence_ids": cited_ids,
        "lexical_overlap_term_count": len(overlap),
        "answer_term_count": len(answer_terms),
        "unsupported_numeric_claims": unsupported_numbers,
        "passed": bool(cited_ids and overlap) and not unsupported_numbers,
    }


def save_grounded_debug_output(raw: str) -> None:
    """Persist a local-only debugging sample when explicitly enabled.

    It is disabled by default because customer questions can contain sensitive
    project information.  The public response never includes this raw output.
    """
    if os.getenv("FACADE_RAG_DEBUG") != "1":
        return
    debug_path = ROOT / "runtime" / "last_grounded_model_output.txt"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(raw, encoding="utf-8")


def normalise_nonfactual_output_fields(value: dict[str, Any]) -> dict[str, Any]:
    """Coerce harmless presentation fields without changing any technical claim.

    Qwen occasionally emits one normalized term as a string.  Citations and
    evidence are never repaired here; they remain subject to strict validation.
    """
    terms = value.get("normalized_terms")
    if isinstance(terms, str):
        value["normalized_terms"] = [{"term": terms, "normalized": terms}]
    elif isinstance(terms, list) and all(isinstance(term, str) for term in terms):
        value["normalized_terms"] = [{"term": term, "normalized": term} for term in terms]
    observations = value.get("image_observations", [])
    if not isinstance(observations, list):
        observations = []
    value["image_observations"] = [
        observation.strip()
        for observation in observations
        if isinstance(observation, str) and observation.strip()
    ][:5]
    next_action = value.get("next_action")
    if isinstance(next_action, list):
        value["next_action"] = "；".join(
            item.strip() for item in next_action if isinstance(item, str) and item.strip()
        ) or "可继续补充产品、工艺、节点或项目条件，以便缩小资料范围。"
    elif not isinstance(next_action, str) or not next_action.strip():
        value["next_action"] = "可继续补充产品、工艺、节点或项目条件，以便缩小资料范围。"
    return value


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok", "implementation": "qwen3-vl-8b-4bit-local-image-rag-v1"}


@app.get("/health/ready")
def ready() -> dict[str, Any]:
    idle_seconds = model_idle_seconds()
    return {
        "status": "ready" if _model is not None else "cold",
        "model_loaded": _model is not None,
        "model_idle_seconds": round(idle_seconds, 1) if idle_seconds is not None else None,
        "model_idle_unload_after_seconds": MODEL_IDLE_UNLOAD_SECONDS,
        "model_path": str(MODEL_PATH),
        "generation_mode": "qwen3_vl_4bit_grounded_rag",
        "customer_image_upload": "local_temporary_file_deleted_after_inference",
        "retrieval_index_available": RAG_INDEX_PATH.exists(),
        "retrieval_index_path": str(RAG_INDEX_PATH),
        "visual_identity_index_available": VISUAL_IDENTITY_INDEX_PATH.exists() and VISUAL_IDENTITY_MANIFEST_PATH.exists(),
        "baidu_ai_search_configured": is_baidu_search_configured(),
        "visual_identity_reference_count": (
            len(json.loads(VISUAL_IDENTITY_MANIFEST_PATH.read_text(encoding="utf-8")))
            if VISUAL_IDENTITY_MANIFEST_PATH.exists()
            else 0
        ),
    }


@app.get("/health/retrieval")
def retrieval_ready() -> dict[str, Any]:
    try:
        retriever = load_retriever()
        return {
            "status": "ready",
            "index_path": str(RAG_INDEX_PATH),
            "metadata": retriever.metadata,
            "gpu_model_loaded": _model is not None,
        }
    except Exception as exc:
        return {"status": "not_ready", "reason": type(exc).__name__, "index_path": str(RAG_INDEX_PATH)}


@app.post("/api/copilot/retrieve")
def retrieve_evidence(request: RetrieveRequest) -> dict[str, Any]:
    """Return locally retrieved evidence without triggering Qwen3 generation."""
    started = time.perf_counter()
    try:
        result = load_retriever().retrieve(request.customer_question, request.top_k, request.visual_k)
        result["meta"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["meta"]["generation_model_loaded"] = _model is not None
        return result
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc), "action": "运行 scripts\\build_rag_index.ps1"})


@app.get("/api/copilot/visual/{asset_id}")
def original_visual(asset_id: str):
    """Serve only an indexed, customer-shareable original PDF crop."""
    asset = load_retriever().visual_asset(asset_id)
    if asset is None:
        return JSONResponse(status_code=404, content={"detail": "未找到可对外返回的图片资料。"})
    image_path = Path(str(asset.get("image_path") or ""))
    if not image_path.exists() or not image_path.is_file():
        return JSONResponse(status_code=410, content={"detail": "原始图片文件当前不可用。"})
    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return FileResponse(
        image_path,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


def _run_general_local_answer(request: DraftRequest, *, allow_public_web: bool | None = None) -> AnswerResponse:
    """Existing non-facade path, now invoked by the LangGraph route."""

    if allow_public_web is False:
        request = request.model_copy(update={"use_online_search": False})
    named_project_lookup = is_named_project_web_query(request.customer_question) and allow_public_web is not False
    online_sources, online_search_meta = maybe_search_online(
        request,
        request.customer_question,
        automatic_named_project_lookup=named_project_lookup,
    )
    if named_project_lookup and not online_sources:
        return AnswerResponse(**public_project_search_unavailable_response(online_search_meta))
    return AnswerResponse(**general_local_chat_answer(request, online_sources, online_search_meta))


def _run_facade_rag_answer(request: DraftRequest, *, allow_public_web: bool | None = None) -> AnswerResponse:
    """Existing evidence-grounded facade route, now invoked by LangGraph."""

    started = time.perf_counter()
    conversation_context = compact_conversation_context(request)
    retrieval_hint = conversation_retrieval_hint(request)
    question_plan = _fallback_question_plan(request.customer_question)
    online_sources: list[dict[str, Any]] = []
    online_search_meta: dict[str, str] = {"status": "not_requested"}
    try:
        commercial_request = any(term in request.customer_question for term in COMMERCIAL_EVIDENCE_TERMS)
        if commercial_request:
            question_plan = {**question_plan, "intent": "quote_delivery", "task_type": "commercial"}
        else:
            question_plan = understand_customer_question(request.customer_question, conversation_context)
        # A model rewrite can add useful synonyms, but it must never replace
        # the customer's own wording. Keeping both prevents a bad rewrite from
        # dropping a product name, location or construction constraint.
        retrieval_query = request.customer_question
        if retrieval_hint:
            retrieval_query = f"{retrieval_hint}\n{request.customer_question}"
        planned_query = str(question_plan["retrieval_query"]).strip()
        if planned_query and planned_query != request.customer_question:
            retrieval_query = f"{retrieval_query}\n{planned_query}"
        has_case_filter = any(question_plan["case_filters"].values())
        task_type = str(question_plan.get("task_type") or "unknown")
        retrieval_mode = task_type if task_type in TASK_TYPES else "factual_lookup"
        case_reference_request = bool(question_plan["case_reference"]) or _is_project_case_question(request.customer_question)
        # Do not expand a public-web query with browser history or local RAG
        # terms.  Baidu receives the complete current question exactly as the
        # customer wrote it, including an explicit project name.
        if allow_public_web is False:
            request = request.model_copy(update={"use_online_search": False})
        online_query = request.customer_question
        named_project_lookup = is_named_project_web_query(
            online_query, case_reference=case_reference_request
        ) and allow_public_web is not False
        online_sources, online_search_meta = maybe_search_online(
            request,
            online_query,
            automatic_named_project_lookup=named_project_lookup,
        )
        retrieval = load_retriever().retrieve(
            retrieval_query,
            top_k=8 if retrieval_mode == "procedure" else 5,
            visual_k=5,
            case_k=20 if has_case_filter or case_reference_request else 5,
            retrieval_mode=retrieval_mode,
        )

        image_identity = default_image_identity()
        if request.image_data_url:
            try:
                with temporary_uploaded_image(request.image_data_url) as image_path:
                    if image_path is not None:
                        image_identity = inspect_customer_image_identity(image_path)
            except Exception:
                # A failed visual identity pass must never become an implicit
                # product match.  We fail closed and keep the image unverified.
                image_identity = {
                    "status": "unverified",
                    "visible_identifiers": [],
                    "visible_subject": "",
                    "message": "图片标识未能完成核验；不能仅凭外观确认其为企业产品产品。",
                }

        explicit_product_request = request_explicitly_names_product(request)
        # A visual match is deliberately only a candidate.  The user must
        # explicitly name/select a product before its technical RAG material
        # can be coupled to an uploaded image.
        if request.image_data_url and not explicit_product_request:
            return AnswerResponse(
                **attach_online_search(
                    unverified_image_identity_response(request, retrieval, image_identity), online_sources, online_search_meta
                )
            )

        if os.getenv("FACADE_RAG_DEBUG") == "1":
            debug_path = ROOT / "runtime" / "last_question_plan.json"
            debug_path.write_text(
                json.dumps(
                    {
                        "question_plan": question_plan,
                        "retrieval_query": retrieval_query,
                        "retrieval_mode": retrieval_mode,
                        "project_case_count": len(retrieval.get("project_cases", [])),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        text_evidence = retrieval.get("text_evidence", [])
        if commercial_request:
            return AnswerResponse(
                **attach_online_search(
                    fallback_grounded_answer(request, retrieval, "commercial_evidence_unavailable", question_plan),
                    online_sources,
                    online_search_meta,
                )
            )
        if case_reference_request:
            retrieval = apply_case_filters(retrieval, question_plan["case_filters"])
        if case_reference_request and not online_sources:
            return AnswerResponse(
                **attach_online_search(
                    catalogue_case_answer(request, retrieval, model_planned=bool(question_plan["model_used"])),
                    online_sources,
                    online_search_meta,
                )
            )
        # A customer image may still receive a strictly observational response
        # when the local RAG has no matching technical text.  Without an
        # image, preserve the existing evidence-first fallback.
        if not text_evidence and not online_sources and not request.image_data_url:
            return AnswerResponse(
                **attach_online_search(
                    fallback_grounded_answer(request, retrieval, "no_retrieved_text_evidence", question_plan),
                    online_sources,
                    online_search_meta,
                )
            )

        evidence_by_id: dict[str, dict[str, Any]] = {}
        evidence_payload: list[dict[str, Any]] = []
        for index, evidence in enumerate(text_evidence, start=1):
            evidence_id = f"T{index}"
            evidence_by_id[evidence_id] = evidence
            evidence_payload.append(
                {
                    "evidence_id": evidence_id,
                    "text": evidence["text"],
                    "citations": evidence["citations"],
                    "source_taxonomy": evidence.get("source_taxonomy") or [],
                    "sales_playbook_use": evidence.get("sales_playbook_use"),
                }
            )
        for source in online_sources:
            evidence_id = str(source.get("source_id") or "").strip()
            excerpt = str(source.get("excerpt") or "").strip()
            if not evidence_id or not excerpt:
                continue
            evidence_by_id[evidence_id] = {
                "text": excerpt,
                "citations": [
                    {
                        "document_name": str(source.get("title") or "网页来源"),
                        "source_page": None,
                        "section_heading": str(source.get("website") or "联网搜索"),
                        "source_url": str(source.get("url") or ""),
                        "source_type": "online",
                    }
                ],
            }
            evidence_payload.append(
                {
                    "evidence_id": evidence_id,
                    "text": excerpt,
                    "citations": evidence_by_id[evidence_id]["citations"],
                }
            )

        payload = {
            "conversation_context": conversation_context,
            "customer_question": request.customer_question,
            "project_context": request.project_context.model_dump(mode="json"),
            "customer_image_present": bool(request.image_data_url),
            "customer_image_policy": (
                "The customer image is private local context. Describe only directly visible content in "
                "image_observations; it is not technical evidence and cannot replace a citation."
            ),
            "customer_image_identity": image_identity,
            "customer_explicitly_named_product": explicit_product_request,
            "online_search": {
                "status": online_search_meta.get("status"),
                "trigger": online_search_meta.get("trigger", "not_requested"),
                "named_project_lookup": named_project_lookup,
                "policy": (
                    "Online evidence is public reference material retrieved from the full current customer question. "
                    "It never verifies private product claims, pricing, delivery, warranty or project applicability."
                ),
            },
            "question_plan": {
                "intent": question_plan["intent"],
                "task_type": task_type,
                "requires_project_conditions": bool(question_plan.get("requires_project_conditions")),
                "target_terms": question_plan.get("target_terms", []),
            },
            "retrieved_text_evidence": evidence_payload,
            "structured_project_cases": retrieval.get("project_cases", [])[:5],
        }
        payload_text = "请处理以下请求：\n" + json.dumps(payload, ensure_ascii=False)
        with temporary_uploaded_image(request.image_data_url) as uploaded_image_path:
            if uploaded_image_path is not None:
                raw = generate_visual_response(
                    system_prompt=GROUNDED_SYSTEM_PROMPT,
                    payload_text=payload_text,
                    image_path=uploaded_image_path,
                    max_new_tokens=680,
                )
            else:
                tokenizer, model = load_model()
                import torch

                messages = [
                    {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                    {"role": "user", "content": payload_text},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with generation_session(), torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=620,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        result = parse_json(raw)
        save_grounded_debug_output(raw)
        result = normalise_nonfactual_output_fields(result)
        result = apply_image_identity_guard(result, image_identity)
        if not is_safe_grounded_answer(result, set(evidence_by_id)):
            return AnswerResponse(
                **attach_online_search(
                    fallback_grounded_answer(request, retrieval, "grounded_output_validation_failed", question_plan),
                    online_sources,
                    online_search_meta,
                )
            )

        result["citations"] = materialize_citations(result["citations"], evidence_by_id)
        result["visual_assets"] = retrieval.get("visual_assets", [])
        result["retrieval"] = customer_visible_retrieval(retrieval)
        result["meta"] = {
            "model_used": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "mode": "grounded_local_rag_with_customer_image" if request.image_data_url else "grounded_local_rag",
            "evidence_count": len(evidence_payload),
            "customer_image_processed_locally": bool(request.image_data_url),
            "image_identity": image_identity,
        }
        return AnswerResponse(**attach_online_search(result, online_sources, online_search_meta))
    except Exception as exc:
        if os.getenv("FACADE_RAG_DEBUG") == "1":
            import traceback

            debug_path = ROOT / "runtime" / "last_answer_exception.txt"
            debug_path.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            task_type = str(question_plan.get("task_type") or "factual_lookup")
            retrieval = load_retriever().retrieve(
                request.customer_question,
                top_k=8 if task_type == "procedure" else 5,
                visual_k=5,
                retrieval_mode=task_type,
            )
        except Exception:
            retrieval = {"text_evidence": [], "visual_assets": [], "meta": {}}
        return AnswerResponse(
            **attach_online_search(
                fallback_grounded_answer(request, retrieval, type(exc).__name__, question_plan),
                online_sources,
                online_search_meta,
            )
        )


def _run_customer_document_answer(request: DraftRequest, plan: ToolPlan) -> AnswerResponse:
    """Answer from uploaded evidence, optionally fused with company RAG and web evidence."""

    started = time.perf_counter()
    document_result = retrieve_customer_documents(
        request.document_session_id or "",
        request.customer_question,
    )
    evidence_by_id: dict[str, dict[str, Any]] = {}
    evidence_payload: list[dict[str, Any]] = []
    for item in document_result.get("evidence", []):
        evidence_id = str(item["evidence_id"])
        evidence_by_id[evidence_id] = item
        evidence_payload.append(
            {
                "evidence_id": evidence_id,
                "document_name": item["document_name"],
                "source_refs": item.get("source_refs", []),
                "text": item["text"],
            }
        )

    selected_visuals = list(document_result.get("selected_visuals", []))
    customer_visual_assets: list[dict[str, Any]] = []
    for index, visual in enumerate(selected_visuals, start=1):
        evidence_id = f"V{index}"
        source = dict(visual.get("source") or {})
        citation = {
            "document_name": str(visual.get("document_name") or "Customer visual"),
            "source_page": source.get("page_number"),
            "section_heading": source.get("section_title") or source.get("section_id") or source.get("sheet_name") or str(visual.get("visual_id")),
            "source_type": "customer_upload_visual",
            "sheet_name": source.get("sheet_name"),
            "source_range": source.get("cell"),
            "row_index": source.get("row_index"),
            "bounding_box": source.get("bounding_box"),
            "parser": source.get("parser"),
            "extraction_confidence": source.get("extraction_confidence"),
        }
        item = {
            "document_name": citation["document_name"],
            "text": (
                f"Visual {index} ({visual.get('kind')}) from {citation['document_name']}; "
                f"page={citation['source_page']}; sheet={citation['sheet_name']}; metadata={visual.get('metadata') or {}}; "
                f"OCR/layout candidate={str(visual.get('searchable_text') or '')[:4000]}"
            ),
            "citations": [citation],
        }
        evidence_by_id[evidence_id] = item
        evidence_payload.append(
            {
                "evidence_id": evidence_id,
                "visual_index": index,
                "visual_id": visual.get("visual_id"),
                "document_name": citation["document_name"],
                "text": item["text"],
            }
        )
        document_id = str(visual.get("document_id") or "")
        visual_id = str(visual.get("visual_id") or "")
        page_number = source.get("page_number")
        sheet_name = source.get("sheet_name")
        source_range = source.get("source_range") or source.get("cell")
        if document_id and visual_id:
            location = (
                f"第 {page_number} 页"
                if page_number is not None
                else f"Sheet：{sheet_name}"
                if sheet_name
                else "附件原图"
            )
            customer_visual_assets.append(
                {
                    "asset_id": f"customer:{document_id}:{visual_id}",
                    "customer_title": f"{citation['document_name']} · {location}",
                    "asset_type": visual.get("kind"),
                    "effective_image_kind": visual.get("kind"),
                    "score": visual.get("score"),
                    "citation": {
                        "document_name": citation["document_name"],
                        "source_page": page_number,
                        "sheet_name": sheet_name,
                        "source_range": source_range,
                    },
                    "visual_endpoint": (
                        f"/api/copilot/documents/{quote(request.document_session_id or '', safe='')}"
                        f"/visual/{quote(document_id, safe='')}/{quote(visual_id, safe='')}"
                    ),
                    "facts_eligible": False,
                    "source_type": "customer_upload_visual",
                }
            )

    company_retrieval: dict[str, Any] = {"text_evidence": [], "visual_assets": [], "meta": {}}
    if "company_rag" in plan.tools:
        company_retrieval = load_retriever().retrieve(
            request.customer_question,
            top_k=5,
            visual_k=4,
            retrieval_mode="factual_lookup",
        )
        for index, item in enumerate(company_retrieval.get("text_evidence", []), start=1):
            evidence_id = f"T{index}"
            evidence_by_id[evidence_id] = item
            evidence_payload.append({"evidence_id": evidence_id, "text": item["text"]})

    online_sources: list[dict[str, Any]] = []
    online_search_meta: dict[str, str] = {"status": "not_requested"}
    if "public_web_search" in plan.tools:
        # The helper sends only the current question. Uploaded/private evidence
        # is never interpolated into the external query.
        online_sources, online_search_meta = maybe_search_online(
            request,
            request.customer_question,
            automatic_named_project_lookup=is_named_project_web_query(request.customer_question),
        )
        for source in online_sources:
            evidence_id = str(source.get("source_id") or "")
            excerpt = str(source.get("excerpt") or "")
            if not evidence_id or not excerpt:
                continue
            item = {
                "text": excerpt,
                "citations": [
                    {
                        "document_name": str(source.get("title") or "Online source"),
                        "source_page": None,
                        "section_heading": str(source.get("website") or "Web search"),
                        "source_url": str(source.get("url") or ""),
                        "source_type": "online",
                    }
                ],
            }
            evidence_by_id[evidence_id] = item
            evidence_payload.append({"evidence_id": evidence_id, "text": excerpt})

    if not evidence_payload:
        return AnswerResponse(
            intent="document_qa",
            normalized_terms=[],
            answerable=False,
            customer_reply="当前附件会话已过期，或没有可用于回答的解析证据。请重新上传文件后再试。",
            key_points=[],
            citations=[],
            missing_information=["可读取的客户文件证据"],
            risk_warnings=[],
            next_action="重新上传文件；一次最多 4 份。",
            meta={"model_used": False, "mode": "customer_document_session_unavailable"},
            image_observations=[],
            visual_assets=[],
            retrieval={"result_count": 0, "supporting_results": [], "visual_count": 0, "strategy": "uploaded_documents"},
            online_sources=[],
        )

    payload = {
        "conversation_context": compact_conversation_context(request),
        "customer_question": request.customer_question,
        "tool_plan": plan.model_dump(mode="json"),
        "rules": [
            "Only make factual claims supported by the supplied evidence IDs.",
            "Customer-uploaded evidence and company knowledge are distinct sources.",
            "If sources conflict, report the conflict instead of silently choosing one.",
            "If evidence is insufficient, answerable must be false.",
        ],
        "evidence": evidence_payload,
        "visual_input_order": [
            {
                "visual_index": index,
                "evidence_id": f"V{index}",
                "visual_id": visual.get("visual_id"),
                "document_name": visual.get("document_name"),
                "source": visual.get("source"),
            }
            for index, visual in enumerate(selected_visuals, start=1)
        ],
    }
    try:
        tokenizer, model = load_model()
        import torch

        payload_text = json.dumps(payload, ensure_ascii=False)
        with temporary_uploaded_image(request.image_data_url) as uploaded_image_path, temporary_visual_files(selected_visuals) as evidence_visual_paths:
            # A freshly uploaded standalone image is already present in the
            # document session. Prefer that direct original once, otherwise use
            # the question-selected PDF/Word/Excel visual assets.
            visual_paths = [uploaded_image_path] if uploaded_image_path is not None else evidence_visual_paths
            if visual_paths and ("visual_inspection" in plan.tools or selected_visuals):
                raw = generate_multi_visual_response(
                    system_prompt=GROUNDED_SYSTEM_PROMPT,
                    payload_text=payload_text,
                    image_paths=visual_paths,
                    max_new_tokens=480,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                        {"role": "user", "content": payload_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with generation_session(), torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=480,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        result = normalise_nonfactual_output_fields(parse_json(raw))
        if not is_safe_grounded_answer(result, set(evidence_by_id)):
            raise ValueError("uploaded_grounded_output_validation_failed")
        support_audit = evidence_support_audit(result, evidence_by_id)
        if support_audit["unsupported_numeric_claims"]:
            raise ValueError("uploaded_answer_contains_unsupported_numeric_claim")
        result["citations"] = materialize_citations(result["citations"], evidence_by_id)
        result["visual_assets"] = [*customer_visual_assets, *company_retrieval.get("visual_assets", [])]
        result["retrieval"] = {
            "result_count": len(evidence_payload),
            "supporting_results": [
                {
                    "result_id": item["evidence_id"],
                    "excerpt": str(item.get("text") or "")[:300],
                    "document_name": evidence_by_id[item["evidence_id"]].get("document_name"),
                }
                for item in evidence_payload[:8]
            ],
            "visual_count": len(customer_visual_assets) + len(company_retrieval.get("visual_assets", [])),
            "strategy": "cross_file_uploaded_evidence_plus_optional_company_rag",
        }
        result["online_sources"] = online_sources
        result["meta"] = {
            "model_used": True,
            "mode": "bounded_multi_document_agent",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "document_input_snapshot": document_result.get("input_snapshot", {}),
            "evidence_support_audit": support_audit,
            "online_search": online_search_meta,
        }
        return AnswerResponse(**result)
    except Exception:
        return AnswerResponse(
            intent="document_qa",
            normalized_terms=[],
            answerable=False,
            customer_reply="已找到相关附件片段，但本次回答未通过证据引用校验，因此没有直接给出结论。",
            key_points=[],
            citations=[],
            missing_information=["可验证的模型结构化回答"],
            risk_warnings=["系统不会把未通过引用校验的内容作为事实返回。"],
            next_action="可缩小问题范围，或指出要核对的文件和字段。",
            meta={"model_used": True, "mode": "uploaded_grounded_validation_failed"},
            image_observations=[],
            visual_assets=[*customer_visual_assets, *company_retrieval.get("visual_assets", [])],
            retrieval={
                "result_count": len(evidence_payload),
                "supporting_results": [],
                "visual_count": len(customer_visual_assets) + len(company_retrieval.get("visual_assets", [])),
                "strategy": "uploaded_documents",
            },
            online_sources=online_sources,
        )


class _CustomerAnswerWorkflowCallbacks:
    """Bridge the finite graph to the local model and guarded tools."""

    @staticmethod
    def plan_tools(request: DraftRequest) -> ToolPlan:
        return plan_customer_tools(request)

    @staticmethod
    def answer_planned(request: DraftRequest, plan: ToolPlan) -> AnswerResponse:
        if "customer_documents" in plan.tools:
            return _run_customer_document_answer(request, plan)
        if "company_rag" in plan.tools:
            return _run_facade_rag_answer(request, allow_public_web="public_web_search" in plan.tools)
        return _run_general_local_answer(request, allow_public_web="public_web_search" in plan.tools)


def customer_answer_graph():
    """Return one process-local graph without a checkpointer or cloud tracing."""

    global _answer_graph
    if _answer_graph is None:
        with _answer_graph_lock:
            if _answer_graph is None:
                _answer_graph = build_customer_answer_graph(_CustomerAnswerWorkflowCallbacks())
    return _answer_graph


@app.post("/api/copilot/answer", response_model=AnswerResponse)
def grounded_answer(request: DraftRequest) -> AnswerResponse:
    """Run one stateless LangGraph request and preserve the existing API contract."""

    try:
        state = customer_answer_graph().invoke({"request": request})
        return AnswerResponse(**state["response"])
    except Exception:
        # Do not turn a workflow-library issue into a failed customer request.
        # The established local paths remain the compatibility fallback.
        plan = fallback_plan(
            has_documents=bool(request.document_session_id and get_session(request.document_session_id)),
            has_image=bool(request.image_data_url),
            facade_related=is_facade_domain_request(request),
            web_requested=request.use_online_search,
        )
        return _CustomerAnswerWorkflowCallbacks.answer_planned(request, plan)


@app.post("/api/copilot/draft", response_model=DraftResponse)
def draft(request: DraftRequest) -> DraftResponse:
    started = time.perf_counter()
    try:
        tokenizer, model = load_model()
        import torch

        payload = request.model_dump(mode="json")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "待处理请求：\n" + json.dumps(payload, ensure_ascii=False)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with generation_session(), torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=420,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        result = parse_json(raw)
        if not is_safe_pre_rag_draft(result):
            return DraftResponse(**fallback_draft(request, "模型输出未通过预 RAG 安全校验"))
        result["meta"] = {
            "model_used": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "mode": "pre_rag_safe_refusal",
        }
        return DraftResponse(**result)
    except Exception as exc:  # keep the prototype available while the model is cold or unavailable
        return DraftResponse(**fallback_draft(request, type(exc).__name__))
