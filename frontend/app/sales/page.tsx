"use client";
/* eslint-disable @next/next/no-img-element -- source images are original, local RAG crops with dynamic dimensions. */

import {
  ChangeEvent,
  ClipboardEvent,
  DragEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  clearBrowserConversations,
  deleteBrowserConversation,
  loadBrowserConversations,
  saveBrowserConversation,
  type BrowserConversation,
} from "./browser-conversations";

// The interface is static and public. Answers and source visuals are served only
// while the workstation's local RAG service is online.
const publicModelApiBase = (process.env.NEXT_PUBLIC_MODEL_API_BASE ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const staticBasePath = (process.env.NEXT_PUBLIC_STATIC_BASE ?? "").replace(/\/$/, "");

type ServiceState = "checking" | "available" | "offline";

type AuthPrincipal = {
  user_id: string;
  username: string;
  display_name: string;
  role: "admin" | "member";
  can_access_internal: boolean;
  access_scopes: string[];
};

type ManagedUser = AuthPrincipal & {
  active: number | boolean;
  created_at: string;
};

const authSessionStorageKey = "facade-copilot-auth-token-v1";
const browserClientStorageKey = "facade-copilot-browser-client-v1";

function loadOrCreateBrowserClientId(): string {
  if (typeof window === "undefined") return "server-render-client";
  const existing = window.localStorage.getItem(browserClientStorageKey);
  if (existing && /^[A-Za-z0-9_-]{16,80}$/.test(existing)) return existing;
  const created = `web_${crypto.randomUUID().replace(/-/g, "")}`;
  window.localStorage.setItem(browserClientStorageKey, created);
  return created;
}

type SourceCitation = {
  evidence_id: string;
  document_name: string;
  source_page: number | null;
  section_heading?: string | null;
  sheet_name?: string | null;
  source_range?: string | null;
  source_url?: string | null;
};

type VisualAsset = {
  asset_id: string;
  customer_title: string;
  visual_endpoint: string;
  effective_image_kind?: string | null;
  visual_role?: string | null;
  gallery_type?: string | null;
  explanation?: string | null;
  project_name?: string | null;
  product_name?: string | null;
  canonical_product?: string | null;
  matched_product?: string | null;
  variant_or_code?: string | null;
  matched_variant_or_code?: string | null;
  installation_method?: string | null;
  area_m2?: string | number | null;
  completion_year?: string | number | null;
  related_case?: VisualCaseContext | null;
  related_project?: VisualProjectContext | null;
  related_product?: VisualProductContext | null;
  project?: VisualProjectContext | string | null;
  product?: VisualProductContext | string | null;
  citation: {
    document_name: string;
    source_page: number | null;
    sheet_name?: string | null;
    source_range?: string | null;
    source_url?: string | null;
  };
};

type VisualCaseContext = {
  case_id?: string | null;
  project_name?: string | null;
  name?: string | null;
  product?: string | null;
  installation_method?: string | null;
  area_m2?: string | number | null;
  completion_year?: string | number | null;
  explanation?: string | null;
};

type VisualProjectContext = {
  project_id?: string | null;
  project_name?: string | null;
  name?: string | null;
  installation_method?: string | null;
  area_m2?: string | number | null;
  completion_year?: string | number | null;
};

type VisualProductContext = {
  product_id?: string | null;
  product_name?: string | null;
  canonical_name?: string | null;
  canonical_product?: string | null;
  name?: string | null;
  model?: string | null;
  variant_or_code?: string | null;
};

type SupportingResult = {
  result_id: string;
  excerpt: string;
  document_name?: string | null;
  source_page?: number | null;
  section_heading?: string | null;
  source_url?: string | null;
};

type RetrievalSummary = {
  incomplete_visual_documents?: string[];
  result_count: number;
  supporting_results: SupportingResult[];
  visual_count: number;
  strategy?: string;
  attachment_status?: "parsed" | "unavailable";
  parsed_document_count?: number;
  document_index_only?: boolean;
};

type ImageIdentity = {
  status: "not_provided" | "appearance_candidate" | "unverified";
  visible_subject: string;
  message: string;
  matches: Array<VisualAsset & { similarity: number }>;
};

type OnlineSource = {
  source_id: string;
  title: string;
  url: string;
  website?: string;
  date?: string;
  excerpt?: string;
  evidence_level?: "verified_page_content" | "official_search_excerpt" | "unverified_search_excerpt" | "unavailable";
  page_fetch_status?: string;
  authority_tier?: string;
  final_score?: number;
};

type CopilotAnswer = {
  answerable: boolean;
  customer_reply: string;
  key_points: string[];
  citations: SourceCitation[];
  visual_assets: VisualAsset[];
  online_sources?: OnlineSource[];
  retrieval?: RetrievalSummary;
  missing_information: string[];
  risk_warnings: string[];
  next_action: string;
  image_observations: string[];
  meta?: {
    execution?: { request_id?: string; state?: string; errors?: Array<{ code: string; stage: string; message: string; action: string; retryable: boolean }> };
    orchestration?: { tools?: string[] };
    model_used?: boolean;
    latency_ms?: number;
    wants_visuals?: boolean;
    visual_scope?: string | null;
    explicit_visual_request?: boolean;
    image_identity?: ImageIdentity;
    online_search?: {
      status?: "not_requested" | "not_configured" | "ok" | "failed" | "quota_exhausted";
      message?: string;
      source_profile?: string;
      cache_hit?: boolean;
      quota?: {
        business_limit?: number;
        hard_limit?: number;
        used?: number;
        remaining?: number;
        reserved?: number;
      };
      api_calls_for_query?: number;
    };
  };
};

function answerSourceLabel(answer: CopilotAnswer): string {
  const tools = answer.meta?.orchestration?.tools ?? [];
  const labels: string[] = [];
  if (tools.includes("customer_documents")) labels.push("附件分析");
  if (tools.includes("company_rag")) labels.push("企业知识库");
  if (tools.includes("visual_inspection")) labels.push("图片理解");
  if ((answer.online_sources?.length ?? 0) > 0) labels.push("联网资料");
  else if (tools.includes("public_web_search")) labels.push("联网未获得可用资料");
  return labels.length ? labels.join(" + ") : "本地模型回答";
}

type AttachedImage = {
  name: string;
  dataUrl: string;
};

type PendingAttachment = {
  id: string;
  file: File;
  previewUrl?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  answer?: CopilotAnswer;
  image?: AttachedImage;
  attachmentNames?: string[];
};

type ConversationContextTurn = {
  role: "user" | "assistant";
  content: string;
};

const legacyConversationSessionKey = "facade-copilot-conversation-v1";
const MAX_MESSAGES_PER_CONVERSATION = 30;
const MAX_ATTACHMENT_COUNT = 4;
const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;
const acceptedAttachmentExtensions = new Set([
  ".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".html", ".htm", ".xml", ".zip",
  ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
]);

const starterQuestions = [
  "旧楼改造有哪些安装方式？",
  "有哪些山东的项目案例？",
  "窗洞口节点图怎么做？",
];

const welcomeMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "你好，我是建材小助手。我会从已导入的产品资料、施工方案、节点图集和项目案例中检索信息，并在回答中展示对应的原始图纸或案例图片。",
};

const defaultConversationMessages: ChatMessage[] = [welcomeMessage];

type LocalConversation = BrowserConversation<ChatMessage>;

const documentSessionByConversation = new Map<string, string>();

function loadLegacyConversation(): ChatMessage[] {
  if (typeof window === "undefined") return [welcomeMessage];
  try {
    const raw = window.sessionStorage.getItem(legacyConversationSessionKey);
    if (!raw) return [welcomeMessage];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [welcomeMessage];
    const safeMessages = parsed
      .filter(
        (message): message is ChatMessage =>
          message &&
          (message.role === "user" || message.role === "assistant") &&
          typeof message.content === "string" &&
          typeof message.id === "string",
      )
      .slice(-MAX_MESSAGES_PER_CONVERSATION)
      // Customer-uploaded images are deliberately not retained in browser memory.
      .map((message) => ({ ...message, image: undefined }));
    return safeMessages.length > 0 ? safeMessages : [welcomeMessage];
  } catch {
    return [welcomeMessage];
  }
}

function makeConversation(messages: ChatMessage[] = defaultConversationMessages): LocalConversation {
  const now = Date.now();
  return {
    id: makeId(),
    title: conversationTitle(messages),
    createdAt: now,
    updatedAt: now,
    messages: persistableMessages(messages),
  };
}

function persistableMessages(messages: ChatMessage[]): ChatMessage[] {
  // Customer-uploaded images are intentionally not saved. They are sent only
  // with the request that needs them and never become browser chat history.
  return messages.slice(-MAX_MESSAGES_PER_CONVERSATION).map((message) => ({
    ...message,
    image: undefined,
    attachmentNames: undefined,
  }));
}

function conversationTitle(messages: ChatMessage[]): string {
  const firstQuestion = messages.find((message) => message.role === "user" && message.content.trim());
  if (!firstQuestion) return "新对话";
  const compact = firstQuestion.content.replace(/\s+/g, " ").trim();
  return compact.length > 22 ? `${compact.slice(0, 22)}…` : compact;
}

function formatConversationTime(timestamp: number): string {
  const value = new Date(timestamp);
  const today = new Date();
  const isToday = value.toDateString() === today.toDateString();
  return isToday
    ? value.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
    : value.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function buildConversationContext(messages: ChatMessage[]): ConversationContextTurn[] {
  return messages
    .filter((message) => message.id !== welcomeMessage.id)
    .slice(-6)
    .map((message) => ({
      role: message.role,
      content: message.content.replace(/\s+/g, " ").trim().slice(0, 800),
    }))
    .filter((message) => message.content.length > 0);
}

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function sourceUrl(asset: VisualAsset) {
  return `${publicModelApiBase}${asset.visual_endpoint}`;
}

function visualLocation(asset: VisualAsset) {
  if (asset.citation.source_page) return `第 ${asset.citation.source_page} 页`;
  if (asset.citation.sheet_name) {
    return `Sheet：${asset.citation.sheet_name}${asset.citation.source_range ? ` · ${asset.citation.source_range}` : ""}`;
  }
  if (asset.citation.source_url) return "官方网站原图";
  return "附件原图";
}

type VisualGalleryKind = "case" | "product" | "component" | "application" | "node" | "process" | "other";

function valueText(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number") return String(value);
  return "";
}

function recordText(value: unknown, keys: string[]): string {
  if (typeof value === "string" || typeof value === "number") return valueText(value);
  if (!value || typeof value !== "object") return "";
  const record = value as Record<string, unknown>;
  for (const key of keys) {
    const text = valueText(record[key]);
    if (text) return text;
  }
  return "";
}

function firstText(...values: unknown[]): string {
  for (const value of values) {
    const text = valueText(value);
    if (text) return text;
  }
  return "";
}

function usableProductText(value: unknown): string {
  const text = valueText(value);
  return ["介绍", "体系", "说明", "产品", "图片", "照片", "配图", "展示"].includes(text) ? "" : text;
}

function productIdentity(asset: VisualAsset): { product: string; variant: string } {
  const productCandidates = [
    asset.product_name,
    asset.canonical_product,
    asset.matched_product,
    recordText(asset.related_product, ["product_name", "canonical_product", "canonical_name", "name"]),
    recordText(asset.product, ["product_name", "canonical_product", "canonical_name", "name"]),
    recordText(asset.related_case, ["product"]),
  ];
  const variantCandidates = [
    asset.variant_or_code,
    asset.matched_variant_or_code,
    recordText(asset.related_product, ["variant_or_code", "model"]),
    recordText(asset.product, ["variant_or_code", "model"]),
  ];
  return {
    product: productCandidates.map(usableProductText).find(Boolean) ?? "",
    variant: variantCandidates.map(usableProductText).find(Boolean) ?? "",
  };
}

function visualGalleryKind(asset: VisualAsset): VisualGalleryKind {
  const declaredRole = firstText(asset.gallery_type, asset.visual_role).toLowerCase();
  const hasProductIdentity = Boolean(productIdentity(asset).product || productIdentity(asset).variant);

  if (declaredRole) {
    if (/case|project|案例|项目/.test(declaredRole)) return "case";
    if (/component|accessory|配件|构件|辅材/.test(declaredRole)) return "component";
    if (/application[_ -]?effect|应用效果|效果展示/.test(declaredRole)) return "application";
    if (/node|detail|drawing|节点|构造/.test(declaredRole)) return "node";
    if (/process|procedure|construction|工艺|施工/.test(declaredRole)) return "process";
    if (/product(?:[_ -]?(?:overview|variant|sample))?|finish|产品|饰面/.test(declaredRole)) return "product";

    // Once the backend supplies a gallery role it is authoritative. Unknown
    // roles remain generic instead of being reclassified by query wording.
    return "other";
  }

  // Legacy assets may not carry the new role contract. In that case only the
  // asset's own audited image kind may provide a conservative fallback. The
  // answer-level requested scope must never reclassify an unknown image.
  const imageKind = valueText(asset.effective_image_kind).toLowerCase();
  if (asset.related_case || asset.related_project || asset.project || /project[_ -]?photo/.test(imageKind)) return "case";
  if (/construction[_ -]?detail|node|drawing/.test(imageKind)) return "node";
  if (/process|procedure/.test(imageKind)) return "process";
  if (/component|accessory/.test(imageKind)) return "component";
  if (/application[_ -]?effect/.test(imageKind)) return "application";
  if (/product[_ -]?photo/.test(imageKind) && hasProductIdentity) return "product";
  return "other";
}

function visualCardTitle(asset: VisualAsset, kind: VisualGalleryKind): string {
  if (kind !== "product") return asset.customer_title;
  const { product, variant } = productIdentity(asset);
  if (product && variant) return `${product} · ${variant}`;
  return product || variant || asset.customer_title;
}

function galleryHeading(kind: VisualGalleryKind): string {
  if (kind === "case") return "项目案例实景";
  if (kind === "product") return "产品与饰面图片";
  if (kind === "component") return "配套构件与辅材";
  if (kind === "application") return "应用效果参考";
  if (kind === "node") return "节点与构造图";
  if (kind === "process") return "施工工艺图片";
  return "相关资料图片";
}

function visualCardDetails(asset: VisualAsset): Array<{ label: string; value: string }> {
  const project = firstText(
    asset.project_name,
    recordText(asset.related_case, ["project_name", "name"]),
    recordText(asset.related_project, ["project_name", "name"]),
    recordText(asset.project, ["project_name", "name"]),
  );
  const { product, variant } = productIdentity(asset);
  const installationMethod = firstText(
    asset.installation_method,
    recordText(asset.related_case, ["installation_method"]),
    recordText(asset.related_project, ["installation_method"]),
    recordText(asset.project, ["installation_method"]),
  );
  const area = firstText(
    asset.area_m2,
    recordText(asset.related_case, ["area_m2"]),
    recordText(asset.related_project, ["area_m2"]),
    recordText(asset.project, ["area_m2"]),
  );
  const year = firstText(
    asset.completion_year,
    recordText(asset.related_case, ["completion_year"]),
    recordText(asset.related_project, ["completion_year"]),
    recordText(asset.project, ["completion_year"]),
  );
  return [
    { label: "项目", value: project },
    { label: "产品", value: product },
    { label: "型号/饰面", value: variant },
    { label: "工艺", value: installationMethod },
    { label: "面积", value: area },
    { label: "年份", value: year },
  ].filter((item) => item.value);
}

function visualExplanation(asset: VisualAsset): string {
  return firstText(asset.explanation, recordText(asset.related_case, ["explanation"]));
}

function InlineVisualGallery({
  assets,
  onOpenImage,
}: {
  assets: VisualAsset[];
  onOpenImage: (asset: VisualAsset) => void;
}) {
  const deduplicated = Array.from(new Map(assets.map((asset) => [asset.asset_id, asset])).values());
  const grouped = deduplicated.reduce<Map<VisualGalleryKind, VisualAsset[]>>((groups, asset) => {
    const kind = visualGalleryKind(asset);
    groups.set(kind, [...(groups.get(kind) ?? []), asset]);
    return groups;
  }, new Map());
  const order: VisualGalleryKind[] = ["product", "component", "application", "case", "node", "process", "other"];

  return (
    <section className="inline-visual-gallery" aria-label="与回答直接相关的图片">
      <div className="detail-heading inline-gallery-heading">
        <span>相关图片</span>
        <small>点击图片可查看带来源的完整原图</small>
      </div>
      {order.map((kind) => {
        const group = grouped.get(kind) ?? [];
        if (group.length === 0) return null;
        return (
          <section className="visual-gallery-group" key={kind}>
            <h3>{galleryHeading(kind)}</h3>
            <div className="visual-grid inline-visual-grid">
              {group.map((asset) => {
                const details = visualCardDetails(asset);
                const explanation = visualExplanation(asset);
                const cardTitle = visualCardTitle(asset, kind);
                return (
                  <article className="inline-visual-card" key={asset.asset_id}>
                    <button className="inline-visual-preview" type="button" onClick={() => onOpenImage(asset)}>
                      <span className="image-frame">
                        <img src={sourceUrl(asset)} alt={asset.customer_title} loading="lazy" />
                        <em>查看原图 ↗</em>
                      </span>
                    </button>
                    <div className="inline-visual-copy">
                      <strong>{cardTitle}</strong>
                      {kind === "product" && cardTitle !== asset.customer_title && (
                        <span className="inline-visual-asset-title">{asset.customer_title}</span>
                      )}
                      {explanation && <p>{explanation}</p>}
                      {details.length > 0 && (
                        <dl>
                          {details.map((item) => (
                            <div key={`${asset.asset_id}-${item.label}`}>
                              <dt>{item.label}</dt>
                              <dd>{item.value}</dd>
                            </div>
                          ))}
                        </dl>
                      )}
                      <small>{asset.citation.document_name} · {visualLocation(asset)}</small>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        );
      })}
    </section>
  );
}

function cleanDisplayText(text: string) {
  return text
    .replace(/&(?:#x20|#32|nbsp);/gi, " ")
    .replace(/\\\s*(?=\n|$)/g, "")
    .trim();
}

function normalizeAnswerForDisplay(answer: CopilotAnswer): CopilotAnswer {
  const cleanList = (items: string[]) => items.map(cleanDisplayText).filter(Boolean);
  const retrieval = answer.retrieval
    ? {
        ...answer.retrieval,
        supporting_results: (answer.retrieval.supporting_results ?? [])
          .map((item) => ({
            ...item,
            result_id: cleanDisplayText(String(item.result_id ?? "")),
            document_name: cleanDisplayText(String(item.document_name ?? "")) || null,
            section_heading: cleanDisplayText(String(item.section_heading ?? "")) || null,
            excerpt: cleanDisplayText(String(item.excerpt ?? "")),
          }))
          .filter((item) => item.result_id || item.document_name || item.excerpt),
      }
    : answer.retrieval;
  return {
    ...answer,
    customer_reply: cleanDisplayText(answer.customer_reply),
    key_points: cleanList(answer.key_points ?? []),
    missing_information: cleanList(answer.missing_information ?? []),
    risk_warnings: cleanList(answer.risk_warnings ?? []),
    next_action: cleanDisplayText(answer.next_action ?? ""),
    image_observations: cleanList(answer.image_observations ?? []),
    retrieval,
  };
}

function ReplyParagraphs({ text }: { text: string }) {
  const cleaned = cleanDisplayText(text);
  return (
    <div className="reply-copy">
      {cleaned
        .split(/\n+/)
        .filter(Boolean)
        .map((paragraph, index) => (
          <p key={`${paragraph}-${index}`}>{paragraph}</p>
        ))}
    </div>
  );
}

function readImageAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("图片读取失败"));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(file);
  });
}

export default function Home() {
  const [browserClientId] = useState(loadOrCreateBrowserClientId);
  const [conversations, setConversations] = useState<LocalConversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [historyReady, setHistoryReady] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [requestStage, setRequestStage] = useState("");
  const [coverageNotice, setCoverageNotice] = useState("");
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [activeImage, setActiveImage] = useState<VisualAsset | null>(null);
  const [attachedImage, setAttachedImage] = useState<AttachedImage | null>(null);
  const [pendingAttachments, setPendingAttachments] = useState<PendingAttachment[]>([]);
  const [isFileDragging, setIsFileDragging] = useState(false);
  // This is permission, not a command: the local Planner still decides
  // whether a turn genuinely needs the metered public-web tool.  Keeping the
  // permission enabled makes current-fact questions work out of the box,
  // while greetings and local product questions remain offline.
  const [useOnlineSearch, setUseOnlineSearch] = useState(true);
  const [useTaskMemory, setUseTaskMemory] = useState(false);
  const [authToken, setAuthToken] = useState<string | null>(() => (
    typeof window === "undefined" ? null : window.sessionStorage.getItem(authSessionStorageKey)
  ));
  const [authPrincipal, setAuthPrincipal] = useState<AuthPrincipal | null>(null);
  const [authStatusResolved, setAuthStatusResolved] = useState(false);
  const [bootstrapRequired, setBootstrapRequired] = useState(false);
  const [authDialogOpen, setAuthDialogOpen] = useState(false);
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const [managedUsers, setManagedUsers] = useState<ManagedUser[]>([]);
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authForm, setAuthForm] = useState({ setupToken: "", username: "", displayName: "", password: "" });
  const [newUserForm, setNewUserForm] = useState({ username: "", displayName: "", password: "", canAccessInternal: true });
  const historyScope = authPrincipal
    ? `user:${authPrincipal.user_id}`
    : authToken
      ? "authenticated-pending"
      : "anonymous";
  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId) ?? null;
  const messages = activeConversation?.messages.length ? activeConversation.messages : defaultConversationMessages;
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);
  const textAreaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragDepthRef = useRef(0);

  useEffect(() => {
    const textarea = textAreaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const maximumHeight = 168;
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, 34), maximumHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maximumHeight ? "auto" : "hidden";
  }, [draft]);

  useEffect(() => {
    const controller = new AbortController();
    setAuthStatusResolved(false);
    const timer = window.setTimeout(() => controller.abort(), 5000);

    fetch(`${publicModelApiBase}/health/live`, { signal: controller.signal })
      .then((response) => setServiceState(response.ok ? "available" : "offline"))
      .catch(() => setServiceState("offline"))
      .finally(() => window.clearTimeout(timer));

    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register(`${staticBasePath}/sw.js`).catch(() => undefined);
    }

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const headers: HeadersInit = {
      "X-Facade-Client-ID": browserClientId,
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    };
    fetch(`${publicModelApiBase}/api/auth/status`, { headers, signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("auth_status_failed");
        return response.json() as Promise<{
          bootstrap_required: boolean;
          authenticated: boolean;
          principal: AuthPrincipal | null;
        }>;
      })
      .then((status) => {
        setBootstrapRequired(status.bootstrap_required);
        setAuthPrincipal(status.authenticated ? status.principal : null);
        if (!status.authenticated && authToken) {
          window.sessionStorage.removeItem(authSessionStorageKey);
          setAuthToken(null);
        }
      })
      .catch(() => {
        // Authentication is optional for public knowledge. A temporarily
        // unavailable status endpoint must not disable anonymous RAG.
      })
      .finally(() => setAuthStatusResolved(true));
    return () => controller.abort();
  }, [authToken, browserClientId]);

  function authenticatedFetch(input: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers);
    if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
    headers.set("X-Facade-Client-ID", browserClientId);
    return fetch(input, { ...init, headers });
  }

  async function submitAuthentication(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      if (bootstrapRequired && authForm.setupToken.trim().length < 20) {
        throw new Error("请粘贴本机 runtime/admin_bootstrap_token.txt 中的完整一次性令牌，不是自定义的短数字。");
      }
      const endpoint = bootstrapRequired ? "/api/auth/bootstrap" : "/api/auth/login";
      const payload = bootstrapRequired
        ? {
            setup_token: authForm.setupToken,
            username: authForm.username,
            display_name: authForm.displayName || authForm.username,
            password: authForm.password,
          }
        : { username: authForm.username, password: authForm.password };
      const response = await fetch(`${publicModelApiBase}${endpoint}`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "X-Facade-Client-ID": browserClientId,
        },
        body: JSON.stringify(payload),
      });
      const body = await response.json().catch(() => ({})) as {
        detail?: string | Array<{ msg?: string }>;
        access_token?: string;
        principal?: AuthPrincipal;
      };
      if (!response.ok || !body.access_token || !body.principal) {
        const detail = typeof body.detail === "string"
          ? body.detail
          : body.detail?.map((item) => item.msg).filter(Boolean).join("；");
        throw new Error(detail || "登录失败");
      }
      window.sessionStorage.setItem(authSessionStorageKey, body.access_token);
      setAuthToken(body.access_token);
      setAuthPrincipal(body.principal);
      setBootstrapRequired(false);
      setAuthDialogOpen(false);
      setAuthForm({ setupToken: "", username: "", displayName: "", password: "" });
    } catch (error) {
      const message = error instanceof Error ? error.message : "登录失败";
      setAuthError(message === "Failed to fetch"
        ? "无法连接本地后端。请确认本机后端与 Tailscale Funnel 正在运行，然后重试。"
        : message);
    } finally {
      setAuthBusy(false);
    }
  }

  async function logout() {
    if (authToken) {
      await authenticatedFetch(`${publicModelApiBase}/api/auth/logout`, { method: "POST" }).catch(() => undefined);
    }
    window.sessionStorage.removeItem(authSessionStorageKey);
    setAuthToken(null);
    setAuthPrincipal(null);
    setAdminPanelOpen(false);
  }

  async function loadManagedUsers() {
    if (!authToken || authPrincipal?.role !== "admin") return;
    setAuthError("");
    const response = await authenticatedFetch(`${publicModelApiBase}/api/auth/admin/users`, { cache: "no-store" });
    const body = await response.json().catch(() => ({})) as { detail?: string; users?: ManagedUser[] };
    if (!response.ok) throw new Error(body.detail || "无法读取人员列表");
    setManagedUsers(body.users ?? []);
  }

  async function openAdminPanel() {
    setAdminPanelOpen(true);
    try {
      await loadManagedUsers();
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "无法读取人员列表");
    }
  }

  async function createManagedUser(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const response = await authenticatedFetch(`${publicModelApiBase}/api/auth/admin/users`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          username: newUserForm.username,
          display_name: newUserForm.displayName || newUserForm.username,
          password: newUserForm.password,
          role: "member",
          can_access_internal: newUserForm.canAccessInternal,
        }),
      });
      const body = await response.json().catch(() => ({})) as { detail?: string };
      if (!response.ok) throw new Error(body.detail || "创建用户失败");
      setNewUserForm({ username: "", displayName: "", password: "", canAccessInternal: true });
      await loadManagedUsers();
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "创建用户失败");
    } finally {
      setAuthBusy(false);
    }
  }

  async function updateManagedUser(user: ManagedUser, changes: Record<string, boolean | string>) {
    setAuthError("");
    const response = await authenticatedFetch(`${publicModelApiBase}/api/auth/admin/users/${encodeURIComponent(user.user_id)}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(changes),
    });
    const body = await response.json().catch(() => ({})) as { detail?: string };
    if (!response.ok) {
      setAuthError(body.detail || "权限更新失败");
      return;
    }
    await loadManagedUsers();
  }

  useEffect(() => {
    if (!authStatusResolved) return;
    let cancelled = false;
    setHistoryReady(false);
    documentSessionByConversation.clear();
    void loadBrowserConversations<ChatMessage>(historyScope).then((savedConversations) => {
      if (cancelled) return;
      let initialConversations = savedConversations;
      if (initialConversations.length === 0) {
        const legacyMessages = loadLegacyConversation();
        const initial = makeConversation(legacyMessages);
        initialConversations = [initial];
        void saveBrowserConversation(initial, historyScope);
        // Move the previous single-session implementation forward once. It
        // remains local to this browser and is then removed from sessionStorage.
        window.sessionStorage.removeItem(legacyConversationSessionKey);
      }
      const active = initialConversations[0];
      initialConversations.forEach((conversation) => {
        if (conversation.documentSessionId) {
          documentSessionByConversation.set(conversation.id, conversation.documentSessionId);
        }
      });
      setConversations(initialConversations);
      setActiveConversationId(active.id);
      setHistoryReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, [authStatusResolved, historyScope]);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  function updateActiveConversationMessages(
    update: (current: ChatMessage[]) => ChatMessage[],
    targetConversationId: string | null = activeConversationId,
  ) {
    if (!historyReady || !targetConversationId) return;
    setConversations((current) => {
      const existing = current.find((conversation) => conversation.id === targetConversationId);
      if (!existing) return current;
      const nextMessages = persistableMessages(update(existing.messages));
      const updated: LocalConversation = {
        ...existing,
        title: conversationTitle(nextMessages),
        updatedAt: Date.now(),
        messages: nextMessages,
      };
      const ordered = [updated, ...current.filter((conversation) => conversation.id !== targetConversationId)];
      const next = ordered.slice(0, 24);
      void saveBrowserConversation(updated, historyScope);
      ordered.slice(24).forEach((conversation) => void deleteBrowserConversation(conversation.id, historyScope));
      return next;
    });
  }

  function updateConversationDocumentSession(
    conversationId: string,
    documentSessionId: string | null,
    attachmentNames: string[] = [],
  ) {
    if (documentSessionId) documentSessionByConversation.set(conversationId, documentSessionId);
    else documentSessionByConversation.delete(conversationId);
    setConversations((current) => {
      const existing = current.find((conversation) => conversation.id === conversationId);
      if (!existing) return current;
      const updated: LocalConversation = {
        ...existing,
        documentSessionId,
        attachmentNames: documentSessionId ? attachmentNames : [],
        updatedAt: Date.now(),
      };
      void saveBrowserConversation(updated, historyScope);
      return [updated, ...current.filter((conversation) => conversation.id !== conversationId)];
    });
  }

  function attachmentExtension(file: File) {
    const dot = file.name.lastIndexOf(".");
    return dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  }

  async function queueAttachments(files: File[]) {
    const selected = files.filter((file) => file.size > 0);
    if (selected.length === 0) return;
    const unsupported = selected.filter((file) => (
      !acceptedAttachmentExtensions.has(attachmentExtension(file))
      && !file.type.startsWith("image/")
    ));
    if (unsupported.length > 0) {
      window.alert(`暂不支持：${unsupported.map((file) => file.name).join("、")}`);
      return;
    }
    if (pendingAttachments.length + selected.length > MAX_ATTACHMENT_COUNT) {
      window.alert(`一次最多上传 ${MAX_ATTACHMENT_COUNT} 份文件。请先移除部分附件。`);
      return;
    }
    if (selected.some((file) => file.size > MAX_ATTACHMENT_BYTES)) {
      window.alert("单个文件请控制在 25 MB 以内。");
      return;
    }
    const existingKeys = new Set(
      pendingAttachments.map(({ file }) => `${file.name}:${file.size}:${file.lastModified}`),
    );
    const additions = selected
      .filter((file) => !existingKeys.has(`${file.name}:${file.size}:${file.lastModified}`))
      .map((file) => ({
      id: makeId(),
      file,
      previewUrl: file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
      }));
    if (additions.length === 0) return;
    setPendingAttachments((current) => [...current, ...additions]);
    const firstVisual = additions.map((item) => item.file).find((file) => /^image\/(jpeg|png|webp)$/.test(file.type));
    if (firstVisual && !attachedImage) {
      try {
        setAttachedImage({ name: firstVisual.name, dataUrl: await readImageAsDataUrl(firstVisual) });
      } catch {
        // The generic parser can still process the attachment without preview.
      }
    }
  }

  async function handleAttachmentSelection(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    event.target.value = "";
    await queueAttachments(selected);
  }

  function isFileDrag(event: DragEvent<HTMLDivElement>) {
    return Array.from(event.dataTransfer.types).includes("Files");
  }

  function handleFileDragEnter(event: DragEvent<HTMLDivElement>) {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    if (isSending) return;
    dragDepthRef.current += 1;
    setIsFileDragging(true);
  }

  function handleFileDragLeave(event: DragEvent<HTMLDivElement>) {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) setIsFileDragging(false);
  }

  function handleFileDragOver(event: DragEvent<HTMLDivElement>) {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = isSending ? "none" : "copy";
  }

  function handleFileDrop(event: DragEvent<HTMLDivElement>) {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    dragDepthRef.current = 0;
    setIsFileDragging(false);
    if (!isSending) void queueAttachments(Array.from(event.dataTransfer.files ?? []));
  }

  function handleClipboardPaste(event: ClipboardEvent<HTMLDivElement>) {
    const files = Array.from(event.clipboardData.files ?? []);
    if (files.length === 0 || isSending) return;
    event.preventDefault();
    void queueAttachments(files);
  }

  function removePendingAttachment(id: string) {
    setPendingAttachments((current) => {
      const target = current.find((item) => item.id === id);
      if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl);
      if (target?.file.name === attachedImage?.name) setAttachedImage(null);
      return current.filter((item) => item.id !== id);
    });
  }

  async function sendQuestion(question: string) {
    const cleanQuestion = question.trim();
    const imageForRequest = attachedImage;
    const attachmentsForRequest = [...pendingAttachments];
    if ((!cleanQuestion && attachmentsForRequest.length === 0) || isSending || !historyReady || !activeConversationId) return;
    const requestConversationId = activeConversationId;
    const customerQuestion = cleanQuestion || "请阅读并概括我刚上传的附件，说明文件结构、主要内容和可核验信息。";

    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: cleanQuestion || "请阅读并概括我刚上传的附件。",
      image: imageForRequest ?? undefined,
      attachmentNames: attachmentsForRequest.map((item) => item.file.name),
    };
    updateActiveConversationMessages((current) => [...current, userMessage], requestConversationId);
    setDraft("");
    setAttachedImage(null);
    attachmentsForRequest.forEach((item) => item.previewUrl && URL.revokeObjectURL(item.previewUrl));
    setPendingAttachments([]);
    setIsSending(true);
    const requestController = new AbortController();
    setRequestStage(attachmentsForRequest.length ? "正在上传并解析附件…" : "正在检查附件并准备回答…");
    setCoverageNotice("");
    let requestTimer = window.setTimeout(() => requestController.abort(), 300_000);

    try {
      let documentSessionId: string | null = null;
      const persistedSessionId = documentSessionByConversation.get(requestConversationId)
        ?? (activeConversation?.id === requestConversationId ? activeConversation.documentSessionId ?? null : null);
      if (attachmentsForRequest.length > 0) {
        const form = new FormData();
        attachmentsForRequest.forEach((item) => form.append("files", item.file));
        if (persistedSessionId) form.append("session_id", persistedSessionId);
        const uploadResponse = await authenticatedFetch(`${publicModelApiBase}/api/copilot/documents`, {
          method: "POST",
          body: form,
          signal: requestController.signal,
        });
        if (!uploadResponse.ok) {
          const failure = await uploadResponse.json().catch(() => null) as { detail?: string; error?: {code?: string; action?: string}; request_id?: string } | null;
          throw new Error([failure?.detail || `附件上传失败（HTTP ${uploadResponse.status}）`,
            failure?.error?.action, failure?.error?.code,
            failure?.request_id ? `请求编号：${failure.request_id}` : null].filter(Boolean).join(" · "));
        }
        const uploaded = (await uploadResponse.json()) as {
          session_id: string;
          documents?: Array<{ file_name?: string; visual_coverage?: { coverage_complete?: boolean; page_count?: number; rendered_page_numbers?: number[] } }>;
        };
        documentSessionId = uploaded.session_id;
        const incomplete = (uploaded.documents ?? []).filter((doc) => doc.visual_coverage?.coverage_complete === false);
        if (incomplete.length) {
          setCoverageNotice(incomplete.map((doc) => `${doc.file_name ?? "附件"}：视觉页面尚未全部处理，后续回答仅基于已解析部分。`).join(" "));
        }
        const uploadedNames = (uploaded.documents ?? [])
          .map((document) => document.file_name?.trim() ?? "")
          .filter(Boolean);
        updateConversationDocumentSession(
          requestConversationId,
          uploaded.session_id,
          uploadedNames.length > 0
            ? uploadedNames
            : Array.from(new Set([
                ...(activeConversation?.attachmentNames ?? []),
                ...attachmentsForRequest.map((item) => item.file.name),
              ])),
        );
      } else {
        documentSessionId = persistedSessionId;
        if (documentSessionId) {
          const sessionResponse = await authenticatedFetch(
            `${publicModelApiBase}/api/copilot/documents/${encodeURIComponent(documentSessionId)}`,
            { method: "GET", cache: "no-store", signal: requestController.signal },
          );
          if (!sessionResponse.ok) {
            updateConversationDocumentSession(requestConversationId, null);
            throw new Error("该对话的附件临时会话已过期或后端已重启。请重新上传附件后再继续提问。");
          }
        }
      }
      window.clearTimeout(requestTimer);
      requestTimer = window.setTimeout(() => requestController.abort(), 115_000);
      setRequestStage("正在检索证据并生成回答…");
      const response = await authenticatedFetch(`${publicModelApiBase}/api/copilot/answer`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          customer_question: customerQuestion,
          image_data_url: imageForRequest?.dataUrl ?? null,
          document_session_id: documentSessionId,
          project_context: { customer_type: "客户/销售" },
          conversation_context: buildConversationContext(messages),
          conversation_id: activeConversationId,
          memory_enabled: useTaskMemory && Boolean(activeConversationId),
          use_online_search: useOnlineSearch,
        }),
        signal: requestController.signal,
      });

      if (!response.ok) {
        const failure = await response.json().catch(() => null) as { detail?: string; error?: {code?: string; action?: string}; request_id?: string } | null;
        throw new Error([failure?.detail || `模型回答失败（HTTP ${response.status}）`, failure?.error?.action,
          failure?.error?.code ? `错误代码：${failure.error.code}` : '',
          failure?.request_id ? `排查编号：${failure.request_id}` : ''].filter(Boolean).join(' '));
      }

      const answer = normalizeAnswerForDisplay((await response.json()) as CopilotAnswer);
      setServiceState("available");
      updateActiveConversationMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content: answer.customer_reply,
          answer,
        },
      ], requestConversationId);
    } catch (error) {
      const isTimeout = error instanceof DOMException && error.name === "AbortError";
      const isNetworkFailure = error instanceof TypeError;
      setServiceState(isNetworkFailure && !isTimeout ? "offline" : "available");
      updateActiveConversationMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content: isTimeout
            ? "本次等待已超时。若附件已解析成功，可在当前临时会话中缩小问题范围后重试；上传未完成时请重新上传。"
            : isNetworkFailure
            ? "本地模型服务暂时没有连接。请确认电脑已开机，并启动本地知识库服务后再重试。"
            : `本次请求未完成：${error instanceof Error ? error.message : "未知错误"}`,
        },
      ], requestConversationId);
    } finally {
      window.clearTimeout(requestTimer);
      setIsSending(false);
      setRequestStage("");
      window.setTimeout(() => textAreaRef.current?.focus(), 0);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendQuestion(draft);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendQuestion(draft);
    }
  }

  function startNewConversation() {
    if (isSending) return;
    const conversation = makeConversation();
    setConversations((current) => [conversation, ...current].slice(0, 24));
    setActiveConversationId(conversation.id);
    setDraft("");
    setAttachedImage(null);
    setPendingAttachments([]);
    setSidebarOpen(false);
    void saveBrowserConversation(conversation, historyScope);
    window.setTimeout(() => textAreaRef.current?.focus(), 0);
  }

  function openConversation(conversation: LocalConversation) {
    if (isSending) return;
    setActiveConversationId(conversation.id);
    setDraft("");
    setAttachedImage(null);
    setPendingAttachments([]);
    setSidebarOpen(false);
    window.setTimeout(() => textAreaRef.current?.focus(), 0);
  }

  function removeConversation(conversationId: string) {
    if (isSending) return;
    const remaining = conversations.filter((conversation) => conversation.id !== conversationId);
    documentSessionByConversation.delete(conversationId);
    void deleteBrowserConversation(conversationId, historyScope);
    if (conversationId !== activeConversationId) {
      setConversations(remaining);
      return;
    }
    if (remaining.length > 0) {
      setConversations(remaining);
      setActiveConversationId(remaining[0].id);
      return;
    }
    const replacement = makeConversation();
    setConversations([replacement]);
    setActiveConversationId(replacement.id);
    void saveBrowserConversation(replacement, historyScope);
  }

  async function clearAllConversations() {
    if (isSending || !window.confirm("清除本浏览器中的全部对话及对应服务器任务记忆？此操作无法恢复。")) return;
    try {
      for (const conversation of conversations) {
        const result = await authenticatedFetch(`${publicModelApiBase}/api/copilot/memory/${encodeURIComponent(conversation.id)}`, { method: "DELETE" });
        if (!result.ok) throw new Error("memory_delete_failed");
      }
    } catch {
      setCoverageNotice("服务器任务记忆未能全部清除。为保留重试入口，暂未删除浏览器聊天记录，请连接恢复后重试。");
      return;
    }
    setUseTaskMemory(false);
    const replacement = makeConversation();
    documentSessionByConversation.clear();
    setConversations([replacement]);
    setActiveConversationId(replacement.id);
    setDraft("");
    setAttachedImage(null);
    setPendingAttachments([]);
    void clearBrowserConversations(historyScope).then(() => saveBrowserConversation(replacement, historyScope));
  }

  const statusLabel = {
    checking: "正在连接本地服务",
    available: "本地模型与知识库在线",
    offline: "本地服务未连接",
  }[serviceState];

  return (
    <main className="copilot-shell">
      <aside className={`conversation-sidebar ${sidebarOpen ? "open" : ""}`} aria-label="本地对话记录">
        <nav className="agent-switcher" aria-label="选择助手">
          <a className="agent-choice active" href="#top" aria-current="page">
            <span className="agent-choice-mark">材</span>
            <span><b>建材销售助手</b><small>资料检索与销售支持</small></span>
          </a>
        </nav>
        <div className="sidebar-heading">
          <span>本地对话</span>
          <small>仅此浏览器</small>
        </div>
        <button className="new-conversation-button" type="button" onClick={startNewConversation} disabled={isSending}>
          <span aria-hidden="true">＋</span>
          新建对话
        </button>
        <p className="sidebar-privacy-note">记录保存在当前浏览器，不写入模型后端、腾讯云或产品知识库。</p>
        <nav className="conversation-list" aria-label="历史对话">
          {conversations.map((conversation) => (
            <div className={`conversation-list-item ${conversation.id === activeConversationId ? "active" : ""}`} key={conversation.id}>
              <button type="button" onClick={() => openConversation(conversation)} disabled={isSending}>
                <span>{conversation.title}</span>
                <small>{formatConversationTime(conversation.updatedAt)}</small>
              </button>
              <button
                className="delete-conversation-button"
                type="button"
                onClick={() => removeConversation(conversation.id)}
                disabled={isSending}
                aria-label={`删除对话：${conversation.title}`}
                title="删除此对话"
              >
                ×
              </button>
            </div>
          ))}
        </nav>
        <button className="clear-history-button" type="button" onClick={clearAllConversations} disabled={isSending}>
          清空本机历史
        </button>
      </aside>
      {sidebarOpen && <button className="sidebar-scrim" type="button" aria-label="关闭对话侧边栏" onClick={() => setSidebarOpen(false)} />}
      <div
        className={`chat-workspace sales-workspace-dropzone${isFileDragging ? " is-dragging" : ""}`}
        onDragEnter={handleFileDragEnter}
        onDragOver={handleFileDragOver}
        onDragLeave={handleFileDragLeave}
        onDrop={handleFileDrop}
        onPaste={handleClipboardPaste}
      >
      <header className="topbar">
        <button className="mobile-sidebar-toggle" type="button" aria-label="打开历史对话" onClick={() => setSidebarOpen(true)}>
          ☰
        </button>
        <a className="brand" href="#top" aria-label="企业小助手首页">
          <span className="brand-symbol" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span>企业<b>小助手</b></span>
        </a>

        <div className={`service-status ${serviceState}`} title="仅在本地电脑开机且服务已启动时可回答问题">
          <span className="status-dot" aria-hidden="true" />
          {statusLabel}
        </div>

        <div className="topbar-actions">
          {authPrincipal ? (
            <>
              <button className="access-button" type="button" onClick={() => authPrincipal.role === "admin" ? void openAdminPanel() : undefined}>
                {authPrincipal.display_name} · {authPrincipal.can_access_internal ? "内部" : "公开"}
              </button>
              <button className="reset-button" type="button" onClick={() => void logout()}>退出</button>
            </>
          ) : (
            <button className="access-button" type="button" onClick={() => setAuthDialogOpen(true)}>
              {bootstrapRequired ? "初始化管理员" : "内部登录"}
            </button>
          )}
          <button className="reset-button" type="button" onClick={startNewConversation} disabled={isSending}>
            新建对话
          </button>
        </div>
      </header>

      <section className="conversation" id="top" aria-live="polite">
        <div className="conversation-column">
          {messages.length === 1 && (
            <section className="welcome-panel">
              <span className="welcome-mark">✦</span>
              <h1>关于建材，有什么想要了解的？</h1>
              <p>可询问产品、施工安装、节点图集或项目案例。回答会附上检索到的原始资料图片。</p>
              <div className="starter-list">
                {starterQuestions.map((question) => (
                  <button key={question} type="button" onClick={() => void sendQuestion(question)}>
                    <span>{question}</span>
                    <b aria-hidden="true">↗</b>
                  </button>
                ))}
              </div>
            </section>
          )}

          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              {message.role === "assistant" && <div className="avatar assistant-avatar">材</div>}
              <div className="message-body">
                {message.role === "assistant" && message.answer?.meta?.model_used && (
                  <div className="answer-meta">
                    {answerSourceLabel(message.answer)}{message.answer.meta.latency_ms ? ` · ${Math.round(message.answer.meta.latency_ms / 100) / 10}s` : ""}
                  </div>
                )}
                {message.role === "assistant" ? (
                  <ReplyParagraphs text={message.content} />
                ) : (
                  <>
                    {message.image && (
                      <img className="customer-upload-preview" src={message.image.dataUrl} alt={`客户上传：${message.image.name}`} />
                    )}
                    {message.attachmentNames && message.attachmentNames.length > 0 && (
                      <small className="customer-attachment-names">附件：{message.attachmentNames.join("、")}</small>
                    )}
                    <p>{message.content}</p>
                  </>
                )}

                {message.answer && (
                  <AnswerDetails
                    answer={message.answer}
                    onOpenImage={setActiveImage}
                    onUseCandidate={(match) => {
                      setDraft(`我选择外观候选“${match.customer_title}”。请按对应真岩产品资料继续说明。`);
                      window.setTimeout(() => textAreaRef.current?.focus(), 0);
                    }}
                  />
                )}
              </div>
              {message.role === "user" && <div className="avatar user-avatar">你</div>}
            </article>
          ))}

          {isSending && (
            <article className="message assistant">
              <div className="avatar assistant-avatar">材</div>
              <span role="status">{requestStage}</span>
              <div className="thinking" aria-label="正在检索资料并生成回答">
                <i />
                <i />
                <i />
              </div>
            </article>
          )}
          <div ref={endOfMessagesRef} />
          {coverageNotice && <p role="status">{coverageNotice}</p>}
        </div>
      </section>

      <div className="composer-wrap">
        <form className="composer" onSubmit={handleSubmit}>
          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            multiple
            accept=".pdf,.docx,.xlsx,.xls,.csv,.txt,.html,.htm,.xml,.zip,image/jpeg,image/png,image/webp,image/gif,image/bmp,image/tiff"
            onChange={handleAttachmentSelection}
          />
          <div className="composer-input-area" title="可点击＋、拖入文件，或从剪贴板粘贴文件和图片">
            {pendingAttachments.map((attachment) => (
              <div className="pending-image" key={attachment.id} aria-label={`待发送附件：${attachment.file.name}`}>
                {attachment.previewUrl ? <img src={attachment.previewUrl} alt="附件预览" /> : <strong>文件</strong>}
                <span>{attachment.file.name}</span>
                <button type="button" onClick={() => removePendingAttachment(attachment.id)} aria-label="移除附件">×</button>
              </div>
            ))}
            {attachedImage && pendingAttachments.length === 0 && (
              <div className="pending-image" aria-label={`待发送图片：${attachedImage.name}`}>
                <img src={attachedImage.dataUrl} alt="待发送图片预览" />
                <span>{attachedImage.name}</span>
                <button type="button" onClick={() => setAttachedImage(null)} aria-label="移除图片">×</button>
              </div>
            )}
            <div className="composer-row">
              <button
                className="image-upload-button"
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isSending}
                aria-label="上传文件或图片"
                title="点击选择文件，也可直接拖入或粘贴"
              >
                <span aria-hidden="true">＋</span>
              </button>
              <textarea
                ref={textAreaRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入问题；文件可点击＋、拖入或粘贴…"
                rows={1}
                aria-label="输入问题"
              />
            </div>
            {activeConversation?.documentSessionId && activeConversation.attachmentNames && activeConversation.attachmentNames.length > 0 && (
              <div className="active-document-session">
                <span>当前对话已关联附件</span>
                <small>{activeConversation.attachmentNames.join("、")}</small>
              </div>
            )}
            <label className="online-search-toggle">
              <input type="checkbox" checked={useTaskMemory} disabled={isSending}
                onChange={(event) => setUseTaskMemory(event.target.checked)} />
              <span>任务记忆</span>
              <small>开启后将本会话问答与条件保存在本地服务器，7天后失效并在后续访问时清理；不作为技术证据。不同项目请新建对话。</small>
            </label>
            <button type="button" disabled={isSending || !activeConversationId} onClick={async () => {
              if (!activeConversationId) return;
              try {
                const result = await authenticatedFetch(`${publicModelApiBase}/api/copilot/memory/${encodeURIComponent(activeConversationId)}`, { method: "DELETE" });
                if (!result.ok) throw new Error("delete_failed");
                setUseTaskMemory(false);
                setCoverageNotice("已清除此会话的服务器任务记忆。聊天记录保留，近期消息仍可用于当前对话；如需完全重新开始，请新建对话。");
              } catch {
                setCoverageNotice("任务记忆清除失败，请检查连接后重试。");
              }
            }}>清除此会话任务记忆</button>
            <label className="online-search-toggle">
              <input
                type="checkbox"
                checked={useOnlineSearch}
                onChange={(event) => setUseOnlineSearch(event.target.checked)}
                disabled={isSending}
              />
              <span>联网补充</span>
              <small>完整发送当前问题；本地资料、图片和历史对话不会上传。具体项目名称会自动检索公开资料。</small>
            </label>
          </div>
          <button type="submit" disabled={(!draft.trim() && pendingAttachments.length === 0) || isSending} aria-label="发送问题">
            <span aria-hidden="true">↑</span>
          </button>
        </form>
        <p className="composer-note">上传图片仅发送至本机模型并在处理后删除；技术结论仍以本地资料页与项目条件为准。</p>
      </div>
      {isFileDragging && (
        <div className="sales-workspace-drop-overlay" aria-hidden="true">
          <div><b>松开以上传文件</b><span>支持 PDF、Word、Excel、文本、网页文件和常见图片</span></div>
        </div>
      )}

      {activeImage && (
        <div className="image-lightbox" role="dialog" aria-modal="true" aria-label="查看资料原图" onClick={() => setActiveImage(null)}>
          <div className="lightbox-card" onClick={(event) => event.stopPropagation()}>
            <div className="lightbox-header">
              <div>
                <strong>{activeImage.customer_title}</strong>
                <span>{activeImage.citation.document_name} · {visualLocation(activeImage)}</span>
              </div>
              <button type="button" onClick={() => setActiveImage(null)} aria-label="关闭原图">×</button>
            </div>
            <img src={sourceUrl(activeImage)} alt={activeImage.customer_title} />
          </div>
        </div>
      )}

      {authDialogOpen && (
        <div className="access-modal" role="dialog" aria-modal="true" aria-label={bootstrapRequired ? "初始化管理员" : "内部登录"}>
          <form className="access-card" onSubmit={submitAuthentication}>
            <div className="access-card-heading">
              <div><strong>{bootstrapRequired ? "初始化管理员" : "内部人员登录"}</strong><small>匿名访客始终只能查询公开资料</small></div>
              <button type="button" onClick={() => setAuthDialogOpen(false)} aria-label="关闭">×</button>
            </div>
            {bootstrapRequired && (
              <>
                <label>一次性初始化令牌<input value={authForm.setupToken} onChange={(event) => setAuthForm({ ...authForm, setupToken: event.target.value })} minLength={20} autoComplete="off" spellCheck={false} required /></label>
                <p className="access-hint">令牌保存在本机 runtime/admin_bootstrap_token.txt，成功初始化后自动删除。</p>
                <label>显示名称<input value={authForm.displayName} onChange={(event) => setAuthForm({ ...authForm, displayName: event.target.value })} placeholder="管理员姓名" /></label>
              </>
            )}
            <label>用户名<input value={authForm.username} onChange={(event) => setAuthForm({ ...authForm, username: event.target.value })} autoComplete="username" required /></label>
            <label>密码<input type="password" value={authForm.password} onChange={(event) => setAuthForm({ ...authForm, password: event.target.value })} autoComplete={bootstrapRequired ? "new-password" : "current-password"} minLength={bootstrapRequired ? 10 : undefined} required /></label>
            {authError && <p className="access-error">{authError}</p>}
            <button className="access-primary" type="submit" disabled={authBusy}>{authBusy ? "处理中…" : bootstrapRequired ? "创建管理员" : "登录"}</button>
          </form>
        </div>
      )}

      {adminPanelOpen && authPrincipal?.role === "admin" && (
        <div className="access-modal" role="dialog" aria-modal="true" aria-label="内部权限管理">
          <section className="access-card admin-access-card">
            <div className="access-card-heading">
              <div><strong>内部权限管理</strong><small>当前内部知识库为空；现有资料均为公开资料</small></div>
              <button type="button" onClick={() => setAdminPanelOpen(false)} aria-label="关闭">×</button>
            </div>
            <form className="new-user-form" onSubmit={createManagedUser}>
              <input value={newUserForm.username} onChange={(event) => setNewUserForm({ ...newUserForm, username: event.target.value })} placeholder="用户名" required />
              <input value={newUserForm.displayName} onChange={(event) => setNewUserForm({ ...newUserForm, displayName: event.target.value })} placeholder="姓名" />
              <input type="password" minLength={10} value={newUserForm.password} onChange={(event) => setNewUserForm({ ...newUserForm, password: event.target.value })} placeholder="初始密码（至少10位）" required />
              <label className="inline-permission"><input type="checkbox" checked={newUserForm.canAccessInternal} onChange={(event) => setNewUserForm({ ...newUserForm, canAccessInternal: event.target.checked })} />允许内部资料</label>
              <button className="access-primary" type="submit" disabled={authBusy}>添加人员</button>
            </form>
            {authError && <p className="access-error">{authError}</p>}
            <div className="managed-user-list">
              {managedUsers.map((user) => (
                <div className="managed-user-row" key={user.user_id}>
                  <div><strong>{user.display_name}</strong><small>{user.username} · {user.role === "admin" ? "管理员" : "成员"}</small></div>
                  <label><input type="checkbox" checked={Boolean(user.can_access_internal)} onChange={(event) => void updateManagedUser(user, { can_access_internal: event.target.checked })} />内部权限</label>
                  <label><input type="checkbox" checked={Boolean(user.active)} onChange={(event) => void updateManagedUser(user, { active: event.target.checked })} />启用</label>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
      </div>
    </main>
  );
}

function AnswerDetails({
  answer,
  onOpenImage,
  onUseCandidate,
}: {
  answer: CopilotAnswer;
  onOpenImage: (asset: VisualAsset) => void;
  onUseCandidate: (asset: VisualAsset) => void;
}) {
  const supportingResults = answer.retrieval?.supporting_results ?? [];
  const resultCount = answer.retrieval?.result_count ?? answer.citations.length;
  const visualCount = answer.retrieval?.visual_count ?? answer.visual_assets.length;
  const requestedVisualScope = answer.meta?.visual_scope?.trim() || null;
  const hasExplicitWantsVisualFlag = typeof answer.meta?.wants_visuals === "boolean";
  const explicitVisualRequest = Boolean(
    answer.meta?.wants_visuals === true
      || answer.meta?.explicit_visual_request === true
      || (!hasExplicitWantsVisualFlag
        && requestedVisualScope
        && !["none", "auto", "not_requested", "mixed"].includes(requestedVisualScope.toLowerCase())),
  );
  const inlineVisualAssets = explicitVisualRequest
    ? Array.from(new Map(answer.visual_assets.map((asset) => [asset.asset_id, asset])).values())
    : [];
  const hasMaterials = supportingResults.length > 0
    || answer.citations.length > 0
    || (!explicitVisualRequest && answer.visual_assets.length > 0);
  const imageIdentity = answer.meta?.image_identity;
  const normalizeForDisplayDedupe = (value: string) => value.replace(/[\s，。；：、,.!?！？（）()《》【】'"“”‘’\-—]/g, "").toLowerCase();
  const normalizedReply = normalizeForDisplayDedupe(answer.customer_reply);
  const cleanedNextAction = cleanDisplayText(answer.next_action ?? "");
  const normalizedNextAction = normalizeForDisplayDedupe(cleanedNextAction);
  const showNextAction = Boolean(
    cleanedNextAction
      && normalizedNextAction
      && !normalizedReply.includes(normalizedNextAction)
      && !normalizedNextAction.includes(normalizedReply),
  );
  const visibleImageObservations = answer.image_observations.filter((observation) => {
    const normalizedObservation = normalizeForDisplayDedupe(observation);
    return normalizedObservation.length > 0 && !normalizedReply.includes(normalizedObservation);
  });
  const onlineSources = answer.online_sources ?? [];
  const onlineSearchStatus = answer.meta?.online_search;
  const attachmentIndexOnly = answer.retrieval?.attachment_status === "parsed"
    && answer.retrieval?.document_index_only === true;
  const evidenceLevelLabel = (source: OnlineSource) => {
    if (source.evidence_level === "verified_page_content") return "已核验网页正文";
    if (source.evidence_level === "official_search_excerpt") return "权威网站搜索摘要";
    if (source.evidence_level === "unverified_search_excerpt") return "搜索摘要，正文未核验";
    return "仅作为搜索线索";
  };

  return (
    <div className="answer-details">
      {!!answer.meta?.execution?.errors?.length && (
        <section role="status" aria-label="处理状态">
          <p>处理状态：{answer.meta.execution.state === "failed" ? "未完成" : "部分能力受限"}</p>
          {answer.meta.execution.errors.slice(0, 4).map((error, index) => (
            <p key={`${error.code}-${index}`}>{error.message} {error.action}（{error.code}）</p>
          ))}
          <small>排查编号：{answer.meta.execution.request_id}</small>
        </section>
      )}
      {!!answer.retrieval?.incomplete_visual_documents?.length && (
        <p role="status">以下附件视觉页面尚未全部处理，回答仅基于已解析部分：{answer.retrieval.incomplete_visual_documents.join("、")}</p>
      )}
      {imageIdentity && imageIdentity.status !== "not_provided" && (
        <section className={`image-identity ${imageIdentity.status}`}>
          <div className="detail-heading">
            <span>图片产品身份核验</span>
            <small>{imageIdentity.status === "appearance_candidate" ? "仅外观候选，尚未确认产品" : "未匹配到可信外观候选"}</small>
          </div>
          <p>{imageIdentity.message}</p>
          {imageIdentity.matches.length > 0 && (
            <div className="identity-match-grid">
              {imageIdentity.matches.map((match) => (
                <div className="identity-match" key={match.asset_id}>
                  <button className="identity-preview" type="button" onClick={() => onOpenImage(match)}>
                    <img src={sourceUrl(match)} alt={match.customer_title} loading="lazy" />
                  </button>
                  <span>{match.customer_title}</span>
                  <small>外观相似度参考 {Math.round(match.similarity * 100)}%</small>
                  <button className="identity-select" type="button" onClick={() => onUseCandidate(match)}>按此候选继续咨询</button>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
      {visibleImageObservations.length > 0 && (
        <section className="image-observations">
          <div className="detail-heading">
            <span>图片中可见的信息</span>
            <small>仅为图片观察，需结合资料与现场条件核验</small>
          </div>
          <ul>
            {visibleImageObservations.map((observation, index) => <li key={`${observation}-${index}`}>{observation}</li>)}
          </ul>
        </section>
      )}
      {inlineVisualAssets.length > 0 && (
        <InlineVisualGallery
          assets={inlineVisualAssets}
          onOpenImage={onOpenImage}
        />
      )}
      {answer.key_points.length > 0 && (
        <ul className="key-points">
          {answer.key_points.map((point, index) => <li key={`${point}-${index}`}>{point}</li>)}
        </ul>
      )}

      {onlineSources.length > 0 && (
        <section className="online-sources">
          <div className="detail-heading">
            <span>联网参考来源</span>
            <small>
              {onlineSearchStatus?.cache_hit ? "已复用联网缓存" : "公开网页信息，仅作补充核验"}
              {typeof onlineSearchStatus?.quota?.remaining === "number" ? ` · 本地今日搜索预算剩余 ${onlineSearchStatus.quota.remaining} 次（非平台余额）` : ""}
            </small>
          </div>
          <ol>
            {onlineSources.map((source) => (
              <li key={source.source_id}>
                <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a>
                <small>
                  {[source.website, source.date, evidenceLevelLabel(source)].filter(Boolean).join(" · ")}
                </small>
                {source.excerpt && <p>{source.excerpt}</p>}
              </li>
            ))}
          </ol>
        </section>
      )}

      {onlineSearchStatus && !["not_requested", "ok"].includes(onlineSearchStatus.status ?? "not_requested") && (
        <p className="online-search-status">
          联网补充暂未返回可用结果：
          {onlineSearchStatus.status === "not_configured"
            ? "本机尚未配置百度搜索密钥。"
            : onlineSearchStatus.status === "quota_exhausted"
              ? "今日业务搜索额度已用完，系统已停止继续调用联网API。"
              : "请稍后重试或检查本机网络与百度接口配置。"}
        </p>
      )}

      {hasMaterials && (
        <details className="retrieved-materials">
          <summary>
            <span>
              {attachmentIndexOnly
                ? `已解析 ${answer.retrieval?.parsed_document_count || 1} 份附件，当前仅定位到结构索引`
                : `已找到 ${resultCount || "相关"} 条资料结果`}
            </span>
            <small>
              {explicitVisualRequest
                ? "展开查看文字证据与资料出处"
                : visualCount > 0
                  ? `含 ${visualCount} 张相关原图，展开查看`
                  : "展开查看资料出处"}
            </small>
          </summary>

          <div className="material-results">
            {supportingResults.length > 0 && (
              <ol className="retrieval-result-list">
                {supportingResults.map((item) => (
                  <li className="retrieval-result" key={item.result_id}>
                    <span className="result-id">{item.result_id}</span>
                    <div>
                      {item.source_url ? (
                        <a href={item.source_url} target="_blank" rel="noreferrer"><b>{item.document_name || "官方网站资料"}</b></a>
                      ) : <b>{item.document_name || "本地资料"}</b>}
                      <span>
                        {item.source_page ? `第 ${item.source_page} 页` : "资料页码待确认"}
                        {item.section_heading ? ` · ${item.section_heading}` : ""}
                      </span>
                      <p>{item.excerpt}</p>
                    </div>
                  </li>
                ))}
              </ol>
            )}

            {!explicitVisualRequest && answer.visual_assets.length > 0 && (
              <section className="visual-section">
                <div className="detail-heading">
                  <span>相关原始资料图片</span>
                  <small>点击图片查看完整原图</small>
                </div>
                <div className="visual-grid">
                  {answer.visual_assets.map((asset) => (
                    <button className="visual-card" type="button" key={asset.asset_id} onClick={() => onOpenImage(asset)}>
                      <span className="image-frame">
                        <img src={sourceUrl(asset)} alt={asset.customer_title} loading="lazy" />
                        <em>查看原图 ↗</em>
                      </span>
                      <strong>{asset.customer_title}</strong>
                      <small>{asset.citation.document_name} · {visualLocation(asset)}</small>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {supportingResults.length === 0 && answer.citations.length > 0 && (
              <ol className="sources">
                {answer.citations.map((citation, index) => (
                  <li key={`${citation.evidence_id}-${citation.document_name}-${index}`}>
                    {citation.source_url ? (
                      <a href={citation.source_url} target="_blank" rel="noreferrer"><b>{citation.document_name}</b></a>
                    ) : <b>{citation.document_name}</b>}
                    <span>
                      {citation.source_page ? `第 ${citation.source_page} 页` : ""}
                      {citation.sheet_name ? `${citation.source_page ? " · " : ""}Sheet：${citation.sheet_name}` : ""}
                      {citation.source_range ? ` · ${citation.source_range}` : ""}
                      {citation.section_heading ? ` · ${citation.section_heading}` : ""}
                      {!citation.source_page && !citation.sheet_name && !citation.source_range && !citation.section_heading ? "来源位置待确认" : ""}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </details>
      )}

      {!answer.answerable && answer.missing_information.length > 0 && (
        <div className="info-note">
          <b>为了进一步细化方案，可补充：</b>
          <span>{cleanDisplayText(answer.missing_information.join("、"))}</span>
        </div>
      )}

      {showNextAction && <p className="next-action">{cleanedNextAction}</p>}
    </div>
  );
}
