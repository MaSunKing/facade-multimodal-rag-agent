"use client";
/* eslint-disable @next/next/no-img-element -- source images are original, local RAG crops with dynamic dimensions. */

import { ChangeEvent, FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
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

type SourceCitation = {
  evidence_id: string;
  document_name: string;
  source_page: number | null;
  section_heading?: string | null;
  sheet_name?: string | null;
  source_range?: string | null;
};

type VisualAsset = {
  asset_id: string;
  customer_title: string;
  visual_endpoint: string;
  citation: {
    document_name: string;
    source_page: number | null;
    sheet_name?: string | null;
    source_range?: string | null;
  };
};

type SupportingResult = {
  result_id: string;
  excerpt: string;
  document_name?: string | null;
  source_page?: number | null;
  section_heading?: string | null;
};

type RetrievalSummary = {
  result_count: number;
  supporting_results: SupportingResult[];
  visual_count: number;
  strategy?: string;
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
    model_used?: boolean;
    latency_ms?: number;
    image_identity?: ImageIdentity;
    online_search?: {
      status?: "not_requested" | "not_configured" | "ok" | "failed";
      message?: string;
    };
  };
};

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
  return "附件原图";
}

function ReplyParagraphs({ text }: { text: string }) {
  return (
    <div className="reply-copy">
      {text
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
  const [conversations, setConversations] = useState<LocalConversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [historyReady, setHistoryReady] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [activeImage, setActiveImage] = useState<VisualAsset | null>(null);
  const [attachedImage, setAttachedImage] = useState<AttachedImage | null>(null);
  const [pendingAttachments, setPendingAttachments] = useState<PendingAttachment[]>([]);
  const [useOnlineSearch, setUseOnlineSearch] = useState(false);
  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId) ?? null;
  const messages = activeConversation?.messages.length ? activeConversation.messages : defaultConversationMessages;
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);
  const textAreaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
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
    let cancelled = false;
    void loadBrowserConversations<ChatMessage>().then((savedConversations) => {
      if (cancelled) return;
      let initialConversations = savedConversations;
      if (initialConversations.length === 0) {
        const legacyMessages = loadLegacyConversation();
        const initial = makeConversation(legacyMessages);
        initialConversations = [initial];
        void saveBrowserConversation(initial);
        // Move the previous single-session implementation forward once. It
        // remains local to this browser and is then removed from sessionStorage.
        window.sessionStorage.removeItem(legacyConversationSessionKey);
      }
      const active = initialConversations[0];
      setConversations(initialConversations);
      setActiveConversationId(active.id);
      setHistoryReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

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
      void saveBrowserConversation(updated);
      ordered.slice(24).forEach((conversation) => void deleteBrowserConversation(conversation.id));
      return next;
    });
  }

  async function handleAttachmentSelection(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (selected.length === 0) return;
    if (pendingAttachments.length + selected.length > 4) {
      window.alert("一次最多上传 4 份文件。请先移除部分附件。");
      return;
    }
    if (selected.some((file) => file.size > 25 * 1024 * 1024)) {
      window.alert("单个文件请控制在 25 MB 以内。");
      return;
    }
    const additions = selected.map((file) => ({
      id: makeId(),
      file,
      previewUrl: file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
    }));
    setPendingAttachments((current) => [...current, ...additions]);
    const firstVisual = selected.find((file) => /^image\/(jpeg|png|webp)$/.test(file.type));
    if (firstVisual && !attachedImage) {
      try {
        setAttachedImage({ name: firstVisual.name, dataUrl: await readImageAsDataUrl(firstVisual) });
      } catch {
        // The generic parser can still process the attachment without preview.
      }
    }
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
    const customerQuestion = cleanQuestion || "请识别图片中直接可见的外墙建材、构造或施工信息，并结合本地资料说明可核验内容。";

    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: cleanQuestion || "请分析这张图片。",
      image: imageForRequest ?? undefined,
      attachmentNames: attachmentsForRequest.map((item) => item.file.name),
    };
    updateActiveConversationMessages((current) => [...current, userMessage], requestConversationId);
    setDraft("");
    setAttachedImage(null);
    attachmentsForRequest.forEach((item) => item.previewUrl && URL.revokeObjectURL(item.previewUrl));
    setPendingAttachments([]);
    setIsSending(true);

    try {
      let documentSessionId: string | null = null;
      if (attachmentsForRequest.length > 0) {
        const form = new FormData();
        attachmentsForRequest.forEach((item) => form.append("files", item.file));
        const existingSession = documentSessionByConversation.get(requestConversationId);
        if (existingSession) form.append("session_id", existingSession);
        const uploadResponse = await fetch(`${publicModelApiBase}/api/copilot/documents`, {
          method: "POST",
          body: form,
        });
        if (!uploadResponse.ok) throw new Error(`UPLOAD HTTP ${uploadResponse.status}`);
        const uploaded = (await uploadResponse.json()) as { session_id: string };
        documentSessionId = uploaded.session_id;
        documentSessionByConversation.set(requestConversationId, uploaded.session_id);
      } else {
        documentSessionId = documentSessionByConversation.get(requestConversationId) ?? null;
      }
      const response = await fetch(`${publicModelApiBase}/api/copilot/answer`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          customer_question: customerQuestion,
          image_data_url: imageForRequest?.dataUrl ?? null,
          document_session_id: documentSessionId,
          project_context: { customer_type: "客户/销售" },
          conversation_context: buildConversationContext(messages),
          use_online_search: useOnlineSearch,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const answer = (await response.json()) as CopilotAnswer;
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
    } catch {
      setServiceState("offline");
      updateActiveConversationMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content:
            "本地模型服务暂时没有连接。前端页面仍可使用；请确认电脑已开机，并启动本地知识库服务后再重试。",
        },
      ], requestConversationId);
    } finally {
      setIsSending(false);
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
    void saveBrowserConversation(conversation);
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
    void deleteBrowserConversation(conversationId);
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
    void saveBrowserConversation(replacement);
  }

  function clearAllConversations() {
    if (isSending || !window.confirm("清除本浏览器中的全部对话记录？此操作无法恢复。")) return;
    const replacement = makeConversation();
    documentSessionByConversation.clear();
    setConversations([replacement]);
    setActiveConversationId(replacement.id);
    setDraft("");
    setAttachedImage(null);
    setPendingAttachments([]);
    void clearBrowserConversations().then(() => saveBrowserConversation(replacement));
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
      <div className="chat-workspace">
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

        <button className="reset-button" type="button" onClick={startNewConversation} disabled={isSending}>
          新建对话
        </button>
      </header>

      <section className="conversation" id="top" aria-live="polite">
        <div className="conversation-column">
          {messages.length === 1 && (
            <section className="welcome-panel">
              <span className="welcome-mark">✦</span>
              <h1>有什么建材资料需要查？</h1>
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
                    本地 RAG 回答{message.answer.meta.latency_ms ? ` · ${Math.round(message.answer.meta.latency_ms / 100) / 10}s` : ""}
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
                      setDraft(`我选择外观候选“${match.customer_title}”。请按对应企业产品产品资料继续说明。`);
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
              <div className="thinking" aria-label="正在检索资料并生成回答">
                <i />
                <i />
                <i />
              </div>
            </article>
          )}
          <div ref={endOfMessagesRef} />
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
          <div className="composer-input-area">
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
                aria-label="上传项目或施工图片"
                title="上传项目、施工、节点或产品图片"
              >
                <span aria-hidden="true">＋</span>
              </button>
              <textarea
                ref={textAreaRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入问题，或上传项目、施工、节点图片…"
                rows={1}
                aria-label="输入问题"
              />
            </div>
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

      {activeImage && (
        <div className="image-lightbox" role="dialog" aria-modal="true" aria-label="查看资料原图" onClick={() => setActiveImage(null)}>
          <div className="lightbox-card" onClick={(event) => event.stopPropagation()}>
            <div className="lightbox-header">
              <div>
                <strong>{activeImage.customer_title}</strong>
                <span>{activeImage.citation.document_name} · 第 {activeImage.citation.source_page} 页</span>
              </div>
              <button type="button" onClick={() => setActiveImage(null)} aria-label="关闭原图">×</button>
            </div>
            <img src={sourceUrl(activeImage)} alt={activeImage.customer_title} />
          </div>
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
  const hasMaterials = supportingResults.length > 0 || answer.visual_assets.length > 0 || answer.citations.length > 0;
  const imageIdentity = answer.meta?.image_identity;
  const onlineSources = answer.online_sources ?? [];
  const onlineSearchStatus = answer.meta?.online_search;

  return (
    <div className="answer-details">
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
      {answer.image_observations.length > 0 && (
        <section className="image-observations">
          <div className="detail-heading">
            <span>图片中可见的信息</span>
            <small>仅为图片观察，需结合资料与现场条件核验</small>
          </div>
          <ul>
            {answer.image_observations.map((observation, index) => <li key={`${observation}-${index}`}>{observation}</li>)}
          </ul>
        </section>
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
            <small>公开网页信息，仅作补充核验</small>
          </div>
          <ol>
            {onlineSources.map((source) => (
              <li key={source.source_id}>
                <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a>
                {(source.website || source.date) && <small>{[source.website, source.date].filter(Boolean).join(" · ")}</small>}
                {source.excerpt && <p>{source.excerpt}</p>}
              </li>
            ))}
          </ol>
        </section>
      )}

      {onlineSearchStatus && !["not_requested", "ok"].includes(onlineSearchStatus.status ?? "not_requested") && (
        <p className="online-search-status">
          联网补充暂未返回可用结果：{onlineSearchStatus.status === "not_configured" ? "本机尚未配置百度搜索密钥。" : "请稍后重试或检查本机网络与百度接口配置。"}
        </p>
      )}

      {hasMaterials && (
        <details className="retrieved-materials">
          <summary>
            <span>已找到 {resultCount || "相关"} 条资料结果</span>
            <small>{visualCount > 0 ? `含 ${visualCount} 张相关原图，展开查看` : "展开查看资料出处"}</small>
          </summary>

          <div className="material-results">
            {supportingResults.length > 0 && (
              <ol className="retrieval-result-list">
                {supportingResults.map((item) => (
                  <li className="retrieval-result" key={item.result_id}>
                    <span className="result-id">{item.result_id}</span>
                    <div>
                      <b>{item.document_name || "本地资料"}</b>
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

            {answer.visual_assets.length > 0 && (
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
                    <b>{citation.document_name}</b>
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
          <span>{answer.missing_information.join("、")}</span>
        </div>
      )}

      {answer.next_action && <p className="next-action">{answer.next_action}</p>}
    </div>
  );
}
