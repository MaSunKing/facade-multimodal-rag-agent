"use client";

import { useEffect, useRef, useState } from "react";
import { compactDetail, importantStage, recorded, stageTokens, normalizeTraceId, preferredTraceId } from "./trace-display";

type Row = { request_id: string; status: string; started_at: string; live?: boolean;
  total_latency_ms?: number; error_type?: string; execution?: Record<string, unknown>;
  summary?: Record<string, unknown>; trace_unavailable?: boolean; visual_label_versions?: Record<string, unknown>[] };
type Props = { baseUrl: string; publicView?: boolean; initialRequestId?: string; activeRequestId?: string; authenticatedFetch: (input: string, init?: RequestInit) => Promise<Response> };
const labels: Record<string, string> = {
  request: "入场/排队", plan_request: "规划", guard_tools: "权限与计划检查",
  customer_documents: "客户附件", document_summary: "分文件概括", company_rag: "企业检索", public_web_search: "联网搜索",
  visual_inspection: "视觉处理", assess_coverage: "覆盖检查", rerank_evidence: "证据排序",
  compose_evidence: "证据组装", support_selection: "片段筛选与去留原因", evidence_evaluation: "支持审核", generate_answer: "回答生成",
  validate_answer: "回答校验", assess_retry: "恢复判断", apply_retry_plan: "应用新计划",
  replan_request: "缺口恢复规划",
  finalize_response: "返回结果",
  evidence_expansion: "正文扩展与预算回退", support_budget: "完整输入预算检查",
  context_preflight: "生成前上下文检查", generation_inputs: "实际生成输入",
  generation_output: "生成输出计量", answer_audit: "回答审核", revise_answer: "一次答案修订",
};
const seconds = (ms: unknown) => typeof ms === "number" ? `${(ms / 1000).toFixed(2)}秒` : "未记录";
const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};

export default function AdminTraces({ baseUrl, authenticatedFetch, publicView = false, initialRequestId = "", activeRequestId = "" }: Props) {
  const tracePath = publicView ? "/api/auth/traces" : "/api/auth/admin/traces";
  const fetchRef = useRef(authenticatedFetch);
  fetchRef.current = authenticatedFetch;
  const [rows, setRows] = useState<Row[]>([]);
  const [selected, setSelected] = useState(() => preferredTraceId(activeRequestId, initialRequestId));
  const [detail, setDetail] = useState<Row | null>(null);
  const [error, setError] = useState("");
  const [lookup, setLookup] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [updated, setUpdated] = useState("");
  const [loading, setLoading] = useState(false);
  const [awaitingAdmission, setAwaitingAdmission] = useState(false);
  const selectionRef = useRef("");
  selectionRef.current = selected;

  useEffect(() => {
    // Only a new executing task auto-selects; polling must respect manual history selection.
    if (activeRequestId) setSelected(activeRequestId);
  }, [activeRequestId]);

  async function read(path: string, signal: AbortSignal) {
    const response = await fetchRef.current(path, { cache: "no-store",
      signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]) });
    const body = await response.json().catch(() => null);
    if (!body || !response.ok) {
      if (response.status === 404) throw new Error('TRACE_NOT_FOUND');
      const message = typeof body?.detail === 'string' ? body.detail :
        response.status === 401 ? '身份已失效，请重新登录后查看轨迹。' :
        `轨迹服务返回异常（HTTP ${response.status}），请刷新重试。`;
      throw new Error(message);
    }
    return body;
  }

  useEffect(() => {
    const controller = new AbortController();
    let pending = false;
    async function poll() {
      if (pending || controller.signal.aborted) return;
      pending = true;
      try {
        const body = await read(`${baseUrl}${tracePath}`, controller.signal);
        if (controller.signal.aborted) return;
        const items: Row[] = [...(Array.isArray(body.active) ? body.active : []),
          ...(Array.isArray(body.requests) ? body.requests : [])];
        setRows([...new Map(items.filter(r => r?.request_id).map(r => [r.request_id,r])).values()]);
        setUpdated(new Date().toLocaleTimeString());
        setError("");
        const first = body.active?.[0] || body.requests?.[0];
        if (!selectionRef.current && first) setSelected(first.request_id);
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "链路读取失败");
      } finally { pending = false; }
    }
    void poll();
    const timer = window.setInterval(() => void poll(), 3000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [baseUrl, tracePath, refresh]);

  useEffect(() => {
    setDetail(null);
    setAwaitingAdmission(false);
    if (!selected) return;
    const controller = new AbortController();
    let pending = false;
    async function load() {
      if (pending || controller.signal.aborted) return;
      pending = true;
      setLoading(true);
      try {
        const body = await read(`${baseUrl}${tracePath}/${encodeURIComponent(selected)}`, controller.signal);
        if (!controller.signal.aborted) { setDetail(body); setError(""); setAwaitingAdmission(false); }
      } catch (e) {
        if (!controller.signal.aborted && e instanceof Error && e.message === 'TRACE_NOT_FOUND' && selected === activeRequestId) {
          setAwaitingAdmission(true); setError("");
        } else if (!controller.signal.aborted) setError(e instanceof Error && e.message === 'TRACE_NOT_FOUND'
          ? '没有找到属于当前身份的请求轨迹。' : e instanceof Error && e.name === 'TimeoutError'
          ? '轨迹读取超时，可刷新重试；不会终止正在运行的任务。' : e instanceof Error ? e.message : "详情读取失败");
      } finally { pending = false; if (!controller.signal.aborted) setLoading(false); }
    }
    void load();
    const timer = window.setInterval(() => void load(), 3000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [baseUrl, tracePath, selected, refresh, activeRequestId]);

  const summary = object(detail?.summary);
  const execution = object(detail?.execution || summary.execution);
  const nodes = Array.isArray(summary.node_trace) ? summary.node_trace.map(object) : [];
  const timings = Array.isArray(execution.stage_timings) ? execution.stage_timings.map(object) : [];
  const errors = Array.isArray(summary.errors) ? summary.errors : execution.errors;
  const stageDetails = (Array.isArray(summary.stage_details) ? summary.stage_details :
    Array.isArray(execution.stage_details) ? execution.stage_details : []).map(object).filter(importantStage);
  const visibleNodes = (stageDetails.length ? stageDetails : nodes.length ? nodes : timings).filter(importantStage);
  return <section className="admin-traces" aria-label="请求链路">
    <div className="trace-toolbar">
      <span>每3秒刷新 · {updated || "正在读取"}</span>
      <button type="button" onClick={() => setRefresh(v => v + 1)}>刷新</button>
    </div>
    <p className="trace-note">{publicView ? "仅显示当前身份的请求轨迹。" : "管理员请求轨迹。"}运行中与失败后均可查看已记录的计划、检索和审核结果；不展示完整Prompt、附件正文或图片像素。旧请求未记录的字段无法补回。</p>
    <form className="trace-lookup" onSubmit={e => { e.preventDefault(); const value = normalizeTraceId(lookup); if (value) {setSelected(value); setRefresh(v => v+1);} }}>
      <input aria-label="请求编号" value={lookup} onChange={e => setLookup(e.target.value)} placeholder="粘贴请求编号或排查编号" maxLength={200} />
      <button type="submit">查看</button>
    </form>
    {error && <p className="access-error" role="alert">{error}</p>}
    <div className="trace-layout">
      <div className="trace-list" aria-label="最近请求">
        {!rows.length && <p>暂无请求记录</p>}
        {rows.map(row => <button type="button" key={row.request_id} className={selected === row.request_id ? "selected" : ""} onClick={() => setSelected(row.request_id)}>
          <strong>{row.live ? "正在执行" : row.status} · {row.live ? seconds(row.execution?.elapsed_ms) : seconds(row.total_latency_ms)}</strong>
          <small>{row.live ? labels[String(row.execution?.stage)] || String(row.execution?.stage) : new Date(row.started_at).toLocaleString()}</small>
          <code>{row.request_id}</code>
        </button>)}
      </div>
      <div className="trace-detail" aria-busy={loading}>
        {!detail ? <p>{awaitingAdmission ? "当前任务正在上传或等待接入，轨迹就绪后自动显示。" : loading ? "读取详情…" : "选择一条请求查看"}</p> : <>
          <h3>{detail.live ? "当前链路" : "请求详情"}</h3>
          <code>{detail.request_id}</code>
          <p>执行状态：{String(execution.state || detail.status)} · 当前/最后阶段：{labels[String(execution.stage)] || String(execution.stage || "未记录")}</p>
          <p>总耗时：{seconds(execution.elapsed_ms ?? detail.total_latency_ms)} · 回答模式：{String(summary.mode || "未记录")}</p>
          {recorded(execution.summary_progress) && <details open><summary>附件概括进度</summary>
            <pre>{JSON.stringify(compactDetail(execution.summary_progress), null, 2)}</pre></details>}
          {recorded(execution.model_contracts) && <details><summary>模型输出协议与格式重试</summary>
            <pre>{JSON.stringify(execution.model_contracts, null, 2)}</pre></details>}
          {Array.isArray(summary.tools) && summary.tools.length > 0 && <p>计划工具：{summary.tools.join(" ＋ ")}{typeof summary.tool_rounds === 'number' ? ` · 工具轮次：${summary.tool_rounds}` : ''}</p>}
          {detail.error_type && <p className="access-error">错误：{detail.error_type}</p>}
          <h4>实际执行的重要阶段</h4>
          <div className="trace-table-wrap"><table><thead><tr><th>阶段</th><th>轮次</th><th>耗时</th><th>Token（含格式重试）</th><th>结果/复用</th></tr></thead><tbody>
            {visibleNodes.map((node, index) => <tr key={index}>
              <td>{labels[String(node.node || node.stage)] || String(node.node || node.stage)}</td>
              <td>{String(node.round ?? "—")}</td><td>{seconds(node.elapsed_ms)}</td>
              <td>{stageTokens(node)}</td>
              <td>{node.error_code ? String(node.error_code) : node.operation_error ? String(node.operation_error) : node.status === 'running' || node.in_progress ? "进行中" : node.status === 'failed' ? "失败" : node.reused ? "复用" : "阶段已结束"}</td>
            </tr>)}
          </tbody></table></div>
          <p className="trace-note">父阶段可能包含子阶段调用，Token不能逐行相加。正文扩展的完整性检查不是事实认证；预算回退保留原证据，不代表原文没有内容。</p>
          <h4>各阶段输入 / 输出</h4>
          {!!detail.visual_label_versions?.length && <details><summary>图片标签：识图 / Final / 审核输入</summary>
            <p>仅显示当前有权限请求的文字标签，含不确定项；不含图片像素。历史未保存的标签无法补回。</p>
            <pre>{JSON.stringify(detail.visual_label_versions, null, 2)}</pre></details>}
          {!stageDetails.length && <p>此记录没有逐阶段详情（旧记录无法追溯未保存的 Planner 输出）。</p>}
          {stageDetails.map((stage,index) => <details key={index} open={stage.stage === "plan_request"}>
            <summary>{labels[String(stage.stage)] || String(stage.stage)} · {String(stage.status)} · {seconds(stage.elapsed_ms)}</summary>
            {recorded(stage.input) && <><p>输入</p><pre>{JSON.stringify(compactDetail(stage.input),null,2)}</pre></>}
            {recorded(stage.output) && <><p>输出</p><pre>{JSON.stringify(compactDetail(stage.output),null,2)}</pre></>}
            {!!stage.error_code && <p className="access-error">{String(stage.error_code)}</p>}
          </details>)}
          {(["evidence_gate", "replanning", "online_search", "context", "generation", "document_summary", "visual_input_manifest", "citation_ids"] as const).filter(key => recorded(summary[key]) && !(key === 'online_search' && object(summary[key]).status === 'not_requested')).map(key => <details key={key}>
            <summary>{{plans:"规划与重规划工具", planner:"Planner耗时与缓存", guard_diagnostics:"权限与工具校验原因", query_policy:"查询策略", evidence_gate:"证据支持审核", task_support_audit:"聊天/视觉审核", replanning:"恢复决策", recovery_actions:"执行恢复记录", online_search:"联网结果", context:"上下文与保留证据", generation:"生成预算与耗时", document_summary:"附件概括与覆盖范围", visual_input_manifest:"实际视觉输入", citation_ids:"最终引用ID"}[key]}</summary>
            <pre>{JSON.stringify(compactDetail(summary[key]), null, 2)}</pre>
          </details>)}
          {recorded(errors) && <details open><summary>错误状态</summary><pre>{JSON.stringify(errors, null, 2)}</pre></details>}
          {detail.trace_unavailable && <p>历史快照不可读取，只能显示基础状态。</p>}
        </>}
      </div>
    </div>
  </section>;
}
