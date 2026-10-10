// Presentation only: never invent stages or alter persisted diagnostics.
export function preferredTraceId(activeRequestId: string, previousRequestId: string): string {
  return activeRequestId || previousRequestId;
}
export const importantStages = new Set([
  'plan_request', 'guard_tools', 'customer_documents', 'document_summary', 'company_rag',
  'public_web_search', 'visual_inspection', 'rerank_evidence',
  'compose_evidence', 'support_selection', 'evidence_evaluation', 'generate_answer',
  'validate_answer', 'assess_retry', 'apply_retry_plan',
  'replan_request',
  'evidence_expansion', 'support_budget', 'context_preflight', 'generation_inputs',
  'generation_output', 'answer_audit', 'answer_audit_admission', 'answer_audit_retry_budget', 'revise_answer',
]);

export function recorded(value: unknown): boolean {
  if (value == null || value === '' || value === 'unknown' || value === '[depth limit]') return false;
  if (Array.isArray(value)) return value.some(recorded);
  if (typeof value === 'object') return Object.values(value).some(recorded);
  return true; // false and 0 are real audit results, not missing fields.
}

export function compactDetail(value: unknown): unknown {
  if (Array.isArray(value)) return value.filter(recorded).map(compactDetail);
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.entries(value).filter(([, v]) => recorded(v)).map(([k, v]) => [k, compactDetail(v)]));
  return value;
}

export function importantStage(row: Record<string, unknown>): boolean {
  return !row.phase && (importantStages.has(String(row.node || row.stage)) || !!row.error_code || !!row.operation_error);
}

export function stageTokens(row: Record<string, unknown>): string {
  const output = row.output && typeof row.output === 'object' ? row.output as Record<string, unknown> : {};
  const usage = (row.token_usage || output.token_usage) as Record<string, unknown> | undefined;
  if (!usage) return '未记录';
  if (usage.model_calls === 0) return '0（程序处理）';
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : '未知';
  const result = typeof usage.output_tokens === 'number' ? usage.output_tokens : '未知';
  return `输入 ${input} / 输出 ${result}${usage.complete === false ? '（不完整）' : ''}`;
}

export function normalizeTraceId(value: string): string {
  const match = value.match(/(?:req_)?[a-f0-9]{32}|(?:task_|client_)[A-Za-z0-9_-]+/i);
  return match?.[0] || value.trim();
}
