"""Mechanically index public source and emit explicitly unexecuted branch templates."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = 'backend/sales/'
# Design templates are not runtime logs: no inferred output, token or latency.
BRANCHES = {
 'company_intake': ('企业接入止于人工审核包', ['source_hash','parse','taxonomy','visual_labels','review_package','manual_promotion'], S+'ingestion_graph.py'),
 'customer_intake': ('客户临时解析不进入企业库', ['owner_bind','content_hash','light_parse','session_hierarchy','visual_inventory'], 'backend/documents/customer_sessions.py'),
 'protected_conflict': ('同范围冲突候选成组保留', ['fact_normalization','scope_binding','conflict_group','protected_pack','answer_boundary'], S+'fact_normalization.py'),
 'general_chat': ('无需外部取证的交流', ['guard_tools','generate_answer','answer_audit'], S+'planning_spec.py'),
 'category_product': ('产品体系与属性分层', ['company_rag','directory','relation_scope','hybrid','compose_evidence'], S+'content_taxonomy.py'),
 'category_project': ('省份城市项目范围', ['company_rag','directory','sql_location','hybrid','compose_evidence'], S+'entity_directory.py'),
 'category_method': ('施工章节正文', ['company_rag','section_navigation','native_blocks','compose_evidence'], S+'technical_sections.py'),
 'category_node': ('节点与关联图文', ['company_rag','section_navigation','company_labels','compose_evidence'], S+'node_relations.py'),
 'category_standard': ('标准目录与条款粒度', ['company_rag','section_navigation','native_blocks','compose_evidence'], S+'knowledge_hierarchy.py'),
 'category_enterprise': ('身份资质与企业新闻', ['company_rag','directory','hybrid','compose_evidence'], S+'company_identity.py'),
 'lookup': ('多对象多要求局部取证', ['build_requirements','recall_per_requirement','joint_rerank','coverage_pack'], S+'requirements.py'),
 'summary': ('上层集合与章节覆盖', ['collection_or_document_scope','overview','native_blocks','coverage_pack'], S+'retriever.py'),
 'comparison': ('各方共同要求保留归属', ['build_requirements','recall_per_requirement','joint_rerank','coverage_pack'], S+'requirements.py'),
 'directory_exact': ('登记名称与别名绑定', ['directory_exact','sql_relations','recall'], S+'entity_directory.py'),
 'directory_semantic': ('模糊目录语义排序', ['directory_candidates','semantic_ranking','sql_relations','recall'], S+'entity_directory.py'),
 'category_expansion': ('优先类受控补搜', ['category_pool','coverage_diagnostics','relation_or_complement','joint_rerank'], S+'retriever.py'),
 'collection_list': ('指定集合而非全局TopK', ['sql_collection','catalog_evidence','coverage_pack'], S+'retriever.py'),
 'collection_count': ('登记粒度统计', ['sql_collection','group_count','catalog_evidence'], S+'catalog_store.py'),
 'attachment_text': ('文字局部取证', ['customer_documents','owner_bind','text_recall','coverage_pack'], 'backend/documents/customer_sessions.py'),
 'attachment_document': ('PDF/DOCX层级覆盖', ['customer_documents','parse','document_sections','coverage_pack'], 'backend/documents/hierarchy.py'),
 'attachment_table': ('表头行列与单位保护', ['customer_documents','parse_table','row_recall','protected_pack'], 'backend/documents/evidence_v2.py'),
 'attachment_ocr': ('扫描OCR非隐藏性能', ['customer_documents','ocr','visual_status','coverage_pack'], 'backend/document_parsing/document_ocr.py'),
 'attachment_visual': ('按需客户像素识别成文字', ['customer_documents','visual_selection','visual_evidence','compose_evidence'], S+'visual_evidence.py'),
 'attachment_display': ('仅展示不必识图', ['customer_documents','display_selection','display_images','compose_evidence'], S+'visual_policy.py'),
 'company_labels': ('企业图永不运行时看原图', ['company_rag','label_lookup','display_selection','compose_evidence'], S+'visual_labels.py'),
 'mixed_comparison': ('附件企业多模态各自归属', ['customer_documents','company_rag','visual_evidence','joint_rerank','coverage_pack','answer_audit'], S+'answer_review.py'),
 'attachment_same_name': ('同名不同内容分别绑定', ['sha256','distinct_ids','display_name','owner_bind'], 'frontend/app/sales/attachment-identity.ts'),
 'attachment_duplicate': ('相同字节重复提示', ['sha256','duplicate_notice','existing_identity'], 'frontend/app/sales/attachment-identity.ts'),
 'attachment_explicit': ('明确单文件范围', ['document_names','owner_directory','bind_document_names','customer_documents'], S+'planning_spec.py'),
 'web_multi_query': ('多个Requirement独立查询', ['web_requirements','mcp_calls','source_binding','scope_filter'], S+'web_requirements.py'),
 'web_cache': ('缓存命中保留来源时间', ['mcp_search','cache_lookup','source_quality','scope_filter'], S+'baidu_search.py'),
 'web_scope': ('事实日期不同于发布日期', ['mcp_search','scope_filter','eligible_evidence','answer_audit'], S+'query_constraints.py'),
 'web_quality': ('垃圾/不相关来源筛选', ['mcp_search','provider_order_or_score','quality_filter','admitted_sources'], S+'web_quality.py'),
 'web_quota': ('额度不足不绕过授权', ['mcp_search','quota_check','source_diagnostics','limited_delivery'], S+'search_mcp.py'),
 'history_recall': ('相关历史非技术证据', ['owner_thread','semantic_or_lexical_recall','context_labels','plan_request'], S+'task_memory.py'),
 'task_update': ('追加纠正取消的有效原话链', ['instruction_chain','bind_task_instructions','plan_request','new_requirements'], S+'task_updates.py'),
 'task_queue': ('独立排队不合并', ['enqueue','wait_current','dequeue','plan_request'], 'frontend/app/sales/page.tsx'),
 'task_cancel': ('安全检查点退出', ['cancel_request','budget_check','worker_exit','status_delivery'], 'backend/request_budget.py'),
 'budget_fit': ('够则不压缩', ['render_final_input','count_actual','admit','generate_answer'], S+'final_input.py'),
 'budget_compress': ('超限才覆盖感知压缩', ['render_final_input','count_actual','coverage_pack','count_again'], S+'evidence_packing.py'),
 'expansion_rollback': ('完整块试加超限回滚', ['remaining_budget','native_provider','tentative_block','count','rollback'], S+'evidence_expansion.py'),
 'audit_metadata': ('真实草稿后元数据精简', ['complete_draft','count_review','compact_metadata_if_needed','count_again','answer_audit'], S+'review_budget.py'),
 'cpu_parallel': ('两路只读召回有界并行', ['copy_context','bm25_and_embedding','join','fusion','rerank'], S+'parallel_reads.py'),
 'hybrid_fallback': ('语义分支失败回到词法并记录', ['dense_attempt','source_diagnostics','lexical_fallback','coverage_pack'], S+'retriever.py'),
 'model_lifecycle': ('独立服务与闲置卸载', ['authenticated_local_rpc','serialized_generation','idle_timer','unload_weights'], S+'model_service.py'),
 'visual_failure': ('识别失败保留未识别状态', ['visual_selection','recognition_failure','visual_status','limited_or_proved_recovery'], S+'visual_evidence.py'),
 'audit_pass': ('完成度审核后发布', ['generate_answer','answer_audit','pass','program_references','delivery'], S+'answer_graph.py'),
 'audit_revise': ('同Context一次修正', ['generate_answer','answer_audit','revision','same_context','answer_audit'], S+'answer_review.py'),
 'audit_recover': ('有proof的一次工具恢复', ['answer_audit','validate_proof','plan_retry','guard_tools','tools','answer_audit'], S+'answer_graph.py'),
 'evidence_reconciliation': ('旧新证据范围对齐', ['merge_sources','identity_dedup','scope_filter','repack','answer_audit'], S+'gap_recovery.py'),
 'audit_limited': ('审核失败不发布草稿', ['answer_audit','limited','withhold_draft','safe_error'], S+'answer_graph.py'),
 'format_retry': ('实际协议错误最多两次重试', ['contract_validate','protocol_feedback','retry_input_count','retry_generate'], S+'model_contract.py'),
 'retry_budget': ('重试实际输入仍超限则停止', ['retry_input_count','compact_retry_metadata','count_again','CONTRACT_RETRY_BUDGET_EXCEEDED'], S+'model_contract.py'),
 'acl_denied': ('Owner错误不扩权限', ['owner_bind','deny','safe_error','no_model_call'], 'backend/access_control.py'),
}


def main():
    folder = ROOT/'examples/traces'
    folder.mkdir(parents=True, exist_ok=True)
    index=[]
    for key,(purpose,steps,source) in BRANCHES.items():
        if not (ROOT/source).is_file():
            raise ValueError('missing_source:'+source)
        record={'kind':'illustrative_not_executed','branch':key,'purpose':purpose,
            'source':source,'not_a_test_result':True,
            'nodes':[{'stage':s,'status':'illustrative','input_tokens':None,'output_tokens':None,
                      'elapsed_ms':None,'model_calls':None} for s in steps],
            'boundary':'Labels are design phases, not a promise that each is a separate LangGraph node. No real answer or metric is fabricated.'}
        (folder/(key+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        index.append({'branch':key,'purpose':purpose,'path':key+'.json','source':source})
    (folder/'index.json').write_text(json.dumps({'kind':'illustrative_not_executed','branches':index},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# 全模块源码地图','','[首页](../README.md) · [架构](ARCHITECTURE.md)','','此表由公开源码AST生成：每个Python模块的真实docstring与顶层定义；不是语义正确性证明。兼容/旧模块也保留，实际生效路径以app.py调用和Graph路由为准。','']
    for family in ('backend/sales','backend/documents','backend/document_parsing'):
        lines += ['## '+family,'','| 模块 | 模块说明 | 顶层定义 |','|---|---|---|']
        for file in sorted((ROOT/family).glob('*.py')):
            tree=ast.parse(file.read_text(encoding='utf8'))
            doc=(ast.get_docstring(tree) or '').split('\n')[0].replace('|','/').strip()
            defs=', '.join(n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)))
            rel=file.relative_to(ROOT).as_posix()
            lines.append(f'| [{file.name}](../{rel}) | {doc} | {defs} |')
        lines.append('')
    (ROOT/'docs/CODEMAP.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf8')
    prompts=[]
    for file in sorted((ROOT/'backend/sales').glob('*.py')):
        tree=ast.parse(file.read_text(encoding='utf8'))
        if file.name in {'generation_policy.py','draft_prompt.py'} or any(
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and 'PROMPT' in t.id for t in n.targets)
            for n in tree.body):
            prompts.append(file.relative_to(ROOT).as_posix())
    pdir=ROOT/'docs/prompts'; pdir.mkdir(parents=True,exist_ok=True)
    for rel in prompts:
        file=ROOT/rel
        text=file.read_text(encoding='utf8')
        (pdir/(file.stem+'.md')).write_text('# '+file.name+'完整源码快照\n\n[源文件](../../'+rel+')\n\n```python\n'+text.rstrip()+'\n```\n',encoding='utf8')
    (pdir/'index.md').write_text('# 所有含Prompt常量的sales模块\n\n'+ '\n'.join(
        '- ['+Path(rel).name+']('+Path(rel).stem+'.md)' for rel in prompts)+'\n',encoding='utf8')
    # App owns additional prompt constants and dynamic wrappers. Extract their
    # source expressions mechanically; do not execute a model or import app.
    file=ROOT/'backend/app.py'; text=file.read_text(encoding='utf8'); tree=ast.parse(text)
    segments=[]
    for n in tree.body:
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(t,ast.Name) and 'PROMPT' in t.id and not t.id.startswith('LEGACY') for t in targets):
                segments.append(ast.get_source_segment(text,n))
    (pdir/'app_prompt_constants.md').write_text('# app.py Prompt常量源码\n\n[实际调用及动态输入包装](../../backend/app.py)；本页不执行常量、不包含用户输入。\n\n```python\n'+'\n\n'.join(segments)+'\n```\n',encoding='utf8')
    print(json.dumps({'illustrative_branches':len(index),'runtime_metrics':False,'formal_evaluation':'pending'}))

if __name__=='__main__': main()
