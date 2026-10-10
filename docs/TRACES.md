# 多模态与不同链路的轨迹

[首页](../README.md) → 本页 → [机器可读示意轨迹](../examples/traces/index.json)

本版展示**设计分支的示意轨迹**，不是实际模型运行日志，不填造耗时、Token或成功率。所有模板kind=illustrative_not_executed，计量值为null。真实运行的Owner轨迹由系统生成；公开前须去除客户内容、账号、路径、Token、临时链接和身份信息。未来正式评测结果另行发布。

## 模态来源边界

| 来源/模态 | 解析与取证 | Final和Audit看到什么 |
|---|---|---|
| 企业文字/表格 | 接入后长期索引，目录/关系/Hybrid | Final已打包正文，Audit执行范围及草稿 |
| 企业图片 | 预先标签 + 关联正文 | Final标签，Audit标签与不确定项；不运行时看原图 |
| 客户TXT/HTML等 | 临时会话文字块 | 当前问题相关文字与来源位置 |
| 客户PDF/DOCX | 文字/表格、必要OCR、提取视觉资产 | 按任务取证，必要像素另走视觉 |
| 客户XLSX/XLS/CSV | 表头、行列、单位、数值原生结构 | 受保护行块，不能裁掉单位/条件 |
| 客户图片/扫描 | 图片清单、OCR、按需识别 | 视觉描述和未知项；Audit不读像素 |
| Web | MCP查询、排序、质量、缓存、受控正文核验 | 公开证据和事实时间；摘要不等于正文核验 |

代码：[customer_sessions.py](../backend/documents/customer_sessions.py)、[visual_inputs.py](../backend/documents/visual_inputs.py)、[visual_evidence.py](../backend/sales/visual_evidence.py)、[visual_policy.py](../backend/sales/visual_policy.py)。

## 分支覆盖目录

每项链接都含阶段、字段、选择理由、回退和边界；完整文件列表在[index.json](../examples/traces/index.json)。该目录覆盖不同设计分支，而非声称穷举任意任务组合。

| 链路族 | 轨迹入口 | 设计差异 |
|---|---|---|
| 无取证交流 | [general_chat](../examples/traces/general_chat.json) | 不为闲聊强制RAG/Web |
| 资料接入 | [企业接入](../examples/traces/company_intake.json)、[客户接入](../examples/traces/customer_intake.json) | 人工审核后晋升长期库 / 临时会话轻解析 |
| 受保护冲突 | [冲突组](../examples/traces/protected_conflict.json) | 同范围候选不同值保留来源，不按新旧自动裁决 |
| 六大类 | [产品](../examples/traces/category_product.json)、[项目](../examples/traces/category_project.json)、[施工](../examples/traces/category_method.json)、[节点](../examples/traces/category_node.json)、[标准](../examples/traces/category_standard.json)、[企业](../examples/traces/category_enterprise.json) | 实体、地域、技术章节、条款、身份不同导航 |
| 三种操作 | [lookup](../examples/traces/lookup.json)、[summary](../examples/traces/summary.json)、[comparison](../examples/traces/comparison.json) | 多要求覆盖 / 集合上层 / 各方归属 |
| 目录定位 | [名称](../examples/traces/directory_exact.json)、[语义](../examples/traces/directory_semantic.json)、[跨类补搜](../examples/traces/category_expansion.json) | 精确别名与模糊语义、优先类不绝对封闭 |
| 集合操作 | [列举](../examples/traces/collection_list.json)、[统计](../examples/traces/collection_count.json) | 登记集合与粒度，不以Top-K计总数 |
| 客户文字 | [文字](../examples/traces/attachment_text.json)、[PDF/DOCX](../examples/traces/attachment_document.json)、[表格](../examples/traces/attachment_table.json) | 局部、章节、行列/单位保护 |
| 客户视觉 | [OCR](../examples/traces/attachment_ocr.json)、[像素识别](../examples/traces/attachment_visual.json)、[仅展示](../examples/traces/attachment_display.json) | OCR不等于完整看图；展示不必识图 |
| 企业图片 | [标签](../examples/traces/company_labels.json) | 永不恢复为运行时看公司原图 |
| 多源多模态 | [附件与企业比较](../examples/traces/mixed_comparison.json) | 客户标签不能移植为公司材料身份 |
| 附件身份 | [同名不同内容](../examples/traces/attachment_same_name.json)、[重复字节](../examples/traces/attachment_duplicate.json)、[明确选定](../examples/traces/attachment_explicit.json) | 哈希去重、原名和display_name、Owner绑定 |
| Web | [独立多查询](../examples/traces/web_multi_query.json)、[缓存](../examples/traces/web_cache.json)、[时效](../examples/traces/web_scope.json)、[垃圾筛选](../examples/traces/web_quality.json)、[额度](../examples/traces/web_quota.json) | MCP不自动提升上游质量，不因缓存改变事实日期 |
| 上下文 | [相关历史](../examples/traces/history_recall.json)、[补充调整](../examples/traces/task_update.json)、[独立排队](../examples/traces/task_queue.json)、[取消](../examples/traces/task_cancel.json) | 历史非事实、有效原话链、独立任务不合并 |
| 压缩与扩展 | [直接放入](../examples/traces/budget_fit.json)、[压缩](../examples/traces/budget_compress.json)、[完整块回滚](../examples/traces/expansion_rollback.json)、[审核元数据](../examples/traces/audit_metadata.json) | 先计数、超限才压缩、不能任意截正文 |
| 模型与并行 | [CPU并行](../examples/traces/cpu_parallel.json)、[Hybrid降级](../examples/traces/hybrid_fallback.json)、[模型生命周期](../examples/traces/model_lifecycle.json)、[视觉失败](../examples/traces/visual_failure.json) | 非强制线程终止；错误不伪装为无资料 |
| 审核恢复 | [pass](../examples/traces/audit_pass.json)、[revise](../examples/traces/audit_revise.json)、[recover](../examples/traces/audit_recover.json)、[重整](../examples/traces/evidence_reconciliation.json)、[limited](../examples/traces/audit_limited.json) | 分开修改、取证与失效范围 |
| 协议与权限 | [格式重试](../examples/traces/format_retry.json)、[重试超预算](../examples/traces/retry_budget.json)、[Owner拒绝](../examples/traces/acl_denied.json) | 最多两次格式重试，不扩权限 |

## 一条详细示意

用户上传两张不同内容的同名image.png，问“它们跟你们的产品类似吗”：

```text
upload → 字节SHA256不同 → 两个document_id，display_name区分，原名仍为image.png
Planner → customer_documents + company_rag / comparison / 产品资料
          document_visual_required=true / wants_images按用户展示要求
Guard → Owner绑定，两附件均在范围，不接受模型造ID
视觉 → 各自像素识别 → 各自文字描述和unknown项 → 明确归属
企业 → 产品上层体系 / 花色 / 饰面 / 标签与相关正文
Context → 两方代表证据、来源边界、预算检查，必要才压缩
Final → 比较外观；不能由外观推品牌、隐藏保温层、耐火等级
Audit → 看两图标签与未知项、真实执行和完整草稿，不看像素
Delivery → 审核通过后发布，程序附资料与可展示图片
```

如果Final把相似写成同一品牌，设计上应revise而非重看公司图片；但模型审核仍可能漏检，详见[限制](LIMITATIONS.md)。

## 真实轨迹可观察字段

节点stage/phase/status、输入摘要与Owner范围、规划来源、实际工具、Requirement/集合覆盖、证据来源、图片归属和recognition_status、压缩/扩展决策、input_tokens/output_tokens/total_tokens/model_calls、耗时、重试与错误码。CPU节点可记录model_calls=0；示意模板所有计量为null，不冒充实际消耗。轨迹可能包含敏感内容，公开版不导出生产数据库。
