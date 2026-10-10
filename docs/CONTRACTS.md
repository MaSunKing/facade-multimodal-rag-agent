# Planner字段、执行契约与回退

[首页](../README.md) → 本页 → [Prompt](PROMPTS.md) → [轨迹](TRACES.md)

## 模型输出

实际schema由[planning_output_schema](../backend/sales/planning_spec.py)动态生成，不把PlanningSpec所有兼容属性都给模型。

| 字段 | 必填条件 | 用途 |
|---|---|---|
| sources | 必填 | general_chat/company_rag/customer_documents/public_web_search/visual_inspection；程序检查授权 |
| operation | 必填 | lookup/summary/comparison |
| categories | 必填，可空 | 企业六类优先范围，不是证据保证 |
| wants_images | 必填bool | 展示图片，独立于理解 |
| document_visual_required | 有附件时必填bool | 是否理解客户像素，即便选择其他来源也不能漏填 |
| source_operations | 按需 | 混合来源不同任务类型 |
| queries | 按需 | 每项source与完整独立query，不能拼成唯一检索串 |
| collection | 明确统计/分组时 | action、group_by等；普通概括不强制模板 |
| document_names | 明确只看某附件时 | 原名选同名集合、display_name选单个，后端绑定ID |
| image_family | 明确形态时 | 展示范围，不猜型号 |
| memory_updates | 记忆启用且明确用户事实 | 最多2条、quote来自当前原话，程序验证 |

不要求targets/filters/aspects、文件ID、answer_basis、实际证据ID或预算。兼容模型仍有这些成员，不能误称当前Planner必须输出它们。[planner_prompt.py](../backend/sales/planner_prompt.py)

```json
{"sources":["customer_documents","company_rag"],"operation":"comparison","categories":["02_product_information"],"wants_images":false,"document_visual_required":true}
```

上例是示意，不是实测。

## 程序派生

| 结构 | 来源 | 边界 |
|---|---|---|
| tools/company_operation/document_operation | 精简计划adapter | 兼容执行，不重复让模型判断 |
| retrieval_requests | queries转换 | 逐来源原子检索要求 |
| requirement_id/target_id/requirement_ids | 工具绑定 | 覆盖跟踪，represented不等于supported |
| target_document_ids/access_scope/owner | 身份会话 | 模型不能编造，歧义不猜 |
| document_scope/company_directory_scope | operation与实际导航 | 内容/集合覆盖，未入Context不等于不存在 |
| evidence_id/source_refs/asset_id | 接入数据库 | 稳定来源及图片归属 |
| visual_status/display_images/visual_evidence | 实际工具执行 | 展示、未识别、已识别、失败、预算不足分别记录 |
| expansion_status/context_budget/input_budget_audit | Context代码 | 真实资源限制，不是Planner授权 |

## 模型角色

| 角色 | 输入 | 输出 |
|---|---|---|
| Planner | 当前原问题、有效补充、必要历史、能力地图、文件名/解析概况 | schema JSON |
| 客户视觉 | 像素、观察要求、来源归属 | 可观察描述和不确定项 |
| Final | 有效要求及预算后的文字/表格/OCR/视觉描述 | 自然语言草稿 |
| Answer Audit | 原话、task_update、必要历史、来源/操作记录、工具范围、标签状态、完整草稿 | decision、coverage_status及必要意见JSON |
| Replanner | 原话、旧计划、可核查proof、资源和缺口 | 一次有界恢复计划 |
| revision | 原Context、草稿、revision意见 | 自然语言修改稿，不调用新工具 |

[json_decoding.py](../backend/sales/json_decoding.py)是约束解码机制，不等同于Prompt要求JSON；环境/配置决定实际可用性，程序仍检查协议。Final不启用JSON包装；用户所要代码/JSON可作为正文，但无权设置后端字段。

## Audit决策

- pass：完成所问，或回答可支持部分并说明缺口；不认证逐句事实。
- revise：漏答、偏题或标签边界矛盾在原Context内可修正，提供revision。
- recover：proof指出scope_mismatch/execution_gap/visual_input，且有权限、资源和恢复额度，才Replan。
- limited：证明不足、权限/预算不足、额度已用或审核异常，不发布未审核草稿。

审核不看大批证据正文或像素；公司能力地图只是组织说明。缺少候选不是未执行，答案短不是工具错误。最多一次工具恢复、一次答案修改、两次额外格式重试分别计数。[answer_review.py](../backend/sales/answer_review.py) · [model_contract.py](../backend/sales/model_contract.py)

来源参考由程序从实际Context回填，不让Final编造ID；参考列表不等于逐句事实认证。[output_contract.py](../backend/sales/output_contract.py)
