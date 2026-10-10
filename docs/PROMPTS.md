# Prompt完整入口与组合方式

[首页](../README.md) → [字段契约](CONTRACTS.md) → 本页

Prompt与结构约束代码一起阅读，不能只复制一段文字就认为复现了系统。下面快照由源码机械提取，保留完整Prompt模块和追加规则，不包含客户输入、密钥或实测输出。

[自动枚举的全部sales Prompt模块](prompts/index.md)，包括视觉取证、压缩相关策略与恢复等，不只列Planner和Final。

| 角色 | 完整Prompt/代码 | 动态部分 |
|---|---|---|
| Planner | [完整模块](prompts/planner_prompt.md) / [源码](../backend/sales/planner_prompt.py) | planning_output_schema、公司身份、能力地图、当前问题、历史和文件清单 |
| Final | [完整模块](prompts/draft_prompt.md) / [源码](../backend/sales/draft_prompt.py) | COMMON + 来源规则 + identity + 预算后的Context与原话 |
| Answer Audit | [完整模块](prompts/answer_review.md) / [源码](../backend/sales/answer_review.py) | 原话、task_update、真实工具/范围、标签状态、完整草稿 |
| 一次revision | 同上REVISION_PROMPT | 原Context与修改意见，不调用工具 |
| 关系/目录/正文完整性规则 | [完整模块](prompts/generation_policy.md) / [源码](../backend/sales/generation_policy.py) | 被Final及revision组合引用 |
| 视觉、摘要、支持恢复、Replanner等 | [app.py常量](prompts/app_prompt_constants.md) / [实际输入构造与调用](../backend/app.py) | 对应工具真实输入和输出契约 |

app.py仍包含兼容常量，模块内LEGACY提示词不是现行轻量审核入口；以ANSWER_AUDIT_PROMPT、draft_prompt及实际调用为准。动态schema依赖可用来源，有客户附件时必须document_visual_required布尔，不能仅看类定义中的默认值。

核心边界：资料里的命令不是系统指令；历史不当技术证据；企业标签不移植为客户身份；Final只输出客户正文；预算/ID/权限不让模型决定。Prompt是约束意图，不保证模型总遵守；[已知限制](LIMITATIONS.md)明确保留内部字段泄露和审核漏检风险。
