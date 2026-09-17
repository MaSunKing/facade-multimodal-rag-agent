# 有限 Agent 与 LangGraph

`WorkflowStep`携带阶段和操作，`WorkflowCursor`暂停生成器；Graph条件路由到下一阶段，执行真实工具并将结果送回，不是在完整回答后补写节点名。

## 谁决定什么

| 组件 | 职责 | 不负责什么 |
|---|---|---|
| Planner模型 | 生成工具选择、来源语言Query、检索目标与回答维度 | 不授予权限、不解除显存/时间上限 |
| 后端Guard与预算 | 复核账号范围、联网授权、工具可用性、可恢复错误与共享额度 | 不按测试句子硬编码答案 |
| LangGraph | 根据Cursor下一阶段调度实际操作，记录执行并进入条件恢复或收尾 | 不等于模型自主执行所有节点、不提供跨进程Checkpointer |
| Context/Composer | 组织各路径Evidence、选择模型输入并记录覆盖缺口 | 不让final读取所有文件、不凭索引代替正文 |
| 生成与校验 | Qwen3-VL读取保留的图文；Python检查结构、引用与启发式支持 | 校验通过不等于所有推理与宣传结论正确 |

企业库、附件、网页和视觉可能共同进入一次回答，但不是各自都经过同一混合检索器；也不是每次都调用全部工具。联网许可与实际搜索执行分开，权限失败不会以重试绕过。

```text
plan_request → guard_tools → 按需工具/视觉
 → compose_evidence → generate_answer → validate_answer
 → collect_response → assess_retry → 有限重试 / finalize_response
```

Compose/Validate边界尚未全部抽成独立可持久化服务。附件视觉可在生成节点中联合输入，不一定单独运行visual_inspection。问候、审核目录和安全兜底有快捷路径，不是每题都运行完整LLM规划。

## 联合来源例子

“结合上传的节点图、施工说明、企业产品资料和公开规范判断做法”：附件工具召回图文、企业工具提供产品/技术证据、网页补公开事实，Compose/Context打包后共同生成。该例是路径说明，不冒充本次模型实测成绩。

## 恢复策略

- 可恢复的检索读操作失败：预算允许可重试一次。
- 附件只有索引且尚未调用模型：CPU扩大范围一次，不重跑Planner。
- 部分目标缺失：满足条件时补检索，受共享额度约束。
- 文本生成OOM：满足条件时降低提示预算，不重复同样大负载。
- 认证/权限错误：不盲目重试；语义错误不保证自动发现/修复。

HTTP请求最多两个恢复动作、每阶段一次，不延长deadline；Graph最多一轮工具重执行。planning_rounds增加也可能只是CPU计划修改。

## 审计

`meta.orchestration.tools`为最终计划；`node_trace`记录实际阶段/耗时/复用/失败；`generation_input_audit.kept_evidence_ids`为保留文本；`visual_input_manifest`为实际图片；支持和覆盖审计报告限制；`recovery_actions`记录真正尝试的恢复。

计划选择、工具成功、进入模型和正确使用是四件不同的事。trace主要记录工具阶段，不列出全部控制节点。finalize_response只补充元数据，不再调用模型。

Graph没有Checkpointer。SQLite记忆/审计不是跨进程执行恢复，也不是无限自主重规划。
