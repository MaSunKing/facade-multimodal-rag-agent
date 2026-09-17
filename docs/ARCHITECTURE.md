# 系统架构

## 两条数据平面

三类回答来源有不同生命周期：企业资料长期索引、客户附件临时隔离、网页按授权获取。企业库用BM25/Dense/RRF/Reranker（受GPU策略约束）；附件用结构窗口/业务行检索与可选重排；网页用去重、时效/来源排序与受限正文验证。三条路径在Evidence/Context阶段汇合，而非强行共用同一召回链。

```text
企业资料：授权 → 解析 → 分类/审核 → 索引与审核图库 → 企业检索
客户附件：上传 → Owner绑定 → 进程内解析 → 独立检索 → TTL/删除
```

企业入库Graph生成审核包，不自动发布新知识。public/internal在候选生成前过滤。客户最多4份文件、默认1小时过期；SQLite不保存附件字节，重启后不能恢复原附件。

当前现有资料均为公开资料，内部区为空/预留；分区能力不等于已经存在内部数据。

## 格式专用结构

公共解析器在 `backend/document_parsing/`，不提供旧的独立财务助手HTTP入口。

| 格式 | 来源对象 |
|---|---|
| PDF | 页、文本、表格、位置、页面/图片资产；可选OCR与视觉回退 |
| DOCX | 段落、表格、页眉页脚、评论、图片；不伪造可靠物理页码 |
| Excel/CSV | Sheet、局部表格、表头、业务行、单元格、公式/缓存、图表元数据 |
| 图片 | 原图、尺寸、哈希、可选OCR/视觉候选 |
| TXT/HTML/XML/ZIP | 文本/表格/成员来源；非任意压缩包执行入口 |

Evidence V2保留格式专用位置，OCR/VLM候选不是官方金标。部分兼容类型仍使用历史命名，不代表产品提供独立财务分析服务。

## 在线路径

1. FastAPI检查容量与总请求预算。
2. Planner生成结构化工具、任务和Query，Guard复核权限/授权/可用性。
3. LangGraph条件执行附件、企业、网页或视觉步骤；当前GPU/工具策略串行。
4. Compose整理来源ID、内容、结构和覆盖信息。
5. Context Engine评分、去重、保护关系与选证据；真实tokenizer检查最终预算。
6. Qwen3-VL进行文字或图文生成。
7. 输出检查、来源物化、按条件有限恢复，收尾返回审计。

`app.py`仍承担较多HTTP集成/模型/校验逻辑，尚未全部独立成可持久化节点服务。核心算法已单独提供代码导航与单元测试。

## 上下文分层

- Canonical Evidence：完整解析，不因检索裁剪修改。
- 本轮输入审计：实际保留证据、Token预算与图片映射。
- 可选任务记忆：用户/会话隔离的状态与相关历史，不是企业技术事实。

最终生成只看到子集；历史也可被预算压缩。收尾节点不做第二次全文分析。

## 图片回填与实际读取

原图与OCR并列。`visual_input_manifest`按SHA绑定上传/附件图片的实际输入序号，不能把检索列表顺序当模型图片顺序。

企业图库分产品、案例、节点、工艺。展示图片、召回元数据、输入像素是不同状态，必须分别审计。

## 资源与部署

NF4 4-bit 8B模型串行运行；生成预留GPU时企业Dense/Reranker可跳过。默认20分钟空闲卸载。Token、图片、像素和输出有预算，OOM降载不是所有路径成功的保证。

腾讯云托管静态前端，Tailscale HTTPS连接本地服务；公开源码使用localhost，不包含生产地址。运行时SQLite、错误日志和可选记忆不公开。

联网不拼接私有Evidence，但当前用户问题本身仍可能敏感；授权开关不能替代隐私治理。Graph没有跨进程Checkpointer，审计持久化不等于断点恢复。

## 代码导航

`backend/sales/answer_graph.py`：条件图；`tool_planner.py`：计划/Guard；`staged_execution.py`：阶段操作；`context_engine.py`：上下文；`retriever.py`：检索；`task_memory.py`：相关历史；`runtime_status.py`和`backend/request_budget.py`：错误/恢复；`backend/documents/`：客户附件；`backend/access_control.py`：权限/审计；`backend/app.py`：回答集成。

新增算法模块：`goal_reranking.py`为目标相关重排；`fact_normalization.py`为实体/指标/范围/单位事实键；`evidence_packing.py`为语义组压缩与原生文本清理；`answer_integrity.py`为回答支持和完整性检查；`recovery_policy.py`为有限恢复策略；`backend/stage_timing.py`为细粒度调用计时。公开75题的指标/边界见[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)。
