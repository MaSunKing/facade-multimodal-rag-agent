# 评测设计与当前结果

本项目的评测目标不是只得到一个“回答准确率”，而是区分文件解析、证据召回、视觉输入、模型回答、引用正确性和 Agent 工具选择。

## 1. 当前建材评测草案

当前从可服务的企业 Canonical Evidence 构建 100 题审核草案：

| 轨道 | 数量 | 评测对象 |
|---|---:|---|
| 企业文字 RAG | 70 | 问题能否召回正确文字／表格 Evidence，并覆盖答案要点 |
| 客户图片直读 | 20 | 给定原始图片后，VLM 能否描述可见内容 |
| 证据不足拒答 | 10 | 知识库无支持时是否拒绝编造 |

状态为 `human_review_draft`。业务人员仍需确认问题表达、答案断言、图片内容和拒答边界，因此不能把它当作正式业务 Benchmark。

## 2. 已验证的诊断结果

- 严格 Evidence Recall@5：**65/70（92.86%）**；
- 客户图片题原始资产可用率：**20/20**；
- 公开仓库 CPU 回归：**33 项通过，1 项按环境跳过**。

严格 Evidence Recall 只认同一个 `chunk_id`。即使召回内容等价的另一条证据，在没有人工标注“等价支持证据”前仍记为未命中。因此它是可复现的召回诊断，不是最终答案准确率。

20 道图片题评测的是“把指定图片作为用户输入后进行视觉理解”，不评测企业知识库是否能从所有图片中主动找到该图；图库召回必须建立独立轨道。

## 3. 分层指标

| 层级 | 推荐指标 |
|---|---|
| 解析 | 页面覆盖率、表格结构准确率、视觉资产保留率、位置映射正确率 |
| 文件选择 | File Recall@1、错误选文件率、无支持文件拒答率 |
| Evidence 检索 | Recall@1/3/5、MRR、Visual Recall@K、Gold-in-Snapshot Rate |
| 回答 | Exact Match、Token F1、数值准确率、原子断言覆盖率 |
| 引用 | Citation Precision/Recall、来源位置正确率、图片证据正确率 |
| 可信性 | Refusal Precision/Recall/F1、Unsupported Answer Rate、冲突识别率 |
| Agent | Tool Selection Accuracy、漏调用率、不必要联网率、Planner JSON 合法率 |
| 工程 | P50/P95 延迟、峰值显存、OOM 率、冷启动时间、输入 Token/像素 |

## 4. 评分协议

评分由答案类型决定，而不是由文件格式决定：

- 金额、数量、日期、型号：规范化 Exact Match 或数值容差；
- 列表：集合 Precision/Recall/F1；
- 多断言回答：人工审核后的 `all_of / any_of / regex / numeric` 断言；
- 开放题：单人业务复核或数据集官方 Judge 协议；
- 拒答：相对当前输入 Evidence 判断 `answerable`，不能因为原文档别处有答案就自动算可回答。

当前建材评测不使用昂贵的三模型多数票；业务开放题由一名审核人依据原始证据判定，确定性题优先代码评分。

## 5. 多文件测试

正式多文件轨道应覆盖：

1. 一份支持文件 + 三份困难干扰；
2. 四份均不支持，模型必须拒答；
3. 两份重复支持；
4. 多份互补支持；
5. 多份内容冲突。

干扰文件必须确认不能独立支持答案，尤其注意短数字、年份和 Yes/No 偶然重复。评测标签应分别保存官方正证据和项目构造的负文件审计，不能把后者伪装成官方标注。

## 6. 错误归因与 Oracle 实验

端到端失败按以下标签归因：

- `parse_failure`：原始信息未进入 Canonical Evidence；
- `file_retrieval_miss`：正确文件未进入候选；
- `evidence_retrieval_miss`：正确 Evidence 未进入 Snapshot；
- `visual_selection_miss`：正确页面／图片未进入视觉输入；
- `model_reasoning_error`：正确证据已经输入但模型答错；
- `grounding_validation_failure`：生成内容未通过引用或 Schema 校验。

对视觉选择或检索失败题，补充 Oracle Evidence：直接给模型正确页面。如果普通输入错、Oracle 对，优先修检索；两者都错，才说明模型视觉理解或推理能力不足。

## 7. 消融实验

建议报告：

1. 无检索的 Qwen3-VL；
2. BM25；
3. BM25 + Dense；
4. BM25 + Dense + Reranker；
5. 完整系统：混合检索 + 视觉图库 + Agent + 引用校验。

所有版本读取同一 Input Snapshot，固定候选顺序、Prompt、Token／像素预算和生成参数，否则提升无法归因。

## 8. 正式冻结条件

在对外报告正式效果前至少满足：

- 业务人员完成 100 题审核；
- Gold Evidence 与等价证据冻结；
- 评测题与调参集按文档隔离；
- Manifest、题目、Evidence 和图片 SHA256 固定；
- Base／RAG／Agent 使用同一输入快照；
- 同时报告问题级成绩、能力切片和主要失败类型。

当前 README 中的 92.86% 仅按以上边界表述为严格检索诊断。
