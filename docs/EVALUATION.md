# 评测设计

## 分层评测

| 层级 | 主要指标 |
|---|---|
| 文档解析 | 页面覆盖率、表格结构准确率、视觉资产保留率、位置映射正确率 |
| 文件选择 | File Recall@1、错误选文件率 |
| 证据检索 | Evidence Recall@1/3/5、MRR、Visual Recall@K |
| 回答生成 | Exact Match、Token F1、数值准确率、开放题原子断言得分 |
| 可信性 | Citation Precision/Recall、Refusal F1、Unsupported Answer Rate |
| Agent | Tool Selection Accuracy、漏调用率、不必要调用率 |
| 工程性能 | P50/P95延迟、峰值显存、OOM率、JSON合法率 |

## 测试场景

单文件测试应覆盖PDF、Word、Excel和图片。跨文件测试建议包含：

1. 一份正确文件与多份困难干扰文件；
2. 所有候选文件都不支持答案；
3. 多份文件重复支持；
4. 多份文件提供互补证据；
5. 多份文件内容冲突。

干扰文件必须人工或程序确认不能独立支持金标答案。短数字、年份和Yes/No答案尤其容易在其他文件中偶然出现。

## 评分方法

- 金额、日期、数量、名称等确定性答案使用规范化Exact Match、F1或数值容差；
- 开放总结题使用官方原子断言或多裁判LLM Judge；
- 评分协议由答案类型决定，而不是简单由PDF、Word或Excel格式决定；
- 所有模型版本必须读取完全相同的Input Snapshot，保证对比公平。

## 数据隔离

按文档而不是按问题划分开发集、冻结测试集和密封保留集。同一文件或近重复版本只能属于一个分区。开发集中观察过的题目不得作为唯一最终成绩。

## 推荐消融实验

1. Qwen3-VL直接回答；
2. BM25 RAG；
3. BM25 + Dense Retrieval；
4. BM25 + Dense + Reranker；
5. 完整系统：混合检索 + 视觉证据 + Agent + 引用校验。

如果Gold Evidence没有进入模型输入，应归类为解析/检索失败，而不是模型理解失败。可以补充Oracle Evidence实验判断模型上限。
