# 评测协议

公开版区分代码/接口校验、离线检索与输入保留、模型业务评测。已发布[75题开发回归集合](EVALUATION_RESULTS.md)，题目和逐题结果在[evaluation](../evaluation/README.md)。该集合不是密封测试，不发布一个混合总体答案准确率。

## 可复现的小规模校验

- `scripts/verify_public_release.py`：语法、CPU回归；显式列出跳过/失败，不加载8B生成模型。
- `scripts/public_smoke.py`：虚构PDF、Word、Excel、PNG；真实上传/健康/权限/原图/删除接口和解析结构。
- `scripts/run_sales_rag_eval.py`：提供授权问题集后顺序评测，结果保存本地；公司问题和金标不包含在仓库中。
- `scripts/verify_evaluation_results.py`：无需模型/下载，按75条已发布结果复核指标和标签对应。
- `scripts/prepare_public_eval.py --download`与`rerun_public_eval_collection.py`：SHA固定的原生来源与虚构文件复现；新增运行不覆盖历史结果。

测试mock隔离外部服务/模型；没有私有索引的业务数据回归显式跳过。可选OCR性能与真实生成不因CPU测试通过而自动合格。

## 分层指标

| 层 | 指标 |
|---|---|
| 解析 | 页面/视觉覆盖、结构/来源正确率 |
| 检索 | File Recall、Evidence Recall@K、Visual Recall@K、MRR |
| 输入 | Gold-in-Snapshot、真实图片对应率、保护关系丢失率 |
| 回答 | Exact Match、列表F1、数值正确率、审核断言覆盖 |
| 引用 | 引用身份有效率、语义支持Precision/Recall、位置正确率 |
| 可信性 | 拒答Precision/Recall、无依据回答率、冲突识别 |
| Agent | 工具选择、漏调用、不必要联网、恢复次数 |
| 工程 | 冷/热态P50/P95、峰值显存、OOM、输入/输出Token |

开放题可人工复核；确定性题优先代码评分。本项目不默认用三裁判多数票，也没有把生产支持审计当作官方评分器。

## 跨文件场景

包含唯一支持+干扰、全部不支持、重复支持、互补支持和冲突。干扰须验证确实不支持问题，不可只凭文件名不同标负例。项目构造标签与官方来源标签分开。

## 控制实验

评测检索器变体时，从相同原始问题/文档运行各自检索并记录各自Snapshot；不能共用已由完整检索器筛好的Snapshot评测检索效果。

单独评测生成模型或校验器时，固定实际输入Snapshot、候选顺序、Prompt、像素与解码参数，才可归因。

## 错误归因

区分解析失败、文件/证据召回遗漏、视觉输入遗漏、模型理解错误和校验失败。普通输入错、Oracle正确证据对，优先修检索；两者错才进一步检查模型/评分。

统计按能力与格式切片，相关多题可同时报告文档宏平均；不能把反复调参的开发题当未见测试，也不能把CPU通过率写成业务准确率。

当前具体校验结果见 [VALIDATION.md](VALIDATION.md)。
