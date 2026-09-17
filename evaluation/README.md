# 公开开发评测

当前主结果为[统一条件75题](results/unified_75_v1.json)，配置与计分规则见[统一评测协议](../docs/UNIFIED_EVALUATION.md)。同批题全部按5000 Token打包预算重跑，多证据题按完整支持评分；另补真实混合检索、OCR与负冲突控制，分别统计，不加入主集合分母。以下原始入口和`results.json`保留为历史记录。

75道唯一问题统一编目：检索45、数值10、条件10、冲突10。题目和失败结果都公开，不替换失败题以追求某个分数。不包含70道旧问题改写变体，也不把历史模型生成或公司业务评测混入本报告。

标签为项目根据原文编写，不冒充数据集官方问答。原生来源包含官方PDF、Word、Excel和HTML；虚构文件和人工Evidence控制题均有出处标记。

第三方原生文件的再分发许可尚未核验，因此只发布URL、SHA256及标签，不发布这些文件或整份解析文本。项目自行生成的虚构文件随仓库提供。数据许可与代码MIT许可分开。

## 无需下载或模型的结果校验

```bash
python scripts/verify_evaluation_results.py
```

此命令独立按75条结果重新计算分母、命中数、MRR和各保留率，并检查问题/ID对应。

## 原始链路复现

```bash
# 下载官方原生来源，SHA变化会报错，不修改冻结标签
python scripts/prepare_public_eval.py --download
# 使用本地Qwen3-VL-8B tokenizer；不加载生成权重
python scripts/rerun_public_eval_collection.py --tokenizer-path /path/to/Qwen3-VL-8B-Instruct
```

准备过程重新解析旧原生文件并验证金标位置绑定，完整Canonical保存在忽略的runtime目录。解析器版本、PDF可选工具和源站文件变化可能影响复现；历史SHA/标签不会被自动重写。

新运行结果保存于`runtime/public_eval`，不会覆盖[已发布历史汇总](results/results.json)。当前汇总由三个历史运行组成，不是同一预算下的一次统一重跑。旧题预算包括650/1800，新30题1800，新15题5000；逐题预算和运行出处均保留。

## 输出索引

- [75题及标签](public_eval_75.json)
- [75题结果与口径](results/results.json)
- [15题1800预算压力对照](results/extra_15_pressure_1800.json)
- [完整指标解释与已知问题](../docs/EVALUATION_RESULTS.md)

以上是CPU附件词法检索与Context/tokenizer打包开发回归，不是完整混合企业RAG、Agent工具选择、OCR、模型视觉回答或业务准确率评测。
