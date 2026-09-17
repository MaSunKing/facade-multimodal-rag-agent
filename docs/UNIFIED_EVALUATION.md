# 统一条件评测

保留原来的75道唯一开发题并全部重跑，不将历史不同预算结果平均，不删除失败题，不使用改写变体。这不是冻结未见测试集。

## 协议

- 候选召回24个窗口，5000估算Token；文本768窗口、96重叠。
- 本地Qwen3-VL-8B-Instruct真实Tokenizer，完整Prompt最多5000 Token，系统指令一致。
- 多证据题：Top-5完整覆盖金标支持集；MRR为首次完整支持的倒数排名。
- 数值：源行目标值、实体、显式单位绑定，不在全文任意找数字。
- 条件：限制/否定文本保留与保护标记分别报告。
- 冲突：原生文件5题与人工Evidence控制5题注明输入类型，只检测目标双方冲突组。
- 70条原生文件加5条人工Evidence控制，后者不经过解析，不宣称同一解析链路。
- 75/75打包预算有效，运行错误0。

## 结果

Recall@5：38/45（84.44%）；MRR：0.5733。数值＋单位10/10；条件文本10/10、保护标记8/10；原生冲突检测5/5、双方保留5/5，人工控制另为5/5。

7道检索失败题保留：`retrieval_001/002/003/004/007`、`fresh_en_021/025`。缺保护标记为`extra_condition_04/05`。

旧评分对部分多证据题使用“任意命中”。本次修正为完整支持，历史MRR不能与本次直接比较，不能选较高的中间输出替换最终结果。原始题目和源文件SHA不变，各次尝试留在本地，以最终完整口径为准。

## 补测，不膨胀主分母

| 轨道 | 范围 | 结果 |
|---|---|---|
| 企业混合检索 | 3303条指纹一致索引，5个已知金标探针 | 真实混合执行5/5，指定金标Top-5为4/5 |
| OCR | 一页人工合成扫描PDF，无可选文字 | 实体、17 mm与原图保留通过，候选进入召回 |
| 非冲突控制 | 5组原生TXT | 等值、单位换算、不同实体/指标、未绑定实体：误报0/5，双方保留5/5 |

企业探针是当前本地知识库执行验收，不是公开Held-out Benchmark。完整企业索引不随仓库提供，不承诺读者直接复现这5个指定ID。OCR只验证一张清晰页，不代表复杂扫描精度，不评颜色/图表语义。所有轨道未运行8B生成、Planner或LLM Judge。

## 运行

无需模型或下载原始文件即可校验已发布数字、题目和代码指纹：

```bash
python scripts/verify_unified_publication.py
```

该命令重算报告数字，不重新检索，也不把指标算术一致性当作源标注正确性。

准备原始文件及本地Tokenizer后：

```bash
python scripts/prepare_public_eval.py --download
python scripts/run_unified_public_eval.py --track offline --tokenizer-path /path/to/Qwen3-VL-8B-Instruct --output outputs/public_eval/new_run
python scripts/verify_unified_public_eval.py --output outputs/public_eval/new_run
python scripts/run_public_negative_conflict_smoke.py --output outputs/public_eval/negative_new_run
python scripts/run_public_ocr_smoke.py --file evaluation/ocr_smoke/synthetic_scanned.pdf --output outputs/public_eval/ocr_new_run.json
```

程序拒绝覆盖既有冻结目录。原始文件下载/许可规则见`evaluation/README.md`。企业混合检索需本地授权索引、匹配Dense索引及Embedding/Reranker，使用`--track hybrid`及新目录。CPU机器需要改变检索设备，因此不能当成本机速度复现。

公开复现入口重建本地Canonical快照并审计固定金标绑定，而不要求仓库提供官方完整解析正文。历史快照与新快照SHA分别记录，不篡改历史冻结Manifest。Tokenizer路径可通过参数指定，不需要复制8B生成权重。

冻结评测文件通过Git属性保留原始字节，复跑不重写既有题目文件。发布代码指纹统一按LF换行计算，避免Windows/Linux检出时仅换行变化导致误报；原始执行代码SHA仍作为当次执行记录保留。

默认开启混合检索，仍保留16GB显存保护，8B热态可能词法降级。必须检查实际通道，不能只看`ready=true`。本次冷态5次真实执行，首次5.28秒、后续0.54—0.64秒，非端到端延迟。测试进程结束后GPU回到约1.25GB基线，未加载8B。

源工作区代码SHA与公开命名空间转换后的SHA分别记录，不声称字节完全相同。公开结果不包含官方整篇正文、完整Context或本地秘密配置。

正在运行的本机服务另经`/api/copilot/retrieve`验收：混合策略与匹配索引已启用，实际CUDA Embedding/Reranker已加载，检索接口约3.46秒，8B未加载。小检索模型在服务中缓存；8B启动时由现有资源调度释放，不能把本次冷态速度当成热态保证。
