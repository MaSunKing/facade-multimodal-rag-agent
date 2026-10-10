# 100题评测设计（尚未完成正式评测）

[首页](../README.md) → 本页 → [原题库](../data/sales/evaluation/agent_eval_100_v1/questions.jsonl) · [冻结清单](../data/sales/evaluation/agent_eval_100_v1/manifest.json)

项目持续迭代，正式端到端语义评测尚待执行与独立判分，本版不提供准确率、通过率或历史结果。结构检查、协议合规、关键词命中和HTTP成功不等于正确回答。

## 为什么分成100题

| 分组 | 数量 | 检验原因 |
|---|---:|---|
| company | 40 | 核心六类、产品层级、型号/属性、地域项目、施工/节点/标准、集合与配图 |
| attachment | 20 | 客户文字、表格、图片、长文、多文件比较，防止套用企业身份 |
| multiturn | 20 | 省略、追加、纠正、取消、切换来源；验证任务状态不等于全历史拼接 |
| web | 15 | 实时/公开信息、多对象查询、日期范围、缓存、来源质量及不联网边界 |
| fault | 5 | 协议、范围、预算与恢复的确定性边界，不代表模型语义任务 |

强调企业场景而非均分，是因为它是主要使用场景；保留附件和多轮较高比例，是为了发现单轮纯文字测试看不到的覆盖/归属问题。Web数量受费用与时效影响，单独分组可控执行。100题是工程诊断集，不是行业通用benchmark，也不具备统计代表性声明。

企业40题内部为产品10、项目8、施工6、节点6、标准5、企业5，偏重产品层级、集合与项目地域，同时覆盖技术资料和身份类，避免只测产品参数lookup。

## 字段和判分

id/group/question固定题目身份；files引用自造fixture；turns描述多轮要求；expected_sources检查规划与真实执行；rubric/expected_points用于人工判分；gold_status保留human_review_required或题目实际状态；timeout_seconds=1200。

拟人工检查：任务覆盖、范围正确、事实支持、图片未知项保持、关系/单位/条件完整、是否诚实说明缺口、是否泄露内部字段。公司和Web题必须对照实际来源，不能只匹配词语；多轮要核对最终有效要求，不强求原计划字段逐字一致。自动脚本不调用额外裁判模型替代独立语义审核。

fixture包括a.txt/b.txt、long.txt、records.csv、shapes.png，均为自造内容；不上传客户原图。公司题需要另行获得有权使用的企业语料；公开repo不含完整知识库，因此不承诺这些题在空库下可答。

## 自动运行

```bash
python scripts/run_agent_eval_100.py                     # 只验证冻结哈希与100条唯一ID
python scripts/run_agent_eval_100.py --run --group company
python scripts/run_agent_eval_100.py --run --group attachment
python scripts/run_agent_eval_100.py --run --group multiturn
python scripts/run_agent_eval_100.py --run --group web    # 显式联网，可能消耗费用/额度
python scripts/run_agent_eval_100.py --run --group fault
python scripts/run_agent_eval_100.py --run --ids AG100-001
```

[run_agent_eval_100.py](../scripts/run_agent_eval_100.py)每题总时限20分钟，包含上传、等待、处理与多轮，不是每个模型调用各20分钟；顺序执行、结果逐题落盘、冻结数据检查、同输出目录锁和断点跳过已记录题目。单题错误通常记录后继续；超时须确认取消并退出，无法确认worker退出则停止，不能让两题叠加抢GPU。启动时不取消用户任务，忙时拒绝或--wait-for-idle等待。fault按test_module执行确定性测试，与模型语义统计分开。

记录真实节点Token/耗时/重试/错误、Final和Owner轨迹；父子Token不能双加，未知统计不假写零。semantic_accuracy默认null，之后独立人工审核方可报告。输出位于outputs且被Git忽略，分享前要另做脱敏和来源许可审查。

## 设计局限

题库有固定日期，例如2026-10-08，它们是范围条件，不应随着机器“今天”自动替换。Web事实变化时需固定运行时间、来源快照与缓存状态，不能直接横比不同时点。厂商宣传与检测结论分开；未来增加无支持、矛盾、同名文件、图片身份未知及长Context分层样本，仍要避免针对固定题目硬编码。
