# 虚构文件示例

运行 `python scripts/public_smoke.py`，在 `runtime/public_smoke/documents/` 生成：

- PDF：两页虚构产品/版本说明，含真实嵌入图片。
- DOCX：标题、说明、检查表及嵌入图片。
- XLSX：两张Sheet分别记录20mm和18mm，展示来源差异。
- PNG：Panel/Support简图，明确不是施工节点设计。

所有内容为项目自建虚构测试材料，不代表真岩石产品参数、检测结果或施工建议。解析校验不会证明模型已理解冲突；生成能力需另行测试。

可尝试的问题在 `demo_queries.json`；这里只提供场景断言，不冒充已运行的模型成绩。

另有[3个真实本地8B案例](real_agent_demo/README.md)，包含原件、实际执行/回答/引用及修复前失败记录。其内容有明确缺口，不作为准确率评测；`demo_queries.json`仍保留为原来的未运行场景清单。
