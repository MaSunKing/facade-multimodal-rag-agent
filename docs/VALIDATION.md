# 公开版本地小规模验证

## 2026-09-17发布更新

| 检查 | 本地结果 | 范围 |
|---|---|---|
| Python语法与CPU回归 | 265通过、54显式跳过、0失败/错误 | 共319项；无模型生成 |
| 进程内HTTP Smoke | 20/20通过 | 真实FastAPI TestClient上传/权限/原图/删除 |
| 前端生产构建 | 通过 | Next.js编译、TypeScript检查、静态页面生成；metadataBase仍有非阻断性提示 |
| 75题结果复算 | 通过 | 问题/ID一致、分子分母、MRR及各能力保留率 |
| 原始文件准备 | 14份官方来源SHA校验通过，旧金标位置绑定有效 | 使用本机SHA匹配原件，没有重新下载或重写金标 |
| 源码/可达历史模式扫描 | 无匹配项 | 不是全面安全证明 |

旧Canonical在本次解析器下重建的Snapshot SHA与历史SHA不同，准备脚本显式记录这一差异；原始文件SHA与金标文档/Chunk绑定仍有效。没有为了让检查通过修改历史freeze manifest。

补齐发布白名单遗漏的`stage_timing`依赖。可选MP-DocVQA训练Adapter不属于本次功能型RAG发布，相关测试明确跳过；企业数据依赖测试也明确跳过，不复制本地索引。分类回归用临时自建taxonomy，HTTP记忆mock支持Graph新增config参数。以上属于发布/测试适配，没有修复本轮发现的单位召回或原生冲突缺陷。

当前公开CPU评测见[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)，接口成功和源码回归通过不能替代模型语义质量评测。以下为上一版本历史验证，不能把其独立Uvicorn、前端构建记录当成此次自动重测结果。

## 2026-09-16历史记录

日期：2026-09-16。范围为发布目录内的代码，不依赖公司原始资料，不加载8B生成模型，不调用付费联网API。

| 检查 | 本地结果 | 含义 |
|---|---|---|
| Python语法 | 通过 | backend/scripts编译检查 |
| CPU回归 | 186通过、53跳过、0失败/错误 | 共239项；业务索引依赖与环境可选项显式跳过 |
| 进程内HTTP Smoke | 20/20通过 | 使用FastAPI TestClient，不是mock上传结果 |
| 独立Uvicorn HTTP Smoke | 20/20通过 | 临时localhost:8001服务，测试结束已停止 |
| 普通问候实际接口 | 返回有效回答，LangGraph轨迹存在 | 确认快捷路径与收尾；模型保持cold |
| 前端生产构建 | 通过 | Next.js编译、TypeScript与静态导出 |
| 源码/历史模式扫描 | 未发现匹配项 | 不打印候选密钥值；不是全面安全证明 |

20项Smoke覆盖：健康/路由、Owner要求、4种格式上传、文件与文字块保留、PDF/Word原图、用户隔离、图像字节一致、Excel双Sheet、Evidence模型结构/稳定性、删除后失效。

其中源码/历史检查由 `audit_public_sources.py --history`重跑；扫描完整性取决于模式和Git可达历史范围。依赖包旧版本有非阻断性警告，未宣称所有平台/版本安装均兼容。

## 本次发现并处理的问题

1. 发布目录遗漏Context Engine、真实阶段执行、错误协议和恢复预算；已同步。
2. 发布文档与代码预算/空闲释放不一致；已更正。
3. 解析器先导入时触发Evidence投影循环依赖；用惰性包导出修复，新增独立进程回归。
4. HTTP记忆测试mock未接受新增config参数；修正测试接口，不改变业务返回。
5. 私有索引依赖测试在空仓库无法执行；显式标注跳过；入库分类测试改用自建临时taxonomy。

当前实际资料均公开，“私有索引依赖”指未随仓库发布的本机业务索引，不表示现有库含保密资料。

## 复现

```bash
python scripts/verify_public_release.py
python scripts/public_smoke.py
python scripts/audit_public_sources.py --history
cd frontend
npm ci
npm run build
```

本地详细报告保存在忽略的 `runtime/public_smoke/`。CI新增CPU与Smoke作业，远端运行状态需以GitHub Actions为准。

**本次没有评测真实模型的业务回答质量、跨文件融合成功率、联网结果质量或生产P95延迟。不能将这些工程校验结果转换为业务准确率。**
