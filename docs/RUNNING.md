# 运行、配置与安全检查

[首页](../README.md)

## 环境

Python 3.11+、Node 22.13+。Qwen3-VL-8B-Instruct权重自行准备；GPU推理需匹配PyTorch/CUDA、Transformers及显存条件。OCR/PP-Structure/MinerU为可选增强，不承诺所有平台开箱即用。领域原始资料、SQL目录与Embedding索引需要自行构建且拥有使用权。

```bash
python -m venv .venv
# 激活环境后按机器安装匹配的torch，再安装依赖
pip install -r requirements.txt
cp .env.example .env
# 修改本地模型路径和允许前端来源；真实密钥仅写本地.env
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --env-file .env
```

Windows可用[scripts/start_backend.ps1](../scripts/start_backend.ps1)。配置默认embedded模式便于单进程入口；独立常驻服务机制见[model_service.py](../backend/sales/model_service.py)与[model_transport.py](../backend/sales/model_transport.py)，需本地token和服务配置，不能公开监听无鉴权端口。

```bash
cd frontend
npm ci
npm run build
npm run dev
```

本发布使用Next静态导出（output: export），构建产物在frontend/out；生产静态预览可在frontend目录运行`python -m http.server 3000 --bind 127.0.0.1 --directory out`，不要用next start。API允许来源须与预览端口一致，否则健康检查会被CORS拒绝；不要为此改成无条件允许所有公网来源。

frontend根路由直接导出sales工作区，不包括财务助手。公开API地址通过NEXT_PUBLIC_MODEL_API_BASE配置。不要把自己的CloudBase地址或私人远程隧道写死进公共源代码。

## SQL与向量配置

依次进行有权语料解析 → canonical evidence → 分类/层级/关联 → SQLite目录 → BM25与Embedding索引；构建脚本在scripts，具体数据输入参数按脚本--help。Qdrant可选，RAG_VECTOR_BACKEND=qdrant、RAG_QDRANT_URL仅loopback；NPZ兼容路径仍在，不由README假设服务自动安装。[知识库](KNOWLEDGE.md)

## 无模型检查与正式评测分开

```bash
python -m compileall -q backend scripts
python scripts/run_agent_eval_100.py
python -m unittest backend.tests.sales.test_review_admission backend.tests.sales.test_model_contract backend.tests.sales.test_staged_answer_graph
python scripts/verify_design_release.py
```

这些是静态/协议/确定性检查，不是100题语义评测。完整suite中部分测试依赖未随仓库发布的企业原料，应显式跳过或准备授权数据，不能写成空库全通过。

## 安全边界

.env、model_service.token、数据库、私钥、图片票据、客户附件、模型权重、向量索引、日志和结果不提交。用户只能查自己的轨迹，管理员全局接口不能为面试临时取消鉴权。公网部署需鉴权、上传限额、Origin限制和HTTPS；本项目不随文档自动部署远程服务。

本发布快照与正在运行的开发后端隔离，更新GitHub无需重启生产模型或使用户附件失效。只有实际业务后端代码变更才按项目工作流检查active task并安全重启。
