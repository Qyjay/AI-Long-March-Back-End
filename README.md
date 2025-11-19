# AI陪伴下的重走长征路 · 后端（FastAPI）

一个基于 FastAPI 的后端服务，提供长征路线与节点数据查询、用户进度保存，以及支持 RAG（检索增强生成）的聊天能力（含 SSE 流式输出）。项目已完成可发布到 GitHub 的工程化优化：统一配置与日志、路由拆分、数据库统一至 SQLAlchemy、Alembic 迁移、基础测试。

## 主要特性
- FastAPI 提供 REST 与 SSE（Server-Sent Events）接口
- RAG 子系统使用远程 API（DashScope 兼容接口），通过环境变量配置模型与地址
- 统一配置：支持 `.env` 与环境变量（不提交密钥）
- 统一日志：`loguru` 结构化日志
- 数据层：SQLAlchemy ORM，Alembic 迁移
  

## 目录结构
```
后端/
├─ App/
│  ├─ api/routers/            # 路由模块
│  │  └─ chat.py              # 聊天与RAG接口（SSE）
│  ├─ core/                   # 核心设施
│  │  ├─ config.py            # Pydantic Settings 配置读取
│  │  └─ logging.py           # 日志初始化（loguru）
│  ├─ db/session.py           # SQLAlchemy 会话封装
│  ├─ models.py               # ORM 模型
│  ├─ ChatModel.py            # DashScope RAG 实现（DOCX源）
│  ├─ init_database.py        # 表初始化（保留，现推荐用Alembic）
│  ├─ main.py                 # 应用入口（聚合路由与数据接口）
│  └─ services/rag/dispatcher.py # RAG选择器（dashscope/ollama）
├─ alembic/                   # Alembic 迁移配置
├─ tests/                     # 基础测试用例
├─ requirements.txt           # 依赖清单
└─ .gitignore
```

## 快速开始（Windows 本地）
- 准备 Python 3.11 虚拟环境并安装依赖：
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- 创建 `.env`（不要提交到仓库）：
```
# 基础
LOG_LEVEL=INFO
CORS_ORIGINS=*

# 数据库（示例：本地Postgres或Neon等）
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/app

# LLM 与 RAG
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=你的DashScope密钥
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# RAG 文档与索引目录（默认即可）
RAG_DOCX_PATH=App\重走长征路.docx
RAG_JSON_PATH=data.json
FAISS_DIR=./faiss_index
```
- 运行服务：
```
uvicorn App.main:app --host 0.0.0.0 --port 8000
```
- 打开浏览器查看交互文档：`http://localhost:8000/docs`

 

## 配置说明
- 配置在 `App/core/config.py` 中定义，统一通过 `get_settings()` 读取
- 关键变量：
  - `DATABASE_URL`：数据库连接串
  - `CORS_ORIGINS`：CORS 允许来源，逗号分隔（生产建议按域严格配置）
  - `LOG_LEVEL`：日志级别（`DEBUG/INFO/WARN/ERROR`）
  - `LLM_PROVIDER`：`dashscope` 或 `ollama`
  - `DASHSCOPE_API_KEY`：DashScope API 密钥（不要提交到仓库）
  - `OPENAI_BASE_URL`：DashScope 兼容模式 Base URL（默认已配置）
  - `RAG_DOCX_PATH`、`RAG_JSON_PATH`、`FAISS_DIR`：RAG的文档与索引目录

## API 概览
- 健康检查：
  - `GET /` 返回服务状态（`App/main.py:414`）
- 数据接口（`App/main.py`）：
  - `GET /nodes` 获取节点列表（`App/main.py:88`）
  - `GET /nodes/{node_id}` 获取单个节点（`App/main.py:97`）
  - `GET /route` 获取路线信息（`App/main.py:106`）
  - `POST /progress` 保存用户进度（`App/main.py:115`）
- 聊天与RAG（SSE，`App/api/routers/chat.py`）：
  - `POST /chat` 通用聊天（`App/api/routers/chat.py:12`）
  - `POST /chat/explanation` 剧情决策讲解（`App/api/routers/chat.py:68`）
  - `POST /chat/poem` 诗词赏析（`App/api/routers/chat.py:131`）
  - `POST /chat/qa` 问答（`App/api/routers/chat.py:195`）

## SSE 流式接口说明
- 响应类型：`text/event-stream`
- 事件格式统一，示例：
```
// 内容块
{"type":"content","content":"局部文本","full_content":"累计文本"}
// 开始
{"type":"start","context_docs":[...]} 
// 完成
{"type":"complete","full_content":"...","usage":{"prompt_tokens":...,"completion_tokens":...,"total_tokens":...},"context_docs":[...]}
// 错误
{"type":"error","content":"","meta":{"message":"错误信息"}}
```
- 前端处理：逐行读取 `data: {json}\n\n`，解析 JSON 后根据 `type` 渲染

## RAG 配置与索引
- 切换实现：在 `.env` 设置 `LLM_PROVIDER=dashscope`（DOCX源，`App/ChatModel.py`）或 `LLM_PROVIDER=ollama`（JSON源，`app.py`）
- 索引目录：默认 `./faiss_index/`（已在 `.gitignore` 中忽略）
- 文档源：
  - DashScope：`App/重走长征路.docx`
  - Ollama：`data.json`

## 数据库与迁移
- ORM 模型：`App/models.py`
- 会话封装：`App/db/session.py`
- 迁移：已配置 Alembic
```
# 生成迁移（自动对比模型变化）
alembic revision --autogenerate -m "init"
# 应用迁移到最新
ailembic upgrade head
```
- 初次建表也可使用保留脚本：`python App/init_database.py`

## 运行测试
```
pytest -vv
```
- 已提供基础用例：
  - `tests/test_health.py` 健康检查
  - `tests/test_chat_routes.py` 聊天路由（验证 SSE 200 响应）

## 安全与发布建议
- 切勿提交 `.env` 与任何密钥到仓库
- 生产环境将 `CORS_ORIGINS` 配置为具体域名列表
- 视需求启用认证（项目已包含 `python-jose`、`passlib` 依赖，后续可加 JWT）

 

## 许可证
- 建议选择开源许可证（如 MIT）。如需，我可以为项目添加 `LICENSE` 文件。

## 致谢
- FastAPI、LangChain、FAISS、DashScope / Ollama、SQLAlchemy、Loguru、Alembic 等开源项目。