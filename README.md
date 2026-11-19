# kb-mesh

「多租户知识库中台」的 Python 后端实现:Milvus × Neo4j 双库融合检索、LangGraph 在线编排(中断/恢复 + 五层幻觉防御)、Celery 离线索引、评估双闭环与全链路可观测。

需求与设计依据见 [`docs/需求清单.md`](docs/需求清单.md) 与 [`docs/技术实现文档.md`](docs/技术实现文档.md);设计回填见技术文档 v0.2。

## 技术栈

| 层 | 选型 |
| --- | --- |
| Web | FastAPI + SSE(`sse-starlette`) |
| 编排 | LangGraph 1.2 + `AsyncPostgresSaver`(schema `lg`) |
| 存储 | PostgreSQL 16(业务/观测/检查点)、Milvus 2.5(向量 + 双 BM25)、Neo4j 5(图谱) |
| 异步 | Celery(Redis broker) |
| 模型 | 百炼兼容接口:qwen3.8-flash / text-embedding-flash / text-rerank / omni-flash |

## 目录

```
src/kbmesh/
  api/            FastAPI 应用、路由、SSE
  agents/         LangGraph 状态图、节点、五层防御、Runtime/检查点
  retrieval/      三路召回、RRF、图谱五步、融合精排
  ingest/         解析/切片/向量/图谱写入/Celery 管道
  eval/           四层指标、可续跑运行器、LLM judge、在线评估
  observability/  版本化单价表、按租户聚合
  db/             模型、仓储、Alembic 迁移
  security/       令牌、密码、租户中间件
scripts/          能力探针、三段式初始化、种子数据
tests/            验收测试(数据层/隔离/恢复全部打真引擎)
```

## 快速开始

```bash
# 1) 依赖(直连 PyPI 会被掐断,走阿里云镜像)
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ UV_HTTP_TIMEOUT=120 \
  uv pip install -p .venv/bin/python -e ".[dev]"

# 2) 本机 Milvus/Neo4j(只绑 127.0.0.1)
docker compose -f deploy/docker-compose.yml up -d

# 3) 建库/迁移/集合/约束(顺序即上线顺序)
.venv/bin/python scripts/bootstrap_stacks.py

# 4) 测试
.venv/bin/pytest

# 5) Lint
.venv/bin/ruff check src tests scripts

# 6) 种子数据(演示租户 + 管理员)
.venv/bin/python scripts/seed_demo.py

# 7) 后端
.venv/bin/uvicorn kbmesh.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

PG / Redis / 业务 MinIO 由部署环境提供,地址与凭据通过 `.env` 注入;本仓库只在本机 Docker 起 Milvus 与 Neo4j。`KBMESH_DASHSCOPE_API_KEY` 仓库内保持占位 `sk-xxx`,真模型验收只在临时注入环境变量的那一条命令上跑。

## 能力探针

改 schema 或怀疑引擎行为时先跑:

```bash
.venv/bin/python scripts/verify_milvus.py
.venv/bin/python scripts/verify_neo4j.py
.venv/bin/python scripts/verify_checkpoint.py   # LangGraph 检查点 C1–C9
.venv/bin/python scripts/verify_models.py       # 真模型四类 structured 输出
```

## API 面(节选)

- `POST /api/auth/login` — 登录换令牌
- `POST /api/kbs/{kb_id}/documents`、`POST /api/documents/{id}/reindex` — 入库/重索引
- `POST /api/ask` — SSE 问答(route / rewrite_candidates / await_confirm / …)
- `POST /api/threads/{id}/resume` — 确认后续跑(支持断线/重启恢复)
- `POST /api/maintenance/expire-stale` — TTL 到期扫描(租户管理员)
- `GET /api/traces/stats` — 按租户聚合:量/P95/token/成本/拒答率/降级率/路命中
- `GET /api/traces/runs/{id}/online-eval`、`GET /api/online-evals/failures`
- 评估:`POST /api/eval-sets/from-feedback`、`POST /api/eval-runs/{id}/execute`、
  `GET /api/eval-runs/{id}/report`、`GET /api/eval-runs/{id}/regressions`、
  `GET /api/eval-runs/{id}/export.csv`

## 评估与可观测口径

- 四层指标(检索/答案/Agent/系统)在 `kbmesh.eval.metrics`,公式唯一,边界有单测。
- 离线运行器每完成一题提交 `eval_run.cursor_case_id`;中断后续跑不重复计分。
- 报告按 `group_key` 切片,并给出相对基线的逐项差值;劣化超过 `regression_gate` 的样本可导出 CSV。
- 在线评估:格式/引用/安全全量规则 + faithfulness/answer_relevancy 抽样 judge;结果落 `platform.eval_online`。
- 成本:`cost = token_in×单价 + token_out×单价 + rerank 次数×单价`;单价表版本号进 `trace_run.config_snapshot`。

## 安全边界(2026-11-06 安全评审后加固)

- 登录:账号不存在也跑一次 dummy bcrypt,避免时序枚举;bcrypt 放线程池并限并发。失败计数按 `(来源 IP, 账号)` 与账号两个维度在事务内原子累加,超阈值只对失败尝试返回 429;**合法口令永远放行**,不会被刷错密码锁死。单来源 IP 超过 `KBMESH_LOGIN_IP_MAX_FAILURES` 后直接 429、不再执行 bcrypt。`X-Forwarded-For` 仅采信 `KBMESH_TRUSTED_PROXIES` 内的代理,并从右往左取第一个非可信地址。
- 上传:默认单文档上限 50 MiB;ASGI 中间件在 multipart 解析前按 Content-Length/实际字节流拒绝,路由层再分块计数兜底,超限 413;部署时反向代理还应设置同级的 `client_max_body_size`。
- 评估导出:CSV 单元格以 `= + - @` 或制表/回车开头时加单引号,防表格公式注入。
- 图谱路:模型返回的 mentions 最多 5 条、label 最多 128 字符;链接词再按 10 条上限去重截断,避免外部查询扇出。
- 越权执行评估:先写 `role_denied` 审计再抛 403,与其它门禁同口径。

## 里程碑

M1–M4(索引/图谱/融合/多租户)、M5 后端(中断恢复)、M6 后端(评估/成本/降级)已完成;双端前端不在本仓库。
