# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

多租户知识库中台后端:FastAPI + LangGraph 在线编排 + Celery 离线索引,Postgres 是权威存储,Milvus 与 Neo4j 是可重建的检索投影。核心能力是三路向量/BM25 召回 + 图谱路五步检索 + RRF 融合精排、问题级人工确认与断点恢复、五层幻觉防御、四层评估指标与按租户可观测。双端前端不在本仓库。

设计与需求的唯一依据是 `docs/需求清单.md` 与 `docs/技术实现文档.md`;仓库级的命令、租户隔离红线和引擎实测约束见 `AGENTS.md`。改跨模块行为前先看对应章节,不要从代码反推设计。

## 常用命令

### 依赖

```bash
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ UV_HTTP_TIMEOUT=120 \
  uv pip install -p .venv/bin/python -e ".[dev]"
```

### 中间件与初始化

```bash
# Milvus / Neo4j 在本机 Docker 跑;PG / Redis / 业务 MinIO 由 .env 指向的部署环境提供
docker compose -f deploy/docker-compose.yml up -d

# 三段式初始化:PG 迁移 → Milvus 集合 → Neo4j 约束(测试库加 KBMESH_ENV=test)
.venv/bin/python scripts/bootstrap_stacks.py
.venv/bin/python scripts/bootstrap_stacks.py --check
KBMESH_ENV=test .venv/bin/python scripts/bootstrap_stacks.py
```

### 测试与 lint

```bash
KBMESH_ENV=test .venv/bin/pytest
# 单条用例
KBMESH_ENV=test .venv/bin/pytest tests/test_rrf.py::test_score_matches_the_formula_in_the_doc -q

.venv/bin/ruff check src scripts tests
```

测试跑在 `KBMESH_TEST_DB_URL` 指向的 `kbmesh_test` 上,并需要 Milvus 与 Neo4j 容器可达;数据层与隔离用例打真引擎,连不上会 fail 而不是 skip,不要加 `skip`。

### 数据库迁移

```bash
KBMESH_ENV=test .venv/bin/alembic upgrade head
KBMESH_ENV=test .venv/bin/alembic check   # 必须输出 No new upgrade operations detected
```

### 能力探针

改 schema、升级引擎或怀疑模型能力时先跑:

```bash
.venv/bin/python scripts/verify_milvus.py
.venv/bin/python scripts/verify_neo4j.py
.venv/bin/python scripts/verify_checkpoint.py
.venv/bin/python scripts/verify_models.py   # 需要临时注入真实 DashScope key;仓库内是占位 sk-xxx
```

### 运行服务

```bash
uvicorn kbmesh.api.app:create_app --factory --host 127.0.0.1 --port 8000
celery -A kbmesh.core.celery_app:celery_app worker -Q kbmesh.ingest -l info
```

## 架构

### 分层与请求链路

```text
HTTP
  → TenantContextMiddleware   # security/middleware.py:验签一次,把 TenantContext 放进 ContextVar
  → api/routers/*             # auth / documents / chat(SSE) / feedback / evals / traces
  → api/deps.py               # 每请求组装 Runtime(SaverHandle + 编译好的图 + GraphDeps)
  → agents/                   # LangGraph 拓扑、节点、五层防御、checkpoint
  → retrieval/                # 三路召回 + 图谱路 + RRF + 融合精排
  → storage/{milvus,neo4j,s3} # 检索投影与对象存储
  → db/                       # Postgres 权威表(platform schema)与仓储
```

`create_app(deps=…, saver=…)` 是测试注入点;生产路径 `agents/wiring.py` 组装唯一的 `GraphDeps`。中间件顺序有讲究:体积上限在外层,身份在内层。

### 在线问答

- 状态图固定 10 个节点:`route_intent → rewrite_query → confirm_interrupt → retrieve_dual → dedupe → rerank_fused → assess_relevance → retry_targeted / refuse / generate_cite`,拓扑见 `agents/graph.py`。
- 人工确认用 `interrupt()`;检查点由 `AsyncPostgresSaver` 落在 `lg` schema,`ask` / `resume` / 流式入口在 `agents/runtime.py`。检查点表由 LangGraph 自建,不归 Alembic;`bootstrap_stacks.py` 只负责先建 schema。
- 五层幻觉防御在 `agents/guards.py`,全是纯函数(不 import langgraph、不碰网络);阈值由节点从 Settings 取一次传入,避免出现两套 θ。
- 节点身份一律从 ContextVar 现取,**不写进 State**:检查点会跨进程恢复,身份不能由旧 checkpoint 决定。

### 检索

- `retrieval/routes.py`:正文 BM25、章节标题 BM25、dense 三路各自独立 RPC;单路开关、单路指标、单路降级都在这里成立。租户过滤只由 `combined_filter(ctx)` 生成。
- `retrieval/graph_route.py`:实体抽取 → 实体链接 → 受控跳数邻域扩展 → 子图线性化 → 回落原文切片;图谱结果必须带 `chunk_id` 指针。
- `retrieval/rrf.py` 做 RRF(`k=60`),`retrieval/fusion.py` 按 `chunk_id` 去重、精排,并回 Postgres 领权威正文;领不到正文的命中直接丢。
- `retrieval/contracts.py` 的 `Hit` / `Citation` 是跨模块契约;不要用文本 hash 做去重键。

### 离线索引

- Celery chain 六阶段:`parse → caption → embed → extract → graph → link`(`ingest/pipeline.py` 的 `PIPELINE_STAGES`),`ingest/tasks.py` 只是薄壳。
- 每个阶段遵守三段式:读 → 无锁外部 I/O(模型/Milvus/Neo4j/S3)→ 短事务写;事务里绝不调模型。
- 幂等靠切片键 upsert、向量 upsert/删孤儿、图谱 MERGE,以及 Redis 租约(`ingest/lease.py`);任何阶段可原样重跑。
- `ingest/graph_extract.py` 的两道白名单校验共用同一份代码,但必须在抽取层与写入层各跑一次;关系类型/跳数不可参数化,白名单是渲染 Cypher 字面量的唯一来源。

### 评估与可观测

- `eval/metrics.py` 是四层指标的唯一计算口径;`eval/runner.py` 用 `cursor_case_id` 支持中断续跑,并输出分组报告、基线差值与 CSV;`eval/runtime_case.py` 把真实 Runtime 桥接成评估器。
- 在线评估(`eval/online.py` + `eval/tasks.py`):格式/引用/安全全量规则,faithfulness/answer_relevancy 按 `KBMESH_ONLINE_EVAL_SAMPLE_RATE` 抽样,异步执行不进主链路。
- `observability/pricing.py` 的单价表带版本号,历史成本不随调价重算;`observability/aggregate.py` 按租户聚合 P50/P95、token、成本、拒答率、降级率与路命中。

### 数据与租户隔离

- `db/models.py` 的业务表都在 `platform` schema,每张表带 `tenant_id NOT NULL` 与 `(tenant_id, …)` 复合索引;Milvus/Neo4j 是投影,坏了可以从 PG 重放。
- 四道防线:存储层租户列/分区键、身份只来自验签令牌、查询层服务端强制注入、验收层用真实多租户数据断言。仓储方法一律收 `ctx: TenantContext`,不接受裸 `tenant_id`;worker 身份从权威 `document` 行构造。
- 角色:`platform_admin` / `tenant_admin` / `reader`;`indexer` 只在 worker 内部使用,不进 `ALL_ROLES`。公共路径只有 `/healthz`、`/readyz`、`/api/auth/login`、`/api/auth/token`。

### 引擎硬约束

- Milvus:BM25 Function 的输出字段只能建 `metric_type=BM25` 的稀疏索引;三路召回必须 `hybrid_search(AnnSearchRequest…)`,`search(anns_field=[...])` 不合法。
- Neo4j:关系类型与变长跳数不能参数化(白名单渲染成字面量);遍历谓词必须逐跳带 `tenant_id`;社区版全文索引的命中不认租户,必须在服务端再收口。

## 关键不变量

- Postgres 是权威,Milvus/Neo4j 可重建;任何引擎字段都必须能从 PG 恢复。
- `TenantContext` 只能由 `security/tokens.py` 从验签 claims 生产;请求体、State、检查点都不能提供身份。
- 测试与 bootstrap 打真引擎,不用 mock 替代数据层/隔离验收。
- 迁移改动后必须让 `alembic check` 无漂移;检查点表不能混进 `platform` schema。
- 口令、API key、完整连接串不写进代码、文档、日志或 trace;`.env` / `deploy/.env` 永不进仓。
