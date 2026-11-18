# AGENTS.md

本仓库是「多租户知识库中台」的 Python 实现(Milvus × Neo4j 双库融合检索、LangGraph 编排、Celery 离线索引、评估闭环)。

需求与设计依据见 [`docs/需求清单.md`](docs/需求清单.md) 与 [`docs/技术实现文档.md`](docs/技术实现文档.md);改动跨模块的行为前先看对应章节,不要从代码反推设计。

## Commands

- 依赖:`UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ UV_HTTP_TIMEOUT=120 uv pip install -p .venv/bin/python -e ".[dev]"`。直连 PyPI 在这台机器上会被掐断,不要退回默认 index。
- 测试:`.venv/bin/pytest`(`asyncio_mode = "auto"`,异步用例不需要装饰器)。跑在 `KBMESH_TEST_DB_URL` 指向的 `kbmesh_test` 上,并且**需要 Milvus 与 Neo4j 容器在跑** —— 数据层与隔离用例打真引擎,连不上就 fail,不要加 skip。
- Lint:`.venv/bin/ruff check src scripts tests` — `line-length = 110`,配置见 `pyproject.toml`。
- 中间件:`docker compose -f deploy/docker-compose.yml up -d`(etcd + Milvus 内部对象存储 + Milvus + Neo4j,全部只绑 `127.0.0.1`)。
- 能力探针(改 schema 或怀疑引擎行为时先跑):`.venv/bin/python scripts/verify_milvus.py`、`.venv/bin/python scripts/verify_neo4j.py`、`.venv/bin/python scripts/verify_models.py`,退出码非 0 表示能力不达标。
- 三段式初始化(PG 迁移 → Milvus 集合 → Neo4j 约束,顺序即 §12 上线顺序):`.venv/bin/python scripts/bootstrap_stacks.py`;只想看现状加 `--check`。
- 单条命令跑迁移:`KBMESH_ENV=test .venv/bin/alembic upgrade head`(test 库)或不带前缀(应用库 `kbmesh`)。`alembic check` 必须报 "No new upgrade operations detected",有漂移就说明模型与库里不一致。

## 中间件落点(硬约定)

- **PG / Redis / 业务 MinIO 由部署环境提供**,地址与凭据通过环境变量注入、不入仓,本机不要为它们再起容器。库 `kbmesh` / `kbmesh_test`,Redis db5/db6,bucket `kbmesh-artifact`。
- **只有 Milvus 与 Neo4j 在本机 Docker 跑**——PG16 不能当向量库用(`pgvector` 与 Milvus 的 BM25/分区键能力不是一回事),Neo4j 也没有可复用的现成实例。
- `deploy/.env` 里的凭据只服务本地容器,已被 `.gitignore` 排除;`milvus-storage` 是 Milvus 的内部后端,和业务 MinIO 无关,不要混着叫。

## 租户与隔离

- 仓储层方法一律收 `ctx: TenantContext`,**绝不接受裸 `tenant_id` 参数**——调用点传的 ID 不可信,ContextVar 里的才可信。
- 四道防线(存储 / 身份来自 token / 服务端注入过滤 / 真实数据验收断言)少一道都不算实现完成;跨租户断言用真数据跑,不用 mock。
- 引擎层的实测约束(会反复踩,细节见技术文档 §4.2 / §4.3):
  - Milvus:`BM25` Function 的输出字段只能建 `metric_type=BM25` 的稀疏索引(写 `IP` 在建索引阶段被拒);三路召回必须 `hybrid_search(AnnSearchRequest…)`,`search(anns_field=[...])` 不合法。
  - Neo4j:关系类型与变长跳数不能参数化(白名单渲染成字面量);遍历谓词必须逐跳带 `tenant_id`(`all(e IN relationships(p) WHERE e.tenant_id = $t)`),只绑起点挡不住漏写属性的边;社区版无可配置分析器,全文只做候选生成且命中不认租户。

## Secrets 与外部服务

- `.env` / `deploy/.env` / `*.key` 永不进仓;口令、API key 不出现在代码、文档、日志、trace 里。**打印连接串也算违规**:脚本只报 `database@host:port`,不要 `print(url)`。
- `KBMESH_DASHSCOPE_API_KEY` 在仓库 `.env` 里保持占位 `sk-xxx`;需要真模型的验收只在那一条命令上临时注入环境变量,且**只跑 test 库**。
- 百炼只读:只调 `Retrieve` / embedding / rerank / chat,不得增删改托管知识库;要写路径的验收一律 mock。

## 文档

`docs/需求清单.md` 与 `docs/技术实现文档.md` 是需求与设计的唯一依据;改动跨模块行为时同步更新对应章节,不要从代码反推设计。
