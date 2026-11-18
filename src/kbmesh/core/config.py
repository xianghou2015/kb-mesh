"""Runtime configuration.

All knobs arrive from the environment (prefix `KBMESH_`) or `.env`; nothing here should
hard-code a credential. Empty/absent model keys are legal: the retrieval and orchestration
logic must stay runnable offline against mocks, which is how the acceptance suite works.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Flags(BaseSettings):
    """Degradation switches — 见技术实现文档 §10 开关矩阵。

    必需项(意图路由 / 正文 BM25 / 检查点持久化)不在此列,任何情况下不可关闭。
    """

    model_config = SettingsConfigDict(env_prefix="KBMESH_FEATURE_", extra="ignore")

    route_heading: bool = True
    route_graph: bool = True
    rerank: bool = True
    crosscheck: bool = True
    online_eval: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KBMESH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    secret_key: str = ""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_ttl_min: int = 120
    #: 登录失败节流:窗口内同一账号或同一来源 IP 达到阈值后返回 429。
    login_max_failures: int = 5
    #: 单来源 IP 的预检上限:达到后直接 429,不再跑 bcrypt(防单 IP 在线爆破)。
    login_ip_max_failures: int = 20
    login_window_seconds: int = 300
    #: 只有来自这些反向代理的 X-Forwarded-For 才可信;留空 = 只信 socket 对端地址。
    trusted_proxies: str = ""
    #: 单文档上传上限;ASGI 中间件在解析表单前按字节流兜底,路由层再做一次分块计数。
    max_document_bytes: int = 50 * 1024 * 1024

    # --- data stores ---
    db_url: str = ""
    test_db_url: str = ""
    #: The PG deployment keeps `vector` in schema `extensions`, and the server default
    #: search_path is `"$user", public` — so a `vector` column does not resolve unless the session
    #: opts in. Setting it per connection instead of per role/database: a database-level search_path
    #: would silently change how other sessions resolve names.
    pg_search_path: str = "platform, lg, extensions, public"
    checkpoint_schema: str = "lg"
    redis_url: str = ""
    celery_broker_url: str = ""
    celery_result_backend: str = ""
    #: 离线阶段的默认队列;金丝雀重索引时把它改成一个只属于若干 worker 的队列。
    celery_queue: str = "kbmesh.ingest"

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_db: str = "default"
    milvus_collection: str = "kb_chunk_index"
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    oss_endpoint: str = ""
    oss_access_key_id: str = ""
    oss_secret_access_key: str = ""
    oss_bucket: str = "kbmesh-artifact"
    oss_addressing: Literal["path", "virtual"] = "path"

    # --- model gateway ---
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.8-flash"
    router_model: str = "qwen3.8-flash"
    vision_model: str = "qwen3.8-omni-flash"
    embedding_model: str = "qwen3.7-text-embedding-flash"
    #: 单请求条数上限:不同档位不一样,批大了整批失败比逐条重试更贵。
    embedding_batch_size: int = 10
    embedding_dim: int = 1024
    rerank_model: str = "qwen3.7-text-rerank"
    #: 实测:qwen3.8-flash 默认开思考,2 字回复计费 29 个 reasoning token、in 从 30 涨到 66。
    #: 意图路由每条问题都要跑一次,不显式关掉就是全线延迟与成本翻倍,所以默认 False。
    llm_enable_thinking: bool = False
    llm_max_concurrency: int = 4
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- retrieval knobs ---
    top_k_route: int = 50
    top_k_graph: int = 30
    top_k_fusion: int = 30
    top_n_context: int = 8
    rrf_k: int = 60
    theta_rel: float = 0.35
    theta_answer: float = 0.45
    max_targeted_retry: int = 2
    graph_max_hops: int = 2
    graph_max_edges: int = 30
    pending_confirm_ttl_min: int = 30
    online_eval_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    regression_gate: float = -0.02

    # --- observability ---
    trace_persist: bool = True

    flags: Flags = Field(default_factory=Flags)

    @field_validator("graph_max_hops")
    @classmethod
    def _hops_whitelist(cls, v: int) -> int:
        # 变长路径不可参数化:跳数必须落在白名单里,而不是被拼进 Cypher
        if v not in (1, 2, 3):
            raise ValueError("graph_max_hops must be one of {1,2,3} (Cypher whitelist)")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def _dim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding_dim must be positive")
        return v

    @model_validator(mode="after")
    def _prod_secrets_strong(self) -> Settings:
        # 短于 32 字节的 HMAC 密钥可被离线爆破(RFC 7518 §3.2);dev/test 图省事允许,prod 拦下。
        if self.env == "prod" and len(self.jwt_secret) < 32:
            raise ValueError("KBMESH_JWT_SECRET 在 prod 至少 32 字节")
        if self.env == "prod" and not self.secret_key:
            raise ValueError("KBMESH_SECRET_KEY 在 prod 必填")
        return self

    @property
    def is_test(self) -> bool:
        return self.env == "test"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """测试专用:改过环境后要重建配置。

    `get_settings.cache_clear()` 在 `pytest.mark.parametrize` / monkeypatch 下会与 pydantic
    的类属性 patch 打架,重建成新实例再清缓存才是确定的。
    """
    get_settings.cache_clear()
