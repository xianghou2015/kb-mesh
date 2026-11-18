"""生产侧 GraphDeps 的唯一组装点:API 与评估 worker 用同一套引擎依赖。

测试注入自己的 deps;生产不许两处各拼一份 —— Milvus collection / Neo4j driver / 在线评估
钩子漏一个,表现都是「主流程能跑但某一路永远降级」,很难在日志里看出来。
"""
from __future__ import annotations

from kbmesh.agents.nodes import GraphDeps
from kbmesh.agents.trace import TraceWriter
from kbmesh.core.config import Settings, get_settings
from kbmesh.core.gateway import Gateway
from kbmesh.db.base import get_session_factory
from kbmesh.storage.milvus import get_client
from kbmesh.storage.neo4j import get_driver


def build_graph_deps(
    settings: Settings | None = None, *, online_eval=None, with_engines: bool = True
) -> GraphDeps:
    settings = settings or get_settings()
    factory = get_session_factory()
    client = None
    driver = None
    if with_engines:
        # 引擎连不上不应让整个 API 起不来:检索层会在路级记 degraded,这是 §10 的降级形态。
        try:
            client = get_client()
        except Exception:  # noqa: BLE001
            client = None
        try:
            driver = get_driver()
        except Exception:  # noqa: BLE001
            driver = None
    return GraphDeps(
        factory=factory,
        gateway=Gateway(settings),
        driver=driver,
        client=client,
        collection=settings.milvus_collection if client is not None else None,
        settings=settings,
        trace=TraceWriter(factory),
        online_eval=online_eval,
    )


def close_graph_deps(deps: GraphDeps) -> None:
    """释放非 DB 引擎;数据库连接由各自的 pool/engine 管。"""
    client = deps.client
    if client is not None and hasattr(client, "close"):
        client.close()
    driver = deps.driver
    if driver is not None and hasattr(driver, "close"):
        driver.close()
