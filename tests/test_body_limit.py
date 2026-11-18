"""ASGI 请求体上限:在 multipart 解析之前就挡掉超大 body。"""
from __future__ import annotations

import httpx
from fastapi import FastAPI, Request

from kbmesh.api.body_limit import MaxBodySizeMiddleware


def _app(limit: int = 10) -> FastAPI:
    app = FastAPI()
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=limit)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"n": len(await request.body())}

    return app


async def _client(limit: int = 10) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(limit)), base_url="http://t")


async def test_content_length_over_limit_is_rejected_before_route() -> None:
    async with await _client(10) as client:
        resp = await client.post("/echo", content=b"01234567890")
        assert resp.status_code == 413
        resp = await client.post("/echo", content=b"0123456789")
        assert resp.status_code == 200 and resp.json() == {"n": 10}


async def test_streamed_body_over_limit_is_rejected() -> None:
    async def chunks():
        yield b"12345"
        yield b"678901"

    async with await _client(10) as client:
        resp = await client.post("/echo", content=chunks())
        assert resp.status_code == 413
