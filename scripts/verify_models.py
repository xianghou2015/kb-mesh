"""百炼模型网关实测:确认 chat / embedding / rerank 三个档位真的能调,并把向量维度钉下来。

为什么要跑:Milvus collection 的 `dense` 维度在建集合时就定死,新旧 embedding 模型的向量空间
不互通 —— 维度或语义猜错,代价是整库重放。rerank 同理:compatible-mode 里并没有 rerank,
原生接口的 URL 与请求体形状要用一次真实调用确认,不要靠记忆写。

跑法(key 只在命令行临时注入,不进仓、不进 .env):
    export KBMESH_DASHSCOPE_API_KEY=...
    .venv/bin/python scripts/verify_models.py

全部是只读推理调用(chat/embedding/rerank),不碰托管知识库的任何写接口。

探针:
    M1 chat(qwen3.8-flash;路由与生成同一档,只跑一次最小补全)
    M2 embedding(维度必须等于 KBMESH_EMBEDDING_DIM,且近义 cos > 异题 cos)
    M3 rerank(两种接口形状逐一试,记下真正可用的那个)
    M4 omni 视觉(qwen3.8-omni-flash 读一张程序生成的中文文字图;确认是否必须流式)
    C7 structured(编排层四类结构化输出各一条真调用;strict json_schema + 带 default 的数组字段
       到底给不给得出合法 JSON —— 单测的 FakeGateway 回答不了这个问题)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx
from openai import OpenAI

from kbmesh.core.config import get_settings

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def probe_chat(client: OpenAI, model: str) -> None:
    """同一句最小补全跑两遍:思考开着 vs 显式关掉。

    这组数字决定路由/生成档位的真实单价 —— qwen3.8-flash 默认把思考 token 计入 completion,
    一句「收到」能烧掉 29 个 reasoning token。意图路由每条问题都跑,不关就是全线延迟与成本翻倍。
    """
    msgs = [{"role": "user", "content": "只回复两个字:收到"}]
    shapes = (("默认(思考开)", {}),
              ("enable_thinking=False", {"extra_body": {"enable_thinking": False}}))
    for label, extra in shapes:
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model, messages=msgs, max_tokens=64, temperature=0, **extra)
        except Exception as exc:  # noqa: BLE001
            report(f"M1 chat {model} {label}", False, f"{type(exc).__name__}: {str(exc)[:160]}")
            continue
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning = getattr(details, "reasoning_tokens", None) if details else None
        text = (resp.choices[0].message.content or "").strip()
        report(
            f"M1 chat {model} {label}",
            bool(text),
            f"{int((time.perf_counter() - t0) * 1000)}ms in={getattr(usage, 'prompt_tokens', '?')}"
            f" out={getattr(usage, 'completion_tokens', '?')} reasoning={reasoning} text={text[:12]!r}",
        )


def probe_embedding(client: OpenAI, model: str, expect_dim: int) -> int:
    """返回真实维度;失败返回 -1。同文本的相似度必须高于异文本,否则向量空间不可用。"""
    a, b, c = "多租户知识库的四道隔离防线", "四道隔离防线:存储、身份、过滤、验收", "今天食堂的红烧肉不错"
    try:
        resp = client.embeddings.create(model=model, input=[a, b, c])
    except Exception as exc:  # noqa: BLE001
        report(f"M2 embedding {model}", False, f"{type(exc).__name__}: {str(exc)[:160]}")
        return -1
    vecs = [e.embedding for e in resp.data]
    dim = len(vecs[0])

    def cos(u, v):
        dot = sum(x * y for x, y in zip(u, v, strict=False))
        nu = sum(x * x for x in u) ** 0.5
        nv = sum(y * y for y in v) ** 0.5
        return dot / (nu * nv)

    near, far = cos(vecs[0], vecs[1]), cos(vecs[0], vecs[2])
    report(
        f"M2 embedding {model}",
        dim == expect_dim and near > far,
        f"dim={dim}(配置 {expect_dim}) 近义 cos={near:.4f} > 异题 cos={far:.4f} → {near > far}",
    )
    return dim


def probe_rerank(key: str, model: str) -> None:
    """两条候选路径都试,记下真正可用的那个形状。"""
    query = "租户隔离怎么做"
    docs = ["本方案用分区键与服务端强制过滤实现多租户隔离", "红烧肉的做法是先焯水再上色"]
    attempts = [
        (
            "原生 text-rerank",
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            {"model": model, "input": {"query": query, "documents": docs},
             "parameters": {"top_n": 2, "return_documents": False}},
            lambda j: j["output"]["results"],
        ),
        (
            "compatible-api reranks",
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            {"model": model, "query": query, "documents": docs, "top_n": 2},
            lambda j: j["results"],
        ),
    ]
    worked = []
    for label, url, body, extract in attempts:
        try:
            r = httpx.post(url, headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"},
                           json=body, timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  · {label}: 网络失败 {type(exc).__name__}")
            continue
        if r.status_code != 200:
            print(f"  · {label}: HTTP {r.status_code} {r.text[:160]}")
            continue
        try:
            out = extract(r.json())
        except (KeyError, ValueError) as exc:
            print(f"  · {label}: 200 但响应形状不符 {type(exc).__name__}; body={r.text[:200]}")
            continue
        top = max(out, key=lambda x: x.get("relevance_score", x.get("score", 0.0)))
        idx = top.get("index", top.get("document", {}).get("idx"))
        worked.append((label, json.dumps(out, ensure_ascii=False)[:120]))
        report(f"M3 rerank {model}({label})", idx == 0, f"最相关文档 index={idx}")
    if not worked:
        report(f"M3 rerank {model}", False, "两种接口形状都没打通(见上面逐条错误)")


def probe_omni(client: OpenAI, model: str) -> None:
    """视觉档位能不能替掉 qwen-vl-ocr:拿一张程序生成的中文文字图,看它读不读得出来。

    omni 系在这条链路上的两个已知怪癖要先确认:是否只允许流式输出、`modalities` 必填否。
    两种形状都试,记下能落地的那个 —— 离线管道的图片回填任务要按这个形状写。
    """
    import base64
    import io

    from PIL import Image, ImageDraw, ImageFont

    text = "多租户隔离:分区键与服务端强制过滤"
    img = Image.new("RGB", (880, 120), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 34)
    except OSError:
        font = ImageFont.load_default()
    draw.text((16, 40), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    content = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": "逐字转录图里的中文,不要解释"},
    ]

    shapes = [
        ("非流式", {"stream": False}),
        ("流式+modalities", {"stream": True, "modalities": ["text"]}),
    ]
    for label, extra in shapes:
        try:
            if extra.get("stream"):
                chunks = []
                usage = None
                for ev in client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": content}],
                    max_tokens=128, temperature=0, stream_options={"include_usage": True}, **extra):
                    if ev.choices and ev.choices[0].delta.content:
                        chunks.append(ev.choices[0].delta.content)
                    if getattr(ev, "usage", None):
                        usage = ev.usage
                got = "".join(chunks).strip()
            else:
                resp = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": content}],
                    max_tokens=128, temperature=0, **extra)
                got = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
        except Exception as exc:  # noqa: BLE001
            print(f"  · {label}: 不可用 {type(exc).__name__}: {str(exc)[:150]}")
            continue
        hit = any(k in got for k in ("分区键", "隔离", "租户"))
        report(f"M4 omni {model}({label})", hit,
               f"转录={got[:40]!r} in={getattr(usage, 'prompt_tokens', '?')}"
               f" out={getattr(usage, 'completion_tokens', '?')}")
        return
    report(f"M4 omni {model}", False, "两种请求形状都没打通(见上面逐条错误)")


def probe_structured() -> None:
    """C7:编排层四类结构化输出,拿**真模型 + 真 prompt 构造器**各跑一条。

    单测里用的是 FakeGateway,它按 schema 类型给可编程返回 —— 也就是说「strict json_schema 加
    带 default 的数组字段,现网模型到底给不给得出合法 JSON」这件事从来没被打过真模型验过。
    `RewriteOut.candidates` 与 `RelevanceOut.missing_info` 正是这种字段:少一条实测,#7 上线当天
    才会知道自修复重试要烧几轮。

    走 `Gateway.structured` 而不是自己拼请求:固化在那一层的东西(`enable_thinking=False`、
    `temperature=0`、schema 进 system、自修复重试)本身就是被验的对象。
    """
    from typing import get_args

    from kbmesh.agents.nodes import assess_prompt, intent_prompt, rewrite_prompt
    from kbmesh.agents.schemas import IntentOut, RelevanceOut, RewriteOut
    from kbmesh.core.gateway import Gateway
    from kbmesh.retrieval.contracts import Citation, Hit
    from kbmesh.retrieval.graph_route import MentionsPayload, mentions_prompt

    query = "多租户知识库的隔离设计怎么做"

    def hit(index: int, text: str) -> Hit:
        cid = f"chunk-vm-{index}"
        return Hit(
            chunk_id=cid,
            score=0.5,
            source_routes={"body"},
            citation=Citation(
                doc_id="doc-vm", chunk_id=cid, heading_path="检索 > 隔离设计", title="设计册"
            ),
            text=text,
            route_scores={"body": 0.5},
        )

    verdicts = get_args(IntentOut.model_fields["verdict"].annotation)
    cases = [
        ("route", intent_prompt(query), IntentOut, lambda o: o.verdict in verdicts),
        ("rewrite", rewrite_prompt(query), RewriteOut, lambda o: len(o.candidates) >= 1),
        ("assess",
         assess_prompt(query, [hit(1, "分区键在服务端强制下推"), hit(2, "邻域扩展每跳重新核对租户")]),
         RelevanceOut, lambda o: len(o.scores) >= 1),
        ("graph_mention", mentions_prompt(query), MentionsPayload, lambda o: True),
    ]
    gateway = Gateway()
    for purpose, prompt, schema, check in cases:
        t0 = time.perf_counter()
        try:
            out, usage = gateway.structured(prompt, schema, purpose=purpose)
        except Exception as exc:  # noqa: BLE001
            report(f"C7 structured {schema.__name__}", False, f"{type(exc).__name__}: {str(exc)[:200]}")
            continue
        ok = bool(check(out))
        report(
            f"C7 structured {schema.__name__}",
            ok,
            f"{int((time.perf_counter() - t0) * 1000)}ms in={usage.token_in} out={usage.token_out}"
            f" → {str(out)[:150]}",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", default=None)
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--rerank", default=None)
    ap.add_argument("--omni", default=None)
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["chat", "embedding", "rerank", "omni", "structured"],
        help="跳过某一档(比如只想验视觉)",
    )
    args = ap.parse_args()

    settings = get_settings()
    key = os.getenv("KBMESH_DASHSCOPE_API_KEY", "")
    if not key or key == "sk-xxx":
        print("KBMESH_DASHSCOPE_API_KEY 未注入(仓库里保持 sk-xxx 占位)。")
        print("按脚本头部的跑法,在命令行临时注入全局 key 再跑,且不要把这个值写回任何文件。")
        return 2

    chat_model = args.chat or settings.llm_model
    emb_model = args.embedding or settings.embedding_model
    rerank_model = args.rerank or settings.rerank_model
    omni_model = args.omni or settings.vision_model
    client = OpenAI(api_key=key, base_url=settings.dashscope_base_url, timeout=60.0, max_retries=0)

    if "chat" not in args.skip:
        probe_chat(client, chat_model)
    if "embedding" not in args.skip:
        dim = probe_embedding(client, emb_model, settings.embedding_dim)
    else:
        dim = settings.embedding_dim
    if "rerank" not in args.skip:
        probe_rerank(key, rerank_model)
    if "omni" not in args.skip:
        probe_omni(client, omni_model)
    if "structured" not in args.skip:
        probe_structured()

    print(f"\n实测维度 {dim} → 建 Milvus collection 时 KBMESH_EMBEDDING_DIM 必须是这个值;"
          f"当前配置 {settings.embedding_dim}")
    failed = [r for r in results if not r[1]]
    print(f"=== {len(results) - len(failed)}/{len(results)} 通过 ===")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
