"""Access tokens: the only place a `TenantContext` is ever created.

防线②:身份来自验签后的 claims,不来自请求体。所以这个模块的边界条件都按「宁可拒绝」写:

- 算法固定为配置里的 HS*,并在验签前挡掉 `alg` 与密钥类型不匹配(避免 RS256→HS256 混淆攻击);
- 未知角色直接拒,不当成「没有这个角色」放行 —— 静默忽略一个拼错的 `platfrom_admin` 等于少一道防线;
- `exp`/`iat` 由 PyJWT 强制校验,缺 `exp` 的 token 视为无效。
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import jwt

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import ALL_ROLES, TenantContext

#: 30s 时钟偏差容忍:再多就是在放大「过期 token 仍可用」的窗口。
LEEWAY_SECONDS = 30

ISSUER = "kbmesh"


class TokenError(Exception):
    """验签/声明不合法 —— 调用方一律转 401,不做「当作匿名继续」的降级。"""


def _signing_key() -> str:
    secret = get_settings().jwt_secret
    if not secret:
        raise TokenError("jwt_secret 未配置:签不出也验不了任何 token")
    return secret


def _algorithm() -> str:
    alg = get_settings().jwt_algorithm
    if not alg.startswith("HS"):
        raise TokenError(f"只允许 HS* 算法,配置为 {alg}")
    return alg


def issue_access_token(
    ctx: TenantContext,
    *,
    ttl_minutes: int | None = None,
    now: datetime | None = None,
) -> str:
    settings = get_settings()
    issued = now or datetime.now(UTC)
    ttl = ttl_minutes if ttl_minutes is not None else settings.access_token_ttl_min
    claims = {
        "iss": ISSUER,
        "sub": ctx.user_id,
        "tid": ctx.tenant_id,
        "roles": list(ctx.roles),
        "kb": sorted(ctx.kb_scope),
        "iat": int(issued.timestamp()),
        "exp": int(issued.timestamp()) + ttl * 60,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, _signing_key(), algorithm=_algorithm())


def context_from_token(token: str) -> TenantContext:
    """`token` → `TenantContext`;全项目唯一的 TenantContext 生产点。

    请求里任何其他形式的 tenant_id/body-tenant/自报 kb_scope 都不会走到这里,因此也不可能有
    第二条注入路径。
    """
    try:
        claims = jwt.decode(
            token,
            _signing_key(),
            algorithms=[_algorithm()],
            issuer=ISSUER,
            leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "iat", "sub", "tid"]},
        )
    except jwt.InvalidTokenError as exc:
        raise TokenError(f"token 无效: {type(exc).__name__}") from exc

    roles = claims.get("roles") or []
    if not isinstance(roles, list) or not set(roles) <= set(ALL_ROLES):
        raise TokenError(f"token 携带未知角色: {roles!r}")

    kb = claims.get("kb") or []
    if not isinstance(kb, list) or not all(isinstance(x, str) for x in kb):
        raise TokenError(f"token 的 kb_scope 形态非法: {kb!r}")

    return TenantContext(
        tenant_id=str(claims["tid"]),
        user_id=str(claims["sub"]),
        roles=tuple(roles),
        kb_scope=frozenset(kb),
    )
