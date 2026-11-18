"""防线②:身份只能来自验签后的 claims。"""
from __future__ import annotations

import time
from datetime import UTC, datetime

import jwt as pyjwt
import pytest

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import ROLE_READER, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.security.tokens import TokenError, context_from_token, issue_access_token

SECRET = get_settings().jwt_secret


def ctx(**over) -> TenantContext:
    base = dict(
        tenant_id="tnt-a", user_id="u-1", roles=(ROLE_TENANT_ADMIN,), kb_scope=frozenset({"kb-1"})
    )
    base.update(over)
    return TenantContext(**base)


def test_round_trip_keeps_every_claim() -> None:
    token = issue_access_token(ctx(roles=(ROLE_TENANT_ADMIN, ROLE_READER)))
    out = context_from_token(token)
    assert out.tenant_id == "tnt-a"
    assert out.user_id == "u-1"
    assert set(out.roles) == {ROLE_TENANT_ADMIN, ROLE_READER}
    assert out.kb_scope == frozenset({"kb-1"})


def test_tenant_swapped_token_is_rejected() -> None:
    """攻击:拿合法 token 把 tid 改成别人的租户 —— 没有密钥就签不出可验的串。"""
    token = issue_access_token(ctx())
    claims = pyjwt.decode(token, options={"verify_signature": False})
    claims["tid"] = "tnt-victim"
    forged = pyjwt.encode(claims, "attacker-guess", algorithm="HS256")
    with pytest.raises(TokenError):
        context_from_token(forged)


def test_alg_none_is_rejected() -> None:
    claims = pyjwt.decode(issue_access_token(ctx()), options={"verify_signature": False})
    claims["exp"] = int(time.time()) + 300
    with pytest.raises(TokenError):
        context_from_token(pyjwt.encode(claims, key="", algorithm="none"))


def test_expired_token_is_rejected() -> None:
    past = datetime(2026, 1, 1, tzinfo=UTC)
    token = issue_access_token(ctx(), ttl_minutes=-10, now=past)
    with pytest.raises(TokenError, match="token 无效"):
        context_from_token(token)


def test_token_without_expiry_is_rejected() -> None:
    token = issue_access_token(ctx())
    claims = pyjwt.decode(token, SECRET, algorithms=["HS256"])
    claims.pop("exp")
    with pytest.raises(TokenError):
        context_from_token(pyjwt.encode(claims, SECRET, algorithm="HS256"))


def test_unknown_role_is_rejected_not_ignored() -> None:
    token = issue_access_token(ctx(roles=("platfrom_admin",)))
    with pytest.raises(TokenError, match="未知角色"):
        context_from_token(token)


def test_malformed_kb_scope_is_rejected() -> None:
    claims = pyjwt.decode(issue_access_token(ctx()), SECRET, algorithms=["HS256"])
    claims["kb"] = {"kb-1": True}
    with pytest.raises(TokenError, match="kb_scope"):
        context_from_token(pyjwt.encode(claims, SECRET, algorithm="HS256"))


def test_missing_jwt_secret_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """没配密钥时必须拒,而不是签一张谁都验不过的 token 或退化成空签名。"""
    monkeypatch.setenv("KBMESH_JWT_SECRET", "")
    get_settings.cache_clear()
    try:
        assert get_settings().jwt_secret == "", "环境没生效:配置不是从 env 重建的"
        with pytest.raises(TokenError, match="jwt_secret"):
            issue_access_token(ctx())
    finally:
        get_settings.cache_clear()


def test_empty_scope_means_whole_tenant() -> None:
    out = context_from_token(issue_access_token(ctx(kb_scope=frozenset())))
    assert out.kb_scope == frozenset()
    assert out.allows_kb("kb-any")
