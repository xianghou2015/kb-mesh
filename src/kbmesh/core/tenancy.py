"""Tenant context carried across a request via ContextVar.

Repositories and stores read this instead of trusting call sites: a missing tenant scope is a
programming error, never a cross-tenant leak. The context is only ever produced by
`security.context_from_token()` — i.e. from verified token claims, never from a request body.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

#: 三角色假设,待 Q-1 评审确认(平台管理员 / 租户管理员 / 终端读者)。
ROLE_PLATFORM_ADMIN = "platform_admin"
ROLE_TENANT_ADMIN = "tenant_admin"
ROLE_READER = "reader"
ALL_ROLES = (ROLE_PLATFORM_ADMIN, ROLE_TENANT_ADMIN, ROLE_READER)

#: 离线索引 worker 专用。故意**不进 `ALL_ROLES`**:`context_from_token()` 会拒未知角色,
#: 所以任何令牌都换不到一个 indexer 上下文 —— 上传文档的人拿不到写图的权限面,
#: 而这个身份只在 worker 内部由权威行(`document.tenant_id`)构造。
ROLE_INDEXER = "indexer"


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    user_id: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    #: Empty means the caller carries no explicit knowledge-base scope, i.e. everything inside
    #: the tenant is reachable. Non-empty restricts retrieval to those kb ids.
    kb_scope: frozenset[str] = field(default_factory=frozenset)

    WILDCARD = "*:*:*"

    def has_role(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)

    def can_write_kb(self) -> bool:
        return self.has_role(ROLE_PLATFORM_ADMIN, ROLE_TENANT_ADMIN, ROLE_INDEXER)

    def allows_kb(self, kb_id: str) -> bool:
        return not self.kb_scope or kb_id in self.kb_scope


def worker_context(tenant_id: str, kb_id: str) -> TenantContext:
    """离线管道的身份来源:租户从**权威行**读出的,不是调用方传的(§5 的防线②形态)。

    scope 收紧到这一个知识库 —— worker 处理一篇文档时能碰的范围就该只有这篇文档所属的库,
    给整租户等于把「读到别库数据」的失误从告警变成静默。
    """
    return TenantContext(
        tenant_id=tenant_id,
        user_id=f"{ROLE_INDEXER}:{kb_id}",
        roles=(ROLE_INDEXER,),
        kb_scope=frozenset({kb_id}),
    )


_ctx: ContextVar[TenantContext | None] = ContextVar("kbmesh_tenant", default=None)


def current_context() -> TenantContext:
    ctx = _ctx.get()
    if ctx is None:
        raise RuntimeError("tenant context is not set for this request")
    return ctx


def current_tenant_id() -> str:
    return current_context().tenant_id


def active_context() -> TenantContext | None:
    """The context for this request, or None when there is not one.

    `current_context()` raising is right for data access — no scope means a bug. A gate deciding
    *whether to allow* has to tell "no operator" apart from "operator without this role", and a
    raise cannot carry that distinction.
    """
    return _ctx.get()


@contextmanager
def use_context(ctx: TenantContext) -> Iterator[TenantContext]:
    token: Token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)
