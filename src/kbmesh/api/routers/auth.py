"""登录与令牌签发(PUBLIC_PATHS 里登记的两个入口)。

登录是唯一「没有租户上下文也要动库」的端点:身份由凭据决定,不由令牌决定。失败的登录也要
留痕(`login_failed`),但**不区分「无此用户」与「密码错」**—— 差异化响应是用户枚举的入口。
"""
from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address, ip_network

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.api.deps import get_db
from kbmesh.api.schemas import LoginIn
from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.models import AppUser, SecurityEvent
from kbmesh.db.repositories import record_security_event
from kbmesh.security.passwords import hash_password, verify_password
from kbmesh.security.tokens import issue_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

#: 用户不存在时也要跑一次 bcrypt:否则「查无此人」会明显快于「密码错」,时序就是枚举入口。
_DUMMY_HASH = hash_password("kbmesh-login-timing-dummy")

#: bcrypt 是 CPU 密集操作:并发上限别跟着请求数走,否则登录端点就是免费的 CPU 放大器。
_BCRYPT_GATE = threading.BoundedSemaphore(4)


def _verify_gated(plain: str, hashed: str) -> bool:
    with _BCRYPT_GATE:
        return verify_password(plain, hashed)


def _trusted_networks() -> list:
    networks = []
    for raw in get_settings().trusted_proxies.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            networks.append(ip_network(raw, strict=False))
        except ValueError:
            continue  # 配置写错时不把任意来源当可信代理
    return networks


def _is_trusted(value: str, networks: list) -> bool:
    try:
        addr = ip_address(value)
    except ValueError:
        return False
    return any(addr in network for network in networks)


def _client_ip(request: Request) -> str:
    """只采信可信代理追加的 XFF:从右往左跳过可信代理,第一个非可信地址才是客户端。"""
    peer = request.client.host if request.client is not None else "unknown"
    networks = _trusted_networks()
    if not networks or not _is_trusted(peer, networks):
        return peer
    forwarded = [
        one.strip()
        for one in request.headers.get("x-forwarded-for", "").split(",")
        if one.strip()
    ]
    for candidate in reversed(forwarded):
        if _is_trusted(candidate, networks):
            continue
        try:
            ip_address(candidate)
        except ValueError:
            return peer  # 链里混进非法值:退回 socket 对端,不接受伪造串
        return candidate
    return peer


def _count_failures(
    db: AsyncSession, *, since: datetime, account_target: str | None, client_ip: str | None
):
    conditions = [
        SecurityEvent.event == "login_failed",
        SecurityEvent.created_at >= since,
    ]
    if account_target is not None:
        conditions.append(SecurityEvent.target == account_target)
    if client_ip is not None:
        conditions.append(SecurityEvent.detail["ip"].astext == client_ip)
    stmt = select(func.count()).select_from(SecurityEvent).where(*conditions)
    return db.execute(stmt)


async def _ip_throttled(db: AsyncSession, *, client_ip: str) -> bool:
    """预检只看来源 IP:达到上限后不再执行 bcrypt,单 IP 无法在线爆破。"""
    settings = get_settings()
    since = datetime.now(UTC) - timedelta(seconds=settings.login_window_seconds)
    failures = int(
        (await _count_failures(db, since=since, account_target=None, client_ip=client_ip)).scalar_one()
    )
    return failures >= settings.login_ip_max_failures


async def _record_login_failure(db: AsyncSession, *, body: LoginIn, client_ip: str) -> bool:
    """在同一个事务里原子地「计数 + 追加失败审计」,返回本次是否应回 429。

    pair(IP+账号)与 account(跨 IP)两个维度分别计数:
    - 先认证再计数,合法口令永远能登录,攻击者不能靠刷错密码把受害者账号锁死;
    - advisory xact lock 让并发请求逐个看到彼此已提交的失败行,消除 check-then-insert 的 TOCTOU。
    """
    settings = get_settings()
    account_target = f"user:{body.tenant_id}:{body.username}"
    account_lock = f"login-account:{body.tenant_id}:{body.username}"
    pair_lock = f"login-pair:{client_ip}:{body.tenant_id}:{body.username}"
    await db.execute(select(func.pg_advisory_xact_lock(func.hashtext(account_lock))))
    await db.execute(select(func.pg_advisory_xact_lock(func.hashtext(pair_lock))))
    since = datetime.now(UTC) - timedelta(seconds=settings.login_window_seconds)
    pair_result = await _count_failures(
        db, since=since, account_target=account_target, client_ip=client_ip
    )
    pair_before = int(pair_result.scalar_one())
    account_result = await _count_failures(
        db, since=since, account_target=account_target, client_ip=None
    )
    account_before = int(account_result.scalar_one())
    blocked = (
        pair_before >= settings.login_max_failures
        or account_before >= settings.login_max_failures
    )
    if not blocked:
        await record_security_event(
            db,
            actor=None,
            event="login_failed",
            target=account_target,
            path="/api/auth/login",
            detail={"ip": client_ip},
        )
        if (
            pair_before + 1 == settings.login_max_failures
            or account_before + 1 == settings.login_max_failures
        ):
            # 只记录一次「进入节流」,不为之后每个被拒请求堆审计行。
            await record_security_event(
                db,
                actor=None,
                event="login_throttled",
                target=account_target,
                path="/api/auth/login",
                detail={"ip": client_ip, "window_s": settings.login_window_seconds},
            )
    await db.commit()  # 显式提交:HTTPException 不在 get_db 的提交白名单里
    return blocked


async def _login(body: LoginIn, db: AsyncSession, *, client_ip: str) -> dict[str, str]:
    if await _ip_throttled(db, client_ip=client_ip):
        raise HTTPException(
            status_code=429,
            detail="too many failed login attempts",
            headers={"Retry-After": str(get_settings().login_window_seconds)},
        )
    row = (
        await db.execute(
            select(AppUser).where(
                AppUser.tenant_id == body.tenant_id, AppUser.username == body.username
            )
        )
    ).scalar_one_or_none()
    # 无此用户与密码错必须付同样的 bcrypt 成本,响应与耗时都不能区分;同步 bcrypt 放到线程里,
    # 且由 `_BCRYPT_GATE` 限并发,避免登录端点阻塞事件循环或放大 CPU。
    stored_hash = row.password_hash if row is not None else _DUMMY_HASH
    password_ok = await asyncio.to_thread(_verify_gated, body.password, stored_hash)
    if row is not None and password_ok:
        ctx = TenantContext(tenant_id=row.tenant_id, user_id=row.id, roles=(row.role,))
        return {"access_token": issue_access_token(ctx), "token_type": "bearer", "role": row.role}

    blocked = await _record_login_failure(db, body=body, client_ip=client_ip)
    if blocked:
        raise HTTPException(
            status_code=429,
            detail="too many failed login attempts",
            headers={"Retry-After": str(get_settings().login_window_seconds)},
        )
    raise HTTPException(status_code=401, detail="invalid credentials")


@router.post("/login")
async def login(
    body: LoginIn, request: Request, db: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    return await _login(body, db, client_ip=_client_ip(request))


@router.post("/token")
async def token(
    body: LoginIn, request: Request, db: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    """同一凭据换同一张令牌:PUBLIC_PATHS 里的第二个名字,留给脚本与测试的语义化入口。"""
    return await _login(body, db, client_ip=_client_ip(request))
