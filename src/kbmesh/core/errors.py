"""Errors that mean the same thing at every layer.

`PermissionDenied` 与 `NotFound` 刻意分开:越权要留痕(§9),而「资源不存在」不需要 —— 但
对外两者都返回 404,免得把一个租户的存在性泄露出去。留痕发生在服务端抛之前,不靠状态码。
"""
from __future__ import annotations


class KbmError(Exception):
    """本项目异常的基类,便于 API 层一处兜底。"""

    http_status = 500


class PermissionDenied(KbmError):
    """令牌身份不允许这次操作 —— 必须写审计,不允许静默返回空。"""

    http_status = 403


class NotFound(KbmError):
    http_status = 404


class IllegalTransition(KbmError):
    """解析状态机禁止跳跃(§5.2):跳段意味着管道有 bug,静默接受会把 bug 埋进数据。"""

    http_status = 409
