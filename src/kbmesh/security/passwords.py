"""口令哈希(bcrypt)。登录端点是唯一消费方;测试用它给 `app_user` 造种子。"""
from __future__ import annotations

import bcrypt

#: 10 轮在验收机上 ~70ms,登录不差这一个量级;调高只会让登录端点变成免费的 CPU 放大器
ROUNDS = 10


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=ROUNDS)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        # 库里混进了非 bcrypt 的历史格式:当密码错处理,不当 500 抛
        return False
