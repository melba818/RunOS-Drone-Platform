"""Authenticated Linux credential storage, shared by platform API workers."""

import os
import stat
import tempfile
from pathlib import Path

PREFIX = b"RHNC1:"


def _key(path: Path, create: bool) -> bytes:
    from cryptography.fernet import Fernet

    path = Path(path)
    if not path.exists():
        if not create:
            raise RuntimeError("服务器凭据密钥缺失，请恢复原密钥后再读取凭据")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".credential-key-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(Fernet.generate_key())
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Publish a complete key atomically; another worker may win this race.
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise RuntimeError("服务器凭据密钥必须是权限为 0600 的普通文件")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            key = handle.read(128).strip()
        try:
            Fernet(key)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("服务器凭据密钥格式无效，请恢复原密钥") from exc
        return key
    finally:
        os.close(fd)


def protect_linux(data: bytes, *, decrypt: bool, key_path: Path) -> bytes:
    from cryptography.fernet import Fernet, InvalidToken

    if decrypt and not data.startswith(PREFIX):
        raise RuntimeError("此凭据不是本服务器加密格式，请在服务器面板重新保存")
    cipher = Fernet(_key(key_path, create=not decrypt))
    if not decrypt:
        return PREFIX + cipher.encrypt(data)
    try:
        return cipher.decrypt(data[len(PREFIX):])
    except InvalidToken as exc:
        raise RuntimeError("服务器凭据无法解密，请恢复匹配的密钥或重新保存凭据") from exc
