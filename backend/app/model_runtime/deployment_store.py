"""DeploymentStore —— Deployment **静态配置**的持久化抽象。

★P0/P1 硬约束★
    * **读取绝不创建文件**：文件不存在时 `load()` 必须返回空集合，
      不能因为 import / startup / list 就在 workspace 下生成 deployments.json。
    * 只有显式调用 `save()`（即未来真正 register/update 时）才允许落盘。
    * 本轮 P0/P1 不执行任何真实写入；单元测试一律用 temp directory。

落盘格式：
    {"version": 1, "deployments": {"<deployment_id>": {...}}}
    读取时同时兼容 list 形式（历史/手写），内部统一转成 dict。

写入：tmp 文件 + os.replace 原子替换；替换前保留最近 N 份 .bak。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["DeploymentStore", "EMPTY_STORE"]

EMPTY_STORE: Dict[str, Any] = {"version": 1, "deployments": {}}
_SCHEMA_VERSION = 1
_BAK_KEEP = 3


class DeploymentStore:
    def __init__(self, path, *, backups: int = _BAK_KEEP):
        self.path = Path(path)
        self.backups = max(0, int(backups))
        self._lock = threading.RLock()

    # ── 读 ──────────────────────────────────────────────────────────────────
    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Dict[str, Any]:
        """文件不存在 → 返回空 store，**不创建任何文件**。"""
        with self._lock:
            try:
                raw = self.path.read_text(encoding="utf-8")
            except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
                return {"version": _SCHEMA_VERSION, "deployments": {}}
            except OSError:
                return {"version": _SCHEMA_VERSION, "deployments": {}}
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                return {"version": _SCHEMA_VERSION, "deployments": {}}
            return self._normalize(data)

    @staticmethod
    def _normalize(data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {"version": _SCHEMA_VERSION, "deployments": {}}
        deps = data.get("deployments", data)
        out: Dict[str, Any] = {}
        if isinstance(deps, dict):
            for k, v in deps.items():
                if isinstance(v, dict):
                    out[str(k)] = v
        elif isinstance(deps, list):
            for v in deps:
                if isinstance(v, dict) and v.get("deployment_id"):
                    out[str(v["deployment_id"])] = v
        version = data.get("version", _SCHEMA_VERSION)
        try:
            version = int(version)
        except (TypeError, ValueError):
            version = _SCHEMA_VERSION
        return {"version": version, "deployments": out}

    def get(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        if not deployment_id:
            return None
        d = self.load().get("deployments", {}).get(str(deployment_id))
        return json.loads(json.dumps(d)) if d else None

    def list(self) -> List[Dict[str, Any]]:
        return list(self.load().get("deployments", {}).values())

    # ── 写 ──────────────────────────────────────────────────────────────────
    def save(self, data: Dict[str, Any]) -> None:
        """原子落盘 + .bak 轮转。**只有显式调用才会创建文件。**"""
        with self._lock:
            payload = self._normalize(data)
            payload["version"] = _SCHEMA_VERSION
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_backup()
            tmp = self.path.with_name(self.path.name + f".tmp.{os.getpid()}.{int(time.time() * 1000)}")
            try:
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(str(tmp), str(self.path))
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    def upsert(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        did = str(deployment.get("deployment_id") or "")
        if not did:
            raise ValueError("deployment_id is required")
        data = self.load()
        data["deployments"][did] = dict(deployment)
        self.save(data)
        return dict(data["deployments"][did])

    def delete(self, deployment_id: str) -> bool:
        data = self.load()
        deps = data.get("deployments", {})
        if str(deployment_id) not in deps:
            return False
        deps.pop(str(deployment_id))
        self.save(data)
        return True

    # ── 备份（预留）──────────────────────────────────────────────────────────
    def _rotate_backup(self) -> None:
        """保留最近 N 份 .bak；只有源文件存在时才轮转。"""
        if self.backups <= 0 or not self.path.exists():
            return
        stamp = time.strftime("%Y%m%d%H%M%S")
        bak = self.path.with_name(self.path.name + f".{stamp}.bak")
        try:
            bak.write_bytes(self.path.read_bytes())
        except OSError:
            return
        existing = sorted(
            self.path.parent.glob(self.path.name + ".*.bak"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in existing[self.backups:]:
            try:
                old.unlink()
            except OSError:
                pass
