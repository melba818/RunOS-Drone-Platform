# -*- coding: utf-8 -*-
"""通过算力节点的 SSH `direct-tcpip` 隧道访问远端 127.0.0.1:<port> 的 HTTP 服务。

为什么要有这个模块：VLX 服务（8021）、OSTrack 服务（8022）、未来的 Qwen 服务，
都跑在 A6000 上、只绑 127.0.0.1，RunOS 本机只能靠 SSH 隧道打过去。

**复用 `lmm.vlx_client` 的共享隧道池**（`_TUNNELS`，按 `(节点, 端口)` 分 key）：
一次 SSH 握手很贵（跳板 + 目标各一次 kex），而 tracker 是高频调用（~7 ticks/s ×
最多 5 个 track），每次新建连接会触发跳板 sshd 的 MaxStartups 限流（VLX 那边
已经踩过这个坑）。8021 / 8022 key 不同，两个服务各占一条隧道，互不干扰。

────────────────────────────────────────────────────────────────────────────
★STEP 2.1★ 新增 `tunnel_http_request_persistent`

实测（`.workbuddy/ostrack_size_sweep.txt`）：每请求新建 channel 要 ~115 ms，
而 OSTrack 服务端推理只有 ~12 ms —— 大头全在"每请求重新握手"和"上传体积"上。
所以这里再包一层：**复用同一条 direct-tcpip channel + HTTP/1.1 keep-alive**。

刻意不碰 VLX 的任何东西：
  * 隧道池 `_TUNNELS` 是共享的（省握手），但**连接槽位表 `_PCONNS` 是 VLX 的，
    本模块用自己的 `_OST_PCONNS`**——两个服务各自一条连接，互不挤占、互不影响；
  * `tunnel_http_request()` 的行为一字未改，VLX 走原路径。

线程安全：`TrackerWorker.step()` 与 VLX 线程的 `_reinit_and_replay()` 会**同时**
调 backend.update，两条线程往同一条 HTTP/1.1 连接写请求会串包，所以每个槽位
配一把 RLock，并发请求排队走同一条连接。
"""
from __future__ import annotations

import http.client
import threading
import time
from contextlib import ExitStack
from typing import Any, Dict, Optional, Tuple

from lmm.vlx_client import (
    TUNNEL_IDLE_TTL,
    VLXSeekClient,
    _PersistentHTTP,
    _TUNNELS,
    _TUNNEL_LOCK,
)

#: 本模块自己的持久连接槽位（与 VLX 的 `_PCONNS` 完全隔离）
_PCONN_LOCK = threading.RLock()
_PCONNS: Dict[str, Dict[str, Any]] = {}


def _find_node(node_id: str):
    from routers.compute_nodes import store
    nodes = store.list()
    node = None
    for n in nodes:
        if node_id and n.get("id") == node_id:
            node = n
            break
        if not node_id and n.get("name") in ("A6000", "3060"):
            node = n
    if node is None and nodes:
        node = nodes[0]
    if node is None:
        raise ConnectionError("未找到可用的算力节点")
    return node


def _acquire_client(node_id: str, port: int, timeout: float):
    """从共享隧道池取一条 SSH 连接（复用旧的或新建）。"""
    from routers.compute_nodes import service, store

    node = _find_node(node_id)
    parts = store.bundle(node)
    key = VLXSeekClient._tunnel_key(node, int(port))
    now = time.monotonic()

    with _TUNNEL_LOCK:
        entry = _TUNNELS.get(key)
        if entry is not None:
            transport = entry["client"].get_transport()
            stale = (now - entry["used"]) > TUNNEL_IDLE_TTL
            if stale or transport is None or not transport.is_active():
                VLXSeekClient._close_entry(entry)
                _TUNNELS.pop(key, None)
                entry = None
        if entry is None:
            stack = ExitStack()
            client = stack.enter_context(service.connect(node, parts))
            entry = {"stack": stack, "client": client, "used": now}
            _TUNNELS[key] = entry
        entry["used"] = now
        client = entry["client"]

    transport = client.get_transport()
    if transport is None or not transport.is_active():
        raise ConnectionError("SSH 隧道不可用")
    return client, transport, key


def tunnel_http_request(node_id: str, port: int, method: str, path: str,
                        body: Optional[bytes] = None,
                        headers: Optional[Dict[str, str]] = None,
                        timeout: float = 30.0) -> Tuple[int, bytes]:
    """复用共享 SSH 隧道，向远端 `127.0.0.1:port` 发一次 HTTP 请求。

    每次请求新建一条 direct-tcpip channel（原行为，VLX 走这条路）。
    返回 `(status, body_bytes)`；连接/握手失败直接抛，由调用方决定降级。
    """
    _client, transport, _key = _acquire_client(node_id, port, timeout)

    channel = transport.open_channel(
        "direct-tcpip", ("127.0.0.1", int(port)), ("127.0.0.1", 0),
        timeout=timeout)
    channel.settimeout(timeout)
    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout)
    conn.sock = channel
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        status = resp.status
    finally:
        try:
            channel.close()
        except Exception:  # noqa: BLE001
            pass
    return status, data


def drop_persistent(node_id: str = "", port: int = 0) -> int:
    """丢弃持久连接（改节点/凭据/排错时调用）。返回丢弃条数。"""
    with _PCONN_LOCK:
        if node_id or port:
            keys = [k for k in _PCONNS
                    if (not node_id or k.startswith(node_id + "|"))
                    and (not port or k.endswith("|%d" % int(port)))]
        else:
            keys = list(_PCONNS)
        entries = [_PCONNS.pop(k, None) for k in keys]
    n = 0
    for e in entries:
        obj = (e or {}).get("obj")
        if obj is not None:
            obj.close()
            n += 1
    return n


def tunnel_http_request_persistent(node_id: str, port: int, method: str, path: str,
                                   body: Optional[bytes] = None,
                                   headers: Optional[Dict[str, str]] = None,
                                   timeout: float = 30.0,
                                   timing: Optional[Dict[str, Any]] = None,
                                   slot_suffix: str = ""
                                   ) -> Tuple[int, bytes]:
    """走**复用**的 direct-tcpip channel + HTTP/1.1 keep-alive 发一次请求。

    ★STEP 2.1★：这是 OSTrack 追帧用的快路径。与 `tunnel_http_request` 的差别
    只有一个 —— channel 与 HTTP 连接不复用的话，每请求要多付 ~115 ms 建连
    （实测值），而服务端推理本身只有 ~12 ms。

    可靠性：
      * 连接断开 / 半开 / 超时 → 丢弃重建，**当前请求最多重试 1 次**；
      * 重建也失败 → 抛异常，由 `RemoteOSTrackBackend` 转成
        `TrackerResult(ok=False)`，绝不让 RunOS 后端退出；
      * `timing` 会写入 `channel_open_ms` / `reused` / `rebuilt` / `write_ms` /
        `wait_first_byte_ms` / `read_ms`，便于现场定位"慢在哪一段"。

    返回 `(status, body_bytes)`。
    """
    node = _find_node(node_id)
    #: ★slot_suffix★：默认情况下一条隧道一个槽位；并发追帧时按 track 分槽，
    #: 每个槽位自己一条 direct-tcpip channel + 自己的锁 —— 这样多个目标的
    #: update 才能真的并行（否则所有请求排在同一把锁上，N 个目标就是 N 倍延迟）。
    slot_key = "%s|%s|%d|%s" % (node.get("id") or "", node.get("host") or "",
                                int(port), str(slot_suffix or ""))

    with _PCONN_LOCK:
        entry = _PCONNS.get(slot_key)
        if entry is None:
            entry = {"obj": None, "lock": threading.RLock()}
            _PCONNS[slot_key] = entry
        lock = entry["lock"]

    last_exc: Optional[BaseException] = None
    with lock:
        for attempt in range(2):
            obj = entry.get("obj")
            if obj is not None:
                reason = obj.stale()
                if reason == "idle_ttl" or (reason is None and not obj.alive()):
                    obj.close()
                    entry["obj"] = None
                    obj = None
                    if timing is not None:
                        timing["rebuild_reason"] = reason or "dead"
            if obj is None:
                t0 = time.perf_counter()
                try:
                    _client, transport, _tkey = _acquire_client(node_id, port, timeout)
                    channel = transport.open_channel(
                        "direct-tcpip", ("127.0.0.1", int(port)), ("127.0.0.1", 0),
                        timeout=timeout)
                    channel.settimeout(timeout)
                    conn = http.client.HTTPConnection(
                        "127.0.0.1", int(port), timeout=timeout)
                    conn.sock = channel
                    obj = _PersistentHTTP(channel, conn)
                    entry["obj"] = obj
                    if timing is not None:
                        timing["channel_open_ms"] = round(
                            (time.perf_counter() - t0) * 1000.0, 3)
                        timing["rebuilt"] = True
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if timing is not None:
                        timing["channel_open_error"] = type(exc).__name__
                    #: 建不出来就退回"每次新建"的旧路径，绝不让调用直接失败
                    return tunnel_http_request(node_id, port, method, path,
                                               body, headers, timeout)
            else:
                if timing is not None:
                    timing["channel_open_ms"] = 0.0
                    timing["reused"] = True
            try:
                try:
                    obj.channel.settimeout(timeout)
                except Exception:  # noqa: BLE001
                    pass
                t0 = time.perf_counter()
                obj.conn.request(method, path, body=body, headers=headers or {})
                t_w = time.perf_counter()
                resp = obj.conn.getresponse()
                t_f = time.perf_counter()
                data = resp.read()
                t_r = time.perf_counter()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                obj.last_error = "%s: %s" % (type(exc).__name__, exc)
                obj.close()
                entry["obj"] = None
                if timing is not None:
                    timing["persistent_retry"] = True
                    timing["persistent_error"] = type(exc).__name__
                continue
            obj.requests += 1
            obj.used = time.monotonic()
            if timing is not None:
                timing["write_ms"] = round((t_w - t0) * 1000.0, 3)
                timing["wait_first_byte_ms"] = round((t_f - t_w) * 1000.0, 3)
                timing["read_ms"] = round((t_r - t_f) * 1000.0, 3)
            return resp.status, data

    raise ConnectionError("持久连接请求失败：%s" % (last_exc or "unknown"))
