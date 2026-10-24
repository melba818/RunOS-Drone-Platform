# -*- coding: utf-8 -*-
"""VLX-Seek 远端推理客户端。

「多模态目标检测」改造后的唯一推理出口：把图像 + 中文目标描述发给云端常驻的
VLX-Seek 服务，拿回真实 region bbox。

传输方式两种（**服务端默认只绑 127.0.0.1，不暴露公网**）：

- ``ssh``    —— 复用 RunOS 已有的算力节点凭据，用 paramiko ``direct-tcpip`` 把
  远端 127.0.0.1:port 隧道到本地再发 HTTP。推荐，也是默认。
- ``direct`` —— 直接 HTTP（仅用于云端服务绑定公网/内网地址的场景）。

服务地址、token 一律走配置，**不写死 IP**：
    lmm_config.json: vlx_seek_transport / vlx_seek_node_id / vlx_seek_base_url /
                     vlx_seek_token / vlx_seek_timeout
环境变量优先:      VLX_SEEK_BASE_URL / VLX_SEEK_TOKEN / VLX_SEEK_TRANSPORT / VLX_SEEK_NODE
"""
from __future__ import annotations

import http.client
import json
import os
import threading
import time
import uuid
from contextlib import ExitStack
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

DEFAULT_TIMEOUT = 300.0
DEFAULT_PORT = 8021
DEFAULT_TASK = "reasoning_detection"
DEFAULT_LANG = "zh"

#: 隧道空闲多久后丢弃重建（服务端也会踢掉长时间空闲的连接）。
TUNNEL_IDLE_TTL = 600.0

# ── ★R2B★ 持久 HTTP 连接（按"节点+端口"一个槽位）──────────────────────────
# 为什么只做这一项：R2B-1 用 /echo 做过三组 A/B，两个流行猜测**都被证伪** ——
#   * SSH channel window：默认已是 2 MB（OpenSSH CHAN_TCP_WINDOW_DEFAULT），
#     反向对照把窗口压到 256 KB，240 KB 写入 667 ms vs 686 ms，毫无变化。
#   * socket SO_SNDBUF/SO_RCVBUF：写入从 761 ms 降到 3.9 ms，"看起来"快了 200 倍，
#     但服务端 body_read 955→771 ms、端到端 1165→1151 ms —— 只是把耗时挪进了
#     wait_first_byte。**只测写入会得出完全错误的结论**，必须看端到端。
# 真正每请求白付的只有一笔：约 148 ms 的 open_channel + HTTP 建连。所以只做复用。
#
# 复用**必须**带健康检查和上限：SSH 通道会因为 sshd 空闲踢人、网络抖动、
# 服务端重启而静默失效；不带 TTL / 次数上限的连接会变成"每隔几百帧随机失败一次"
# 这种最难排查的故障。失败时丢弃重建，当前请求**最多重试 1 次**。
PERSIST_IDLE_TTL = 120.0          # 空闲多久丢弃（秒）
PERSIST_MAX_REQUESTS = 500        # 单条连接最多服务多少次请求
_PCONN_LOCK = threading.RLock()
_PCONNS: Dict[str, Dict[str, Any]] = {}


class _PersistentHTTP:
    """一条复用的 direct-tcpip channel + HTTP/1.1 连接。

    刻意不做连接池：R1 的 InferenceWorker 只有 1 个，多线程同时往一条
    HTTP/1.1 连接上写请求会串包。所以是「一个槽位 + 一把锁」，并发请求排队走
    同一条连接 —— 真需要更高并发时应该开多条**独立**连接，而不是共享这一条。
    """

    __slots__ = ("channel", "conn", "used", "requests", "created", "last_error")

    def __init__(self, channel: Any, conn: Any) -> None:
        self.channel = channel
        self.conn = conn
        self.used = time.monotonic()
        self.requests = 0
        self.created = time.time()
        self.last_error: Optional[str] = None

    def alive(self) -> bool:
        """连接是否还能用（**只做本地可判定的检查**，不发探测包）。

        真正的健康检查是"发请求试试"：SSH 通道半开的情况本地看不出来，所以
        这里返回 True 也不代表一定成功 —— 失败由 `_request_persistent` 兜底。
        """
        try:
            if self.conn is None or self.channel is None:
                return False
            if getattr(self.channel, "closed", False):
                return False
            if self.channel.exit_status_ready():
                return False
            tr = self.channel.get_transport()
            if tr is None or not tr.is_active():
                return False
        except Exception:  # noqa: BLE001
            return False
        return True

    def stale(self) -> Optional[str]:
        if self.requests >= PERSIST_MAX_REQUESTS:
            return "max_requests"
        if (time.monotonic() - self.used) > PERSIST_IDLE_TTL:
            return "idle_ttl"
        return None

    def close(self) -> None:
        for obj in (self.conn, self.channel):
            try:
                if obj is not None:
                    obj.close()
            except Exception:  # noqa: BLE001
                pass
        self.conn = None
        self.channel = None


def drop_persistent() -> None:
    """丢弃全部持久连接（改节点/凭据/排错时调用）。"""
    with _PCONN_LOCK:
        entries = list(_PCONNS.values())
        _PCONNS.clear()
    for e in entries:
        conn = e.get("obj")
        if conn is not None:
            conn.close()


# ── 共享 SSH 隧道池 ────────────────────────────────────────────────────────
# 为什么必须复用：一次 VLX 请求 = 跳板机 + 目标机各一次 SSH 握手。视频任务
# 775 帧就是 1500+ 次握手，跳板 sshd 的 MaxStartups 一旦限流，握手会在 kex
# 完成前被掐断 —— 表现是「跑到第 N 帧后每帧都报 No existing session」
# （paramiko 在 kex 未完成、transport 未 active 时取 host key 抛的这个错）。
# 所以隧道按 (节点, 端口) 复用整段任务，用完不关，空闲超过 TTL 才重建。
_TUNNEL_LOCK = threading.RLock()
_TUNNELS: Dict[Tuple, Dict[str, Any]] = {}


class VLXError(RuntimeError):
    """VLX-Seek 调用失败（网络、鉴权、推理、超时统一归一到这一种异常）。"""

    def __init__(self, message: str, *, status: Optional[int] = None,
                 detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


# ── 配置 ───────────────────────────────────────────────────────────────────

def _env_flag(env_name: str, stored: Any, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        raw = stored
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def resolve_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并 环境变量 > 显式入参 > lmm_config.json > 默认值。"""
    try:
        from lmm import config as lmm_config
        stored = lmm_config.load() if hasattr(lmm_config, "load") else {}
    except Exception:  # noqa: BLE001
        stored = {}

    cfg = {
        "transport": (os.environ.get("VLX_SEEK_TRANSPORT")
                      or stored.get("vlx_seek_transport") or "ssh"),
        "node_id": (os.environ.get("VLX_SEEK_NODE")
                    or stored.get("vlx_seek_node_id") or ""),
        "base_url": (os.environ.get("VLX_SEEK_BASE_URL")
                     or stored.get("vlx_seek_base_url") or ""),
        "token": (os.environ.get("VLX_SEEK_TOKEN")
                  or stored.get("vlx_seek_token") or ""),
        "port": int(stored.get("vlx_seek_port") or DEFAULT_PORT),
        "timeout": float(stored.get("vlx_seek_timeout") or DEFAULT_TIMEOUT),
        "task": stored.get("vlx_seek_task") or DEFAULT_TASK,
        "lang": stored.get("vlx_seek_lang") or DEFAULT_LANG,
        "extra_instruction": stored.get("vlx_seek_extra_instruction") or "",
        #: 也放进 cfg，让 VLXEngine 与单图接口共用同一份默认值（否则两条路径的
        #: task / max_new_tokens 会各写各的，同一句 Prompt 表现不一致）。
        "max_new_tokens": int(os.environ.get("VLX_SEEK_MAX_NEW_TOKENS")
                              or stored.get("vlx_seek_max_new_tokens") or 256),
        #: SSH 隧道复用（默认开）。关掉就退回「每次请求新建连接」，只在排错时用。
        "reuse_tunnel": _env_flag("VLX_SEEK_REUSE_TUNNEL", stored.get("vlx_seek_reuse_tunnel"), True),
        #: ★R2B★ Fast Transport 总开关：复用 direct-tcpip channel + HTTP keep-alive。
        #: 默认 0（旧路径），一键可回退。环境变量优先，便于 A/B 不用改配置文件。
        "fast_transport": _env_flag("VLX_FAST_TRANSPORT",
                                    stored.get("vlx_seek_fast_transport"), False),
    }
    cfg.update({k: v for k, v in (overrides or {}).items() if v not in (None, "")})
    return cfg


def local_availability(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check saved VLX routing locally; this never probes or opens a connection."""
    cfg = resolve_config(config)
    try:
        from routers.compute_nodes import store
        nodes = store.list()
    except Exception:
        nodes = []

    transport = str(cfg.get("transport") or "ssh").strip().lower()
    if transport == "direct":
        base_url = str(cfg.get("base_url") or "").strip()
        if base_url and "://" not in base_url:
            base_url = "http://" + base_url
        host = (urlsplit(base_url).hostname or "").strip().lower()
        if not host:
            return {"available": False, "reason": "未配置 VLX-Seek 直连地址。"}
        return {"available": True, "reason": "手动选择并启动检测后才会调用 VLX-Seek。"}

    if transport != "ssh":
        return {"available": False, "reason": "VLX-Seek 传输方式配置无效。"}

    node_id = str(cfg.get("node_id") or "")
    node = next((n for n in nodes if node_id and n.get("id") == node_id), None)
    if node is None and not node_id:
        node = next((n for n in nodes if n.get("name") == "3060"), None)
        if node is None and nodes:
            node = nodes[0]
    if node is None:
        return {"available": False, "reason": "未配置 VLX-Seek 算力节点。"}
    return {"available": True, "reason": "手动选择并启动检测后才会调用 VLX-Seek。"}


# ── multipart ──────────────────────────────────────────────────────────────

def build_multipart(fields: Dict[str, Any], file_field: str, filename: str,
                    payload: bytes, content_type: str = "image/jpeg"
                    ) -> Tuple[bytes, str]:
    boundary = "----RunOSVLX" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode("utf-8"))
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
         f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").encode("utf-8"))
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# ── 客户端 ─────────────────────────────────────────────────────────────────

class VLXSeekClient:
    """线程安全（每次请求自建连接；SSH 隧道按需新建）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = resolve_config(config)

    # -- 低层：两种传输都归一到 (status, body_bytes) --
    def _request(self, method: str, path: str, body: Optional[bytes] = None,
                 headers: Optional[Dict[str, str]] = None,
                 timing: Optional[Dict[str, Any]] = None) -> Tuple[int, bytes]:
        transport = (self.cfg.get("transport") or "ssh").lower()
        if transport == "direct":
            return self._request_direct(method, path, body, headers)
        #: ★R2B★ Fast Transport：先试复用连接，返回 None（建不出来 / 两次都失败）
        #: 就**原样退回旧路径**，绝不因为优化而让推理失败。
        if self.cfg.get("fast_transport"):
            got = self._request_persistent(method, path, body, headers, timing)
            if got is not None:
                return got
        return self._request_ssh(method, path, body, headers, timing=timing)

    def _request_direct(self, method, path, body, headers):
        base = self.cfg.get("base_url") or ""
        if not base:
            raise VLXError("未配置 VLX_SEEK_BASE_URL，无法直连 VLX-Seek 服务")
        base = base.rstrip("/")
        if "://" not in base:
            base = "http://" + base
        scheme, _, hostport = base.partition("://")
        host, _, port = hostport.partition(":")
        conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(host, int(port or (443 if scheme == "https" else 80)),
                        timeout=self.cfg["timeout"])
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            return resp.status, resp.read()
        except OSError as exc:
            raise VLXError(f"无法连接 VLX-Seek 服务（{base}）：{exc}") from exc
        finally:
            conn.close()

    # -- SSH 隧道：按 (节点, 端口) 复用，坏了自动重建 --
    @staticmethod
    def _tunnel_key(node: Dict[str, Any], port: int) -> Tuple:
        return (node.get("id") or "", node.get("host") or "", node.get("port"),
                node.get("jump_host") or "", port)

    @staticmethod
    def _close_entry(entry: Dict[str, Any]) -> None:
        try:
            entry["stack"].close()
        except Exception:  # noqa: BLE001
            pass

    def _acquire_tunnel(self, node, parts, port, timing: Optional[Dict[str, Any]] = None) -> Any:
        """取一条可用隧道（复用旧的或新建一条），返回 paramiko SSHClient。

        ★R2A★ `timing["tunnel_reused"]` 记这次到底**复用**了还是**重新握手**了。
        两者差一次完整 SSH kex（几百毫秒量级），不记这个标记的话，"隧道获取耗时
        忽大忽小"就只能归因为网络抖动，其实是复用开关没生效。
        """
        key = self._tunnel_key(node, port)
        now = time.monotonic()
        reused = False
        with _TUNNEL_LOCK:
            entry = _TUNNELS.get(key)
            if entry is not None:
                transport = entry["client"].get_transport()
                stale = (now - entry["used"]) > TUNNEL_IDLE_TTL
                if stale or transport is None or not transport.is_active():
                    self._close_entry(entry)
                    _TUNNELS.pop(key, None)
                    entry = None
                else:
                    reused = True
            if entry is None:
                from routers.compute_nodes import service
                stack = ExitStack()
                try:
                    client = stack.enter_context(service.connect(node, parts))
                except Exception:  # noqa: BLE001
                    stack.close()
                    raise
                entry = {"stack": stack, "client": client, "used": now}
                _TUNNELS[key] = entry
            entry["used"] = now
        if timing is not None:
            timing["tunnel_reused"] = reused
        return entry["client"]

    @classmethod
    def _release_tunnel(cls, key: Tuple) -> None:
        """丢弃一条隧道（请求失败时用：下次请求会重新握手）。"""
        with _TUNNEL_LOCK:
            entry = _TUNNELS.pop(key, None)
        if entry is not None:
            cls._close_entry(entry)

    @classmethod
    def drop_tunnels(cls) -> None:
        """全部丢弃（改了节点/端口/凭据后调用，避免复用旧连接）。"""
        with _TUNNEL_LOCK:
            entries = list(_TUNNELS.values())
            _TUNNELS.clear()
        for entry in entries:
            cls._close_entry(entry)

    # -- ★R2B★ 持久连接：拿到一条可复用的 (channel, HTTPConnection) --
    def _acquire_persistent(self, node, parts, port,
                            timing: Optional[Dict[str, Any]] = None
                            ) -> Optional[_PersistentHTTP]:
        """取当前槽位里的连接；没有/过期/已断就新建一条。

        `timing` 里记录 `channel_open_ms`：复用态是 0，新建态才是真实握手耗时 ——
        这个数就是 R2B 要省掉的那 148 ms，必须能看出来省没省。
        """
        key = "%s|%s|%d" % (node.get("id") or "", node.get("host") or "", port)
        with _PCONN_LOCK:
            entry = _PCONNS.get(key)
            obj = entry.get("obj") if entry else None
            if obj is not None:
                reason = obj.stale()
                if reason == "idle_ttl" or (reason is None and not obj.alive()):
                    obj.close()
                    _PCONNS.pop(key, None)
                    obj = None
                    entry = None
            if obj is None:
                t_build = time.time()
                try:
                    client = self._acquire_tunnel(node, parts, port, timing=None)
                    transport = client.get_transport()
                    if transport is None or not transport.is_active():
                        return None
                    channel = transport.open_channel(
                        "direct-tcpip", ("127.0.0.1", port), ("127.0.0.1", 0),
                        timeout=self.cfg["timeout"])
                    channel.settimeout(self.cfg["timeout"])
                    conn = http.client.HTTPConnection(
                        "127.0.0.1", port, timeout=self.cfg["timeout"])
                    conn.sock = channel
                    obj = _PersistentHTTP(channel, conn)
                    _PCONNS[key] = {"obj": obj}
                    #: 只有**新建**那一次才真的付了握手钱，记下来（复用态=0）。
                    #: 不记的话，B1 的第一笔请求会把 148 ms 隐藏掉，看起来像"零成本建连"。
                    if timing is not None:
                        timing["persistent_build_ms"] = round(
                            (time.time() - t_build) * 1000.0, 3)
                except Exception as exc:  # noqa: BLE001
                    #: 建不出来就退回旧路径（每次新建），绝不让推理直接失败
                    if timing is not None:
                        timing["persistent_build_error"] = type(exc).__name__
                    return None
            obj.used = time.monotonic()
            borrowed = obj
        if timing is not None:
            timing["persistent_reused"] = borrowed.requests > 0
        return borrowed

    def _request_persistent(self, method, path, body, headers,
                            timing: Optional[Dict[str, Any]] = None):
        """走复用连接发一次请求；失败即丢弃连接并**最多重试 1 次**。"""
        node_id = self.cfg.get("node_id") or ""
        try:
            from routers.compute_nodes import store  # 延迟导入避免循环
        except Exception as exc:  # noqa: BLE001
            raise VLXError(f"无法加载算力节点模块：{exc}") from exc
        nodes = store.list()
        node = None
        for n in nodes:
            if node_id and n.get("id") == node_id:
                node = n
                break
            if not node_id and n.get("name") == "3060":
                node = n
        if node is None and not node_id and nodes:
            node = nodes[0]
        if node is None:
            raise VLXError("未找到可用的算力节点，无法建立 VLX-Seek 隧道")

        parts = store.bundle(node)
        port = int(self.cfg.get("port") or DEFAULT_PORT)

        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            obj = self._acquire_persistent(node, parts, port, timing)
            if obj is None:
                return None          # 让调用方退回旧路径
            t0 = time.time()
            try:
                obj.conn.request(method, path, body=body, headers=headers or {})
                t_w = time.time()
                resp = obj.conn.getresponse()
                t_f = time.time()
                payload = resp.read()
                t_r = time.time()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                obj.last_error = f"{type(exc).__name__}: {exc}"
                obj.close()
                with _PCONN_LOCK:
                    _PCONNS.pop("%s|%s|%d" % (node.get("id") or "",
                                              node.get("host") or "", port), None)
                if timing is not None:
                    timing["persistent_retry"] = True
                continue
            obj.requests += 1
            obj.used = time.monotonic()
            if timing is not None:
                timing["tunnel_acquire_ms"] = 0.0
                timing["ssh_channel_open_ms"] = 0.0
                timing["http_connection_create_ms"] = 0.0
                timing["request_write_ms"] = round((t_w - t0) * 1000.0, 3)
                timing["wait_first_byte_ms"] = round((t_f - t_w) * 1000.0, 3)
                timing["response_read_ms"] = round((t_r - t_f) * 1000.0, 3)
                timing["channel_reused"] = True
                timing["http_keepalive"] = True
            return resp.status, payload
        if timing is not None:
            timing["persistent_failed"] = f"{type(last_exc).__name__}"
        return None

    def _request_ssh(self, method, path, body, headers,
                     timing: Optional[Dict[str, Any]] = None):
        """通过 SSH direct-tcpip 隧道访问远端 127.0.0.1:port。

        ★R2A★ 这一段过去只有一个 `io_ms`（含全部等待），"传输慢"和"服务端慢"
        永远分不开。现在按**实际发生的动作**切六刀：

            tunnel_acquire   取（或建）SSH 隧道 —— 复用时接近 0，新建要一次 kex
            ssh_channel_open 在隧道上开 direct-tcpip channel（每次请求一条，不复用）
            http_conn_create 建 HTTPConnection 对象并挂上 channel（无 socket 动作）
            request_write    写请求头 + body
            wait_first_byte  等到服务端回第一个字节
            response_read    把响应体读完

        其中 `wait_first_byte` 才是**真正的"路上 + 服务端干活"**，它和服务端自报
        的 `server_total_ms` 一减，剩下的就是纯线路开销（wire_overhead）。
        另外三个复用标记（`tunnel_reused` / `channel_reused` / `http_keepalive`）
        一并记录：本实现每次请求都新建 channel 与 HTTP 连接，**两个恒为 False**，
        这是 R2B 的明确优化点，先把它量出来再动。
        """
        node_id = self.cfg.get("node_id") or ""
        try:
            from routers.compute_nodes import service, store  # 延迟导入避免循环
        except Exception as exc:  # noqa: BLE001
            raise VLXError(f"无法加载算力节点模块：{exc}") from exc

        nodes = store.list()
        node = None
        for n in nodes:
            if node_id and n.get("id") == node_id:
                node = n
                break
            if not node_id and n.get("name") == "3060":
                node = n
        if node is None and not node_id and nodes:
            node = nodes[0]
        if node is None:
            raise VLXError("未找到可用的算力节点，无法建立 VLX-Seek 隧道")
        from routers.compute_nodes import required  # 跳板机节点的凭据完整性校验
        parts = store.bundle(node)
        required(node, parts, '建立隧道')
        port = int(self.cfg.get("port") or DEFAULT_PORT)
        key = self._tunnel_key(node, port)
        reuse = bool(self.cfg.get("reuse_tunnel", True))
        attempts = 2 if reuse else 1

        last_exc = None
        for attempt in range(attempts):
            req_stack = ExitStack()
            try:
                t_a = time.time()
                if reuse:
                    client = self._acquire_tunnel(node, parts, port, timing=timing)
                else:
                    client = req_stack.enter_context(service.connect(node, parts))
                    if timing is not None:
                        timing["tunnel_reused"] = False
                t_tun = time.time()
                transport = client.get_transport()
                if transport is None or not transport.is_active():
                    raise OSError("SSH 隧道已断开")
                channel = transport.open_channel(
                    "direct-tcpip", ("127.0.0.1", port), ("127.0.0.1", 0),
                    timeout=self.cfg["timeout"])
                t_ch = time.time()
                channel.settimeout(self.cfg["timeout"])
                conn = http.client.HTTPConnection("127.0.0.1", port,
                                                  timeout=self.cfg["timeout"])
                conn.sock = channel
                t_conn = time.time()
                req_stack.callback(conn.close)
                conn.request(method, path, body=body, headers=headers or {})
                t_wr = time.time()
                resp = conn.getresponse()
                t_first = time.time()
                payload = resp.read()
                t_read = time.time()
                if timing is not None:
                    timing["tunnel_acquire_ms"] = round((t_tun - t_a) * 1000.0, 3)
                    timing["ssh_channel_open_ms"] = round((t_ch - t_tun) * 1000.0, 3)
                    timing["http_connection_create_ms"] = round((t_conn - t_ch) * 1000.0, 3)
                    timing["request_write_ms"] = round((t_wr - t_conn) * 1000.0, 3)
                    timing["wait_first_byte_ms"] = round((t_first - t_wr) * 1000.0, 3)
                    timing["response_read_ms"] = round((t_read - t_first) * 1000.0, 3)
                    # 本实现的现状（不是变量）：每个请求一条新 channel、一个新
                    # HTTPConnection。记下来，R2B 改复用之后可以直接对比。
                    timing["channel_reused"] = False
                    timing["http_keepalive"] = False
                return resp.status, payload
            except VLXError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # 隧道可能已经死了：丢掉它，用一条全新的重试一次。
                # 这一步是「跑到第 N 帧后每帧都失败」的关键兜底。
                if reuse:
                    self._release_tunnel(key)
                    if attempt + 1 < attempts:
                        time.sleep(1.0)
                        continue
            finally:
                req_stack.close()
        raise VLXError(f"VLX-Seek SSH 隧道请求失败：{last_exc}") from last_exc

    def _json_request(self, method: str, path: str, body: Optional[bytes] = None,
                      headers: Optional[Dict[str, str]] = None,
                      timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """发请求并解析 JSON。

        `timing` 传入一个 dict 时，会把**本层可观测的细分耗时**写进去（毫秒）：
        `io_ms`（等响应）/ `decode_ms`（字节解码）/ `parse_ms`（JSON 解析）。
        这条链路里"传输"和"解析"经常被混为一谈，而判断 4.4 秒到底花在
        网络还是解码上必须能分开看 —— 所以埋点放在这一层，而不是让上层估算。

        ★R2A★ 新增：`response_bytes`（**客户端实测**的响应体字节数）。服务端
        刻意不报这个数 —— 它是"发出去多少字节"，而它自己又要占用字节，服务端
        算必然自指；客户端量 `len(payload)` 天然精确。
        """
        hdrs = dict(headers or {})
        if self.cfg.get("token"):
            hdrs["x-vlx-token"] = self.cfg["token"]
        t0 = time.time()
        status, payload = self._request(method, path, body, hdrs, timing=timing)
        t_io = time.time()
        text = payload.decode("utf-8", "replace")
        t_decode = time.time()
        if status != 200:
            raise VLXError(f"VLX-Seek 返回 HTTP {status}", status=status, detail=text[:500])
        try:
            result = json.loads(text)
            t_parse = time.time()
        except json.JSONDecodeError as exc:
            raise VLXError("VLX-Seek 返回的不是合法 JSON", status=status,
                           detail=text[:500]) from exc
        if timing is not None:
            timing["io_ms"] = round((t_io - t0) * 1000.0, 3)
            timing["decode_ms"] = round((t_decode - t_io) * 1000.0, 3)
            timing["parse_ms"] = round((t_parse - t_decode) * 1000.0, 3)
            # R2A 新命名（与旧名同值，旧名保留以兼容 R0/R1 的消费者）
            timing["bytes_decode_ms"] = timing["decode_ms"]
            timing["json_parse_ms"] = timing["parse_ms"]
            timing["response_bytes"] = len(payload)
            if method == "POST" and body:
                timing["request_bytes"] = len(body)
        return result

    def _prompt_with_instruction(self, prompt: str) -> str:
        """Append the saved instruction only when a VLX request is explicitly made."""
        instruction = str(self.cfg.get("extra_instruction") or "").strip()
        return f"{prompt}\n\n补充要求：{instruction}" if instruction else prompt

    # -- 公开 API --
    def health(self) -> Dict[str, Any]:
        return self._json_request("GET", "/health")

    def model_info(self) -> Dict[str, Any]:
        return self._json_request("GET", "/model-info")

    def infer_image(self, image_bytes: bytes, prompt: str, *, task: Optional[str] = None,
                    lang: Optional[str] = None, debug: bool = False,
                    max_new_tokens: int = 1024,
                    filename: str = "frame.jpg",
                    timing: Optional[Dict[str, Any]] = None,
                    request_id: Optional[str] = None) -> Dict[str, Any]:
        """单张图片推理。返回服务端统一 JSON（detections 里是**原图像素 xyxy**）。

        `timing` 见 `_json_request`：传出 io/decode/parse 的细分耗时（R0 性能探针用）。

        ★R2A★ `request_id`：一次请求的全局追踪 ID，作为 multipart 字段发给服务端，
        服务端原样回显并写进自己的日志。于是 RunOS 日志 / HTTP / A6000 服务端日志
        能对上同一条线，不用靠时间戳去猜哪条响应对应哪次请求。
        """
        fields = {
            "prompt": self._prompt_with_instruction(prompt),
            "task": task or self.cfg.get("task") or DEFAULT_TASK,
            "lang": lang or self.cfg.get("lang") or DEFAULT_LANG,
            "debug": "true" if debug else "false",
            "max_new_tokens": str(max_new_tokens),
            "request_id": request_id or "",
        }
        t_mp = time.time()
        body, ctype = build_multipart(fields, "image", filename, image_bytes)
        if timing is not None:
            timing["multipart_ms"] = round((time.time() - t_mp) * 1000.0, 3)
        return self._json_request("POST", "/infer", body,
                                 {"Content-Type": ctype,
                                  "Content-Length": str(len(body)),
                                  "x-vlx-request-id": request_id or ""},
                                 timing=timing)

    def infer_at(self, frame_id: Optional[int] = None, prompt: str = "",
                 *, media_time_ms: Optional[float] = None,
                 pts_ms: Optional[float] = None,
                 task: Optional[str] = None, lang: Optional[str] = None,
                 max_new_tokens: int = 1024,
                 timing: Optional[Dict[str, Any]] = None,
                 request_id: Optional[str] = None) -> Dict[str, Any]:
        """★服务器侧共享帧★：只发 `frame_id`，**不上传图片**。

        VLX 服务端收到后自己向 StreamHub 要这一帧再推理 —— 上行只有几十字节，
        而不是旧链路的整张 JPEG（1080p q90 ≈ 745 KB）。

        ★这是"VLX 与 OSTrack 取到同一帧"的关键★：两者拿的是同一个 `frame_id`，
        StreamHub 是唯一取帧者，不存在"各拉一遍导致像素不同"的可能。

        返回与 `infer_image` 同构，并额外回带 `frame_id` / `media_time_ms`
        （服务端实际用的那一帧），调用方据此与 tracker 的帧号对齐。
        """
        payload: Dict[str, Any] = {
            "prompt": self._prompt_with_instruction(prompt),
            "task": task or self.cfg.get("task") or DEFAULT_TASK,
            "lang": lang or self.cfg.get("lang") or DEFAULT_LANG,
            "max_new_tokens": int(max_new_tokens),
        }
        if frame_id is not None:
            payload["frame_id"] = int(frame_id)
        if media_time_ms is not None:
            payload["media_time_ms"] = float(media_time_ms)
        #: ★真实 PTS★（STEP1）：纯附加，服务端不认识就忽略，不影响功能。
        if pts_ms is not None:
            payload["pts_ms"] = float(pts_ms)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if timing is not None:
            timing["request_bytes"] = len(body)
        return self._json_request("POST", "/infer_at", body,
                                 {"Content-Type": "application/json",
                                  "Content-Length": str(len(body)),
                                  "x-vlx-request-id": request_id or ""},
                                 timing=timing)

    def echo(self, payload: bytes, timing: Optional[Dict[str, Any]] = None,
             request_id: Optional[str] = None) -> Dict[str, Any]:
        """★R2A★ 纯传输回声：把 body 原样发到 `/echo`，服务端不碰模型就回。

        与 `infer_image` 用**同一张 JPEG**各打一次，两者之差 = 服务端真实干活的
        时间（收包 + 拆包 + 解码 + 模型 + 序列化）。这是把"线路开销"和"服务端
        开销"分开的唯一硬办法 —— 只靠客户端减服务端自报值，会把两边的时钟偏差
        和采样误差一起算进线路里。
        """
        return self._json_request("POST", "/echo", payload,
                                 {"Content-Type": "application/octet-stream",
                                  "Content-Length": str(len(payload)),
                                  "x-vlx-request-id": request_id or ""},
                                 timing=timing)
