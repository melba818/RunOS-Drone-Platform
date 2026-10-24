import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


# 改动这些字段意味着「连接路径变了」，保存时必须作废旧凭据、要求重新录入。
CONNECTION_KEYS = ('host', 'port', 'username', 'auth_type',
                   'jump_enabled', 'jump_host', 'jump_port', 'jump_username', 'jump_auth_type')

# 凭据 blob 统一形状：{'target': {...}, 'jump': {...}|None}。
# 历史数据里 target 槽直接就是凭据本体（含 password / private_key），读取时按此识别。


def split_secret(raw):
    """把凭据归一成 {'target':…, 'jump':…} 两槽。未知来源一律视为 target。"""
    if not raw: return {'target': None, 'jump': None}
    if isinstance(raw, dict) and ('target' in raw or 'jump' in raw):
        return {'target': raw.get('target') or None, 'jump': raw.get('jump') or None}
    return {'target': raw or None, 'jump': None}


def protect(data: bytes, decrypt=False, key_path=None) -> bytes:
    """Windows DPAPI or authenticated server-side encryption; never expose secrets to UI."""
    if os.name != 'nt':
        if key_path is None:
            raise RuntimeError('服务器凭据加密路径未配置')
        from compute.credential_crypto import protect_linux
        return protect_linux(data, decrypt=decrypt, key_path=Path(key_path))
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buf = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    output = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise RuntimeError('Windows 凭据加密/解密失败，请使用保存凭据的 Windows 账户')
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel = ctypes.WinDLL('kernel32')
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))


class NodeStore:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'nodes.sqlite3'
        self.key_path = root.resolve() / '.credentials.key'
        self.lock = threading.RLock()
        self.temporary = {}
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS credentials(id TEXT PRIMARY KEY, encrypted BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS host_keys(target TEXT PRIMARY KEY, algorithm TEXT, key TEXT, fingerprint TEXT);
            ''')

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection: yield connection
        finally: connection.close()

    def get(self, id):
        with self.db() as db:
            row = db.execute('SELECT config FROM nodes WHERE id=?', (id,)).fetchone()
        if not row: raise KeyError('节点不存在')
        return self.defaults(json.loads(row[0]))

    @staticmethod
    def defaults(node):
        """老节点的配置 JSON 里没有跳板机字段，读出来一律补默认值。"""
        for key, value in (('jump_enabled', False), ('jump_host', ''), ('jump_port', 22),
                           ('jump_username', ''), ('jump_auth_type', 'password')):
            node.setdefault(key, value)
        return node

    def list(self):
        with self.db() as db:
            return [self.defaults(json.loads(r[0])) for r in db.execute('SELECT config FROM nodes ORDER BY rowid')]

    def public(self, node):
        result = dict(node)
        cid = node['credential_id']
        rows = self.rows(node)
        saved = {'target': cid in rows, 'jump': cid + '#jump' in rows}
        if cid in rows:
            # 历史 blob 里可能同时挤着 target 与 jump，拆开统计
            parts = split_secret(json.loads(protect(rows[cid], decrypt=True, key_path=self.key_path)))
            saved['target'] = parts['target'] is not None
            saved['jump'] = saved['jump'] or parts['jump'] is not None
        temporary = self.temporary.get(node['id'], {})
        result['credential_saved'] = saved['target']
        result['jump_credential_saved'] = saved['jump']
        result['credential_available'] = saved['target'] or temporary.get('target') is not None
        result['jump_credential_available'] = saved['jump'] or temporary.get('jump') is not None
        return result

    def save(self, config, id=None, secret=None, persist=True):
        with self.lock:
            old = self.get(id) if id else {}
            id = id or uuid.uuid4().hex
            changed = any(old.get(k) != config.get(k) for k in CONNECTION_KEYS)
            parts = split_secret(secret)
            cid = 'runos_ssh_' + id
            node = {**config, 'id': id, 'credential_id': cid,
                    'created_at': old.get('created_at', now()), 'updated_at': now(),
                    'status': 'unknown', 'last_check_time': None, 'report': None}
            if old and config == {k: old.get(k) for k in config}:
                for k in ('status', 'last_check_time', 'report'): node[k] = old.get(k)
            with self.db() as db:
                # 目标机与跳板机分开存：换其中一个不牵连另一个。
                # 连接路径变了而某槽又没重录 → 该槽作废旧凭据（避免出现只有一半凭据的半残状态）。
                for slot, suffix in (('target', ''), ('jump', '#jump')):
                    given = parts[slot]
                    if not (changed or given is not None): continue
                    db.execute('DELETE FROM credentials WHERE id=?', (cid + suffix,))
                    self.temporary.get(id, {}).pop(slot, None)
                    if given is None: continue
                    if persist: db.execute('INSERT INTO credentials VALUES(?,?)', (cid + suffix, protect(json.dumps(given).encode(), key_path=self.key_path)))
                    else: self.temporary.setdefault(id, {})[slot] = given
                db.execute('INSERT OR REPLACE INTO nodes VALUES(?,?)', (id, json.dumps(node)))
            return self.public(node)

    def rows(self, node):
        cid = node['credential_id']
        with self.db() as db:
            return {row[0]: row[1] for row in db.execute(
                'SELECT id,encrypted FROM credentials WHERE id IN (?,?)', (cid, cid + '#jump'))}

    def bundle(self, node):
        """{'target':…, 'jump':…} —— SSH 连接入口只认这个形状。缺槽返回 None，由调用方决定报错口径。"""
        rows = self.rows(node)
        cid = node['credential_id']
        read = lambda key: json.loads(protect(rows[key], decrypt=True, key_path=self.key_path)) if key in rows else None
        parts = {'target': read(cid), 'jump': read(cid + '#jump')}
        for slot in ('target', 'jump'):
            value = self.temporary.get(node['id'], {}).get(slot)
            if value is not None: parts[slot] = value
        legacy = split_secret(parts['target'] or {})
        parts['target'] = legacy['target']
        parts['jump'] = parts['jump'] or legacy['jump']
        return parts

    def credential(self, node):
        target = self.bundle(node)['target']
        if target is None: raise ValueError('未保存凭据，请输入密码或 SSH Key')
        return target

    def report(self, id, report):
        with self.lock:
            node = self.get(id)
            node.update(status=report['status'], last_check_time=report['checked_at'], report=report)
            with self.db() as db:
                db.execute('UPDATE nodes SET config=? WHERE id=?', (json.dumps(node), id))

    def delete(self, id):
        with self.lock:
            node = self.get(id)
            with self.db() as db:
                db.execute('DELETE FROM credentials WHERE id IN (?,?)', (node['credential_id'], node['credential_id'] + '#jump'))
                db.execute('DELETE FROM nodes WHERE id=?', (id,))
            self.temporary.pop(id, None)

    def host_key(self, target):
        with self.db() as db:
            return db.execute('SELECT algorithm,key,fingerprint FROM host_keys WHERE target=?', (target,)).fetchone()

    def trust(self, target, algorithm, key, fingerprint):
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO host_keys VALUES(?,?,?,?)', (target, algorithm, key, fingerprint))
