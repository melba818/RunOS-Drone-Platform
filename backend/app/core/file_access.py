"""Private worker filesystem boundaries. Shared model policy is deliberately separate."""
from contextlib import contextmanager
import mimetypes
import os
from pathlib import Path, PureWindowsPath
import stat

from fastapi import HTTPException
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse


def owned_path(value, *, root=None, exists=False, directory=False):
    """Authorize against a server-owned root, never a client tenant identifier."""
    from config import settings
    boundary = settings.WORKSPACE_DIR.resolve()
    scope = Path(root).resolve() if root is not None else boundary
    try:
        if not scope.is_relative_to(boundary):
            raise ValueError("Unapproved private root")
        raw = str(value).replace("\\", "/")
        p = Path(raw).expanduser()
        if ".." in p.parts or (PureWindowsPath(raw).drive and os.name != "nt"):
            raise ValueError("Traversal or foreign drive")
        candidate = (p if p.is_absolute() else scope / p).resolve()
        if not candidate.is_relative_to(scope):
            raise ValueError("Outside authorized root")
        if exists and not candidate.exists():
            raise FileNotFoundError
        if directory and candidate.exists() and not candidate.is_dir():
            raise NotADirectoryError
        return candidate
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(403, "File access is outside this account's permitted directory or is unavailable") from None


@contextmanager
def owned_directory(value):
    """Pin a Linux directory by walking relative components without symlink following.

    The descriptor pins this traversal, not all later pathname-based parser work.
    Independent UIDs/mount namespaces are still required for hostile same-UID peers.
    """
    from config import settings
    path = owned_path(value, exists=True, directory=True)
    if os.name != "posix" or os.open not in os.supports_dir_fd:
        raise HTTPException(503, "Secure directory access is unavailable on this platform")
    root = settings.WORKSPACE_DIR.resolve()
    fd = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(root, flags)
        for part in path.relative_to(root).parts:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield path, fd
    except OSError:
        raise HTTPException(403, "Directory access was denied") from None
    finally:
        if fd is not None:
            os.close(fd)


def checked_tree(value):
    """Reject external symlink entries before a parser or archive traverses data."""
    path = owned_path(value, exists=True, directory=True)
    pending = [path]
    visited = set()
    entries = 0
    while pending:
        current = pending.pop()
        with owned_directory(current) as (current, descriptor):
            metadata = os.fstat(descriptor)
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in visited:
                continue
            visited.add(identity)
            for entry in os.scandir(descriptor):
                entries += 1
                if entries > 100000:
                    raise HTTPException(413, "Directory inspection budget exceeded")
                child = owned_path(current / entry.name, exists=True)
                info = child.stat()
                if stat.S_ISDIR(info.st_mode):
                    pending.append(child)
                elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise HTTPException(403, "Directory contains an unsafe file")
    return path


def owned_parent(value):
    from config import settings
    path = owned_path(value)
    return path.parent if path != settings.WORKSPACE_DIR.resolve() else None


def owned_children(value):
    with owned_directory(value) as (path, descriptor):
        names = sorted(entry.name for entry in os.scandir(descriptor))
        children = []
        for name in names:
            try:
                owned_path(path / name, exists=True)
            except HTTPException:
                # Shared models have a separate API/policy. A private-data picker
                # must not traverse these aliases or fail listing its own root.
                continue
            children.append(path / name)
        return children


def owned_file_response(value, *, media_type=None, filename=None, headers=None):
    """Stream the verified descriptor; never reopen a checked pathname for download."""
    from config import settings
    path = owned_path(value, exists=True)
    root = settings.WORKSPACE_DIR.resolve()
    fd = None
    try:
        with owned_directory(path.parent) as (_, parent):
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("Not a private regular file")
    except OSError:
        if fd is not None:
            os.close(fd)
        raise HTTPException(403, "File access was denied") from None
    closed = False
    def close():
        nonlocal closed
        if not closed:
            closed = True
            os.close(fd)
    def content():
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                yield chunk
        finally:
            close()
    response_headers = dict(headers or {})
    response_headers["Content-Length"] = str(metadata.st_size)
    if filename:
        from urllib.parse import quote
        response_headers["Content-Disposition"] = "attachment; filename*=utf-8''" + quote(filename)
    return StreamingResponse(content(), media_type=media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream", headers=response_headers, background=BackgroundTask(close))


def owned_file_bytes(value, *, limit=128 * 1024 * 1024):
    """Read a pinned private regular file with a bounded parser-input budget."""
    path = owned_path(value, exists=True)
    fd = None
    try:
        with owned_directory(path.parent) as (_, parent):
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("Unsafe parser input")
        if info.st_size > limit:
            raise HTTPException(413, "Parser input exceeds its byte budget")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, limit + 1 - total))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise HTTPException(413, "Parser input exceeds its byte budget")
    except OSError:
        raise HTTPException(403, "File access was denied") from None
    finally:
        if fd is not None:
            os.close(fd)


def save_model_upload(name, content):
    """Confine a write to the configured model root; this grants no shared role rights."""
    from config import settings
    import secrets
    raw = str(name or "")
    if not raw or raw in (".", "..") or "/" in raw or "\\" in raw or PureWindowsPath(raw).drive or any(ord(c) < 32 for c in raw):
        raise HTTPException(403, "Model filename must be a single permitted name")
    scope = settings.MODELS_DIR.resolve()
    parent = fd = None
    temporary = ".upload-" + secrets.token_hex(16)
    created = False
    try:
        parent = os.open(scope, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            old = os.stat(raw, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            old = None
        if old and (not stat.S_ISREG(old.st_mode) or old.st_nlink != 1):
            raise OSError("Unsafe existing model target")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o666, dir_fd=parent)
        created = True
        view = memoryview(content)
        while view:
            view = view[os.write(fd, view):]
        if old:
            os.fchmod(fd, stat.S_IMODE(old.st_mode))
        os.close(fd)
        fd = None
        # rename replaces a directory entry, never follows a raced destination symlink.
        os.replace(temporary, raw, src_dir_fd=parent, dst_dir_fd=parent)
        created = False
        return scope / raw
    except OSError:
        raise HTTPException(403, "Model file write was denied") from None
    finally:
        if fd is not None:
            os.close(fd)
        if parent is not None:
            if created:
                os.unlink(temporary, dir_fd=parent)
            os.close(parent)
