"""Reentrant locks shared by annotation writers and review decisions."""
import hashlib, threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from config import settings
try:
    import fcntl
except ImportError:
    fcntl = None
_guard = threading.Lock()
_locks = {}
_local = threading.local()

@contextmanager
def file_lock(path):
    path = Path(path)
    key = str(path.resolve())
    with _guard:
        mutex = _locks.setdefault(key, threading.RLock())
    with mutex:
        held = getattr(_local, 'held', {})
        if held.get(key):
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+b') as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            held[key] = True
            _local.held = held
            try:
                yield
            finally:
                held.pop(key, None)
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

@contextmanager
def annotation_lock(dataset_path):
    key = hashlib.sha256(str(Path(dataset_path).resolve()).encode()).hexdigest()
    with file_lock(settings.WORKSPACE_DIR / '.annotation-locks' / (key + '.lock')):
        yield

def locked_annotation_write(function):

    @wraps(function)
    def run(self, dataset_path, *args, **kwargs):
        with annotation_lock(dataset_path):
            return function(self, dataset_path, *args, **kwargs)
    return run
