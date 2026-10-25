"""Local account settings storage for standalone RunOS.

Settings are kept in a JSON file under the data root (one file per
admin account). This replaces the RunHive portal settings service
used by the original embedded deployment.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _settings_file(admin_id: int) -> Path:
    data_root = os.environ.get("RUNOS_DATA_ROOT", "").strip()
    if data_root:
        base = Path(data_root).expanduser()
    else:
        base = Path(__file__).parent.parent / "workspace"
    return base / f"account_settings_{int(admin_id)}.json"


def read_account_settings(admin_id: int) -> dict[str, Any]:
    path = _settings_file(admin_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_account_settings(admin_id: int, settings: dict[str, Any]) -> dict[str, Any]:
    path = _settings_file(admin_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"account_settings_{int(admin_id)}.json.tmp")
    with _LOCK:
        tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    return dict(settings)
