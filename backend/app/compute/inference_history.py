"""Persistent video task references and last observed status; no SSH secrets."""
import json
import threading
import time
from pathlib import Path

_lock=threading.RLock()

def location():
    from config import settings
    return settings.WORKSPACE_DIR/'compute_nodes'/'inference_history.json'

def listing():
    with _lock:
        path=location()
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

def save(node_id,job_id,snapshot=None):
    with _lock:
        rows=listing()
        row=next((r for r in rows if r['node_id']==node_id and r['job_id']==job_id),None)
        if row is None:
            row={'node_id':node_id,'job_id':job_id,'created_at':time.time(),'snapshot':{}}
            rows.insert(0,row)
        if snapshot:row['snapshot']=snapshot
        path=location();path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8');temp.replace(path)
        return row
