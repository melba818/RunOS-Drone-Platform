"""Persistent human review of saved candidates; no model inference."""
import base64, copy, hashlib, json, math, os, re, tempfile, time
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
from PIL import Image
from fastapi import HTTPException
from config import settings
from core.annotation_lock import annotation_lock, file_lock
YOLO_FORMATS = {'yolo', 'yolov5', 'yolov8', 'yolov9', 'yolov10', 'yolov11', 'yolov12', 'generic-images'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}

def _store():
    return settings.WORKSPACE_DIR / 'review-queue'

def _lock():
    return file_lock(_store() / '.lock')

def _path(job_id):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', job_id):
        raise HTTPException(404, '复核任务不存在')
    return _store() / (job_id + '.json')

def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def _save(data):
    atomic_bytes(_path(data['job_id']), json.dumps(data, ensure_ascii=False, indent=2).encode())

def _context(data):
    dataset_id = str(data.get('dataset_id') or '')
    dataset = settings.active_datasets.get(dataset_id)
    if not dataset:
        raise HTTPException(404, '对应数据集不存在')
    root = (settings.DATASETS_DIR / dataset_id).resolve()
    if not root.is_relative_to(settings.DATASETS_DIR.resolve()):
        raise HTTPException(404, '对应数据集不存在')
    return (root, dataset)

def _image(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404, '复核图片不存在或路径无效')
    return path

def _job_state(data):
    if data.get('kind') == 'batch':
        job = settings._text_annotate_jobs.get(data['job_id'])
        if not job:
            return ('deleted', '任务已删除，仅保留历史记录')
        status = job.get('status', 'done')
        if status == 'undone':
            return (status, '任务已撤销，复核记录仅供查看')
        if status in ('running', 'paused'):
            return (status, '请等待批量任务结束后再复核')
        return (status, '')
    return (data.get('status', 'done'), '')

def _summary(data):
    counts = {'pending': 0, 'accepted': 0, 'rejected': 0}
    for c in data['candidates']:
        status = c.get('review_status', 'pending')
        if status in counts:
            counts[status] += 1
    _, dataset = _context(data)
    status, reason = _job_state(data)
    return dict(job_id=data['job_id'], dataset_id=data['dataset_id'], dataset_name=dataset.get('name', data['dataset_id']), kind=data.get('kind', 'batch'), created=data.get('created', 0), status=status, readonly=bool(reason), readonly_reason=reason, summary=dict(counts, total=sum(counts.values())), revision=data.get('revision', 0))

def _candidate(key, relative, index, ann):
    identity = json.dumps([key, index, ann.get('bbox'), ann.get('class_name'), ann.get('evidence', {}).get('teachers', [])], sort_keys=True, ensure_ascii=False)
    return dict(candidate_id=hashlib.sha256(identity.encode()).hexdigest()[:32], image_key=key, image=relative, annotation=copy.deepcopy(ann), review_status='pending')

def _sync_batch(job_id):
    job = settings._text_annotate_jobs.get(job_id)
    if not job:
        raise HTTPException(404, '批量任务不存在')
    path = _path(job_id)
    data = json.loads(path.read_text()) if path.exists() else dict(job_id=job_id, dataset_id=job['dataset_id'], kind='batch', created=job.get('started_at') or time.time(), revision=0, candidates=[])
    known = {c['candidate_id'] for c in data['candidates']}
    changed = not path.exists()
    for file in sorted((settings.WORKSPACE_DIR / 'tile-pass' / job_id).glob('*.json')):
        if not re.fullmatch('[a-f0-9]{20}', file.stem):
            continue
        try:
            saved = json.loads(file.read_text())
        except (ValueError, OSError):
            raise HTTPException(409, '候选记录读取失败，请保留原记录并重试')
        for index, ann in enumerate(saved.get('annotations', [])):
            if ann.get('decision') != 'REVIEW':
                continue
            c = _candidate(file.stem, saved['image'], index, ann)
            if c['candidate_id'] not in known:
                data['candidates'].append(c)
                known.add(c['candidate_id'])
                changed = True
    if changed:
        data['revision'] += 1
        _save(data)
    return data

def register_preview(job_id, dataset_id, image_file, annotations, *, model_id='', verify_model_id=''):
    with _lock():
        root, _ = _context(dict(dataset_id=dataset_id))
        relative = Path(image_file).resolve().relative_to(root).as_posix()
        key = hashlib.sha256(relative.encode()).hexdigest()[:20]
        data = dict(job_id=job_id, dataset_id=dataset_id, kind='preview', status='done', created=time.time(), revision=1, model_id=model_id, verify_model_id=verify_model_id, candidates=[_candidate(key, relative, i, a) for i, a in enumerate(annotations) if a.get('decision') == 'REVIEW'])
        _save(data)
        return _summary(data)

def _load(job_id):
    path = _path(job_id)
    if job_id in settings._text_annotate_jobs:
        data = _sync_batch(job_id)
    elif path.exists():
        data = json.loads(path.read_text())
    else:
        raise HTTPException(404, '复核任务不存在')
    if data.get('pending_write'):
        root, _ = _context(data)
        with annotation_lock(root):
            _recover(data, root)
    return data

def _digest(content):
    return hashlib.sha256(content).hexdigest()

def _target(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(409, '标签路径无效')
    return path

def _restore_changes(changes, root):
    for c in changes:
        path = _target(root, c['path'])
        current = path.read_bytes() if path.exists() else b''
        if _digest(current) != c['after_hash']:
            continue
        if c['existed']:
            atomic_bytes(path, base64.b64decode(c['before']))
        else:
            path.unlink(missing_ok=True)

def _finish(data, transaction):
    c = next((c for c in data['candidates'] if c['candidate_id'] == transaction['candidate_id']))
    c.update(review_status='accepted', resolved_annotation=transaction['annotation'], reviewed_at=transaction['at'])
    data.pop('pending_write', None)
    data['revision'] += 1
    _save(data)

def _recover(data, root):
    tx = data['pending_write']
    changes = tx['changes']
    if all((_digest(_target(root, c['path']).read_bytes() if _target(root, c['path']).exists() else b'') == c['after_hash'] for c in changes)):
        _finish(data, tx)
    else:
        _restore_changes(changes, root)
        data.pop('pending_write', None)
        _save(data)

def _classes(root, dataset):
    classes = settings.model_manager._get_dataset_classes(root, dataset.get('format', 'yolo')) or dataset.get('classes') or []
    return [str(c) for c in (classes.values() if isinstance(classes, dict) else classes)]

def _bbox(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise HTTPException(422, '候选框需要 x、y、宽、高四个值')
    try:
        x, y, w, h = [float(v) for v in box]
    except (TypeError, ValueError):
        raise HTTPException(422, '候选框坐标必须是数字')
    if not all((math.isfinite(v) for v in (x, y, w, h))) or x < 0 or y < 0 or (w <= 0) or (h <= 0) or (x + w > width + 0.01) or (y + h > height + 0.01):
        raise HTTPException(422, '候选框必须位于图片内，宽高必须大于零')
    return [round(x, 6), round(y, 6), round(w, 6), round(h, 6)]

def _iou(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, right - x) * max(0, bottom - y)
    return inter / max(1e-12, a[2] * a[3] + b[2] * b[3] - inter)

def _duplicate(existing, ann):
    return any((a.get('class_name') == ann['class_name'] and a.get('bbox') and (_iou(a['bbox'], ann['bbox']) >= 0.98) for a in existing))

def _native(root, dataset, image):
    """Locate the exact file and read current labels without rewriting unrelated data."""
    fmt = dataset.get('format', 'yolo')
    relative = image.relative_to(root)
    with Image.open(image) as im:
        width, height = im.size
    classes = _classes(root, dataset)
    if fmt in YOLO_FORMATS:
        parts = list(relative.parts)
        pos = next((i for i, p in enumerate(parts) if p.lower() in ('images', 'imgs')), None)
        if pos is not None:
            parts[pos] = 'labels'
            path = root.joinpath(*parts).with_suffix('.txt')
        else:
            path = root / 'labels' / image.with_suffix('.txt').name
        mapping = {i: n for i, n in enumerate(classes)}
        for file in list(root.glob('*.yaml')) + list(root.glob('*.yml')):
            cfg = yaml.safe_load(file.read_text()) or {}
            if isinstance(cfg.get('names'), dict):
                mapping = {int(i): str(n) for i, n in cfg['names'].items()}
            if 'names' in cfg:
                break
        path = _target(root, path.relative_to(root))
        content = path.read_text() if path.exists() else ''
        existing = []
        for line in content.splitlines():
            try:
                fields = line.split()
                cls = int(fields[0])
                nums = [float(v) for v in fields[1:]]
                if len(nums) == 4:
                    xc, yc, w, h = nums
                    box = [(xc - w / 2) * width, (yc - h / 2) * height, w * width, h * height]
                elif len(nums) >= 6 and len(nums) % 2 == 0:
                    xs = nums[::2]
                    ys = nums[1::2]
                    box = [min(xs) * width, min(ys) * height, (max(xs) - min(xs)) * width, (max(ys) - min(ys)) * height]
                else:
                    continue
                existing.append(dict(class_name=mapping.get(cls, str(cls)), bbox=box))
            except (ValueError, IndexError):
                continue
        return (path, dict(format=fmt, content=content, mapping=mapping, width=width, height=height), existing)
    if fmt == 'coco':
        for file in list(root.glob('*.json')) + list((root / 'annotations').glob('*.json')):
            try:
                doc = json.loads(_target(root, file.relative_to(root)).read_text())
            except (ValueError, OSError):
                continue
            if not all((k in doc for k in ('images', 'annotations', 'categories'))):
                continue
            exact = [a for a in doc['images'] if str(a.get('file_name', '')).replace('\\', '/').lstrip('./') == relative.as_posix()]
            if not exact:
                exact = [a for a in doc['images'] if Path(a.get('file_name', '')).name == image.name]
            if len(exact) != 1:
                raise HTTPException(409, 'COCO 图片路径不唯一或缺少记录')
            rec = exact[0]
            cats = {c['id']: c['name'] for c in doc['categories']}
            existing = [dict(class_name=cats.get(a['category_id'], str(a['category_id'])), bbox=a['bbox']) for a in doc['annotations'] if a['image_id'] == rec['id'] and a.get('bbox')]
            return (file, dict(format=fmt, document=doc, image_id=rec['id'], width=width, height=height), existing)
        raise HTTPException(409, '没有可用的 COCO 标注文件')
    if fmt in ('voc', 'pascal-voc'):
        matches = list(root.glob('**/' + image.stem + '.xml'))
        if len(matches) > 1:
            raise HTTPException(409, 'VOC 标注文件同名，请先消除歧义')
        path = matches[0] if matches else root / 'Annotations' / (image.stem + '.xml')
        path = _target(root, path.relative_to(root))
        doc = ET.fromstring(path.read_bytes()) if path.exists() else ET.Element('annotation')
        existing = []
        for obj in doc.findall('object'):
            box = obj.find('bndbox')
            if box is None:
                continue
            x1, y1, x2, y2 = [float(box.findtext(k, '0')) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            existing.append(dict(class_name=obj.findtext('name', ''), bbox=[x1, y1, x2 - x1, y2 - y1]))
        return (path, dict(format=fmt, document=doc, width=width, height=height, image_name=image.name), existing)
    if fmt == 'labelme':
        preferred = image.with_suffix('.json')
        matches = [p for p in root.glob('**/' + image.stem + '.json') if p.name != 'dataset_metadata.json']
        if preferred.exists():
            path = preferred
        elif len(matches) == 1:
            path = matches[0]
        elif len(matches) > 1:
            raise HTTPException(409, 'LabelMe 标注文件同名，请先消除歧义')
        else:
            path = preferred
        path = _target(root, path.relative_to(root))
        doc = json.loads(path.read_text()) if path.exists() else dict(version='5.0.1', flags={}, shapes=[], imagePath=image.name, imageData=None, imageWidth=width, imageHeight=height)
        existing = []
        for s in doc.get('shapes', []):
            pts = s.get('points') or []
            if len(pts) < 2:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            existing.append(dict(class_name=s.get('label', ''), bbox=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]))
        return (path, dict(format=fmt, document=doc, width=width, height=height), existing)
    raise HTTPException(409, '当前数据集格式暂不支持复核写入')

def _changes(root, dataset, image, ann, candidate_id):
    path, state, existing = _native(root, dataset, image)
    if _duplicate(existing, ann):
        return ([], True)
    x, y, w, h = ann['bbox']
    name = ann['class_name']
    fmt = state['format']
    extras = []
    if fmt in YOLO_FORMATS:
        cls = next((i for i, n in state['mapping'].items() if n == name), None)
        if cls is None:
            raise HTTPException(422, '类别没有对应的 YOLO 编号')
        line = f'{cls} {(x + w / 2) / state['width']:.6f} {(y + h / 2) / state['height']:.6f} {w / state['width']:.6f} {h / state['height']:.6f}\n'
        old = state['content']
        content = (old + ('' if not old or old.endswith('\n') else '\n') + line).encode()
        if fmt == 'generic-images':
            if not list(root.glob('*.yaml')) and (not list(root.glob('*.yml'))):
                extras.append((root / 'data.yaml', yaml.safe_dump({'names': list(state['mapping'].values()), 'nc': len(state['mapping'])}, allow_unicode=True).encode()))
            meta = _target(root, 'dataset_metadata.json')
            doc = json.loads(meta.read_text()) if meta.exists() else dict(dataset)
            doc['format'] = 'yolo'
            extras.append((meta, json.dumps(doc, ensure_ascii=False, indent=2).encode()))
    elif fmt == 'coco':
        doc = state['document']
        category = next((c['id'] for c in doc['categories'] if c['name'] == name), None)
        if category is None:
            raise HTTPException(422, '类别不在 COCO 类别表中')
        ids = [int(a.get('id', 0)) for a in doc['annotations'] if str(a.get('id', 0)).isdigit()]
        doc['annotations'].append(dict(id=max(ids, default=0) + 1, image_id=state['image_id'], category_id=category, bbox=[x, y, w, h], area=w * h, iscrowd=0))
        content = json.dumps(doc, ensure_ascii=False, indent=2).encode()
    elif fmt in ('voc', 'pascal-voc'):
        doc = state['document']
        if doc.find('filename') is None:
            ET.SubElement(doc, 'filename').text = state['image_name']
        if doc.find('size') is None:
            size = ET.SubElement(doc, 'size')
            for k, v in [('width', state['width']), ('height', state['height']), ('depth', 3)]:
                ET.SubElement(size, k).text = str(v)
        obj = ET.SubElement(doc, 'object')
        ET.SubElement(obj, 'name').text = name
        ET.SubElement(obj, 'difficult').text = '0'
        box = ET.SubElement(obj, 'bndbox')
        for k, v in [('xmin', x), ('ymin', y), ('xmax', x + w), ('ymax', y + h)]:
            ET.SubElement(box, k).text = str(round(v, 6))
        content = ET.tostring(doc, encoding='utf-8', xml_declaration=True)
    else:
        doc = state['document']
        doc.setdefault('shapes', []).append(dict(label=name, points=[[x, y], [x + w, y + h]], group_id=None, shape_type='rectangle', flags={}))
        content = json.dumps(doc, ensure_ascii=False, indent=2).encode()
    changes = []
    for target, after in [(path, content)] + extras:
        before = target.read_bytes() if target.exists() else b''
        changes.append(dict(path=target.relative_to(root).as_posix(), existed=target.exists(), before=base64.b64encode(before).decode(), after=base64.b64encode(after).decode(), before_hash=_digest(before), after_hash=_digest(after)))
    return (changes, False)

def list_jobs(dataset_id=None):
    with _lock():
        ids = set((p.stem for p in _store().glob('*.json'))) if _store().exists() else set()
        ids.update((jid for jid, j in settings._text_annotate_jobs.items() if j.get('tile_pass') or j.get('review_images')))
        jobs = []
        for jid in ids:
            try:
                data = _load(jid)
                if dataset_id and data['dataset_id'] != dataset_id:
                    continue
                result = _summary(data)
                if result['summary']['total']:
                    jobs.append(result)
            except HTTPException as e:
                if e.status_code != 404:
                    raise

        def created(j):
            value = j.get('created', 0)
            try:
                return float(value)
            except (TypeError, ValueError):
                try:
                    return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
                except ValueError:
                    return 0
        jobs.sort(key=created, reverse=True)
        return jobs

def enrich_job(job):
    if not job.get('tile_pass') and (not job.get('review_images')):
        return job
    with _lock():
        try:
            summary = _summary(_load(job['job_id']))
        except HTTPException as error:
            if error.status_code != 404:
                raise
            return dict(job, review_readonly=True, review_error='对应数据集或候选记录不存在')
        return dict(job, review_summary=summary['summary'], review_readonly=summary['readonly'])

def get_job(job_id):
    with _lock():
        data = _load(job_id)
        result = _summary(data)
        root, dataset = _context(data)
        result['classes'] = _classes(root, dataset)
        result['format'] = dataset.get('format', 'yolo')
        result['candidates'] = []
        for c in data['candidates']:
            ann = c.get('resolved_annotation') or c['annotation']
            result['candidates'].append(dict(candidate_id=c['candidate_id'], image_key=c['image_key'], image=c['image'], bbox=ann.get('bbox'), class_name=ann.get('class_name'), review_status=c['review_status'], decision_reasons=c['annotation'].get('decision_reasons', []), evidence=c['annotation'].get('evidence', {}), reviewed_at=c.get('reviewed_at')))
        return result

def candidate_detail(job_id, candidate_id):
    with _lock():
        data = _load(job_id)
        c = next((c for c in data['candidates'] if c['candidate_id'] == candidate_id), None)
        if not c:
            raise HTTPException(404, '候选框不存在')
        root, dataset = _context(data)
        image = _image(root, c['image'])
        with Image.open(image) as im:
            width, height = im.size
        with annotation_lock(root):
            try:
                _, _, existing = _native(root, dataset, image)
            except HTTPException:
                existing = []
        return dict(width=width, height=height, existing_annotations=existing, image_url=f'/api/reviews/jobs/{job_id}/candidates/{candidate_id}/image')

def candidate_image(job_id, candidate_id):
    with _lock():
        data = _load(job_id)
        c = next((c for c in data['candidates'] if c['candidate_id'] == candidate_id), None)
        if not c:
            raise HTTPException(404, '候选框不存在')
        root, _ = _context(data)
        return _image(root, c['image'])

def decide(job_id, candidate_id, action, revision, *, bbox=None, class_name=None):
    with _lock():
        data = _load(job_id)
        root, dataset = _context(data)
        with annotation_lock(root):
            _, reason = _job_state(data)
            if reason:
                raise HTTPException(409, reason)
            c = next((c for c in data['candidates'] if c['candidate_id'] == candidate_id), None)
            if not c:
                raise HTTPException(404, '候选框不存在')
            image = _image(root, c['image'])
            with Image.open(image) as im:
                width, height = im.size
            ann = copy.deepcopy(c['annotation'])
            ann['class_name'] = class_name if class_name is not None else ann['class_name']
            target = 'accepted' if action == 'accept' else 'rejected'
            if action == 'accept':
                ann['bbox'] = _bbox(bbox if bbox is not None else ann['bbox'], width, height)
            if c['review_status'] != 'pending':
                old = c.get('resolved_annotation') or c['annotation']
                if c['review_status'] == target and (action != 'accept' or (old['bbox'] == ann['bbox'] and old['class_name'] == ann['class_name'])):
                    return dict(success=True, idempotent=True, job=_summary(data))
                raise HTTPException(409, '该候选已处理，请刷新复核列表')
            if revision != data['revision']:
                raise HTTPException(409, '复核进度已更新，请刷新后重试')
            if action == 'reject':
                c.update(review_status='rejected', reviewed_at=time.time())
                data['revision'] += 1
                _save(data)
                return dict(success=True, job=_summary(data))
            classes = _classes(root, dataset)
            if ann['class_name'] not in classes:
                raise HTTPException(422, '请选择当前数据集已有类别')
            ann.update(class_id=classes.index(ann['class_name']), decision='ACCEPT', decision_reasons=['human_review'])
            changes, duplicate = _changes(root, dataset, image, ann, candidate_id)
            tx = dict(candidate_id=candidate_id, annotation=ann, at=time.time(), changes=changes)
            data['pending_write'] = tx
            _save(data)
            try:
                for change in changes:
                    atomic_bytes(_target(root, change['path']), base64.b64decode(change['after']))
            except OSError:
                _restore_changes(changes, root)
                data.pop('pending_write', None)
                _save(data)
                raise HTTPException(503, '标签保存失败，候选仍待复核，请重试')
            _finish(data, tx)
            if dataset.get('format') == 'generic-images':
                dataset['format'] = 'yolo'
            settings._images_cache.pop(data['dataset_id'], None)
            return dict(success=True, deduplicated=duplicate, job=_summary(data))
