"""Original-resolution Tile Pass. Geometry and provenance never use VLM scores."""
from __future__ import annotations
import copy
import math
import os
import tempfile
from pathlib import Path
from merge_ops import Candidate, MergedCandidate, iou_xyxy
from tiling import generate_tiles, from_tile

class TilePassCancelled(RuntimeError):
    pass

def validate_options(tile_size=640, tile_overlap=128, tile_batch_size=64):
    if tile_size not in (512, 640, 768, 1024):
        raise ValueError("切片尺寸必须是 512、640、768 或 1024")
    if not 0 <= tile_overlap < tile_size // 2:
        raise ValueError("重叠像素必须小于切片尺寸的一半")
    if not 1 <= tile_batch_size <= 64:
        raise ValueError("每轮切片数量必须在 1 到 64 之间")

def checkpoint(control=None):
    if control:
        if control['stop'].is_set():
            raise TilePassCancelled('任务已取消')
        control['pause'].wait()
        if control['stop'].is_set():
            raise TilePassCancelled('任务已取消')

def internal_edge(box, tile, image_size, margin=8):
    x, y, x2, y2 = tile.x0, tile.y0, tile.x1, tile.y1
    w, h = image_size
    return bool((x > 0 and box[0] <= margin) or
                (y > 0 and box[1] <= margin) or
                (x2 < w and x2-x-box[2] <= margin) or
                (y2 < h and y2-y-box[3] <= margin))

def collect_tiles(image, detect, *, tile_size=640, tile_overlap=128,
                  tile_batch_size=64, control=None, progress=None):
    validate_options(tile_size, tile_overlap, tile_batch_size)
    w, h = image.size
    tiles = generate_tiles(w, h, tile_size, tile_overlap)
    # A single tile equal to the image would repeat the full-image inference.
    if len(tiles) == 1 and (tiles[0].x0, tiles[0].y0, tiles[0].x1, tiles[0].y1) == (0, 0, w, h):
        tiles = []
    state = dict(enabled=True, tile_size=tile_size, overlap=tile_overlap,
                 total=len(tiles), completed=0, failed=0, rounds=math.ceil(len(tiles)/tile_batch_size),
                 round=0, candidates=0, coverage_complete=False, errors=[])
    out = []
    def report():
        if progress:
            progress(copy.deepcopy(state))
    report()
    for index, tile in enumerate(tiles):
        checkpoint(control)
        state['round'] = index // tile_batch_size + 1
        path = None
        try:
            fd, name = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            path = Path(name)
            image.crop((tile.x0, tile.y0, tile.x1, tile.y1)).save(path, 'PNG')
            got, error = detect(path)
            if error:
                raise RuntimeError(error)
            for candidate in got:
                local = candidate.bbox
                candidate.bbox = from_tile(local, tile, w, h)
                if candidate.bbox[2] <= candidate.bbox[0] or candidate.bbox[3] <= candidate.bbox[1]:
                    continue
                candidate.tile_index = index
                candidate.edge = internal_edge(local, tile, (w, h))
                candidate.raw.update(pass_type='tile', tile_index=index, tile_box=[tile.x0, tile.y0, tile.x1, tile.y1],
                                     local_box=list(local), edge=candidate.edge)
                out.append(candidate)
        except TilePassCancelled:
            raise
        except Exception as exc:
            state['failed'] += 1
            state['errors'].append(dict(tile_index=index, error=str(exc)[:200]))
        finally:
            if path:
                path.unlink(missing_ok=True)
        state['completed'] += 1
        state['candidates'] = len(out)
        report()
        # A dead model must not consume the remaining tile requests.
        if state['failed'] >= 2 and state['completed'] == state['failed']:
            state['aborted'] = '模型连续不可用，保留全图结果并报告覆盖不完整'
            break
    state['coverage_complete'] = state['completed'] == state['total'] and state['failed'] == 0
    report()
    return out, state

def _area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])

def _same(a, b, threshold):
    if a.label != b.label:
        return False
    if iou_xyxy(a.bbox, b.bbox) >= threshold:
        return True
    # Containment is allowed only for a tile-edge fragment and a complete box.
    if a.edge == b.edge:
        return False
    fragment, complete = (a, b) if a.edge else (b, a)
    if fragment.tile_index is None or _area(fragment.bbox) > _area(complete.bbox):
        return False
    aa, bb = fragment.bbox, complete.bbox
    inter = max(0,min(aa[2],bb[2])-max(aa[0],bb[0])) * max(0,min(aa[3],bb[3])-max(aa[1],bb[1]))
    fw, fh = aa[2]-aa[0], aa[3]-aa[1]
    cw, ch = bb[2]-bb[0], bb[3]-bb[1]
    return (inter / max(1e-6,_area(aa)) >= .85 and
            max(fw/cw, fh/ch) >= .8 and min(fw/cw,fh/ch) >= .25)

def strict_merge(candidates, iou_threshold=.5):
    # Complete-link clusters prevent an A-B-C overlap chain from swallowing neighbours.
    ordered = sorted(candidates, key=lambda c: (c.tile_index is not None, c.edge, -_area(c.bbox)))
    clusters = []
    for candidate in ordered:
        for cluster in clusters:
            if all(_same(candidate, member, iou_threshold) for member in cluster):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    result = []
    for cluster in clusters:
        fulls = [c for c in cluster if c.tile_index is None]
        tiles = [c for c in cluster if c.tile_index is not None]
        main = (fulls or [c for c in cluster if not c.edge] or cluster)[0]
        result.append(MergedCandidate(bbox=main.bbox, label=main.label,
            sources=sorted({c.source for c in cluster}), raw_boxes=[c.bbox for c in cluster],
            merge_reason='tile_geometry:%d|main=%s' % (len(cluster), 'full' if fulls else 'complete_tile'),
            n_full=len(fulls), n_tile=len(tiles), edge_only=bool(tiles) and not fulls and all(c.edge for c in tiles),
            extra={'members': cluster}))
    return result

def observation(c):
    return dict(bbox=list(c.bbox), source=c.source, model_id=c.raw.get('model_id'),
                pass_type=c.raw.get('pass_type', 'full'), tile_index=c.tile_index,
                tile_box=c.raw.get('tile_box'), local_box=c.raw.get('local_box'), edge=c.edge)

def consolidate(candidates):
    # Correlated full/tile observations remain ONE teacher, with all passes retained.
    output = []
    for source in dict.fromkeys(c.source for c in candidates):
        for merged in strict_merge([c for c in candidates if c.source == source]):
            members = merged.extra['members']
            main = next(c for c in members if c.bbox == merged.bbox)
            candidate = copy.deepcopy(main)
            candidate.raw['observations'] = [observation(c) for c in members]
            output.append(candidate)
    return output


def review_overlapping_tile_boxes(annotations):
    """Retain ambiguous smaller tile boxes for review, without merging possible neighbours."""
    from math import isfinite
    result = copy.deepcopy(annotations)
    def geometry(annotation):
        box = annotation.get('bbox')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            x, y, w, h = map(float, box)
        except (TypeError, ValueError):
            return None
        if not all(isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            return None
        return x, y, x + w, y + h
    fulls = [(index, annotation, geometry(annotation)) for index, annotation in enumerate(result)
             if annotation.get('decision') == 'ACCEPT'
             and 'full' in (annotation.get('evidence') or {}).get('passes', [])]
    conflicts = []
    for index, annotation in enumerate(result):
        passes = (annotation.get('evidence') or {}).get('passes', [])
        if annotation.get('decision') != 'ACCEPT' or 'tile' not in passes or 'full' in passes:
            continue
        box = geometry(annotation)
        if box is None:
            continue
        area = _area(box)
        matches = []
        for full_index, full, full_box in fulls:
            if full_box is None or full.get('class_name') != annotation.get('class_name'):
                continue
            full_area = _area(full_box)
            if area > full_area * .8:
                continue
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            if not (full_box[0] <= cx <= full_box[2] and full_box[1] <= cy <= full_box[3]):
                continue
            inter = max(0, min(box[2], full_box[2])-max(box[0], full_box[0])) * max(0, min(box[3], full_box[3])-max(box[1], full_box[1]))
            coverage = inter / max(area, 1e-6)
            if coverage < .75:
                continue
            matches.append(dict(full_index=full_index, full_bbox=list(full['bbox']),
                                tile_coverage=round(coverage, 4), iou=round(iou_xyxy(box, full_box), 4)))
        if matches:
            annotation['decision'] = 'REVIEW'
            reasons = annotation.setdefault('decision_reasons', [])
            if 'tile_overlaps_full_requires_review' not in reasons:
                reasons.append('tile_overlaps_full_requires_review')
            annotation.setdefault('evidence', {})['overlap_with_full'] = matches
            conflicts.append(dict(annotation_index=index, bbox=list(annotation['bbox']), matches=matches))
    return result, conflicts
