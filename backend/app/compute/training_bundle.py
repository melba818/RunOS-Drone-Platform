"""Read-only YOLO dataset snapshots for remote detection training."""
import hashlib
import json
from pathlib import Path
import zipfile
import yaml

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}


def bundle_dataset(root: Path, destination: Path, remote_dataset: str):
    root=root.resolve()
    config=yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8-sig'))
    names=config.get('names')
    if not isinstance(names,(dict,list)) or not names: raise ValueError('数据集 data.yaml 缺少类别 names')
    base=Path(config.get('path') or root)
    if not base.is_absolute(): base=root/base
    if base.resolve()!=root: raise ValueError('data.yaml 必须指向当前选择的数据集')
    result={'path':remote_dataset,'names':names}
    counts={}
    def within(path):
        path=path.resolve()
        if not path.is_relative_to(root): raise ValueError('数据集引用了目录外文件，请先整理为独立 Split 数据集')
        return path
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED) as archive:
        for split in ('train','val','test'):
            entries=config.get(split)
            if not entries:
                if split in ('train','val'): raise ValueError(f'缺少 {split} 数据集')
                continue
            paths=[]
            for entry in entries if isinstance(entries,list) else [entries]:
                path=within(base/entry)
                if path.is_dir(): paths.extend(within(p) for p in sorted(path.rglob('*')) if p.suffix.lower() in IMAGE_EXTS)
                elif path.suffix.lower()=='.txt':
                    for line in path.read_text(encoding='utf-8-sig').splitlines():
                        if not line.strip(): continue
                        item=Path(line.strip())
                        paths.append(within(item if item.is_absolute() else path.parent/item))
                else: raise ValueError(f'{split} 图片路径不存在或格式不支持')
            paths=sorted(set(paths))
            if not paths: raise ValueError(f'{split} 没有可上传的图片')
            count=0
            for image in paths:
                if not image.is_file(): raise ValueError('图片清单包含不存在的文件')
                relative=image.relative_to(root)
                # Unique flat names prevent collisions across merged folders.
                stem=hashlib.sha256(relative.as_posix().encode()).hexdigest()[:20]
                archive.write(image,f'images/{split}/{stem}{image.suffix.lower()}')
                parts=list(relative.parts)
                if 'images' in parts:
                    parts[parts.index('images')]='labels'
                    label=within(root/Path(*parts).with_suffix('.txt'))
                else: label=within(image.with_suffix('.txt'))
                if label.exists():
                    text=label.read_text(encoding='utf-8-sig')
                    for line in text.splitlines():
                        if line.strip() and len(line.split())!=5: raise ValueError('远程训练当前只支持 YOLO 检测框标签')
                    archive.writestr(f'labels/{split}/{stem}.txt',text)
                    count+=bool(text.strip())
                else: archive.writestr(f'labels/{split}/{stem}.txt','')
            result[split]=f'images/{split}'
            counts[split]={'images':len(paths),'positive_labels':count}
        if not counts['train']['positive_labels']: raise ValueError('训练集没有目标标注')
        archive.writestr('data.yaml',yaml.safe_dump(result,allow_unicode=True))
    return counts
