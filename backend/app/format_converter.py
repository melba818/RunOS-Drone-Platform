"""
Format Converter - Convert between different CV annotation formats
"""

import os
import json
import math
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw


class FormatConverter:
    """Convert between different annotation formats"""
    
    SUPPORTED_FORMATS = {
        "yolo": {"name": "YOLO/YOLOv5/v8/v9/v10/v11", "extensions": [".txt"], "task": ["detection", "segmentation"]},
        "yolo-obb": {"name": "YOLO OBB", "extensions": [".txt"], "task": ["obb-detection"]},
        "coco": {"name": "COCO JSON", "extensions": [".json"], "task": ["detection", "segmentation", "keypoint"]},
        "coco-panoptic": {"name": "COCO Panoptic", "extensions": [".json", ".png"], "task": ["panoptic-segmentation"]},
        "pascal-voc": {"name": "Pascal VOC XML", "extensions": [".xml"], "task": ["detection"]},
        "createml": {"name": "CreateML JSON", "extensions": [".json"], "task": ["detection"]},
        "tensorflow-csv": {"name": "TensorFlow CSV", "extensions": [".csv"], "task": ["detection"]},
        "tfrecord": {"name": "TFRecord", "extensions": [".record"], "task": ["detection", "classification"]},
        "labelme": {"name": "LabelMe JSON", "extensions": [".json"], "task": ["detection", "segmentation"]},
        "classification-folder": {"name": "Classification Folders", "extensions": [], "task": ["classification"]},
        "cityscapes": {"name": "Cityscapes", "extensions": [".json", ".png"], "task": ["segmentation"]},
        "ade20k": {"name": "ADE20K", "extensions": [".png"], "task": ["segmentation"]},
        "dota": {"name": "DOTA", "extensions": [".txt"], "task": ["obb-detection"]},
        "csv": {"name": "CSV", "extensions": [".csv"], "task": ["detection"]},
    }
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
    
    def list_formats(self) -> List[Dict[str, Any]]:
        """List all supported formats"""
        return [
            {"id": fmt_id, **fmt_info}
            for fmt_id, fmt_info in self.SUPPORTED_FORMATS.items()
        ]
    
    def convert(
        self,
        source_path: Path,
        output_path: Path,
        source_format: str,
        target_format: str
    ):
        """Convert a dataset from one format to another"""
        source_path = Path(source_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # First, load the dataset into a unified internal format
        unified_data = self._load_unified(source_path, source_format)
        
        # Then, export to target format
        self._export_unified(unified_data, output_path, target_format)
        
        # Copy images
        self._copy_images(source_path, output_path, unified_data, target_format)
    
    def _load_unified(self, path: Path, format_name: str) -> Dict[str, Any]:
        """Load dataset into unified internal format"""
        loaders = {
            "yolo": self._load_yolo,
            "yolov5": self._load_yolo,
            "yolov8": self._load_yolo,
            "yolov9": self._load_yolo,
            "yolov10": self._load_yolo,
            "yolov11": self._load_yolo,
            "yolov12": self._load_yolo,
            "yolo_seg": self._load_yolo,
            "yolo-seg": self._load_yolo,
            "yolo-obb": self._load_yolo_obb,
            "yolo_obb": self._load_yolo_obb,
            "yolov8-obb": self._load_yolo_obb,
            "coco": self._load_coco,
            "coco-panoptic": self._load_coco_panoptic,
            "coco_panoptic": self._load_coco_panoptic,
            "pascal-voc": self._load_voc,
            "voc": self._load_voc,
            "createml": self._load_createml,
            "tensorflow-csv": self._load_tensorflow_csv,
            "csv": self._load_tensorflow_csv,
            "labelme": self._load_labelme,
            "classification-folder": self._load_classification,
            "classification": self._load_classification,
            "cityscapes": self._load_cityscapes,
            "ade20k": self._load_ade20k,
            "dota": self._load_dota,
            "tfrecord": self._load_tfrecord,
        }
        
        loader = loaders.get(format_name, self._load_generic)
        return loader(path)
    
    def _export_unified(self, data: Dict[str, Any], output_path: Path, format_name: str):
        """Export unified data to target format"""
        exporters = {
            "yolo": self._export_yolo,
            "yolov5": self._export_yolo,
            "yolov8": self._export_yolo,
            "yolov9": self._export_yolo,
            "yolov10": self._export_yolo,
            "yolov11": self._export_yolo,
            "yolov12": self._export_yolo,
            "yolo_seg": self._export_yolo,
            "yolo-seg": self._export_yolo,
            "yolo-obb": self._export_yolo_obb,
            "yolo_obb": self._export_yolo_obb,
            "yolov8-obb": self._export_yolo_obb,
            "coco": self._export_coco,
            "coco-panoptic": self._export_coco_panoptic,
            "coco_panoptic": self._export_coco_panoptic,
            "pascal-voc": self._export_voc,
            "voc": self._export_voc,
            "createml": self._export_createml,
            "tensorflow-csv": self._export_tensorflow_csv,
            "csv": self._export_tensorflow_csv,
            "labelme": self._export_labelme,
            "classification-folder": self._export_classification,
            "classification": self._export_classification,
            "cityscapes": self._export_cityscapes,
            "ade20k": self._export_ade20k,
            "dota": self._export_dota,
            "tfrecord": self._export_tfrecord,
        }
        
        exporter = exporters.get(format_name, self._export_yolo)
        exporter(data, output_path)
    
    # ============== LOADERS ==============
    
    def _load_yolo(self, path: Path) -> Dict[str, Any]:
        """Load YOLO format into unified format"""
        data = {
            "classes": [],
            "images": []
        }
        
        # Load classes from yaml
        for yaml_file in list(path.glob("*.yaml")) + list(path.glob("*.yml")):
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                    if "names" in config:
                        if isinstance(config["names"], dict):
                            data["classes"] = list(config["names"].values())
                        else:
                            data["classes"] = config["names"]
                    break
            except Exception:
                pass
        
        # Find images and labels
        image_dirs = ["images", "train/images", "val/images", "test/images", "images/train", "images/val", "images/test", ""]
        
        for img_dir in image_dirs:
            img_path = path / img_dir if img_dir else path
            if not img_path.exists():
                continue
            
            label_dir = str(img_dir).replace("images", "labels") if img_dir else "labels"
            label_path = path / label_dir if label_dir else path
            
            for img_file in img_path.iterdir():
                if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                    # Get image dimensions
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size
                    except Exception:
                        width, height = 0, 0
                    
                    image_data = {
                        "id": img_file.stem,
                        "filename": img_file.name,
                        "path": str(img_file.relative_to(path)),
                        "width": width,
                        "height": height,
                        "annotations": []
                    }
                    
                    # Load annotations
                    label_file = label_path / f"{img_file.stem}.txt"
                    if label_file.exists():
                        with open(label_file) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    
                                    if len(parts) == 5:
                                        # Bounding box: class x_center y_center width height
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        w = float(parts[3])
                                        h = float(parts[4])
                                        
                                        # Convert to absolute coordinates
                                        x_min = (x_center - w/2) * width
                                        y_min = (y_center - h/2) * height
                                        x_max = (x_center + w/2) * width
                                        y_max = (y_center + h/2) * height
                                        
                                        image_data["annotations"].append({
                                            "type": "bbox",
                                            "class_id": class_id,
                                            "class_name": data["classes"][class_id] if class_id < len(data["classes"]) else f"class_{class_id}",
                                            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min]
                                        })
                                    else:
                                        # Polygon segmentation
                                        points = [float(p) for p in parts[1:]]
                                        # Convert normalized to absolute
                                        abs_points = []
                                        for i in range(0, len(points), 2):
                                            abs_points.append(points[i] * width)
                                            abs_points.append(points[i+1] * height)
                                        
                                        image_data["annotations"].append({
                                            "type": "polygon",
                                            "class_id": class_id,
                                            "class_name": data["classes"][class_id] if class_id < len(data["classes"]) else f"class_{class_id}",
                                            "segmentation": [abs_points]
                                        })
                    
                    data["images"].append(image_data)
        
        return data
    
    def _load_coco(self, path: Path) -> Dict[str, Any]:
        """Load COCO format into unified format"""
        data = {
            "classes": [],
            "images": []
        }
        
        for json_file in list(path.glob("*.json")) + list(path.glob("annotations/*.json")):
            try:
                with open(json_file) as f:
                    coco_data = json.load(f)
                
                if all(key in coco_data for key in ["images", "annotations", "categories"]):
                    # Build class list
                    cat_map = {}
                    for cat in coco_data["categories"]:
                        cat_map[cat["id"]] = len(data["classes"])
                        data["classes"].append(cat["name"])
                    
                    # Build annotation map
                    ann_map = {}
                    for ann in coco_data["annotations"]:
                        img_id = ann["image_id"]
                        if img_id not in ann_map:
                            ann_map[img_id] = []
                        
                        annotation = {
                            "class_id": cat_map.get(ann["category_id"], 0),
                            "class_name": data["classes"][cat_map.get(ann["category_id"], 0)] if cat_map.get(ann["category_id"], 0) < len(data["classes"]) else "unknown",
                            "bbox": ann.get("bbox", [])
                        }
                        
                        if ann.get("segmentation"):
                            annotation["type"] = "polygon"
                            annotation["segmentation"] = ann["segmentation"]
                        else:
                            annotation["type"] = "bbox"
                        
                        ann_map[img_id].append(annotation)
                    
                    # Build image list
                    for img in coco_data["images"]:
                        data["images"].append({
                            "id": str(img["id"]),
                            "filename": img["file_name"],
                            "path": img["file_name"],
                            "width": img.get("width", 0),
                            "height": img.get("height", 0),
                            "annotations": ann_map.get(img["id"], [])
                        })
                    
                    break
            except Exception:
                pass
        
        return data
    
    def _load_voc(self, path: Path) -> Dict[str, Any]:
        """Load Pascal VOC format into unified format"""
        data = {
            "classes": [],
            "images": []
        }
        
        class_set = set()
        
        for xml_file in path.glob("**/*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                if root.tag == "annotation":
                    filename_elem = root.find("filename")
                    size_elem = root.find("size")
                    
                    if filename_elem is not None:
                        width = int(size_elem.find("width").text) if size_elem is not None else 0
                        height = int(size_elem.find("height").text) if size_elem is not None else 0
                        
                        image_data = {
                            "id": xml_file.stem,
                            "filename": filename_elem.text,
                            "path": filename_elem.text,
                            "width": width,
                            "height": height,
                            "annotations": []
                        }
                        
                        for obj in root.findall("object"):
                            name = obj.find("name")
                            bndbox = obj.find("bndbox")
                            
                            if name is not None and bndbox is not None:
                                class_name = name.text
                                class_set.add(class_name)
                                
                                x_min = int(float(bndbox.find("xmin").text))
                                y_min = int(float(bndbox.find("ymin").text))
                                x_max = int(float(bndbox.find("xmax").text))
                                y_max = int(float(bndbox.find("ymax").text))
                                
                                image_data["annotations"].append({
                                    "type": "bbox",
                                    "class_name": class_name,
                                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min]
                                })
                        
                        data["images"].append(image_data)
            except Exception:
                pass
        
        data["classes"] = sorted(list(class_set))
        
        # Update class IDs
        class_to_id = {name: idx for idx, name in enumerate(data["classes"])}
        for img in data["images"]:
            for ann in img["annotations"]:
                ann["class_id"] = class_to_id.get(ann["class_name"], 0)
        
        return data
    
    def _load_createml(self, path: Path) -> Dict[str, Any]:
        """Load CreateML format into unified format"""
        data = {
            "classes": [],
            "images": []
        }
        
        class_set = set()
        
        for json_file in path.glob("*.json"):
            try:
                with open(json_file) as f:
                    createml_data = json.load(f)
                
                if isinstance(createml_data, list):
                    for item in createml_data:
                        if "image" in item:
                            image_data = {
                                "id": Path(item["image"]).stem,
                                "filename": item["image"],
                                "path": item["image"],
                                "width": 0,
                                "height": 0,
                                "annotations": []
                            }
                            
                            for ann in item.get("annotations", []):
                                class_name = ann.get("label", "unknown")
                                class_set.add(class_name)
                                
                                coords = ann.get("coordinates", {})
                                x = coords.get("x", 0)
                                y = coords.get("y", 0)
                                w = coords.get("width", 0)
                                h = coords.get("height", 0)
                                
                                image_data["annotations"].append({
                                    "type": "bbox",
                                    "class_name": class_name,
                                    "bbox": [x - w/2, y - h/2, w, h]
                                })
                            
                            data["images"].append(image_data)
            except Exception:
                pass
        
        data["classes"] = sorted(list(class_set))
        
        # Update class IDs
        class_to_id = {name: idx for idx, name in enumerate(data["classes"])}
        for img in data["images"]:
            for ann in img["annotations"]:
                ann["class_id"] = class_to_id.get(ann["class_name"], 0)
        
        return data
    
    def _load_tensorflow_csv(self, path: Path) -> Dict[str, Any]:
        """Load TensorFlow CSV format into unified format"""
        data = {
            "classes": [],
            "images": []
        }
        
        class_set = set()
        image_map = {}
        
        for csv_file in path.glob("*.csv"):
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row.get("filename", "")
                        class_name = row.get("class", "unknown")
                        class_set.add(class_name)
                        
                        if filename not in image_map:
                            image_map[filename] = {
                                "id": Path(filename).stem,
                                "filename": filename,
                                "path": filename,
                                "width": int(row.get("width", 0)),
                                "height": int(row.get("height", 0)),
                                "annotations": []
                            }
                        
                        image_map[filename]["annotations"].append({
                            "type": "bbox",
                            "class_name": class_name,
                            "bbox": [
                                int(row.get("xmin", 0)),
                                int(row.get("ymin", 0)),
                                int(row.get("xmax", 0)) - int(row.get("xmin", 0)),
                                int(row.get("ymax", 0)) - int(row.get("ymin", 0))
                            ]
                        })
            except Exception:
                pass
        
        data["classes"] = sorted(list(class_set))
        data["images"] = list(image_map.values())
        
        # Update class IDs
        class_to_id = {name: idx for idx, name in enumerate(data["classes"])}
        for img in data["images"]:
            for ann in img["annotations"]:
                ann["class_id"] = class_to_id.get(ann["class_name"], 0)
        
        return data
    
    def _load_labelme(self, path: Path) -> Dict[str, Any]:
        """Load LabelMe format into unified format"""
        data = {
            "classes": [],
            "images": []
        }
        
        class_set = set()
        
        for json_file in path.glob("**/*.json"):
            try:
                with open(json_file) as f:
                    labelme_data = json.load(f)
                
                if "shapes" in labelme_data:
                    image_data = {
                        "id": json_file.stem,
                        "filename": labelme_data.get("imagePath", json_file.stem),
                        "path": labelme_data.get("imagePath", ""),
                        "width": labelme_data.get("imageWidth", 0),
                        "height": labelme_data.get("imageHeight", 0),
                        "annotations": []
                    }
                    
                    for shape in labelme_data["shapes"]:
                        class_name = shape.get("label", "unknown")
                        class_set.add(class_name)
                        
                        shape_type = shape.get("shape_type", "polygon")
                        points = shape.get("points", [])
                        
                        annotation = {
                            "class_name": class_name
                        }
                        
                        if shape_type == "rectangle" and len(points) >= 2:
                            x_min = min(points[0][0], points[1][0])
                            y_min = min(points[0][1], points[1][1])
                            x_max = max(points[0][0], points[1][0])
                            y_max = max(points[0][1], points[1][1])
                            annotation["type"] = "bbox"
                            annotation["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
                        else:
                            flat_points = []
                            for pt in points:
                                flat_points.extend(pt)
                            annotation["type"] = "polygon"
                            annotation["segmentation"] = [flat_points]
                        
                        image_data["annotations"].append(annotation)
                    
                    data["images"].append(image_data)
            except Exception:
                pass
        
        data["classes"] = sorted(list(class_set))
        
        # Update class IDs
        class_to_id = {name: idx for idx, name in enumerate(data["classes"])}
        for img in data["images"]:
            for ann in img["annotations"]:
                ann["class_id"] = class_to_id.get(ann["class_name"], 0)
        
        return data
    
    def _load_classification(self, path: Path) -> Dict[str, Any]:
        """Load classification folder format into unified format"""
        data = {
            "classes": [],
            "images": []
        }
        
        for subdir in sorted(path.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("."):
                class_name = subdir.name
                class_id = len(data["classes"])
                data["classes"].append(class_name)
                
                for img_file in subdir.iterdir():
                    if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                        data["images"].append({
                            "id": img_file.stem,
                            "filename": img_file.name,
                            "path": str(img_file.relative_to(path)),
                            "width": 0,
                            "height": 0,
                            "annotations": [{
                                "type": "classification",
                                "class_id": class_id,
                                "class_name": class_name
                            }]
                        })
        
        return data
    
    def _load_generic(self, path: Path) -> Dict[str, Any]:
        """Load generic/unknown format"""
        data = {
            "classes": [],
            "images": []
        }
        
        for img_file in path.glob("**/*"):
            if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                data["images"].append({
                    "id": img_file.stem,
                    "filename": img_file.name,
                    "path": str(img_file.relative_to(path)),
                    "width": 0,
                    "height": 0,
                    "annotations": []
                })
        
        return data
    
    # ============== EXPORTERS ==============
    
    def _export_yolo(self, data: Dict[str, Any], output_path: Path):
        """Export to YOLO format"""
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data.yaml
        yaml_data = {
            "path": str(output_path.absolute()),
            "train": "images",
            "val": "images",
            "names": {i: name for i, name in enumerate(data["classes"])}
        }
        
        with open(output_path / "data.yaml", "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        # Create label files
        class_to_id = {name: idx for idx, name in enumerate(data["classes"])}
        
        for img in data["images"]:
            if not img["annotations"]:
                continue
            
            width = img.get("width", 1) or 1
            height = img.get("height", 1) or 1
            
            label_file = labels_dir / f"{img['id']}.txt"
            with open(label_file, "w") as f:
                for ann in img["annotations"]:
                    class_id = ann.get("class_id", class_to_id.get(ann.get("class_name", ""), 0))
                    
                    if ann.get("type") == "polygon" and ann.get("segmentation"):
                        # Segmentation
                        points = ann["segmentation"][0] if ann["segmentation"] else []
                        normalized = []
                        for i in range(0, len(points), 2):
                            normalized.append(points[i] / width)
                            normalized.append(points[i+1] / height)
                        
                        f.write(f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized) + "\n")
                    elif ann.get("bbox"):
                        # Bounding box
                        bbox = ann["bbox"]
                        x_center = (bbox[0] + bbox[2]/2) / width
                        y_center = (bbox[1] + bbox[3]/2) / height
                        w = bbox[2] / width
                        h = bbox[3] / height
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    def _export_coco(self, data: Dict[str, Any], output_path: Path):
        """Export to COCO format"""
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for idx, class_name in enumerate(data["classes"]):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "none"
            })
        
        # Add images and annotations
        ann_id = 1
        class_to_id = {name: idx for idx, name in enumerate(data["classes"])}
        
        for img_idx, img in enumerate(data["images"]):
            img_id = img_idx + 1
            
            coco_data["images"].append({
                "id": img_id,
                "file_name": img["filename"],
                "width": img.get("width", 0),
                "height": img.get("height", 0)
            })
            
            for ann in img["annotations"]:
                class_id = ann.get("class_id", class_to_id.get(ann.get("class_name", ""), 0))
                
                coco_ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "iscrowd": 0
                }
                
                if ann.get("bbox"):
                    coco_ann["bbox"] = ann["bbox"]
                    coco_ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                
                if ann.get("segmentation"):
                    coco_ann["segmentation"] = ann["segmentation"]
                
                coco_data["annotations"].append(coco_ann)
                ann_id += 1
        
        with open(output_path / "annotations.json", "w") as f:
            json.dump(coco_data, f, indent=2)
    
    def _export_voc(self, data: Dict[str, Any], output_path: Path):
        """Export to Pascal VOC format"""
        annotations_dir = output_path / "Annotations"
        images_dir = output_path / "JPEGImages"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for img in data["images"]:
            # Create XML annotation
            annotation = ET.Element("annotation")
            
            folder = ET.SubElement(annotation, "folder")
            folder.text = "JPEGImages"
            
            filename = ET.SubElement(annotation, "filename")
            filename.text = img["filename"]
            
            size = ET.SubElement(annotation, "size")
            width_elem = ET.SubElement(size, "width")
            width_elem.text = str(img.get("width", 0))
            height_elem = ET.SubElement(size, "height")
            height_elem.text = str(img.get("height", 0))
            depth = ET.SubElement(size, "depth")
            depth.text = "3"
            
            for ann in img["annotations"]:
                if ann.get("bbox"):
                    obj = ET.SubElement(annotation, "object")
                    
                    name = ET.SubElement(obj, "name")
                    name.text = ann.get("class_name", "unknown")
                    
                    pose = ET.SubElement(obj, "pose")
                    pose.text = "Unspecified"
                    
                    truncated = ET.SubElement(obj, "truncated")
                    truncated.text = "0"
                    
                    difficult = ET.SubElement(obj, "difficult")
                    difficult.text = "0"
                    
                    bndbox = ET.SubElement(obj, "bndbox")
                    bbox = ann["bbox"]
                    
                    xmin = ET.SubElement(bndbox, "xmin")
                    xmin.text = str(int(bbox[0]))
                    ymin = ET.SubElement(bndbox, "ymin")
                    ymin.text = str(int(bbox[1]))
                    xmax = ET.SubElement(bndbox, "xmax")
                    xmax.text = str(int(bbox[0] + bbox[2]))
                    ymax = ET.SubElement(bndbox, "ymax")
                    ymax.text = str(int(bbox[1] + bbox[3]))
            
            # Pretty print XML
            xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
            with open(annotations_dir / f"{img['id']}.xml", "w") as f:
                f.write(xml_str)
    
    def _export_createml(self, data: Dict[str, Any], output_path: Path):
        """Export to CreateML format"""
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        createml_data = []
        
        for img in data["images"]:
            item = {
                "image": img["filename"],
                "annotations": []
            }
            
            for ann in img["annotations"]:
                if ann.get("bbox"):
                    bbox = ann["bbox"]
                    item["annotations"].append({
                        "label": ann.get("class_name", "unknown"),
                        "coordinates": {
                            "x": bbox[0] + bbox[2]/2,
                            "y": bbox[1] + bbox[3]/2,
                            "width": bbox[2],
                            "height": bbox[3]
                        }
                    })
            
            createml_data.append(item)
        
        with open(output_path / "annotations.json", "w") as f:
            json.dump(createml_data, f, indent=2)
    
    def _export_tensorflow_csv(self, data: Dict[str, Any], output_path: Path):
        """Export to TensorFlow CSV format"""
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for img in data["images"]:
            for ann in img["annotations"]:
                if ann.get("bbox"):
                    bbox = ann["bbox"]
                    rows.append({
                        "filename": img["filename"],
                        "width": img.get("width", 0),
                        "height": img.get("height", 0),
                        "class": ann.get("class_name", "unknown"),
                        "xmin": int(bbox[0]),
                        "ymin": int(bbox[1]),
                        "xmax": int(bbox[0] + bbox[2]),
                        "ymax": int(bbox[1] + bbox[3])
                    })
        
        with open(output_path / "annotations.csv", "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def _export_labelme(self, data: Dict[str, Any], output_path: Path):
        """Export to LabelMe format"""
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for img in data["images"]:
            labelme_data = {
                "version": "5.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": img["filename"],
                "imageData": None,
                "imageHeight": img.get("height", 0),
                "imageWidth": img.get("width", 0)
            }
            
            for ann in img["annotations"]:
                shape = {
                    "label": ann.get("class_name", "unknown"),
                    "flags": {},
                    "group_id": None
                }
                
                if ann.get("type") == "polygon" and ann.get("segmentation"):
                    points = ann["segmentation"][0] if ann["segmentation"] else []
                    shape["shape_type"] = "polygon"
                    shape["points"] = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
                elif ann.get("bbox"):
                    bbox = ann["bbox"]
                    shape["shape_type"] = "rectangle"
                    shape["points"] = [
                        [bbox[0], bbox[1]],
                        [bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    ]
                else:
                    continue
                
                labelme_data["shapes"].append(shape)
            
            with open(output_path / f"{img['id']}.json", "w") as f:
                json.dump(labelme_data, f, indent=2)
    
    def _export_classification(self, data: Dict[str, Any], output_path: Path):
        """Export to classification folder format"""
        for class_name in data["classes"]:
            (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    def _copy_images(self, source_path: Path, output_path: Path, data: Dict[str, Any], target_format: str):
        """Copy images to output directory"""
        if target_format in ("pascal-voc", "voc"):
            dest_dir = output_path / "JPEGImages"
        elif target_format in ("classification-folder", "classification"):
            for img in data["images"]:
                if img["annotations"]:
                    class_name = img["annotations"][0].get("class_name", "unknown")
                    dest_dir = output_path / class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    src_file = source_path / img["path"]
                    if src_file.exists():
                        shutil.copy(src_file, dest_dir / img["filename"])
            return
        elif target_format == "cityscapes":
            dest_dir = output_path / "leftImg8bit" / "train" / "dataset"
        else:
            dest_dir = output_path / "images"

        dest_dir.mkdir(parents=True, exist_ok=True)

        for img in data["images"]:
            src_file = source_path / img["path"]
            if src_file.exists():
                shutil.copy(src_file, dest_dir / img["filename"])
            else:
                for found_file in source_path.glob(f"**/{img['filename']}"):
                    shutil.copy(found_file, dest_dir / img["filename"])
                    break
    
    # ============== HELPER METHODS ==============

    def _obb_to_bbox(self, cx: float, cy: float, w: float, h: float, angle_deg: float) -> List[float]:
        """Convert oriented bounding box to axis-aligned bbox [x, y, w, h]"""
        rad = math.radians(angle_deg)
        cos_a = abs(math.cos(rad))
        sin_a = abs(math.sin(rad))
        new_w = w * cos_a + h * sin_a
        new_h = w * sin_a + h * cos_a
        return [cx - new_w / 2, cy - new_h / 2, new_w, new_h]

    def _polygon_to_bbox(self, pts: List[float]) -> List[float]:
        """Convert flat polygon coords [x1,y1,x2,y2,...] to bbox [x,y,w,h]"""
        if not pts or len(pts) < 4:
            return [0, 0, 0, 0]
        xs = pts[0::2]
        ys = pts[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return [xmin, ymin, xmax - xmin, ymax - ymin]

    def _ann_to_bbox(self, ann: Dict[str, Any]) -> Optional[List[float]]:
        """Extract an axis-aligned bbox from any annotation type"""
        if ann.get("bbox"):
            return ann["bbox"]
        if ann.get("type") == "obb" and ann.get("obb"):
            return self._obb_to_bbox(*ann["obb"])
        if ann.get("segmentation"):
            pts = ann["segmentation"][0] if ann["segmentation"] else []
            return self._polygon_to_bbox(pts)
        return None

    # ============== NEW LOADERS ==============

    def _load_yolo_obb(self, path: Path) -> Dict[str, Any]:
        """Load YOLO OBB format (oriented bounding boxes)"""
        data = {"classes": [], "images": []}

        for yaml_file in list(path.glob("*.yaml")) + list(path.glob("*.yml")):
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                    if "names" in config:
                        data["classes"] = (list(config["names"].values())
                                           if isinstance(config["names"], dict)
                                           else config["names"])
                        break
            except Exception:
                pass

        image_dirs = ["images", "train/images", "val/images", "test/images",
                      "images/train", "images/val", "images/test", ""]
        visited = set()

        for img_dir in image_dirs:
            img_path = path / img_dir if img_dir else path
            key = str(img_path.resolve()) if img_path.exists() else None
            if key is None or key in visited:
                continue
            visited.add(key)

            if img_dir and "images" in img_dir:
                label_dir = img_dir.replace("images", "labels")
            else:
                label_dir = "labels"
            label_path = path / label_dir

            for img_file in img_path.iterdir():
                if img_file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(img_file) as im:
                        width, height = im.size
                except Exception:
                    width, height = 0, 0

                image_data = {
                    "id": img_file.stem,
                    "filename": img_file.name,
                    "path": str(img_file.relative_to(path)),
                    "width": width,
                    "height": height,
                    "annotations": [],
                }

                label_file = label_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 2:
                                continue
                            class_id = int(parts[0])
                            cname = (data["classes"][class_id]
                                     if class_id < len(data["classes"])
                                     else f"class_{class_id}")
                            vals = [float(v) for v in parts[1:]]

                            if len(vals) == 5:
                                # class_id cx cy w h angle
                                cx = vals[0] * width
                                cy = vals[1] * height
                                w = vals[2] * width
                                h = vals[3] * height
                                angle = vals[4]
                                image_data["annotations"].append({
                                    "type": "obb",
                                    "class_id": class_id,
                                    "class_name": cname,
                                    "obb": [cx, cy, w, h, angle],
                                    "bbox": self._obb_to_bbox(cx, cy, w, h, angle),
                                })
                            elif len(vals) == 8:
                                # class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                                abs_pts = []
                                for i in range(0, 8, 2):
                                    abs_pts.append(vals[i] * width)
                                    abs_pts.append(vals[i + 1] * height)
                                image_data["annotations"].append({
                                    "type": "polygon",
                                    "class_id": class_id,
                                    "class_name": cname,
                                    "segmentation": [abs_pts],
                                    "bbox": self._polygon_to_bbox(abs_pts),
                                })

                data["images"].append(image_data)

        return data

    def _load_coco_panoptic(self, path: Path) -> Dict[str, Any]:
        """Load COCO Panoptic format"""
        data = {"classes": [], "images": []}

        json_files = (list(path.glob("annotations/panoptic_*.json"))
                      + list(path.glob("panoptic_*.json")))
        if not json_files:
            json_files = list(path.glob("annotations/*.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    pan_data = json.load(f)

                if "categories" not in pan_data:
                    continue

                cat_map: Dict[int, int] = {}
                for cat in pan_data["categories"]:
                    cat_map[cat["id"]] = len(data["classes"])
                    data["classes"].append(cat["name"])

                ann_by_image: Dict[int, Any] = {}
                for ann in pan_data.get("annotations", []):
                    ann_by_image[ann["image_id"]] = ann

                masks_dir = path / "panoptic_masks"

                for img in pan_data.get("images", []):
                    img_id = img["id"]
                    img_entry = {
                        "id": str(img_id),
                        "filename": img["file_name"],
                        "path": img["file_name"],
                        "width": img.get("width", 0),
                        "height": img.get("height", 0),
                        "annotations": [],
                    }

                    ann = ann_by_image.get(img_id)
                    if ann:
                        mask_file = masks_dir / ann.get("file_name", "")
                        for seg in ann.get("segments_info", []):
                            seg_id = seg["id"]
                            cat_idx = cat_map.get(seg["category_id"], 0)
                            cname = (data["classes"][cat_idx]
                                     if cat_idx < len(data["classes"])
                                     else "unknown")
                            bbox = seg.get("bbox", [])

                            # Try to read the mask to get a tighter bbox
                            if not bbox and mask_file.exists():
                                try:
                                    mask_img = Image.open(mask_file).convert("RGB")
                                    r, g, b = (
                                        seg_id % 256,
                                        (seg_id // 256) % 256,
                                        (seg_id // 65536) % 256,
                                    )
                                    # Find bbox of this segment in the mask
                                    pixels = mask_img.load()
                                    mw, mh = mask_img.size
                                    xs, ys = [], []
                                    for y in range(mh):
                                        for x in range(mw):
                                            pr, pg, pb = pixels[x, y]
                                            if pr == r and pg == g and pb == b:
                                                xs.append(x)
                                                ys.append(y)
                                    if xs:
                                        xmin, xmax = min(xs), max(xs)
                                        ymin, ymax = min(ys), max(ys)
                                        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                                except Exception:
                                    pass

                            img_entry["annotations"].append({
                                "type": "bbox",
                                "class_id": cat_idx,
                                "class_name": cname,
                                "bbox": bbox if bbox else [0, 0, 0, 0],
                            })

                    data["images"].append(img_entry)

                break  # use first valid panoptic JSON
            except Exception:
                pass

        return data

    def _load_cityscapes(self, path: Path) -> Dict[str, Any]:
        """Load Cityscapes format (gtFine polygon JSONs)"""
        data = {"classes": [], "images": []}
        class_set: set = set()

        for json_file in path.glob("**/*_gtFine_polygons.json"):
            try:
                with open(json_file) as f:
                    cs_data = json.load(f)

                img_w = cs_data.get("imgWidth", 0)
                img_h = cs_data.get("imgHeight", 0)

                # Derive image filename from JSON name
                img_stem = json_file.name.replace("_gtFine_polygons.json", "_leftImg8bit")
                img_filename = img_stem + ".png"
                img_path_str = img_filename

                image_data = {
                    "id": json_file.stem,
                    "filename": img_filename,
                    "path": img_path_str,
                    "width": img_w,
                    "height": img_h,
                    "annotations": [],
                }

                for obj in cs_data.get("objects", []):
                    label = obj.get("label", "unknown")
                    class_set.add(label)
                    polygon = obj.get("polygon", [])
                    if polygon:
                        flat = []
                        for pt in polygon:
                            flat.extend(pt)
                        image_data["annotations"].append({
                            "type": "polygon",
                            "class_name": label,
                            "segmentation": [flat],
                            "bbox": self._polygon_to_bbox(flat),
                        })

                data["images"].append(image_data)
            except Exception:
                pass

        data["classes"] = sorted(class_set)
        class_to_id = {n: i for i, n in enumerate(data["classes"])}
        for img in data["images"]:
            for ann in img["annotations"]:
                ann["class_id"] = class_to_id.get(ann.get("class_name", ""), 0)

        return data

    def _load_ade20k(self, path: Path) -> Dict[str, Any]:
        """Load ADE20K segmentation format (PNG masks, pixel = class ID)"""
        data = {"classes": [], "images": []}
        class_ids_seen: set = set()

        ann_dir = path / "annotations"
        if not ann_dir.exists():
            ann_dir = path

        img_dir = path / "images"
        if not img_dir.exists():
            img_dir = path

        # Build stem → annotation file map
        ann_map: Dict[str, Path] = {}
        for ann_file in ann_dir.glob("**/*.png"):
            ann_map[ann_file.stem] = ann_file

        for img_file in img_dir.glob("**/*"):
            if img_file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue

            ann_file = ann_map.get(img_file.stem)
            try:
                with Image.open(img_file) as im:
                    width, height = im.size
            except Exception:
                width, height = 0, 0

            image_data = {
                "id": img_file.stem,
                "filename": img_file.name,
                "path": str(img_file.relative_to(path)),
                "width": width,
                "height": height,
                "annotations": [],
            }

            if ann_file and ann_file.exists():
                try:
                    mask = Image.open(ann_file).convert("L")
                    unique_vals = set(mask.getdata()) - {0}
                    for cls_id in sorted(unique_vals):
                        class_ids_seen.add(cls_id)
                        binary = mask.point(lambda p, c=cls_id: 255 if p == c else 0)
                        bbox = binary.getbbox()
                        if bbox:
                            x, y, x2, y2 = bbox
                            image_data["annotations"].append({
                                "type": "bbox",
                                "class_id": int(cls_id),
                                "class_name": f"class_{cls_id}",
                                "bbox": [x, y, x2 - x, y2 - y],
                            })
                except Exception:
                    pass

            data["images"].append(image_data)

        # Build class list from seen IDs
        all_ids = sorted(class_ids_seen)
        data["classes"] = [f"class_{i}" for i in all_ids]
        id_to_idx = {cid: idx for idx, cid in enumerate(all_ids)}
        for img in data["images"]:
            for ann in img["annotations"]:
                raw_id = ann["class_id"]
                ann["class_id"] = id_to_idx.get(raw_id, raw_id)

        return data

    def _load_dota(self, path: Path) -> Dict[str, Any]:
        """Load DOTA format (labelTxt/ directory, absolute quad coordinates)"""
        data = {"classes": [], "images": []}
        class_set: set = set()

        label_dir = path / "labelTxt"
        if not label_dir.exists():
            label_dir = path

        img_dir = path / "images"
        if not img_dir.exists():
            img_dir = path

        # Build stem → image file map
        img_map: Dict[str, Path] = {}
        for ext in self.IMAGE_EXTENSIONS:
            for img_file in img_dir.glob(f"*{ext}"):
                img_map[img_file.stem] = img_file

        for txt_file in label_dir.glob("*.txt"):
            img_file = img_map.get(txt_file.stem)
            if img_file:
                try:
                    with Image.open(img_file) as im:
                        width, height = im.size
                except Exception:
                    width, height = 0, 0
                img_filename = img_file.name
                img_path_str = str(img_file.relative_to(path))
            else:
                width, height = 0, 0
                img_filename = txt_file.stem + ".png"
                img_path_str = img_filename

            image_data = {
                "id": txt_file.stem,
                "filename": img_filename,
                "path": img_path_str,
                "width": width,
                "height": height,
                "annotations": [],
            }

            try:
                with open(txt_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("imagesource") or line.startswith("gsd"):
                            continue
                        parts = line.split()
                        if len(parts) < 9:
                            continue
                        try:
                            coords = [float(v) for v in parts[:8]]
                            class_name = parts[8]
                            class_set.add(class_name)
                            image_data["annotations"].append({
                                "type": "polygon",
                                "class_name": class_name,
                                "segmentation": [coords],
                                "bbox": self._polygon_to_bbox(coords),
                            })
                        except (ValueError, IndexError):
                            continue
            except Exception:
                pass

            data["images"].append(image_data)

        data["classes"] = sorted(class_set)
        class_to_id = {n: i for i, n in enumerate(data["classes"])}
        for img in data["images"]:
            for ann in img["annotations"]:
                ann["class_id"] = class_to_id.get(ann.get("class_name", ""), 0)

        return data

    def _load_tfrecord(self, path: Path) -> Dict[str, Any]:
        """Load TFRecord format (requires tensorflow)"""
        try:
            import tensorflow as tf  # type: ignore
        except ImportError:
            raise RuntimeError(
                "TFRecord format requires TensorFlow. "
                "Install it with: pip install tensorflow"
            )

        data = {"classes": [], "images": []}
        class_set: set = set()

        # Load label map if present
        label_map: Dict[int, str] = {}
        for pbtxt in path.glob("*.pbtxt"):
            try:
                content = pbtxt.read_text()
                import re
                items = re.findall(
                    r"item\s*\{[^}]*id:\s*(\d+)[^}]*name:\s*['\"]([^'\"]+)['\"]",
                    content,
                    re.DOTALL,
                )
                for item_id, item_name in items:
                    label_map[int(item_id)] = item_name
            except Exception:
                pass

        for record_file in path.glob("*.record"):
            try:
                dataset = tf.data.TFRecordDataset(str(record_file))
                for raw_record in dataset:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    feat = example.features.feature

                    def _get(key, default=None):
                        if key in feat:
                            f = feat[key]
                            if f.HasField("bytes_list"):
                                return f.bytes_list.value
                            if f.HasField("float_list"):
                                return list(f.float_list.value)
                            if f.HasField("int64_list"):
                                return list(f.int64_list.value)
                        return default

                    filename_bytes = _get("image/filename") or _get("image/source_id")
                    filename = (filename_bytes[0].decode("utf-8")
                                if filename_bytes else "unknown.jpg")

                    widths = _get("image/width") or [0]
                    heights = _get("image/height") or [0]
                    w = int(widths[0]) if widths else 0
                    h = int(heights[0]) if heights else 0

                    xmins = _get("image/object/bbox/xmin") or []
                    xmaxs = _get("image/object/bbox/xmax") or []
                    ymins = _get("image/object/bbox/ymin") or []
                    ymaxs = _get("image/object/bbox/ymax") or []
                    labels = _get("image/object/class/label") or []
                    label_texts = _get("image/object/class/text") or []

                    annotations = []
                    for i in range(len(xmins)):
                        class_id = int(labels[i]) if i < len(labels) else 0
                        class_name = (label_texts[i].decode("utf-8")
                                      if i < len(label_texts)
                                      else label_map.get(class_id, f"class_{class_id}"))
                        class_set.add(class_name)
                        x1 = float(xmins[i]) * w
                        y1 = float(ymins[i]) * h
                        x2 = float(xmaxs[i]) * w
                        y2 = float(ymaxs[i]) * h
                        annotations.append({
                            "type": "bbox",
                            "class_id": class_id,
                            "class_name": class_name,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                        })

                    data["images"].append({
                        "id": Path(filename).stem,
                        "filename": filename,
                        "path": filename,
                        "width": w,
                        "height": h,
                        "annotations": annotations,
                    })
            except Exception:
                pass

        data["classes"] = sorted(class_set)
        class_to_id = {n: i for i, n in enumerate(data["classes"])}
        for img in data["images"]:
            for ann in img["annotations"]:
                ann["class_id"] = class_to_id.get(ann.get("class_name", ""), ann.get("class_id", 0))

        return data

    # ============== NEW EXPORTERS ==============

    def _export_yolo_obb(self, data: Dict[str, Any], output_path: Path):
        """Export to YOLO OBB format"""
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        yaml_data = {
            "path": str(output_path.absolute()),
            "train": "images",
            "val": "images",
            "names": {i: n for i, n in enumerate(data["classes"])},
        }
        with open(output_path / "data.yaml", "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        class_to_id = {n: i for i, n in enumerate(data["classes"])}

        for img in data["images"]:
            if not img["annotations"]:
                continue
            width = img.get("width", 1) or 1
            height = img.get("height", 1) or 1

            label_file = labels_dir / f"{img['id']}.txt"
            with open(label_file, "w") as f:
                for ann in img["annotations"]:
                    cid = ann.get("class_id", class_to_id.get(ann.get("class_name", ""), 0))

                    if ann.get("type") == "obb" and ann.get("obb"):
                        cx, cy, w, h, angle = ann["obb"]
                        f.write(
                            f"{cid} {cx/width:.6f} {cy/height:.6f} "
                            f"{w/width:.6f} {h/height:.6f} {angle:.4f}\n"
                        )
                    elif ann.get("type") == "polygon" and ann.get("segmentation"):
                        pts = ann["segmentation"][0]
                        if len(pts) >= 8:
                            quad = pts[:8]
                            norm = []
                            for i in range(0, 8, 2):
                                norm.append(quad[i] / width)
                                norm.append(quad[i + 1] / height)
                            f.write(f"{cid} " + " ".join(f"{v:.6f}" for v in norm) + "\n")
                    elif ann.get("bbox"):
                        bbox = ann["bbox"]
                        cx = (bbox[0] + bbox[2] / 2) / width
                        cy = (bbox[1] + bbox[3] / 2) / height
                        w = bbox[2] / width
                        h = bbox[3] / height
                        f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} 0.0000\n")

    def _export_coco_panoptic(self, data: Dict[str, Any], output_path: Path):
        """Export to COCO Panoptic format"""
        images_dir = output_path / "images"
        ann_dir = output_path / "annotations"
        masks_dir = output_path / "panoptic_masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        categories = [
            {"id": i, "name": n, "supercategory": "none", "isthing": 1}
            for i, n in enumerate(data["classes"])
        ]
        class_to_id = {n: i for i, n in enumerate(data["classes"])}

        pan_images = []
        pan_annotations = []

        for img_idx, img in enumerate(data["images"]):
            img_id = img_idx + 1
            w = img.get("width", 0) or 1
            h = img.get("height", 0) or 1
            pan_images.append({
                "id": img_id,
                "file_name": img["filename"],
                "width": w,
                "height": h,
            })

            mask_name = f"{Path(img['filename']).stem}.png"
            mask_img = Image.new("RGB", (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(mask_img)

            segments_info = []
            for seg_idx, ann in enumerate(img["annotations"]):
                seg_id = seg_idx + 1
                cid = ann.get("class_id", class_to_id.get(ann.get("class_name", ""), 0))
                r = seg_id % 256
                g = (seg_id // 256) % 256
                b = (seg_id // 65536) % 256
                color = (r, g, b)

                bbox = self._ann_to_bbox(ann) or [0, 0, 0, 0]
                area = int(bbox[2] * bbox[3])

                # Draw the segment in the mask
                if ann.get("type") == "polygon" and ann.get("segmentation"):
                    pts = ann["segmentation"][0]
                    poly = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
                    if len(poly) >= 3:
                        draw.polygon(poly, fill=color)
                else:
                    x, y, bw, bh = [int(v) for v in bbox]
                    draw.rectangle([x, y, x + bw, y + bh], fill=color)

                segments_info.append({
                    "id": seg_id,
                    "category_id": cid,
                    "area": area,
                    "bbox": [int(v) for v in bbox],
                    "iscrowd": 0,
                })

            mask_img.save(masks_dir / mask_name)
            pan_annotations.append({
                "image_id": img_id,
                "file_name": mask_name,
                "segments_info": segments_info,
            })

        panoptic_json = {
            "images": pan_images,
            "annotations": pan_annotations,
            "categories": categories,
        }
        with open(ann_dir / "panoptic_train.json", "w") as f:
            json.dump(panoptic_json, f, indent=2)

    def _export_cityscapes(self, data: Dict[str, Any], output_path: Path):
        """Export to Cityscapes format"""
        city = "dataset"
        gt_dir = output_path / "gtFine" / "train" / city
        gt_dir.mkdir(parents=True, exist_ok=True)

        class_to_id = {n: i for i, n in enumerate(data["classes"])}

        for img in data["images"]:
            stem = img["id"]
            cs_name = f"{stem}_gtFine_polygons.json"

            objects = []
            for ann in img["annotations"]:
                cname = ann.get("class_name", "unlabeled")
                if ann.get("type") == "polygon" and ann.get("segmentation"):
                    pts = ann["segmentation"][0]
                    polygon = [[pts[i], pts[i + 1]] for i in range(0, len(pts) - 1, 2)]
                elif ann.get("bbox"):
                    x, y, w, h = ann["bbox"]
                    polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                else:
                    continue
                objects.append({"label": cname, "polygon": polygon})

            cs_json = {
                "imgHeight": img.get("height", 0),
                "imgWidth": img.get("width", 0),
                "objects": objects,
            }
            with open(gt_dir / cs_name, "w") as f:
                json.dump(cs_json, f, indent=2)

    def _export_ade20k(self, data: Dict[str, Any], output_path: Path):
        """Export to ADE20K format (grayscale PNG masks)"""
        images_dir = output_path / "images"
        ann_dir = output_path / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        class_to_id = {n: i + 1 for i, n in enumerate(data["classes"])}  # ADE20K starts at 1

        for img in data["images"]:
            w = img.get("width", 1) or 1
            h = img.get("height", 1) or 1
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)

            for ann in img["annotations"]:
                cname = ann.get("class_name", "")
                pixel_val = class_to_id.get(cname, ann.get("class_id", 0) + 1)
                pixel_val = min(pixel_val, 255)

                if ann.get("type") == "polygon" and ann.get("segmentation"):
                    pts = ann["segmentation"][0]
                    poly = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
                    if len(poly) >= 3:
                        draw.polygon(poly, fill=pixel_val)
                elif ann.get("bbox"):
                    x, y, bw, bh = [int(v) for v in ann["bbox"]]
                    draw.rectangle([x, y, x + bw, y + bh], fill=pixel_val)

            mask.save(ann_dir / f"{img['id']}.png")

    def _export_dota(self, data: Dict[str, Any], output_path: Path):
        """Export to DOTA format (absolute quad coordinates in labelTxt/)"""
        images_dir = output_path / "images"
        label_dir = output_path / "labelTxt"
        images_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for img in data["images"]:
            w = img.get("width", 1) or 1
            h = img.get("height", 1) or 1
            lines = []

            for ann in img["annotations"]:
                cname = ann.get("class_name", "unknown")

                if ann.get("type") == "polygon" and ann.get("segmentation"):
                    pts = ann["segmentation"][0]
                    if len(pts) >= 8:
                        quad = pts[:8]
                    else:
                        # Pad to 8 values by repeating last point
                        quad = pts + pts[-2:] * ((8 - len(pts)) // 2 + 1)
                        quad = quad[:8]
                elif ann.get("type") == "obb" and ann.get("obb"):
                    cx, cy, bw, bh, angle = ann["obb"]
                    rad = math.radians(angle)
                    cos_a, sin_a = math.cos(rad), math.sin(rad)
                    hw, hh = bw / 2, bh / 2
                    corners = [
                        (cx + cos_a * hw - sin_a * hh, cy + sin_a * hw + cos_a * hh),
                        (cx - cos_a * hw - sin_a * hh, cy - sin_a * hw + cos_a * hh),
                        (cx - cos_a * hw + sin_a * hh, cy - sin_a * hw - cos_a * hh),
                        (cx + cos_a * hw + sin_a * hh, cy + sin_a * hw - cos_a * hh),
                    ]
                    quad = [v for corner in corners for v in corner]
                elif ann.get("bbox"):
                    x, y, bw, bh = ann["bbox"]
                    quad = [x, y, x + bw, y, x + bw, y + bh, x, y + bh]
                else:
                    continue

                coord_str = " ".join(f"{v:.2f}" for v in quad)
                lines.append(f"{coord_str} {cname} 0")

            with open(label_dir / f"{img['id']}.txt", "w") as f:
                f.write("\n".join(lines))

    def _export_tfrecord(self, data: Dict[str, Any], output_path: Path):
        """Export to TFRecord format (requires tensorflow)"""
        try:
            import tensorflow as tf  # type: ignore
        except ImportError:
            raise RuntimeError(
                "TFRecord export requires TensorFlow. "
                "Install it with: pip install tensorflow"
            )

        output_path.mkdir(parents=True, exist_ok=True)

        # Write label map
        label_map_lines = []
        for i, name in enumerate(data["classes"]):
            label_map_lines.append(
                f"item {{\n  id: {i}\n  name: '{name}'\n}}"
            )
        with open(output_path / "label_map.pbtxt", "w") as f:
            f.write("\n\n".join(label_map_lines))

        class_to_id = {n: i for i, n in enumerate(data["classes"])}
        writer = tf.io.TFRecordWriter(str(output_path / "train.record"))

        for img in data["images"]:
            w = img.get("width", 0)
            h = img.get("height", 0)
            xmins, xmaxs, ymins, ymaxs, labels, label_texts = [], [], [], [], [], []

            for ann in img["annotations"]:
                bbox = self._ann_to_bbox(ann)
                if not bbox:
                    continue
                x, y, bw, bh = bbox
                xmins.append(max(0.0, x / w) if w else 0.0)
                ymins.append(max(0.0, y / h) if h else 0.0)
                xmaxs.append(min(1.0, (x + bw) / w) if w else 0.0)
                ymaxs.append(min(1.0, (y + bh) / h) if h else 0.0)
                cname = ann.get("class_name", "unknown")
                cid = ann.get("class_id", class_to_id.get(cname, 0))
                labels.append(cid)
                label_texts.append(cname.encode("utf-8"))

            feature = {
                "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                "image/filename": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img["filename"].encode("utf-8")])
                ),
                "image/source_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img["id"].encode("utf-8")])
                ),
                "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
                "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
                "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
                "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
                "image/object/class/label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=labels)
                ),
                "image/object/class/text": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=label_texts)
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    def update_data_yaml(
        self, 
        dataset_path: Path, 
        classes: List[str],
        train_path: str = "train/images",
        val_path: str = "val/images",
        test_path: Optional[str] = "test/images"
    ):
        """Create or update data.yaml file for YOLO-format datasets"""
        dataset_path = Path(dataset_path)
        
        # Check if train/val/test directories exist
        has_train = (dataset_path / "train").exists()
        has_val = (dataset_path / "val").exists() or (dataset_path / "valid").exists()
        has_test = (dataset_path / "test").exists()
        
        yaml_data = {
            "path": str(dataset_path.absolute()),
            "nc": len(classes),
            "names": {i: name for i, name in enumerate(classes)}
        }
        
        if has_train:
            yaml_data["train"] = train_path
        if has_val:
            val_dir = "val" if (dataset_path / "val").exists() else "valid"
            yaml_data["val"] = f"{val_dir}/images"
        if has_test and test_path:
            yaml_data["test"] = test_path
        
        # If no splits, use root images folder
        if not has_train and not has_val:
            yaml_data["train"] = "images"
            yaml_data["val"] = "images"
        
        with open(dataset_path / "data.yaml", "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        return yaml_data
