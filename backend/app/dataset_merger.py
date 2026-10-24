"""
Dataset Merger - Merge multiple datasets into one
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image

from format_converter import FormatConverter


class DatasetMerger:
    """Merge multiple datasets into a single dataset"""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
    
    def __init__(self):
        self.converter = FormatConverter()
    
    def merge(
        self,
        datasets: List[Dict[str, Any]],
        output_path: Path,
        output_format: str,
        class_mapping: Optional[Dict[str, str]] = None
    ):
        """Merge multiple datasets into one"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all data in unified format
        merged_data = {
            "classes": [],
            "images": []
        }
        
        class_map = {}  # Map old class names to new class IDs
        image_id_counter = 0
        
        for dataset in datasets:
            dataset_path = Path(dataset["path"])
            dataset_format = dataset["format"]
            
            # Load dataset in unified format
            unified = self.converter._load_unified(dataset_path, dataset_format)
            
            # Merge classes
            for class_name in unified["classes"]:
                if class_name not in class_map:
                    class_map[class_name] = len(merged_data["classes"])
                    merged_data["classes"].append(class_name)
            
            # Merge images
            for img in unified["images"]:
                image_id_counter += 1
                
                # Update class IDs in annotations
                for ann in img["annotations"]:
                    old_class_name = ann.get("class_name", "unknown")
                    ann["class_id"] = class_map.get(old_class_name, 0)
                    ann["class_name"] = old_class_name
                
                # Generate unique filename to avoid conflicts
                original_filename = img["filename"]
                base_name = Path(original_filename).stem
                ext = Path(original_filename).suffix
                new_filename = f"{dataset['info']['name']}_{base_name}_{image_id_counter}{ext}"
                
                merged_data["images"].append({
                    "id": Path(new_filename).stem if output_format in ("yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12") else str(image_id_counter),
                    "filename": new_filename,
                    "original_path": img["path"],
                    "source_dataset": str(dataset_path),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "annotations": img["annotations"]
                })
        
        # Export merged data to output format
        self.converter._export_unified(merged_data, output_path, output_format)
        
        # Copy images
        self._copy_merged_images(merged_data, output_path, output_format)
    
    def _copy_merged_images(
        self,
        merged_data: Dict[str, Any],
        output_path: Path,
        output_format: str
    ):
        """Copy images from all source datasets to merged dataset"""
        if output_format == "pascal-voc":
            images_dir = output_path / "JPEGImages"
        elif output_format == "classification-folder":
            # Handle separately
            return
        else:
            images_dir = output_path / "images"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for img in merged_data["images"]:
            source_path = Path(img["source_dataset"]) / img["original_path"]
            
            if source_path.exists():
                dest_path = images_dir / img["filename"]
                shutil.copy(source_path, dest_path)
            else:
                # Try to find the image in the source dataset
                source_dir = Path(img["source_dataset"])
                original_name = Path(img["original_path"]).name
                
                for found in source_dir.glob(f"**/{original_name}"):
                    dest_path = images_dir / img["filename"]
                    shutil.copy(found, dest_path)
                    break
    
    def split_dataset(
        self,
        dataset_path: Path,
        format_name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        shuffle: bool = True
    ) -> Dict[str, Any]:
        """Split a dataset into train/val/test sets"""
        import random
        
        dataset_path = Path(dataset_path)
        
        # Load dataset
        unified = self.converter._load_unified(dataset_path, format_name)
        
        images = unified["images"]
        
        if shuffle:
            random.shuffle(images)
        
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Create split directories
        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }
        
        for split_name, split_images in splits.items():
            if not split_images:
                continue
            
            split_path = dataset_path / split_name
            split_path.mkdir(parents=True, exist_ok=True)
            
            split_data = {
                "classes": unified["classes"],
                "images": split_images
            }
            
            self.converter._export_unified(split_data, split_path, format_name)
            
            # Copy images
            if format_name == "pascal-voc":
                images_dir = split_path / "JPEGImages"
            else:
                images_dir = split_path / "images"
            
            images_dir.mkdir(parents=True, exist_ok=True)
            
            for img in split_images:
                source = dataset_path / img["path"]
                if source.exists():
                    shutil.copy(source, images_dir / img["filename"])
        
        # Update data.yaml if YOLO format
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            yaml_data = {
                "path": str(dataset_path.absolute()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images" if test_images else None,
                "names": {i: name for i, name in enumerate(unified["classes"])}
            }
            
            if not yaml_data["test"]:
                del yaml_data["test"]
            
            with open(dataset_path / "data.yaml", "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
        
        return {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images)
        }
    
    def augment_dataset(
        self,
        dataset_path: Path,
        format_name: str,
        augmentations: List[str],
        num_augmented: int = 2
    ) -> Dict[str, Any]:
        """Augment dataset with transformed copies"""
        try:
            import albumentations as A
            import numpy as np
            import cv2
        except ImportError:
            return {"error": "albumentations package not installed"}
        
        dataset_path = Path(dataset_path)
        
        # Build augmentation pipeline
        transforms_list = []
        
        aug_map = {
            "flip_horizontal": A.HorizontalFlip(p=0.5),
            "flip_vertical": A.VerticalFlip(p=0.5),
            "rotate": A.Rotate(limit=30, p=0.5),
            "brightness": A.RandomBrightnessContrast(p=0.5),
            "blur": A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            "noise": A.GaussNoise(p=0.3),
            "crop": A.RandomCrop(width=512, height=512, p=0.3),
            "scale": A.RandomScale(scale_limit=0.2, p=0.5),
            "color": A.ColorJitter(p=0.5),
        }
        
        for aug_name in augmentations:
            if aug_name in aug_map:
                transforms_list.append(aug_map[aug_name])
        
        if not transforms_list:
            return {"error": "No valid augmentations specified"}
        
        transform = A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        # Load dataset
        unified = self.converter._load_unified(dataset_path, format_name)
        
        augmented_count = 0
        
        for img in unified["images"]:
            image_path = dataset_path / img["path"]
            if not image_path.exists():
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Prepare bboxes in YOLO format
            bboxes = []
            class_labels = []
            
            for ann in img["annotations"]:
                if ann.get("bbox"):
                    bbox = ann["bbox"]
                    # Convert to YOLO format: x_center, y_center, width, height (normalized)
                    x_center = (bbox[0] + bbox[2]/2) / width
                    y_center = (bbox[1] + bbox[3]/2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height
                    bboxes.append([x_center, y_center, w, h])
                    class_labels.append(ann.get("class_id", 0))
            
            # Apply augmentations
            for i in range(num_augmented):
                try:
                    transformed = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    # Save augmented image
                    aug_filename = f"{img['id']}_aug_{i}{Path(img['filename']).suffix}"
                    aug_path = image_path.parent / aug_filename
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_path), aug_image_bgr)
                    
                    # Save augmented annotations
                    if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
                        labels_dir = dataset_path / "labels"
                        if (dataset_path / "train" / "labels").exists():
                            labels_dir = dataset_path / "train" / "labels"
                        
                        label_file = labels_dir / f"{img['id']}_aug_{i}.txt"
                        with open(label_file, "w") as f:
                            for bbox, label in zip(aug_bboxes, aug_labels):
                                f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    
                    augmented_count += 1
                    
                except Exception as e:
                    continue
        
        return {
            "augmented_count": augmented_count,
            "original_count": len(unified["images"])
        }
    
    def balance_dataset(
        self,
        dataset_path: Path,
        format_name: str,
        target_per_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Balance dataset by oversampling minority classes"""
        import random
        
        dataset_path = Path(dataset_path)
        
        # Load dataset
        unified = self.converter._load_unified(dataset_path, format_name)
        
        # Count images per class
        class_counts = {}
        class_images = {}
        
        for img in unified["images"]:
            for ann in img["annotations"]:
                class_name = ann.get("class_name", "unknown")
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    class_images[class_name] = []
                class_counts[class_name] += 1
                class_images[class_name].append(img)
        
        if not class_counts:
            return {"error": "No annotations found"}
        
        # Determine target count
        if target_per_class is None:
            target_per_class = max(class_counts.values())
        
        # Oversample minority classes
        balanced_images = []
        
        for class_name, images in class_images.items():
            current_count = len(images)
            
            if current_count >= target_per_class:
                # Undersample
                balanced_images.extend(random.sample(images, target_per_class))
            else:
                # Oversample
                balanced_images.extend(images)
                needed = target_per_class - current_count
                oversampled = random.choices(images, k=needed)
                balanced_images.extend(oversampled)
        
        return {
            "original_distribution": class_counts,
            "target_per_class": target_per_class,
            "balanced_count": len(balanced_images)
        }
