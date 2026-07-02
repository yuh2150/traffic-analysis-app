import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any, Optional
import cv2
import xml.etree.ElementTree as ET
import logging
import numpy as np

logger_traffic = logging.getLogger("TrafficDataset")

CLASS_MAP = {
    "car": 0,
    "van": 1,
    "bus": 2,
    "others": 3,
}

CLASS_NAMES = ["car", "van", "bus", "others"]


class TrafficDataset(Dataset):
    def __init__(self, img_dir: str, anno_dir: str, img_size: Tuple[int, int] = (640, 640)):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.img_size = img_size
        self.num_classes = len(CLASS_NAMES)

        self.images: List[Dict[str, Any]] = []
        self.annotations: Dict[int, List[Dict[str, Any]]] = {}
        self._load_detrac_annotations()
        
        total_anns = sum(len(anns) for anns in self.annotations.values())
        logger_traffic.info(f"Loaded TrafficDataset: {len(self.images)} images and {total_anns} annotations.")

    def _load_detrac_annotations(self):
        img_id = 0
        xml_files = sorted([f for f in os.listdir(self.anno_dir) if f.endswith(".xml")])
        for xml_file in xml_files:
            xml_path = os.path.join(self.anno_dir, xml_file)
            seq_name = xml_file.replace(".xml", "")
            seq_dir = os.path.join(self.img_dir, seq_name)
            if not os.path.isdir(seq_dir):
                continue
            
            # Optimization: pre-scan directory once and do set lookups
            existing_files = set(os.listdir(seq_dir))
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except ET.ParseError as e:
                logger_traffic.warning(f"Failed to parse XML file {xml_path}: {e}")
                continue
            for frame in root.findall("frame"):
                frame_num = int(frame.get("num", "0"))
                img_file = f"img{frame_num:05d}.jpg"
                if img_file not in existing_files:
                    continue
                h_img, w_img = 540, 960
                self.images.append({
                    "id": img_id,
                    "file_name": os.path.join(seq_name, img_file),
                    "width": w_img,
                    "height": h_img,
                })
                self.annotations[img_id] = []
                target_list = frame.find("target_list")
                targets = target_list.findall("target") if target_list is not None else frame.findall("target")
                for target in targets:
                    box = target.find("box")
                    attr = target.find("attribute")
                    if box is None:
                        continue
                    left = float(box.get("left", "0"))
                    top = float(box.get("top", "0"))
                    width = float(box.get("width", "0"))
                    height = float(box.get("height", "0"))
                    x1 = left / w_img
                    y1 = top / h_img
                    x2 = (left + width) / w_img
                    y2 = (top + height) / h_img
                    vehicle_type = "others"
                    if attr is not None:
                        vehicle_type = attr.get("vehicle_type", "others").lower()
                    cat_id = CLASS_MAP.get(vehicle_type, CLASS_MAP["others"])
                    self.annotations[img_id].append({
                        "bbox": [x1, y1, x2, y2],
                        "category_id": cat_id,
                    })
                img_id += 1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        anns = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
            "file_name": img_info["file_name"],
        }
        return img_tensor, target


import logging
logger_coco = logging.getLogger("CocoDataset")

class CocoDataset(Dataset):
    """Dataset class supporting MS COCO format with continuous 80-class mapping and synthetic fallback."""

    COCO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    COCO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, img_dir: str, anno_file: str, img_size: Tuple[int, int] = (640, 640), max_samples: Optional[int] = None, normalized: bool = True):
        self.img_dir = img_dir
        self.anno_file = anno_file
        self.img_size = img_size
        self.num_classes = 80

        self.images: List[Dict[str, Any]] = []
        self.annotations: Dict[int, List[Dict[str, Any]]] = {}

        # Standard COCO category ID (1 to 90) mapping to continuous YOLO indices (0 to 79)
        self.coco_cat_to_idx = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
            11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18,
            21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
            33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36,
            42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45,
            52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
            61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
            74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
            84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
        }

        self.use_synthetic = False
        if not os.path.exists(img_dir):
            self.use_synthetic = True
            logger_coco.warning(f"COCO img_dir not found: {img_dir}. Using synthetic data fallback.")
            self._load_synthetic()
        elif anno_file.lower().endswith(".json"):
            if not os.path.exists(anno_file):
                self.use_synthetic = True
                logger_coco.warning(f"COCO JSON anno_file not found: {anno_file}. Using synthetic data fallback.")
                self._load_synthetic()
            else:
                try:
                    self._load_coco_annotations()
                except Exception as e:
                    logger_coco.error(f"Error loading COCO JSON annotations: {e}. Falling back to synthetic.")
                    self.use_synthetic = True
                    self._load_synthetic()
        else:
            # Assume it is a directory containing YOLO .txt annotations (or try to auto-resolve parallel directory)
            yolo_anno_path = anno_file
            if not os.path.exists(yolo_anno_path) or not os.path.isdir(yolo_anno_path):
                # Attempt to find a parallel labels folder (e.g. replacing 'images' with 'labels' in path)
                parent_dir = os.path.dirname(img_dir)
                last_dir = os.path.basename(img_dir)
                candidate_labels = os.path.join(os.path.dirname(parent_dir), "labels", last_dir)
                if os.path.isdir(candidate_labels):
                    yolo_anno_path = candidate_labels
                    logger_coco.info(f"Auto-resolved parallel YOLO labels directory: {yolo_anno_path}")
                else:
                    self.use_synthetic = True
                    logger_coco.warning(f"YOLO annotation path not found: {anno_file}. Using synthetic fallback.")
                    self._load_synthetic()

            if not self.use_synthetic:
                try:
                    self._load_yolo_annotations(img_dir, yolo_anno_path, normalized=normalized)
                except Exception as e:
                    logger_coco.error(f"Error loading YOLO TXT annotations: {e}. Falling back to synthetic.")
                    self.use_synthetic = True
                    self._load_synthetic()

        total_anns = sum(len(anns) for anns in self.annotations.values())
        logger_coco.info(f"Loaded CocoDataset: {len(self.images)} images and {total_anns} annotations.")

    def _load_yolo_annotations(self, img_dir: str, anno_dir: str, normalized: bool = True):
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)])

        img_id = 0
        for f in img_files:
            base_name = os.path.splitext(f)[0]
            label_path = os.path.join(anno_dir, base_name + ".txt")

            # Default dimensions. We only determine actual image dimensions if normalized is False.
            w_img, h_img = 640, 640
            if not normalized:
                img_path = os.path.join(img_dir, f)
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        w_img, h_img = img.size
                except Exception:
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is not None:
                        h_img, w_img = img_bgr.shape[:2]

            self.images.append({
                "id": img_id,
                "file_name": f,
                "width": w_img,
                "height": h_img,
            })

            self.annotations[img_id] = []
            if os.path.exists(label_path):
                with open(label_path, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_idx = int(parts[0])
                            xc = float(parts[1])
                            yc = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])

                            if not normalized:
                                xc /= w_img
                                yc /= h_img
                                w /= w_img
                                h /= h_img

                            # Convert cx, cy, w, h normalized coordinates to relative x1, y1, x2, y2
                            x1 = xc - w / 2
                            y1 = yc - h / 2
                            x2 = xc + w / 2
                            y2 = yc + h / 2

                            # Clamp values
                            x1 = max(0.0, min(1.0, x1))
                            y1 = max(0.0, min(1.0, y1))
                            x2 = max(0.0, min(1.0, x2))
                            y2 = max(0.0, min(1.0, y2))

                            self.annotations[img_id].append({
                                "bbox": [x1, y1, x2, y2],
                                "category_id": class_idx
                            })
            img_id += 1

    def _load_coco_annotations(self):
        import json
        with open(self.anno_file, "r") as f:
            data = json.load(f)

        images_dict = {img["id"]: img for img in data["images"]}
        for img_id in images_dict:
            self.annotations[img_id] = []

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in images_dict:
                continue
            cat_id = ann["category_id"]
            if cat_id not in self.coco_cat_to_idx:
                continue
            class_idx = self.coco_cat_to_idx[cat_id]
            bbox = ann["bbox"]  # [x_min, y_min, width, height]
            w_img = images_dict[img_id]["width"]
            h_img = images_dict[img_id]["height"]

            # Normalize to relative coordinates [x1, y1, x2, y2]
            x1 = bbox[0] / w_img
            y1 = bbox[1] / h_img
            x2 = (bbox[0] + bbox[2]) / w_img
            y2 = (bbox[1] + bbox[3]) / h_img

            # Clamp coordinates
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            self.annotations[img_id].append({
                "bbox": [x1, y1, x2, y2],
                "category_id": class_idx
            })

        # Collect image items with non-empty annotations
        valid_img_ids = [img_id for img_id, anns in self.annotations.items() if len(anns) > 0]
        for img_id in valid_img_ids:
            self.images.append(images_dict[img_id])

    def _load_synthetic(self):
        num_samples = 10
        import random
        for i in range(num_samples):
            self.images.append({
                "id": i,
                "file_name": f"synthetic_{i}.jpg",
                "width": 640,
                "height": 640
            })
            # Generate 1-3 random bounding boxes per image
            random.seed(42 + i)
            self.annotations[i] = []
            for _ in range(random.randint(1, 3)):
                x1 = random.uniform(0.1, 0.5)
                y1 = random.uniform(0.1, 0.5)
                x2 = random.uniform(x1 + 0.1, 0.9)
                y2 = random.uniform(y1 + 0.1, 0.9)
                class_idx = random.randint(0, self.num_classes - 1)
                self.annotations[i].append({
                    "bbox": [x1, y1, x2, y2],
                    "category_id": class_idx
                })

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_info = self.images[idx]
        img_id = img_info["id"]

        if self.use_synthetic:
            import numpy as np
            np.random.seed(42 + idx)
            img_resized = np.random.randint(0, 256, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - torch.tensor(self.COCO_MEAN)[:, None, None]) / torch.tensor(self.COCO_STD)[:, None, None]
        else:
            img_path = os.path.join(self.img_dir, img_info["file_name"])
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"COCO Image not found: {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            target_short_side = int(self.img_size[0]) if isinstance(self.img_size, tuple) else int(self.img_size)
            max_size = 1333
            scale = target_short_side / min(h, w)
            if h < w:
                new_h = target_short_side
                new_w = int(round(w * scale))
            else:
                new_w = target_short_side
                new_h = int(round(h * scale))
            if max(new_h, new_w) > max_size:
                scale = max_size / max(new_h, new_w)
                new_h = int(round(new_h * scale))
                new_w = int(round(new_w * scale))
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor(self.COCO_MEAN)[:, None, None]
            std = torch.tensor(self.COCO_STD)[:, None, None]
            img_tensor = (img_tensor - mean) / std

        anns = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
            "file_name": img_info["file_name"]
        }
        return img_tensor, target


def get_data_loader(
    img_dir: str,
    anno_dir: str,
    batch_size: int = 2,
    img_size: Tuple[int, int] = (640, 640),
    shuffle: bool = False,
) -> DataLoader:
    dataset = TrafficDataset(img_dir=img_dir, anno_dir=anno_dir, img_size=img_size)

    def collate_fn(batch):
        return tuple(zip(*batch))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)

