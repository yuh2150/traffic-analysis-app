import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any, Optional
import cv2
import xml.etree.ElementTree as ET

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

    def _load_detrac_annotations(self):
        img_id = 0
        xml_files = sorted([f for f in os.listdir(self.anno_dir) if f.endswith(".xml")])
        for xml_file in xml_files:
            xml_path = os.path.join(self.anno_dir, xml_file)
            seq_name = xml_file.replace(".xml", "")
            seq_dir = os.path.join(self.img_dir, seq_name)
            if not os.path.isdir(seq_dir):
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for frame in root.findall("frame"):
                frame_num = int(frame.get("num", "0"))
                img_file = f"img{frame_num:05d}.jpg"
                img_path = os.path.join(seq_dir, img_file)
                if not os.path.exists(img_path):
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
