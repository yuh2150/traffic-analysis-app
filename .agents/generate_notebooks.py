import json
import os

def make_markdown_cell(source):
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines.pop()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def make_code_cell(source):
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines.pop()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

# Common blocks
PIP_INSTALL = """# 1. Cài đặt các thư viện cần thiết
!pip install -q ultralytics fvcore pycocotools pandas matplotlib tqdm opencv-python-headless
print("Đã cài đặt xong các thư viện cần thiết!")"""

DATASET_PREP = """# 2. Tải và cấu hình Dataset COCO 2017 val
import os
import json
import cv2
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url

# Cấu hình thư mục dữ liệu
DATA_DIR = "/kaggle/working/coco" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else "./data/coco"
os.makedirs(DATA_DIR, exist_ok=True)

VAL_ROOT = os.path.join(DATA_DIR, "val2017")
VAL_ANN  = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")

# Kiểm tra các đường dẫn dữ liệu phổ biến trên Kaggle trước
kaggle_paths = [
    "/kaggle/input/coco2017",
    "/kaggle/input/coco-2017-dataset",
    "/kaggle/input/coco-2017-dataset/coco2017",
    "/kaggle/input/coco2017-dataset/coco2017"
]

dataset_found = False
for kp in kaggle_paths:
    candidate_img = os.path.join(kp, "val2017")
    candidate_ann = os.path.join(kp, "annotations", "instances_val2017.json")
    if os.path.isdir(candidate_img) and os.path.isfile(candidate_ann):
        VAL_ROOT = candidate_img
        VAL_ANN = candidate_ann
        dataset_found = True
        print(f"Đã tìm thấy tập dữ liệu COCO 2017 tại: {kp}")
        break

if not dataset_found:
    print("Không tìm thấy dữ liệu COCO sẵn có trên Kaggle. Tiến hành tải tự động...")
    if not os.path.isdir(VAL_ROOT):
        print("Đang tải ảnh COCO val2017...")
        download_url("http://images.cocodataset.org/zips/val2017.zip", DATA_DIR, "val2017.zip")
        with zipfile.ZipFile(os.path.join(DATA_DIR, "val2017.zip"), "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(os.path.join(DATA_DIR, "val2017.zip"))
        print("Tải ảnh hoàn tất!")

    if not os.path.isfile(VAL_ANN):
        print("Đang tải annotations COCO...")
        download_url("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", DATA_DIR, "annotations.zip")
        with zipfile.ZipFile(os.path.join(DATA_DIR, "annotations.zip"), "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(os.path.join(DATA_DIR, "annotations.zip"))
        print("Tải annotations hoàn tất!")
        
    print("Dữ liệu COCO val2017 đã sẵn sàng!")"""

COCO_DATASET_CLASS = """# 3. Định nghĩa class CocoDataset và collate_fn
class CocoDataset(Dataset):
    def __init__(self, img_dir, anno_file, img_size=(640, 640), max_samples=None):
        self.img_dir = img_dir
        self.anno_file = anno_file
        self.img_size = img_size
        self.num_classes = 80
        
        # Ánh xạ từ COCO category IDs sang chỉ số liên tục 0-79 cho YOLO
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
        
        self.images = []
        self.annotations = {}
        self._load_annotations()
        if max_samples is not None:
            self.images = self.images[:max_samples]

    def _load_annotations(self):
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
            bbox = ann["bbox"] # [x_min, y_min, width, height]
            w_img = images_dict[img_id]["width"]
            h_img = images_dict[img_id]["height"]
            
            x1 = bbox[0] / w_img
            y1 = bbox[1] / h_img
            x2 = (bbox[0] + bbox[2]) / w_img
            y2 = (bbox[1] + bbox[3]) / h_img
            
            self.annotations[img_id].append({
                "bbox": [max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
                         max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))],
                "category_id": class_idx
            })
            
        valid_img_ids = [img_id for img_id, anns in self.annotations.items() if len(anns) > 0]
        for img_id in valid_img_ids:
            self.images.append(images_dict[img_id])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
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
            "file_name": img_info["file_name"]
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# Khởi tạo dataloaders cho nhanh (100 mẫu đầu)
max_samples = 100
train_dataset = CocoDataset(VAL_ROOT, VAL_ANN, max_samples=max_samples)
val_dataset = CocoDataset(VAL_ROOT, VAL_ANN, max_samples=max_samples)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
print(f"DataLoader đã sẵn sàng: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")"""

MODEL_WRAPPER = """# 4. YOLOv5s Model Wrapper
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv5Wrapper(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = 640
        
        # Tải YOLOv5s từ torch.hub
        print("Đang tải YOLOv5s từ torch.hub...")
        self.model = torch.hub.load("ultralytics/yolov5:v7.0", "yolov5s", pretrained=True, autoshape=False)
        
        # Đảm bảo các tham số có thể huấn luyện
        for param in self.model.parameters():
            param.requires_grad = True
            
        self._adapt_head(num_classes)
        
    def _get_detect(self):
        m = self.model
        if hasattr(m, "model"):
            inner = m.model
            if hasattr(inner, "model") and isinstance(inner.model, nn.Sequential):
                return inner.model[-1]
            if isinstance(inner, nn.Sequential):
                return inner[-1]
        if isinstance(m, nn.Sequential):
            return m[-1]
        raise AttributeError("Cannot locate Detect module")

    def _adapt_head(self, num_classes):
        detect = self._get_detect()
        if detect.nc == num_classes:
            return
        detect.nc = num_classes
        detect.no = 5 + num_classes
        for i in range(len(detect.m)):
            old_conv = detect.m[i]
            na = old_conv.out_channels // (5 + 80)
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                na * (5 + num_classes),
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=old_conv.bias is not None
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                new_conv.bias.data.zero_()
            detect.m[i] = new_conv

    def decode_predictions(self, raw_outputs):
        detect = self._get_detect()
        strides = detect.stride
        anchors = detect.anchors * strides.view(-1, 1, 1)
        decoded_list = []
        for i, x in enumerate(raw_outputs):
            B, na, ny, nx, no = x.shape
            device = x.device
            grid_y, grid_x = torch.meshgrid(
                torch.arange(ny, device=device),
                torch.arange(nx, device=device),
                indexing="ij"
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, ny, nx, 2).float()
            xy = (torch.sigmoid(x[..., 0:2]) * 2.0 - 0.5 + grid) * strides[i]
            wh = (torch.sigmoid(x[..., 2:4]) * 2.0) ** 2 * anchors[i].view(1, na, 1, 1, 2)
            conf = torch.sigmoid(x[..., 4:5])
            cls_prob = torch.sigmoid(x[..., 5:])
            decoded_scale = torch.cat((xy, wh, conf, cls_prob), dim=-1)
            decoded_list.append(decoded_scale.view(B, -1, no))
        return torch.cat(decoded_list, dim=1)

    def forward(self, x):
        if self.model.training or self.training:
            raw = self.model(x)
            if isinstance(raw, tuple):
                raw = raw[0]
            return self.decode_predictions(raw)
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x)
            if isinstance(out, (tuple, list)):
                decoded = out[0]
            else:
                decoded = out
            boxes, scores, class_ids = self._decode(decoded, x.shape[2:])
            return {"boxes": boxes, "scores": scores, "class_ids": class_ids}

    def _decode(self, decoded, img_shape):
        device = decoded.device
        B = decoded.shape[0]
        img_h, img_w = img_shape
        dets_list = [[] for _ in range(B)]
        for b in range(B):
            preds = decoded[b]
            obj = preds[:, 4:5]
            cls = preds[:, 5:]
            scores_i, ids_i = (obj * cls).max(dim=1)
            conf_mask = scores_i > 0.05
            if not conf_mask.any():
                dets_list[b].append(torch.zeros(0, 6, device=device))
                continue
            xc, yc, w, h = preds[conf_mask, 0], preds[conf_mask, 1], preds[conf_mask, 2], preds[conf_mask, 3]
            scores_i, ids_i = scores_i[conf_mask], ids_i[conf_mask]
            x1, y1 = xc - w / 2, yc - h / 2
            x2, y2 = xc + w / 2, yc + h / 2
            dets_list[b].append(torch.stack([x1, y1, x2, y2, scores_i, ids_i.float()], dim=1))

        final_boxes, final_scores, final_ids = [], [], []
        for b in range(B):
            cat = torch.cat(dets_list[b], dim=0)
            if cat.numel() == 0:
                final_boxes.append(torch.zeros(0, 4, device=device))
                final_scores.append(torch.zeros(0, device=device))
                final_ids.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            keep = self._nms(cat[:, :4], cat[:, 4], 0.5)
            final_boxes.append(cat[keep, :4] / torch.tensor([img_w, img_h] * 2, device=device))
            final_scores.append(cat[keep, 4])
            final_ids.append(cat[keep, 5].long())

        max_dets = max(b.shape[0] for b in final_boxes)
        if max_dets == 0:
            return (
                torch.zeros(B, 0, 4, device=device),
                torch.zeros(B, 0, device=device),
                torch.zeros(B, 0, dtype=torch.long, device=device),
            )
        pb = torch.zeros(B, max_dets, 4, device=device)
        ps = torch.zeros(B, max_dets, device=device)
        pi = torch.zeros(B, max_dets, dtype=torch.long, device=device)
        for i in range(B):
            n = final_boxes[i].shape[0]
            pb[i, :n] = final_boxes[i]
            ps[i, :n] = final_scores[i]
            pi[i, :n] = final_ids[i]
        return pb, ps, pi

    @staticmethod
    def _nms(boxes, scores, iou_thresh=0.5):
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(-1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            order = order[1:][iou <= iou_thresh]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def get_params_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def calculate_flops(self, input_size=(3, 640, 640)) -> int:
        try:
            from fvcore.nn import FlopCountAnalysis
            x = torch.zeros(1, *input_size, device=next(self.parameters()).device)
            flops = FlopCountAnalysis(self.model, x)
            return flops.total()
        except Exception:
            active_params = sum(p.numel() for p in self.parameters() if p.requires_grad and (p != 0).any())
            return int(active_params * 2 * 10)"""

BASE_PRUNER_CLASS = """# 5. Định nghĩa BasePruner cho các thuật toán Cắt tỉa
class BasePruner:
    def __init__(self, model, pruning_ratio=0.0):
        self.model = model
        self.pruning_ratio = pruning_ratio

    def discover_prunable_layers(self):
        prunable = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable.append((name, module))
        return prunable

    def calculate_sparsity(self):
        total_weights = 0
        zero_weights = 0
        for name, module in self.discover_prunable_layers():
            w = module.weight.data
            total_weights += w.numel()
            zero_weights += (w == 0.0).sum().item()
        return zero_weights / total_weights if total_weights > 0 else 0.0

    def register_mask(self, module, mask):
        if hasattr(module, "pruning_mask"):
            delattr(module, "pruning_mask")
        module.register_buffer("pruning_mask", mask.float())

    def apply_mask(self, module):
        if hasattr(module, "pruning_mask"):
            module.weight.data.mul_(module.pruning_mask)
            if module.bias is not None and module.pruning_mask.shape[0] == module.bias.shape[0]:
                if module.pruning_mask.ndim == 1:
                    module.bias.data.mul_(module.pruning_mask)
                elif all(d == 1 for d in module.pruning_mask.shape[1:]):
                    module.bias.data.mul_(module.pruning_mask.view(-1))

    def apply_all_masks(self):
        for name, module in self.discover_prunable_layers():
            self.apply_mask(module)

    def enforce_sparsity(self):
        for name, module in self.discover_prunable_layers():
            if hasattr(module, "pruning_mask"):
                module.weight.data.mul_(module.pruning_mask)

    def zero_gradients(self):
        for name, module in self.discover_prunable_layers():
            if hasattr(module, "pruning_mask") and module.weight.grad is not None:
                module.weight.grad.data.mul_(module.pruning_mask)

    def collect_statistics(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        active_params = 0
        for name, module in self.discover_prunable_layers():
            if hasattr(module, "pruning_mask"):
                active_params += int(module.pruning_mask.sum().item() * (module.weight.numel() / module.pruning_mask.numel()))
            else:
                active_params += module.weight.numel()
                
        all_conv_linear_params = sum(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                                     for name, module in self.discover_prunable_layers())
        other_params = total_params - all_conv_linear_params
        active_total_params = active_params + other_params
        
        size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        return {
            "total_params": total_params,
            "active_params": int(active_total_params),
            "sparsity": 1.0 - (active_total_params / total_params) if total_params > 0 else 0.0,
            "size_mb": size_mb,
        }"""

EVALUATION_HELPERS = """# 6. Các hàm hỗ trợ Đánh giá Hiệu năng (Latency, FPS, mAP)
import time
import numpy as np

def bbox_iou_numpy(box1, box2):
    N = box1.shape[0]
    M = box2.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M))
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-8, None)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for imgs, targets in loader:
            batch_imgs = torch.stack(imgs).to(device)
            outputs = model(batch_imgs)
            for i in range(len(imgs)):
                img_id = targets[i]["image_id"].item()
                gt_boxes = targets[i]["boxes"].cpu().numpy()
                gt_labels = targets[i]["labels"].cpu().numpy()
                
                pred_boxes = outputs["boxes"][i].cpu().numpy()
                pred_scores = outputs["scores"][i].cpu().numpy()
                pred_labels = outputs["class_ids"][i].cpu().numpy()
                
                keep = pred_scores >= 0.05
                all_preds.append({
                    "image_id": img_id,
                    "boxes": pred_boxes[keep],
                    "scores": pred_scores[keep],
                    "labels": pred_labels[keep],
                })
                all_gts.append({
                    "image_id": img_id,
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                })
                
    num_classes = 80
    aps = []
    total_tp = 0
    total_fp = 0
    total_gts_count = 0
    
    for cls in range(num_classes):
        cls_preds = []
        num_cls_gts = 0
        for pred, gt in zip(all_preds, all_gts):
            cls_mask = pred["labels"] == cls
            p_boxes = pred["boxes"][cls_mask]
            p_scores = pred["scores"][cls_mask]
            for box, score in zip(p_boxes, p_scores):
                cls_preds.append({"image_id": pred["image_id"], "box": box, "score": score})
            gt_mask = gt["labels"] == cls
            num_cls_gts += gt_mask.sum()
            
        total_gts_count += num_cls_gts
        if num_cls_gts == 0:
            continue
            
        cls_preds = sorted(cls_preds, key=lambda x: x["score"], reverse=True)
        gt_tracked = {}
        for gt in all_gts:
            img_id = gt["image_id"]
            cls_mask = gt["labels"] == cls
            gt_tracked[img_id] = {
                "boxes": gt["boxes"][cls_mask],
                "matched": np.zeros(gt["boxes"][cls_mask].shape[0], dtype=bool)
            }
            
        tps = np.zeros(len(cls_preds))
        fps = np.zeros(len(cls_preds))
        
        for idx, pred in enumerate(cls_preds):
            img_id = pred["image_id"]
            p_box = pred["box"]
            gts_img = gt_tracked[img_id]["boxes"]
            if len(gts_img) == 0:
                fps[idx] = 1
                continue
            ious = bbox_iou_numpy(p_box[None, :], gts_img)[0]
            best_gt_idx = ious.argmax()
            if ious[best_gt_idx] >= 0.5 and not gt_tracked[img_id]["matched"][best_gt_idx]:
                tps[idx] = 1
                gt_tracked[img_id]["matched"][best_gt_idx] = True
                total_tp += 1
            else:
                fps[idx] = 1
                total_fp += 1
                
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recalls = tp_cum / max(num_cls_gts, 1e-8)
        precisions = tp_cum / np.clip(tp_cum + fp_cum, 1e-8, None)
        
        ap = 0.0
        for r in np.linspace(0.0, 1.0, 11):
            p = precisions[recalls >= r]
            ap += (p.max() if len(p) > 0 else 0.0) / 11.0
        aps.append(ap)
        
    mAP50 = np.mean(aps) if aps else 0.0
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gts_count, 1)
    
    # Đo đạc Latency & FPS
    x = torch.zeros((1, 3, 640, 640), device=device)
    for _ in range(10):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    runs = 50
    for _ in range(runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start_time) * 1000 / runs
    fps = 1000 / latency if latency > 0 else 0.0
    
    return {
        "Params": model.get_params_count(),
        "FLOPs": model.calculate_flops(),
        "Size (MB)": model.get_model_size_mb(),
        "Latency (ms)": latency,
        "FPS": fps,
        "mAP50": mAP50,
        "Precision": precision,
        "Recall": recall
    }"""

TRAINING_HELPERS = """# 7. Hàm hỗ trợ huấn luyện hồi phục (Recovery Training)
def bbox_iou_pytorch(box1, box2):
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    box1_area = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    box2_area = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area + 1e-8
    return inter_area / union_area

def compute_loss(pred, targets, device, num_classes=80):
    B = pred.shape[0]
    S = pred.shape[1]
    img_size = 640
    total_box_loss = torch.tensor(0.0, device=device)
    total_cls_loss = torch.tensor(0.0, device=device)
    total_obj_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        p = pred[b]
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)
        p_boxes = p[:, :4]
        p_conf = p[:, 4]
        p_cls = p[:, 5:]
        obj_targets = torch.zeros(S, device=device)

        if gt_boxes.numel() > 0:
            gt_xc = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 * img_size
            gt_yc = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 * img_size
            gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]) * img_size
            gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]) * img_size
            gt_pixel_cxcywh = torch.stack((gt_xc, gt_yc, gt_w, gt_h), dim=-1)
            
            ious = bbox_iou_pytorch(gt_pixel_cxcywh, p_boxes)
            matched_preds, matched_gts = [], []
            for gt_idx in range(gt_pixel_cxcywh.shape[0]):
                best_pred_idx = ious[gt_idx].argmax().item()
                matched_preds.append(best_pred_idx)
                matched_gts.append(gt_idx)
                obj_targets[best_pred_idx] = ious[gt_idx, best_pred_idx].detach()
                
            matched_preds_t = torch.tensor(matched_preds, dtype=torch.long, device=device)
            matched_gts_t = torch.tensor(matched_gts, dtype=torch.long, device=device)
            
            pos_preds_box = p_boxes[matched_preds_t]
            pos_gts_box = gt_pixel_cxcywh[matched_gts_t]
            total_box_loss += F.l1_loss(pos_preds_box, pos_gts_box, reduction="mean")
            
            pos_preds_cls = p_cls[matched_preds_t]
            pos_gts_labels = gt_labels[matched_gts_t]
            target_one_hot = torch.zeros((len(matched_gts), num_classes), device=device)
            target_one_hot.scatter_(1, pos_gts_labels.unsqueeze(1), 1.0)
            total_cls_loss += F.binary_cross_entropy(pos_preds_cls, target_one_hot, reduction="mean")
            
        total_obj_loss += F.binary_cross_entropy(p_conf, obj_targets, reduction="mean")

    box_w, cls_w, obj_w = 0.05, 0.5, 1.0
    return (total_box_loss / B) * box_w + (total_cls_loss / B) * cls_w + (total_obj_loss / B) * obj_w

def fine_tune_model(model, train_loader, device, epochs=3, lr=1e-4):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    
    is_pruned = any(hasattr(m, "pruning_mask") for m in model.modules())
    
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, targets in train_loader:
            batch_imgs = torch.stack(imgs).to(device)
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(batch_imgs)
                
            with torch.amp.autocast("cuda", enabled=False):
                loss = compute_loss(outputs, targets, device)
                
            scaler.scale(loss).backward()
            
            # Đóng băng gradient cho các trọng số đã bị prune
            if is_pruned:
                for module in model.modules():
                    if hasattr(module, "pruning_mask") and module.weight.grad is not None:
                        module.weight.grad.data.mul_(module.pruning_mask)
                        
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Enforce zero weight constraints in pruned slots
            if is_pruned:
                for module in model.modules():
                    if hasattr(module, "pruning_mask"):
                        module.weight.data.mul_(module.pruning_mask)
                        
            running_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {running_loss / len(train_loader):.4f}")"""

BASELINE_RUN = """# 8. Benchmark Baseline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Khởi tạo mô hình và chuyển sang GPU/CPU
model = YOLOv5Wrapper(num_classes=80).to(device)

print("\\nĐang benchmark mô hình Baseline...")
baseline_results = evaluate_model(model, val_loader, device)

print("\\nKẾT QUẢ BASELINE:")
for k, v in baseline_results.items():
    if isinstance(v, float):
        print(f"  {k:<20}: {v:.4f}")
    else:
        print(f"  {k:<20}: {v}")"""

# Define specific pruner structures
CHANNEL_PRUNER_CODE = """class ChannelPruner(BasePruner):
    def prune(self):
        if self.pruning_ratio <= 0.0:
            return self.model
        for name, module in self.discover_prunable_layers():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                num_channels = w.size(1)
                channel_norms = w.abs().sum(dim=[0, 2, 3])
                num_to_prune = int(num_channels * self.pruning_ratio)
                if num_to_prune == 0:
                    continue
                threshold = torch.topk(channel_norms, num_to_prune, largest=False).values[-1].item()
                mask = channel_norms >= threshold
                mask_expanded = mask.view(1, -1, 1, 1).to(w.device).float()
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
            elif isinstance(module, nn.Linear):
                w = module.weight.data
                num_channels = w.size(1)
                channel_norms = w.abs().sum(dim=0)
                num_to_prune = int(num_channels * self.pruning_ratio)
                if num_to_prune == 0:
                    continue
                threshold = torch.topk(channel_norms, num_to_prune, largest=False).values[-1].item()
                mask = channel_norms >= threshold
                mask_expanded = mask.view(1, -1).to(w.device).float()
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
        return self.model"""

FILTER_PRUNER_CODE = """class FilterPruner(BasePruner):
    def prune(self):
        if self.pruning_ratio <= 0.0:
            return self.model
        for name, module in self.discover_prunable_layers():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                num_filters = w.size(0)
                filter_means = w.abs().mean(dim=[1, 2, 3])
                num_to_prune = int(num_filters * self.pruning_ratio)
                if num_to_prune == 0:
                    continue
                threshold = torch.topk(filter_means, num_to_prune, largest=False).values[-1].item()
                mask = filter_means >= threshold
                mask_expanded = mask.view(-1, 1, 1, 1).to(w.device).float()
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
        return self.model"""

L1NORM_PRUNER_CODE = """class L1NormPruner(BasePruner):
    def prune(self):
        if self.pruning_ratio <= 0.0:
            return self.model
        for name, module in self.discover_prunable_layers():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                num_filters = w.size(0)
                l1_norms = w.abs().sum(dim=[1, 2, 3])
                num_to_prune = int(num_filters * self.pruning_ratio)
                if num_to_prune == 0:
                    continue
                threshold = torch.topk(l1_norms, num_to_prune, largest=False).values[-1].item()
                mask = l1_norms >= threshold
                mask_expanded = mask.view(-1, 1, 1, 1).to(w.device).float()
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
        return self.model"""

MAGNITUDE_PRUNER_CODE = """class MagnitudePruner(BasePruner):
    def compute_mask(self, module):
        w = module.weight.data.abs().view(-1)
        k = int((1.0 - self.pruning_ratio) * w.numel())
        if k < 1: k = 1
        threshold = w.topk(k).values.min() if k <= w.numel() else 0.0
        return (module.weight.data.abs() >= threshold).float()

    def prune(self):
        for name, module in self.discover_prunable_layers():
            mask = self.compute_mask(module)
            self.register_mask(module, mask)
            self.apply_mask(module)
        return self.model"""

LAYER_PRUNER_CODE = """def get_c3_bottlenecks(model):
    bottlenecks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'C3':
            if hasattr(module, 'm') and isinstance(module.m, nn.Sequential):
                for idx, bottleneck in enumerate(module.m):
                    if bottleneck.__class__.__name__ == 'Bottleneck':
                        if hasattr(bottleneck, 'cv2'):
                            conv = bottleneck.cv2.conv
                            conv_weight = conv.weight.data.abs().cpu()
                            weight_alpha = torch.mean(conv_weight.view(conv_weight.size(0), -1), dim=1)
                            if hasattr(bottleneck.cv2, 'bn') and bottleneck.cv2.bn is not None:
                                bn = bottleneck.cv2.bn
                                bn_weight = bn.weight.data.abs().cpu()
                                importance_tensor = 10 * weight_alpha * bn_weight
                            else:
                                importance_tensor = weight_alpha
                            mean_importance = torch.mean(importance_tensor).item()
                            bottlenecks.append({
                                'c3_name': name,
                                'c3_module': module,
                                'bottleneck_idx': idx,
                                'bottleneck_module': bottleneck,
                                'importance': mean_importance
                            })
    return bottlenecks

def prune_yolov5_c3_layers(model, num_to_prune):
    import copy
    model = copy.deepcopy(model)
    bottlenecks = get_c3_bottlenecks(model)
    total_bottlenecks = len(bottlenecks)
    if total_bottlenecks == 0:
        return model
    if num_to_prune >= total_bottlenecks:
        num_to_prune = total_bottlenecks - 1
    sorted_bottlenecks = sorted(bottlenecks, key=lambda x: x['importance'])
    pruned_info = sorted_bottlenecks[:num_to_prune]
    c3_kept_map = {}
    for item in bottlenecks:
        is_pruned = any(
            p['c3_name'] == item['c3_name'] and p['bottleneck_idx'] == item['bottleneck_idx']
            for p in pruned_info
        )
        if not is_pruned:
            if item['c3_name'] not in c3_kept_map:
                c3_kept_map[item['c3_name']] = []
            c3_kept_map[item['c3_name']].append(item['bottleneck_module'])
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'C3':
            kept_modules = c3_kept_map.get(name, [])
            module.m = nn.Sequential(*kept_modules)
    return model

class LayerPruner(BasePruner):
    def prune(self):
        num_layers_to_prune = max(1, int(self.pruning_ratio * 10.0))
        self.model = prune_yolov5_c3_layers(self.model, num_layers_to_prune)
        return self.model"""

# Loop and execution blocks
def get_execution_loop(pruner_class, file_prefix):
    return f"""# 9. Thực hiện Pruning và Đánh giá (Pruned Benchmarks)
import copy

sparsities = [0.3, 0.5, 0.7]
pruned_results = {{}}
pruned_benchmarks = {{}}

for sp in sparsities:
    print(f"\\n--- Áp dụng cắt tỉa với tỷ lệ Sparsity = {{sp*100:.0f}}% ---")
    model_to_prune = copy.deepcopy(model)
    
    # Khởi tạo pruner và thực hiện cắt tỉa
    pruner = {pruner_class}(model_to_prune, pruning_ratio=sp)
    pruned_model = pruner.prune()
    
    # Đo các chỉ số cấu trúc
    stats = pruner.collect_statistics()
    print(f"  Tham số active: {{stats['active_params']:,}} / {{stats['total_params']:,}}")
    print(f"  Tỷ lệ thưa thực tế: {{stats['sparsity']*100:.2f}}%")
    print(f"  Kích thước file ước lượng: {{stats['size_mb']:.2f}} MB")
    
    pruned_results[sp] = {{
        "model": pruned_model,
        "pruner": pruner,
        "stats": stats
    }}
    
    # Đánh giá hiệu năng và mAP
    print(f"  Đang benchmark mô hình Pruned (Sparsity {{sp}})...")
    res = evaluate_model(pruned_model, val_loader, device)
    pruned_benchmarks[sp] = res"""

def get_recovery_loop(file_prefix):
    return """# 10. Thực hiện Recovery Fine-tuning
import os

recovered_results = {}
recovered_benchmarks = {}

EPOCHS = 3
LR = 1e-4

# Tạo thư mục checkpoints
os.makedirs("checkpoints", exist_ok=True)

for sp, data in pruned_results.items():
    print(f"\\n==================================================")
    print(f"Bắt đầu huấn luyện hồi phục (Recovery) - Sparsity {sp}")
    print(f"==================================================")
    
    model_to_recover = copy.deepcopy(data["model"])
    
    # Huấn luyện khôi phục
    fine_tune_model(model_to_recover, train_loader, device, epochs=EPOCHS, lr=LR)
    
    # Lưu trọng số
    save_path = f"checkpoints/yolov5s_recovered_sp_{sp}.pt"
    torch.save(model_to_recover.state_dict(), save_path)
    print(f"  Đã lưu checkpoint phục hồi tại: {save_path}")
    
    recovered_results[sp] = model_to_recover
    
    # Đánh giá sau phục hồi
    print(f"  Đang benchmark mô hình sau Recovery (Sparsity {sp})...")
    res = evaluate_model(model_to_recover, val_loader, device)
    recovered_benchmarks[sp] = res"""

def get_report_block(file_prefix):
    return f"""# 11. Tổng hợp kết quả và xuất báo cáo CSV
import pandas as pd
from IPython.display import display

report_data = []

# Thêm baseline
report_data.append({{
    "Stage": "Baseline",
    "Pruning Ratio": 0.0,
    "Params": baseline_results["Params"],
    "FLOPs (G)": baseline_results["FLOPs"] / 1e9,
    "Size (MB)": baseline_results["Size (MB)"],
    "Latency (ms)": baseline_results["Latency (ms)"],
    "FPS": baseline_results["FPS"],
    "mAP50": baseline_results["mAP50"]
}})

# Thêm pruned và recovered
for sp in sparsities:
    p_res = pruned_benchmarks[sp]
    r_res = recovered_benchmarks[sp]
    
    report_data.append({{
        "Stage": f"Pruned (Sparsity {{sp}})",
        "Pruning Ratio": sp,
        "Params": p_res["Params"],
        "FLOPs (G)": p_res["FLOPs"] / 1e9,
        "Size (MB)": p_res["Size (MB)"],
        "Latency (ms)": p_res["Latency (ms)"],
        "FPS": p_res["FPS"],
        "mAP50": p_res["mAP50"]
    }})
    
    report_data.append({{
        "Stage": f"Recovered (Sparsity {{sp}})",
        "Pruning Ratio": sp,
        "Params": r_res["Params"],
        "FLOPs (G)": r_res["FLOPs"] / 1e9,
        "Size (MB)": r_res["Size (MB)"],
        "Latency (ms)": r_res["Latency (ms)"],
        "FPS": r_res["FPS"],
        "mAP50": r_res["mAP50"]
    }})

df = pd.DataFrame(report_data)
display(df)

df.to_csv("{file_prefix}_pruning_report.csv", index=False)
print(f"Đã xuất báo cáo vào {file_prefix}_pruning_report.csv")"""

def get_plot_block(pruner_name, file_prefix):
    return f"""# 12. Trực quan hóa kết quả bằng biểu đồ
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("So sánh hiệu năng YOLOv5s với {pruner_name}", fontsize=16)

# Biểu đồ mAP50
axes[0].plot(sparsities, [baseline_results["mAP50"]]*len(sparsities), label="Baseline", linestyle="--", color="black")
axes[0].plot(sparsities, [pruned_benchmarks[sp]["mAP50"] for sp in sparsities], marker="o", label="Pruned")
axes[0].plot(sparsities, [recovered_benchmarks[sp]["mAP50"] for sp in sparsities], marker="s", label="Recovered")
axes[0].set_title("Độ chính xác mAP50")
axes[0].set_xlabel("Sparsity Ratio")
axes[0].set_ylabel("mAP50")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Biểu đồ Latency
axes[1].plot(sparsities, [baseline_results["Latency (ms)"]]*len(sparsities), label="Baseline", linestyle="--", color="black")
axes[1].plot(sparsities, [pruned_benchmarks[sp]["Latency (ms)"] for sp in sparsities], marker="o", label="Pruned")
axes[1].plot(sparsities, [recovered_benchmarks[sp]["Latency (ms)"] for sp in sparsities], marker="s", label="Recovered")
axes[1].set_title("Độ trễ Latency (ms)")
axes[1].set_xlabel("Sparsity Ratio")
axes[1].set_ylabel("Latency (ms)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Biểu đồ kích thước (Size MB)
axes[2].plot(sparsities, [baseline_results["Size (MB)"]]*len(sparsities), label="Baseline", linestyle="--", color="black")
axes[2].plot(sparsities, [pruned_benchmarks[sp]["Size (MB)"] for sp in sparsities], marker="o", label="Pruned")
axes[2].plot(sparsities, [recovered_benchmarks[sp]["Size (MB)"] for sp in sparsities], marker="s", label="Recovered")
axes[2].set_title("Kích thước mô hình (MB)")
axes[2].set_xlabel("Sparsity Ratio")
axes[2].set_ylabel("Size (MB)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("{file_prefix}_performance_comparison.png", dpi=150)
plt.show()
print("Đồ thị so sánh đã được lưu vào {file_prefix}_performance_comparison.png")"""


def generate_notebook_content(pruner_name, pruner_class, pruner_code, file_prefix):
    cells = []
    
    # 1. Introduction Markdown
    cells.append(make_markdown_cell(f"""# YOLOv5s {pruner_name} Pipeline trên Kaggle (Độc Lập)

Notebook này thực hiện tối ưu hóa mô hình YOLOv5s bằng phương pháp **{pruner_name}** trên tập dữ liệu COCO validation 2017. 
*Notebook được thiết kế hoàn toàn độc lập và không phụ thuộc vào bất kỳ import nào bên ngoài.*

### Quy trình chạy:
1. **Môi trường**: Cài đặt các gói pip.
2. **Dữ liệu**: Tải/Đọc COCO val 2017 và tạo DataLoader.
3. **Mô hình**: Tải YOLOv5s pretrained từ `torch.hub`.
4. **Benchmark Baseline**: Đo chất lượng ban đầu của mô hình.
5. **Cắt tỉa (Pruning)**: Thực hiện thuật toán `{pruner_class}` với các mức độ thưa (0.3, 0.5, 0.7).
6. **Benchmark sau Pruning**: Đánh giá hiệu năng ngay sau khi cắt tỉa.
7. **Recovery Fine-tuning**: Huấn luyện hồi phục ngắn hạn (3 epochs) để phục hồi mAP.
8. **Benchmark sau Recovery**: Đánh giá kết quả cuối cùng.
9. **Trực quan hóa**: Xuất bảng báo cáo CSV và biểu đồ đồ thị so sánh."""))

    # 2. Setup pip
    cells.append(make_code_cell(PIP_INSTALL))

    # 3. Download & Prepare Dataset
    cells.append(make_code_cell(DATASET_PREP))

    # 4. CocoDataset Class definition & dataloaders
    cells.append(make_code_cell(COCO_DATASET_CLASS))

    # 5. YOLOv5s Wrapper definition
    cells.append(make_code_cell(MODEL_WRAPPER))

    # 6. BasePruner definition
    cells.append(make_code_cell(BASE_PRUNER_CLASS))

    # 7. Specific Pruner implementation
    cells.append(make_code_cell(f"# 5.1. Định nghĩa thuật toán {pruner_name}\n" + pruner_code))

    # 8. Evaluation Helpers
    cells.append(make_code_cell(EVALUATION_HELPERS))

    # 9. Training Helpers
    cells.append(make_code_cell(TRAINING_HELPERS))

    # 10. Run Baseline Benchmark
    cells.append(make_code_cell(BASELINE_RUN))

    # 11. Run Pruning
    cells.append(make_code_cell(get_execution_loop(pruner_class, file_prefix)))

    # 12. Run Recovery
    cells.append(make_code_cell(get_recovery_loop(file_prefix)))

    # 13. Summary Report
    cells.append(make_code_cell(get_report_block(file_prefix)))

    # 14. Performance Plotting
    cells.append(make_code_cell(get_plot_block(pruner_name, file_prefix)))

    return make_notebook(cells)

notebook_configs = [
    ("Channel Pruning", "ChannelPruner", CHANNEL_PRUNER_CODE, "channel_pruning"),
    ("Filter Pruning", "FilterPruner", FILTER_PRUNER_CODE, "filter_pruning"),
    ("Layer Pruning", "LayerPruner", LAYER_PRUNER_CODE, "layer_pruning"),
    ("Magnitude Pruning", "MagnitudePruner", MAGNITUDE_PRUNER_CODE, "magnitude_pruning"),
    ("L1-norm based Pruning", "L1NormPruner", L1NORM_PRUNER_CODE, "l1_norm_pruning")
]

notebooks_dir = "d:/Project/traffic-analysis-app/notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

for name, cls, code, prefix in notebook_configs:
    nb_content = generate_notebook_content(name, cls, code, prefix)
    nb_path = os.path.join(notebooks_dir, f"{prefix}.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=1, ensure_ascii=False)
    print(f"Generated self-contained notebook: {nb_path}")
