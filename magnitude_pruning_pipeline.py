import os, sys, copy, time, json, tempfile, warnings, zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url
from torchvision.transforms import functional as TF
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

warnings.filterwarnings('ignore')
print("All imports OK.")

# ===========================================================================
#  UNIFIED BASE MODEL INTERFACE
# ===========================================================================
class BaseModel(nn.Module):
    def forward(self, images): raise NotImplementedError
    def get_raw_model(self): raise NotImplementedError
    def get_params_count(self):
        return sum(p.numel() for p in self.get_raw_model().parameters())
    def get_input_size(self): raise NotImplementedError


def calculate_model_sparsity(model):
    total = 0; zeros = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / max(total, 1)


# ===========================================================================
#  BASE PRUNER + MAGNITUDE PRUNER
# ===========================================================================
class BasePruner:
    def __init__(self, model, pruning_ratio=0.0):
        self.model = model
        self.pruning_ratio = pruning_ratio

    def discover_prunable_layers(self):
        return [(n, m) for n, m in self.model.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))]

    def register_mask(self, module, mask):
        if 'pruning_mask' in module._buffers:
            del module._buffers['pruning_mask']
        module.register_buffer('pruning_mask', mask.float())

    def apply_mask(self, module):
        if hasattr(module, 'pruning_mask'):
            module.weight.data.mul_(module.pruning_mask)

    def apply_all_masks(self):
        for _, module in self.discover_prunable_layers():
            self.apply_mask(module)

    def calculate_sparsity(self):
        t = 0; z = 0
        for _, m in self.discover_prunable_layers():
            w = m.weight.data; t += w.numel(); z += (w == 0.0).sum().item()
        return z / t if t > 0 else 0.0


class MagnitudePruner(BasePruner):
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


print("Base classes defined.")

# ===========================================================================
#  DETR MODEL WRAPPER
# ===========================================================================
class DETRModel(BaseModel):
    DETR_IMG_SIZE = 800

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        self.num_classes = self.model.class_embed.out_features - 1

    def forward(self, images):
        if images.shape[-1] != self.DETR_IMG_SIZE or images.shape[-2] != self.DETR_IMG_SIZE:
            images = F.interpolate(images, size=(self.DETR_IMG_SIZE, self.DETR_IMG_SIZE),
                                   mode='bilinear', align_corners=False)
        out = self.model(images)
        logits = out['pred_logits']
        boxes = out['pred_boxes']
        prob = F.softmax(logits, dim=-1)
        scores, labels = prob[..., :-1].max(dim=-1)
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2).clamp(0, 1); y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1); y2 = (cy + h / 2).clamp(0, 1)
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        results = []
        for b in range(images.shape[0]):
            dets = torch.stack([
                boxes_xyxy[b, :, 0], boxes_xyxy[b, :, 1],
                boxes_xyxy[b, :, 2], boxes_xyxy[b, :, 3],
                scores[b], labels[b].float()
            ], dim=-1)
            results.append(dets)
        return results

    def forward_train(self, images):
        B, _, H, W = images.shape
        if W != self.DETR_IMG_SIZE or H != self.DETR_IMG_SIZE:
            images = F.interpolate(images, size=(self.DETR_IMG_SIZE, self.DETR_IMG_SIZE),
                                   mode='bilinear', align_corners=False)
        out = self.model(images)
        return out['pred_logits'], out['pred_boxes']

    def get_raw_model(self): return self.model
    def get_input_size(self): return (3, self.DETR_IMG_SIZE, self.DETR_IMG_SIZE)


print("DETRModel defined.")

# ===========================================================================
#  YOLOv5s MODEL WRAPPER
# ===========================================================================
class YOLOv5Model(BaseModel):
    YOLO_IMG_SIZE = 640

    def __init__(self):
        super().__init__()
        from ultralytics import YOLO
        _hub = YOLO('yolov5s.pt')
        self.raw_model = _hub.model.model
        self.conf = 0.001
        self.iou = 0.65
        self.num_classes = self._infer_classes()

    def _infer_classes(self):
        detect = self._get_detect()
        return detect.no - 5 if (detect is not None and hasattr(detect, 'no')) else 80

    def _get_detect(self):
        m = self.raw_model
        for i in range(len(m) - 1, -1, -1):
            if hasattr(m[i], 'no'):
                return m[i]
        return None

    def _run_backbone_neck(self, x):
        m = self.raw_model
        detect_idx = -1
        for i in range(len(m) - 1, -1, -1):
            if hasattr(m[i], 'no'):
                detect_idx = i
                break
        layer_outputs = []
        y = x
        for i, layer in enumerate(m):
            if i == detect_idx:
                break
            if hasattr(layer, 'f') and layer.f != -1:
                f = layer.f
                if isinstance(f, int):
                    inp = layer_outputs[f] if f >= 0 else y
                else:
                    inp = [layer_outputs[j] if j >= 0 else y for j in f]
            else:
                inp = y
            y = layer(inp)
            layer_outputs.append(y)
        return [layer_outputs[j] for j in m[detect_idx].f if j >= 0]

    def _decode(self, x):
        detect = self._get_detect()
        if detect is None: return x
        z = []
        for i, xi in enumerate(x):
            bs, _, ny, nx = xi.shape
            xi = xi.view(bs, detect.no, ny, nx).permute(0, 2, 3, 1).contiguous()
            if detect.grid[i].shape[2:4] != (ny, nx):
                detect.grid[i], detect.anchor_grid[i] = detect._make_grid(nx, ny, i)
            y = xi.sigmoid()
            nch = 5 + self.num_classes
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + detect.grid[i].to(xi.device)) * detect.stride[i].to(xi.device)
            y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * detect.anchor_grid[i].to(xi.device)
            z.append(y.view(bs, -1, nch))
        return torch.cat(z, 1)

    def _decode_to_xyxy(self, x, img_w, img_h):
        if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], torch.Tensor) and x[0].dim() == 3:
            pred = x[0]
        elif isinstance(x, torch.Tensor) and x.dim() == 3:
            pred = x
        else:
            pred = self._decode(x)
        box_cxcy = pred[..., 0:2]; box_wh = pred[..., 2:4]
        x1y1 = box_cxcy - box_wh / 2.0; x2y2 = box_cxcy + box_wh / 2.0
        boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)
        obj_conf = pred[..., 4:5]; cls_conf = pred[..., 5:]
        scores, labels = cls_conf.max(dim=-1, keepdim=True)
        scores = scores * obj_conf
        out = torch.cat([boxes_xyxy, scores, labels.float()], dim=-1)
        out[..., 0] /= img_w; out[..., 1] /= img_h
        out[..., 2] /= img_w; out[..., 3] /= img_h
        return out

    def _nms(self, pred, conf_thres=0.001, iou_thres=0.65):
        from torchvision.ops import nms
        out = []
        for i in range(pred.shape[0]):
            det = pred[i]
            mask = det[:, 4] > conf_thres
            det = det[mask]
            if det.shape[0] == 0:
                out.append(torch.zeros((0, 6), device=pred.device))
                continue
            keep = nms(det[:, :4], det[:, 4], iou_thres)
            out.append(det[keep])
        return out

    def forward(self, images):
        B, _, H, W = images.shape
        x = images * 255.0
        if W != self.YOLO_IMG_SIZE or H != self.YOLO_IMG_SIZE:
            x = F.interpolate(x, size=(self.YOLO_IMG_SIZE, self.YOLO_IMG_SIZE),
                              mode='bilinear', align_corners=False)
        fpn_maps = self._run_backbone_neck(x)
        detect = self._get_detect()
        raw_out = detect(fpn_maps)
        pred = self._decode_to_xyxy(raw_out, self.YOLO_IMG_SIZE, self.YOLO_IMG_SIZE)
        return self._nms(pred, self.conf, self.iou)

    def forward_train(self, images):
        B, _, H, W = images.shape
        x = images * 255.0
        if W != self.YOLO_IMG_SIZE or H != self.YOLO_IMG_SIZE:
            x = F.interpolate(x, size=(self.YOLO_IMG_SIZE, self.YOLO_IMG_SIZE),
                              mode='bilinear', align_corners=False)
        fpn_maps = self._run_backbone_neck(x)
        detect = self._get_detect()
        raw_out = detect(fpn_maps)
        if isinstance(raw_out, tuple) and len(raw_out) == 2:
            raw_out = raw_out[1]
        return self._decode(raw_out)

    def get_raw_model(self): return self.raw_model
    def get_input_size(self): return (3, self.YOLO_IMG_SIZE, self.YOLO_IMG_SIZE)


print("YOLOv5Model defined.")

# ===========================================================================
#  COCO VALIDATION DATASET  (returns orig_w, orig_h for absolute pixel rescale)
# ===========================================================================
class CocoValDataset(Dataset):
    def __init__(self, root, annFile, img_size=640, max_samples=None, model_type='yolo'):
        self.coco = COCO(annFile)
        self.root = root
        self.img_size = img_size
        self.model_type = model_type
        self.ids = list(self.coco.imgs.keys())
        if max_samples is not None:
            self.ids = self.ids[:max_samples]

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        fpath = os.path.join(self.root, img_info['file_name'])
        img = Image.open(fpath).convert('RGB')
        orig_w, orig_h = img.size
        img = TF.resize(img, (self.img_size, self.img_size))
        tensor = TF.to_tensor(img)
        # CRITICAL FIX: DETR was trained with ImageNet normalization
        # Without it, the model receives OOD input and outputs garbage (mAP ~ 0)
        if self.model_type == 'detr':
            tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, img_id, orig_w, orig_h


# ===========================================================================
#  COCO TRAINING DATASET  (recovery fine-tuning)
# ===========================================================================
class CocoTrainDataset(Dataset):
    def __init__(self, root, annFile, img_size=640, offset=0, max_samples=None, model_type='yolo'):
        self.coco = COCO(annFile)
        self.root = root
        self.img_size = img_size
        self.model_type = model_type
        self.ids = list(self.coco.imgs.keys())
        if offset > 0: self.ids = self.ids[offset:]
        if max_samples is not None: self.ids = self.ids[:max_samples]
        
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

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        fpath = os.path.join(self.root, img_info['file_name'])
        img = Image.open(fpath).convert('RGB')
        orig_w, orig_h = img.size
        img = TF.resize(img, (self.img_size, self.img_size))
        img_tensor = TF.to_tensor(img)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []; labels = []
        for ann in anns:
            if ann.get('ignore', 0) or ann.get('iscrowd', 0): continue
            cat_id = ann['category_id']
            if self.model_type == 'yolo':
                if cat_id not in self.coco_cat_to_idx: continue
                class_label = self.coco_cat_to_idx[cat_id]
            else:
                class_label = cat_id - 1
            x, y, w, h = ann['bbox']
            x1 = x / orig_w; y1 = y / orig_h
            x2 = (x + w) / orig_w; y2 = (y + h) / orig_h
            boxes.append([x1, y1, x2, y2])
            labels.append(class_label)
        if not boxes:
            boxes = torch.zeros((0, 4)); labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.long)
        return img_tensor, {'boxes': boxes, 'labels': labels}


print("Datasets defined.")

# ===========================================================================
#  LOSS FUNCTIONS
# ===========================================================================
def self_bbox_iou(box1, box2):
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    return inter / (area1 + area2 - inter + 1e-9)


def compute_yolo_loss(pred, targets, num_classes=80):
    if isinstance(pred, dict):
        if 'output' in pred:
            pred = pred['output']
        elif 'pred' in pred:
            pred = pred['pred']
        else:
            pred = list(pred.values())[0]
    
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(pred)}")
    
    B = pred.shape[0]
    device = pred.device
    IMG_S = float(YOLOv5Model.YOLO_IMG_SIZE)
    
    actual_num_classes = pred.shape[-1] - 5
    if actual_num_classes != num_classes:
        num_classes = actual_num_classes
    
    total_box = total_cls = total_obj = 0.0
    for b in range(B):
        gb = targets['boxes'][b].to(device)
        gl = targets['labels'][b].to(device)
        if gb.shape[0] == 0: continue
        
        pb_px = pred[b, :, :4]
        cx, cy = pb_px[:, 0], pb_px[:, 1]
        bw, bh = pb_px[:, 2], pb_px[:, 3]
        pb_rel = torch.stack([
            ((cx - bw/2)/IMG_S).clamp(0,1), ((cy - bh/2)/IMG_S).clamp(0,1),
            ((cx + bw/2)/IMG_S).clamp(0,1), ((cy + bh/2)/IMG_S).clamp(0,1),
        ], dim=-1)
        iou = self_bbox_iou(pb_rel.unsqueeze(1), gb.unsqueeze(0))
        best_iou, best_idx = iou.max(dim=-1)
        pos_mask = best_iou > 0.5
        if pos_mask.sum() == 0:
            fb = best_iou.argmax()
            pos_mask = torch.zeros(pb_rel.shape[0], dtype=torch.bool, device=device)
            pos_mask[fb] = True
        matched_labels = gl[best_idx[pos_mask]]
        matched_boxes = gb[best_idx[pos_mask]]
        matched_iou = best_iou[pos_mask]
        
        pred_boxes_pos = pb_px[pos_mask]
        matched_boxes_pixel = matched_boxes * IMG_S
        mx1, my1, mx2, my2 = matched_boxes_pixel.unbind(-1)
        matched_cxcywh = torch.stack([
            (mx1 + mx2) / 2,
            (my1 + my2) / 2,
            mx2 - mx1,
            my2 - my1
        ], dim=-1)
        box_loss = F.l1_loss(pred_boxes_pos, matched_cxcywh, reduction='sum')
        
        cls_logits = pred[b, pos_mask, 5:5+num_classes]
        cls_target = F.one_hot(matched_labels, num_classes).float()
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction='sum')
        
        obj_target = torch.zeros(pred.shape[1], device=device)
        obj_target[pos_mask] = matched_iou.detach()
        obj_loss = F.binary_cross_entropy_with_logits(pred[b, :, 4], obj_target, reduction='sum')
        
        total_box += box_loss; total_cls += cls_loss; total_obj += obj_loss
    return 0.05 * total_box / max(B, 1) + 0.5 * total_cls / max(B, 1) + 1.0 * total_obj / max(B, 1)


def compute_detr_loss(pred_logits, pred_boxes, targets, num_classes=91):
    B, N = pred_logits.shape[:2]; device = pred_logits.device
    gt_boxes = targets["boxes"]; gt_labels = targets["labels"]
    total_box = total_cls = 0.0
    for b in range(B):
        gb = gt_boxes[b].to(device); gl = gt_labels[b].to(device)
        if gb.shape[0] == 0: continue
        pb_box = pred_boxes[b]; pb_logits = pred_logits[b]
        cx, cy, w, h = pb_box.unbind(-1)
        x1 = (cx - w/2).clamp(0,1); y1 = (cy - h/2).clamp(0,1)
        x2 = (cx + w/2).clamp(0,1); y2 = (cy + h/2).clamp(0,1)
        pb_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        iou = self_bbox_iou(pb_xyxy.unsqueeze(1), gb.unsqueeze(0))
        best_iou, best_idx = iou.max(dim=-1)
        pos_mask = best_iou > 0.5
        if pos_mask.sum() == 0:
            fb = best_iou.argmax()
            pos_mask = torch.zeros(N, dtype=torch.bool, device=device); pos_mask[fb] = True
        matched_labels = gl[best_idx[pos_mask]]
        matched_boxes = gb[best_idx[pos_mask]]
        mx1, my1, mx2, my2 = matched_boxes.unbind(-1)
        matched_cxcywh = torch.stack([(mx1+mx2)/2, (my1+my2)/2, mx2-mx1, my2-my1], dim=-1)
        total_box += F.l1_loss(pb_box[pos_mask], matched_cxcywh, reduction='sum')
        total_cls += F.cross_entropy(pb_logits[pos_mask, :-1], matched_labels, reduction='sum')
    return 5.0*total_box + 2.0*total_cls / max(B, 1)


print("Loss functions defined.")

# ===========================================================================
#  SPARSITY ENFORCEMENT + RECOVERY TRAINING
# ===========================================================================
def enforce_sparsity(model):
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, 'pruning_mask'):
            mod.weight.data.mul_(mod.pruning_mask)


def recover_model(model, pruner, train_loader, device, epochs=3, lr=1e-4,
                  report_every=50, loss_fn=None, num_classes=80, model_type='yolo'):
    model.train()
    pruner.apply_all_masks()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Epoch {epoch+1}/{epochs}')
        for step, (images, targets) in pbar:
            images = images.to(device)
            if model_type == 'detr':
                targets = {
                    'boxes': [b.to(device) for b in targets['boxes']],
                    'labels': [l.to(device) for l in targets['labels']]
                }
            else:  # yolo
                targets = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in targets.items()
                }
            optimizer.zero_grad()
            pred = model.forward_train(images)
            if model_type == 'yolo':
                loss = (loss_fn or compute_yolo_loss)(pred, targets, num_classes=num_classes)
            else:
                loss = (loss_fn or compute_detr_loss)(pred[0], pred[1], targets, num_classes=num_classes)
            loss.backward()
            enforce_sparsity(model)
            optimizer.step()
            enforce_sparsity(model)
            running_loss += loss.item()
            if step % report_every == 0 and step > 0:
                pbar.set_postfix({'loss': f'{running_loss/(step+1):.4f}'})
    return running_loss / max(len(train_loader), 1)


print("Recovery training function defined.")

# ===========================================================================
#  BENCHMARKING HELPERS
# ===========================================================================
def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(model.get_raw_model().state_dict(), f.name)
        sz = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return sz


def theoretical_sparse_size_mb(model):
    total_bytes = 0
    for p in model.get_raw_model().parameters():
        total_bytes += (p.data != 0).sum().item() * p.element_size()
    return total_bytes / (1024 * 1024)


def benchmark_model(model, dataloader, device, desc='', num_batches=200):
    model.eval(); model.to(device)
    latencies, total_frames, total_time = [], 0, 0.0
    is_cuda = device.type == 'cuda'
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=desc,
                                       total=min(num_batches, len(dataloader)))):
            if i >= num_batches: break
            images = batch[0].to(device)
            if images.dim() == 3: images = images.unsqueeze(0)
            if is_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(images)
            if is_cuda: torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            total_frames += images.size(0); total_time += elapsed
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    return {'latency_ms': avg_lat * 1000, 'fps': total_frames / max(total_time, 1e-9)}


def compute_flops(model, device='cpu'):
    input_size = (1,) + model.get_input_size()
    try:
        from fvcore.nn import FlopCountAnalysis
        import logging
        logging.getLogger("fvcore").setLevel(logging.ERROR)
        m = model.get_raw_model().to(device).eval()
        dummy = torch.randn(*input_size, device=device)
        flops = FlopCountAnalysis(m, dummy)
        return flops.total() / 1e9
    except Exception:
        try:
            m = model.get_raw_model().to(device).eval()
            dummy = torch.randn(*input_size, device=device)
            hooks = []
            total_macs = 0

            def conv_hook(module, inp, out):
                nonlocal total_macs
                n, c, h, w = inp[0].shape
                out_c, out_h, out_w = out.shape[1], out.shape[2], out.shape[3]
                k = module.kernel_size[0] * module.kernel_size[1]
                groups = module.groups
                macs = out_c * out_h * out_w * (c // groups) * k
                total_macs += macs

            def linear_hook(module, inp, out):
                nonlocal total_macs
                macs = inp[0].shape[-1] * out.shape[-1]
                total_macs += macs

            def register_hooks(module):
                for child in module.children():
                    if isinstance(child, nn.Conv2d):
                        hooks.append(child.register_forward_hook(conv_hook))
                    elif isinstance(child, nn.Linear):
                        hooks.append(child.register_forward_hook(linear_hook))
                    register_hooks(child)

            register_hooks(m)
            with torch.no_grad():
                m(dummy)
            for h in hooks:
                h.remove()
            return total_macs * 2 / 1e9
        except Exception:
            return float('nan')


def evaluate_coco(model, dataloader, coco_gt, device, desc='eval'):
    model.eval(); model.to(device)
    results = []
    
    # Category ID mapping for YOLOv5 (maps 0-79 to 1-90)
    coco_cat_to_idx = {
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
    idx_to_coco_cat = {v: k for k, v in coco_cat_to_idx.items()}
    is_yolo = hasattr(model, 'YOLO_IMG_SIZE')
    
    with torch.no_grad():
        for images, img_ids, orig_ws, orig_hs in tqdm(dataloader, desc=desc):
            images = images.to(device)
            if images.dim() == 3: images = images.unsqueeze(0)
            preds_per_image = model(images)
            for batch_idx, dets in enumerate(preds_per_image):
                img_id = (img_ids[batch_idx].item()
                          if isinstance(img_ids, torch.Tensor) else img_ids[batch_idx])
                orig_w = (orig_ws[batch_idx].item()
                          if isinstance(orig_ws, torch.Tensor) else orig_ws[batch_idx])
                orig_h = (orig_hs[batch_idx].item()
                          if isinstance(orig_hs, torch.Tensor) else orig_hs[batch_idx])
                if dets.shape[0] == 0: continue
                dets = dets.cpu().numpy()
                for d in dets:
                    x1, y1, x2, y2, score, cls_id = d
                    w_abs = (x2 - x1) * orig_w; h_abs = (y2 - y1) * orig_h
                    
                    if is_yolo:
                        category_id = idx_to_coco_cat.get(int(cls_id), int(cls_id) + 1)
                    else:
                        category_id = int(cls_id) + 1
                    
                    results.append({
                        'image_id': int(img_id),
                        'category_id': category_id,
                        'bbox': [float(x1 * orig_w), float(y1 * orig_h),
                                 float(w_abs), float(h_abs)],
                        'score': float(score)
                    })
    if not results:
        print(f"  [WARN] No detections submitted for {desc}!")
        return {'mAP50': 0.0, 'mAP': 0.0, 'precision': 0.0, 'recall': 0.0}
    else:
        print(f"  [DEBUG] {desc}: {len(results)} detections submitted")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return {
        'mAP50': float(coco_eval.stats[1]),
        'mAP': float(coco_eval.stats[0]),
        'precision': float(coco_eval.stats[4]),
        'recall': float(coco_eval.stats[5])
    }


# ===========================================================================
#  ENHANCEMENT: Theoretical sparse FLOPs estimation
#  Shows the gap between real FLOPs (unchanged) and what's possible
#  with perfect sparse hardware.
# ===========================================================================
def compute_sparse_flops(model, device='cpu'):
    base_flops = compute_flops(model, device)
    sparsity = calculate_model_sparsity(model.get_raw_model())
    return base_flops * (1 - sparsity)


# ===========================================================================
#  ENHANCEMENT: Structured L1-Norm Pruner — actually REMOVES channels
#  Unlike MagnitudePruner (zeroes weights), this prunes entire filters
#  so the architecture changes and FLOPs/latency actually decrease.
# ===========================================================================
class StructuredL1Pruner:
    def __init__(self, model, pruning_ratio=0.0):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.pruned = {}

    def prune_conv(self, name, module):
        w = module.weight.data
        l1_norm = w.abs().sum(dim=[1, 2, 3])
        k = int(w.size(0) * self.pruning_ratio)
        if k == 0:
            return
        _, indices = torch.topk(l1_norm, k, largest=False)
        keep_mask = torch.ones(w.size(0), dtype=torch.bool)
        keep_mask[indices] = False
        module.weight.data = module.weight.data[keep_mask]
        if module.bias is not None:
            module.bias.data = module.bias.data[keep_mask]
        self.pruned[name] = (~keep_mask).sum().item()

    def prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.prune_conv(name, module)


print("Benchmarking helpers defined.")

# ===========================================================================
#  MAIN EXECUTION BLOCK
# ===========================================================================
if __name__ == "__main__":
    # ===========================================================================
    #  COCO VAL2017 DOWNLOAD
    # ===========================================================================
    DATA_DIR = "/kaggle/working/coco"
    os.makedirs(DATA_DIR, exist_ok=True)

    VAL_ROOT = os.path.join(DATA_DIR, "val2017")
    VAL_ANN  = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")

    if not os.path.isdir(VAL_ROOT):
        print("Downloading COCO val2017 images...")
        download_url("http://images.cocodataset.org/zips/val2017.zip",
                     DATA_DIR, "val2017.zip")
        with zipfile.ZipFile(os.path.join(DATA_DIR, "val2017.zip"), "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(os.path.join(DATA_DIR, "val2017.zip"))

    if not os.path.isfile(VAL_ANN):
        print("Downloading COCO annotations...")
        download_url(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            DATA_DIR, "annotations.zip")
        with zipfile.ZipFile(os.path.join(DATA_DIR, "annotations.zip"), "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(os.path.join(DATA_DIR, "annotations.zip"))

    print(f"COCO val2017 ready: {VAL_ROOT} | {VAL_ANN}")

    # ===========================================================================
    #  LOAD MODELS
    # ===========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading DETR-ResNet50...")
    detr = DETRModel().to(device).eval()
    print(f"  DETR params: {detr.get_params_count():,}")

    print("Loading YOLOv5s...")
    yolo = YOLOv5Model().to(device).eval()
    print(f"  YOLOv5s params: {yolo.get_params_count():,}")

    print(f"\nDETR   initial sparsity: {calculate_model_sparsity(detr):.4f}")
    print(f"YOLOv5 initial sparsity: {calculate_model_sparsity(yolo):.4f}")

    # ===========================================================================
    #  VALIDATION DATALOADERS — separate per model (FIXED: no double-resize)
    # ===========================================================================
    VAL_MAX_SAMPLES = 500

    val_ds_640 = CocoValDataset(VAL_ROOT, VAL_ANN, img_size=640, max_samples=VAL_MAX_SAMPLES, model_type='yolo')
    val_ds_800 = CocoValDataset(VAL_ROOT, VAL_ANN, img_size=800, max_samples=VAL_MAX_SAMPLES, model_type='detr')

    val_loader_640 = DataLoader(val_ds_640, batch_size=4, shuffle=False, num_workers=2)
    val_loader_800 = DataLoader(val_ds_800, batch_size=2, shuffle=False, num_workers=2)

    coco_gt = COCO(VAL_ANN)
    print(f"Val samples: {len(val_ds_640)}")
    print(f"YOLO loader: batch=4, img=640x640")
    print(f"DETR loader: batch=2, img=800x800")

    # ===========================================================================
    #  COLLATE FUNCTIONS
    # ===========================================================================
    def collate_detr(batch):
        """Collate function cho DETR training - giữ boxes và labels dạng lists"""
        imgs = torch.stack([b[0] for b in batch])
        boxes = [b[1]['boxes'] for b in batch]
        labels = [b[1]['labels'] for b in batch]
        return imgs, {'boxes': boxes, 'labels': labels}


    def collate_yolo(batch):
        """Collate function cho YOLO training - giữ boxes và labels dạng lists"""
        imgs = torch.stack([b[0] for b in batch])
        boxes = [b[1]['boxes'] for b in batch]
        labels = [b[1]['labels'] for b in batch]
        return imgs, {'boxes': boxes, 'labels': labels}


    # ===========================================================================
    #  MAIN EXPERIMENT LOOP
    # ===========================================================================
    PRUNING_RATIOS = [0.0, 0.3, 0.5, 0.7, 0.9]
    REPORT = []

    RECOVERY_OFFSET = VAL_MAX_SAMPLES
    RECOVERY_SAMPLES = 200

    MODEL_CONFIGS = [
        ("YOLOv5", yolo, val_loader_640),
        ("DETR",   detr, val_loader_800),
    ]

    for model_name, model_obj, val_loader in MODEL_CONFIGS:
        print(f"\n{'='*65}")
        print(f"MODEL: {model_name}  |  input: {model_obj.get_input_size()}")
        print(f"{'='*65}")

        baseline_params = model_obj.get_params_count()

        for ratio in PRUNING_RATIOS:
            print(f"\n--- ratio={ratio} ---")

            if ratio == 0.0:
                row = {"model": model_name, "pruning_ratio": ratio, "stage": "baseline"}
                sp = calculate_model_sparsity(model_obj.get_raw_model())
                row["params"] = baseline_params
                row["active_params"] = baseline_params
                row["sparsity"] = sp
                row["compression_ratio"] = 1.0
                row["model_size_mb"] = model_size_mb(model_obj)
                row["sparse_size_mb"] = theoretical_sparse_size_mb(model_obj)
                row["flops_g"] = compute_flops(model_obj, device=str(device))

                bench = benchmark_model(model_obj, val_loader, device,
                                        desc=f"{model_name} baseline bench", num_batches=200)
                row.update(bench)
                metrics = evaluate_coco(model_obj, val_loader, coco_gt, device,
                                         desc=f"{model_name} baseline eval")
                row.update(metrics)
                REPORT.append(row)
                print(f"  Baseline: mAP={metrics['mAP']:.3f}, mAP50={metrics['mAP50']:.3f}, "
                      f"FPS={row['fps']:.1f}, Sparsity={sp*100:.1f}%")

                # ENHANCEMENT: Validate baseline before pruning
                if metrics['mAP'] < 0.01:
                    print(f"  [WARN] Baseline mAP={metrics['mAP']:.4f} is near zero!")
                    print(f"  [WARN] Check: input normalization, forward pass, NMS settings.")
                    print(f"  [WARN] DETR needs ImageNet Normalize(mean, std).")
                    print(f"  [WARN] YOLO needs eval mode (no layer.train() on BatchNorm).")
            else:
                cloned = copy.deepcopy(model_obj)
                raw_cloned = cloned.get_raw_model()
                pruner = MagnitudePruner(raw_cloned, pruning_ratio=ratio)
                pruner.prune()
                actual_sparsity = pruner.calculate_sparsity()

                row = {"model": model_name, "pruning_ratio": ratio, "stage": "after_pruning"}
                row["params"] = cloned.get_params_count()
                row["active_params"] = int(row["params"] * (1 - actual_sparsity))
                row["sparsity"] = actual_sparsity
                row["compression_ratio"] = 1.0 / (1.0 - actual_sparsity + 1e-9)
                row["model_size_mb"] = model_size_mb(cloned)
                row["sparse_size_mb"] = theoretical_sparse_size_mb(cloned)
                row["flops_g"] = compute_flops(cloned, device=str(device))
                # ENHANCEMENT: Show theoretical sparse FLOPs vs real FLOPs
                # Demonstrates the gap: FLOPs stay the same even at 90% sparsity
                row["sparse_flops_g"] = compute_sparse_flops(cloned, device=str(device))

                bench = benchmark_model(cloned, val_loader, device,
                                        desc=f"{model_name} pruned {ratio}", num_batches=200)
                row.update(bench)
                metrics = evaluate_coco(cloned, val_loader, coco_gt, device,
                                         desc=f"{model_name} pruned {ratio} eval")
                row.update(metrics)
                REPORT.append(row)
                print(f"  Pruned ({ratio:.0%}): mAP={metrics['mAP']:.3f}, mAP50={metrics['mAP50']:.3f}, "
                      f"FPS={row['fps']:.1f}, Sparsity={actual_sparsity*100:.1f}%"
                      f" | FLOPs={row['flops_g']:.1f}G (real), sparse={row['sparse_flops_g']:.1f}G (theoretical)")

                # Recovery fine-tuning
                print(f"  Recovery training on {RECOVERY_SAMPLES} samples...")
                if model_name == "YOLOv5":
                    batch_size = 4
                    collate_fn = collate_yolo
                    img_size = YOLOv5Model.YOLO_IMG_SIZE
                else:
                    batch_size = 2
                    collate_fn = collate_detr
                    img_size = DETRModel.DETR_IMG_SIZE

                train_loader = DataLoader(
                    CocoTrainDataset(
                        VAL_ROOT, VAL_ANN,
                        img_size=img_size,
                        offset=RECOVERY_OFFSET, max_samples=RECOVERY_SAMPLES,
                        model_type='yolo' if model_name == "YOLOv5" else 'detr'
                    ), batch_size=batch_size, shuffle=True, num_workers=2,
                    collate_fn=collate_fn
                )

                if model_name == "YOLOv5":
                    _ = recover_model(cloned, pruner, train_loader, device,
                                      epochs=3, lr=1e-4, num_classes=cloned.num_classes,
                                      model_type='yolo')
                else:
                    _ = recover_model(cloned, pruner, train_loader, device,
                                      epochs=3, lr=1e-4, num_classes=cloned.num_classes,
                                      model_type='detr',
                                      loss_fn=lambda p, t, nc: compute_detr_loss(*p, t, num_classes=nc))

                # after_recovery
                row = {"model": model_name, "pruning_ratio": ratio, "stage": "after_recovery"}
                row["params"] = cloned.get_params_count()
                row["active_params"] = int(row["params"] * (1 - actual_sparsity))
                row["sparsity"] = actual_sparsity
                row["compression_ratio"] = 1.0 / (1.0 - actual_sparsity + 1e-9)
                row["model_size_mb"] = model_size_mb(cloned)
                row["sparse_size_mb"] = theoretical_sparse_size_mb(cloned)
                row["flops_g"] = compute_flops(cloned, device=str(device))
                row["sparse_flops_g"] = compute_sparse_flops(cloned, device=str(device))

                bench = benchmark_model(cloned, val_loader, device,
                                        desc=f"{model_name} recovered {ratio}", num_batches=200)
                row.update(bench)
                metrics = evaluate_coco(cloned, val_loader, coco_gt, device,
                                         desc=f"{model_name} recovered {ratio} eval")
                row.update(metrics)
                REPORT.append(row)
                print(f"  Recovered ({ratio:.0%}): mAP={metrics['mAP']:.3f}, "
                      f"mAP50={metrics['mAP50']:.3f}, FPS={row['fps']:.1f}"
                      f" | FLOPs={row['flops_g']:.1f}G (still unchanged!)")


    print(f"\n{'='*65}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*65}")

    # ===========================================================================
    #  RESULTS TABLE
    # ===========================================================================
    df = pd.DataFrame(REPORT)
    if not df.empty:
        stage_order = {"baseline": 0, "after_pruning": 1, "after_recovery": 2}
        df["stage_order"] = df["stage"].map(stage_order)
        df = df.sort_values(["model", "stage_order", "pruning_ratio"]).drop(columns="stage_order")

        display_cols = ["model", "stage", "pruning_ratio", "params", "active_params",
                        "sparsity", "compression_ratio", "model_size_mb", "sparse_size_mb",
                        "flops_g", "sparse_flops_g", "latency_ms", "fps", "mAP50", "mAP"]
        display_cols = [c for c in display_cols if c in df.columns]
        print("\n" + df[display_cols].to_string(index=False))

        csv_path = "/kaggle/working/magnitude_pruning_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nReport saved to {csv_path}")

        # ===========================================================================
        #  BENCHMARK SUMMARY: Before vs After Pruning (per ratio)
        # ===========================================================================
        print("\n\n========== BENCHMARK: BEFORE vs AFTER PRUNING ==========\n")
        for model_name in ["DETR", "YOLOv5"]:
            sub = df[df["model"] == model_name]
            baseline = sub[sub["stage"] == "baseline"]
            if baseline.empty: continue
            print(f"\n--- {model_name} ---")
            hdr = f"{'Ratio':<8} {'Stage':<18} {'Params':<12} {'Active':<12} {'Sparsity':<10} "
            hdr += f"{'Compress':<10} {'SizeMB':<8} {'SparseMB':<10} {'FLOPs(G)':<10} "
            hdr += f"{'SparseF(G)':<10} {'Lat(ms)':<10} {'FPS':<8} {'mAP50':<8} {'mAP':<8}"
            print(hdr)
            print("-" * 145)

            b = baseline.iloc[0]
            print(f"{'0.0':<8} {'baseline':<18} {b['params']:<12} {b['active_params']:<12} "
                  f"{b['sparsity']*100:<9.1f}% {b['compression_ratio']:<10.2f} "
                  f"{b['model_size_mb']:<8.1f} {b['sparse_size_mb']:<10.2f} "
                  f"{b['flops_g']:<10.2f} {b.get('sparse_flops_g', b['flops_g']):<10.2f} "
                  f"{b['latency_ms']:<10.2f} {b['fps']:<8.1f} "
                  f"{b['mAP50']:<8.3f} {b['mAP']:<8.3f}")

            for ratio in PRUNING_RATIOS[1:]:
                pruned = sub[(sub["stage"] == "after_pruning") & (sub["pruning_ratio"] == ratio)]
                if pruned.empty: continue
                p = pruned.iloc[0]
                sf = p.get('sparse_flops_g', p['flops_g'])
                print(f"{ratio:<8} {'after_pruning':<18} {p['params']:<12} {p['active_params']:<12} "
                      f"{p['sparsity']*100:<9.1f}% {p['compression_ratio']:<10.2f} "
                      f"{p['model_size_mb']:<8.1f} {p['sparse_size_mb']:<10.2f} "
                      f"{p['flops_g']:<10.2f} {sf:<10.2f} "
                      f"{p['latency_ms']:<10.2f} {p['fps']:<8.1f} "
                      f"{p['mAP50']:<8.3f} {p['mAP']:<8.3f}")

                recovered = sub[(sub["stage"] == "after_recovery") & (sub["pruning_ratio"] == ratio)]
                if not recovered.empty:
                    r = recovered.iloc[0]
                    sf = r.get('sparse_flops_g', r['flops_g'])
                    print(f"{ratio:<8} {'after_recovery':<18} {r['params']:<12} {r['active_params']:<12} "
                          f"{r['sparsity']*100:<9.1f}% {r['compression_ratio']:<10.2f} "
                          f"{r['model_size_mb']:<8.1f} {r['sparse_size_mb']:<10.2f} "
                          f"{r['flops_g']:<10.2f} {sf:<10.2f} "
                          f"{r['latency_ms']:<10.2f} {r['fps']:<8.1f} "
                          f"{r['mAP50']:<8.3f} {r['mAP']:<8.3f}")
                print()

        # ===========================================================================
        #  VISUALISATION — 6 subplots: mAP, FPS, FLOPs gap, Latency, Sparse Size, Sparse FLOPs
        # ===========================================================================
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Magnitude Pruning: DETR-ResNet50 vs YOLOv5s | COCO val2017", fontsize=13)

        COLORS  = {"DETR": "#2196F3", "YOLOv5": "#FF5722"}
        METRICS = [
            ("mAP",             "mAP (IoU 0.50:0.95)",  axes[0, 0]),
            ("fps",             "FPS",                   axes[0, 1]),
            ("flops_g",         "FLOPs (G) — unchanged", axes[0, 2]),
            ("latency_ms",      "Latency (ms/batch)",    axes[1, 0]),
            ("sparse_size_mb",  "Sparse size (MB)",      axes[1, 1]),
            ("sparse_flops_g",  "Theoretical sparse FLOPs", axes[1, 2]),
        ]

        for metric, title, ax in METRICS:
            if metric not in df.columns:
                ax.set_visible(False); continue
            for mname in ["DETR", "YOLOv5"]:
                sub_b = df[(df["model"] == mname) & (df["stage"] == "baseline")]
                sub_p = df[(df["model"] == mname) & (df["stage"] == "after_pruning")]
                sub_r = df[(df["model"] == mname) & (df["stage"] == "after_recovery")]

                if sub_b.empty: continue

                xs_p = [0.0] + list(sub_p["pruning_ratio"])
                ys_p = list(sub_b[metric]) + list(sub_p[metric])
                ax.plot(xs_p, ys_p, marker='o', label=f"{mname} pruned", color=COLORS[mname])

                if not sub_r.empty:
                    xs_r = [0.0] + list(sub_r["pruning_ratio"])
                    ys_r = list(sub_b[metric]) + list(sub_r[metric])
                    ax.plot(xs_r, ys_r, marker='s', linestyle='--',
                            label=f"{mname} recovered", color=COLORS[mname], alpha=0.6)

            # ENHANCEMENT: Add annotation explaining the FLOPs gap
            if metric == 'flops_g':
                ax.annotate('Unstructured pruning: FLOPs do NOT change\nbecause the compute graph is unchanged',
                            xy=(0.5, 0.5), xycoords='axes fraction', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_title(title); ax.set_xlabel("Pruning ratio"); ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = "/kaggle/working/magnitude_pruning_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to {plot_path}")

        print("\nAll done. Output files:")
        print(f"  /kaggle/working/magnitude_pruning_report.csv")
        print(f"  /kaggle/working/magnitude_pruning_plots.png")
    else:
        print("No data available to display or save.")
