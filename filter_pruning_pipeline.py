import os, sys, copy, time, json, tempfile, warnings, zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence

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

import torch_pruning as tp

warnings.filterwarnings('ignore')
print("All imports OK, including torch_pruning.")

# ===========================================================================
#  UNIFIED BASE MODEL INTERFACE
# ===========================================================================
class BaseModel(nn.Module):
    def forward(self, images): raise NotImplementedError
    def get_raw_model(self): raise NotImplementedError
    def get_params_count(self):
        return sum(p.numel() for p in self.get_raw_model().parameters())
    def get_input_size(self): raise NotImplementedError


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


# ===========================================================================
#  YOLOv5s MODEL WRAPPER
# ===========================================================================
class YOLOv5Model(BaseModel):
    YOLO_IMG_SIZE = 640

    def __init__(self):
        super().__init__()
        from ultralytics import YOLO
        _hub = YOLO('yolov5s.pt')
        self.raw_model = _hub.model
        self.conf = 0.001
        self.iou = 0.65
        self.num_classes = 80

    def train(self, mode: bool = True):
        self.training = mode
        self.raw_model.train(mode)
        return self

    def forward(self, images):
        self.raw_model.eval()
        with torch.no_grad():
            B, _, H, W = images.shape
            x = images * 255.0
            if W != self.YOLO_IMG_SIZE or H != self.YOLO_IMG_SIZE:
                x = F.interpolate(x, size=(self.YOLO_IMG_SIZE, self.YOLO_IMG_SIZE),
                                  mode='bilinear', align_corners=False)
            
            raw_out = self.raw_model(x)
            
            if isinstance(raw_out, dict):
                raw_out = raw_out.get('output', raw_out.get('pred', list(raw_out.values())[0]))
            elif isinstance(raw_out, (list, tuple)):
                raw_out = raw_out[0]

            if raw_out.dim() == 3 and raw_out.shape[1] < raw_out.shape[2]:
                raw_out = raw_out.transpose(1, 2)

            return self._custom_torchvision_nms(raw_out)

    def forward_train(self, images):
        with torch.enable_grad():
            B, _, H, W = images.shape
            x = images * 255.0
            if W != self.YOLO_IMG_SIZE or H != self.YOLO_IMG_SIZE:
                x = F.interpolate(x, size=(self.YOLO_IMG_SIZE, self.YOLO_IMG_SIZE),
                                  mode='bilinear', align_corners=False)
            
            if not x.requires_grad:
                x.requires_grad_(True)
            
            self.raw_model.train()
            raw_out = self.raw_model(x)
            
            if isinstance(raw_out, dict):
                raw_out = raw_out.get('output', raw_out.get('pred', list(raw_out.values())[0]))
            elif isinstance(raw_out, (list, tuple)):
                raw_out = raw_out[0]
            
            if isinstance(raw_out, torch.Tensor):
                if raw_out.requires_grad is False:
                    raw_out.requires_grad_(True)
                
                if raw_out.dim() == 3 and raw_out.shape[1] < raw_out.shape[2]:
                    raw_out = raw_out.transpose(1, 2)
                    
                return raw_out
            else:
                raise TypeError(f"Expected torch.Tensor, got {type(raw_out)}")

    def _custom_torchvision_nms(self, raw_out):
        from torchvision.ops import nms
        B = raw_out.shape[0]
        results = []
        
        for b in range(B):
            pred = raw_out[b]
            bboxes = pred[:, :4]
            obj_conf = pred[:, 4]
            class_probs = pred[:, 5:5 + self.num_classes]
            max_probs, class_ids = class_probs.max(dim=-1)
            scores = max_probs * obj_conf
            
            conf_mask = scores > self.conf
            if not conf_mask.any():
                results.append(torch.zeros((0, 6), device=raw_out.device))
                continue
                
            f_boxes = bboxes[conf_mask]
            f_scores = scores[conf_mask]
            f_class_ids = class_ids[conf_mask]
            
            cx, cy, w, h = f_boxes.unbind(-1)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            keep = nms(xyxy_boxes, f_scores, self.iou)
            
            if len(keep) == 0:
                results.append(torch.zeros((0, 6), device=raw_out.device))
            else:
                final_boxes = xyxy_boxes[keep] / self.YOLO_IMG_SIZE
                final_scores = f_scores[keep].unsqueeze(-1)
                final_classes = f_class_ids[keep].unsqueeze(-1).float()
                
                image_res = torch.cat([final_boxes, final_scores, final_classes], dim=-1)
                results.append(image_res)
                
        return results

    def get_raw_model(self): return self.raw_model
    def get_input_size(self): return (3, self.YOLO_IMG_SIZE, self.YOLO_IMG_SIZE)


# ===========================================================================
#  CUSTOM PRUNER FOR FROZEN BATCH NORM
# ===========================================================================
class FrozenBatchNormPruner(tp.pruner.BasePruningFunc):
    """Custom pruner handler for DETR's frozen batch norm layers."""
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep = sorted(set(range(layer.weight.shape[0])) - set(idxs))
        for attr in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(layer, attr, None)
            if t is not None:
                new = t.data[keep].clone()
                if isinstance(t, nn.Parameter):
                    new = nn.Parameter(new)
                setattr(layer, attr, new)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.weight.shape[0]

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.weight.shape[0]


# ===========================================================================
#  HARD STRUCTURED FILTER PRUNER
# ===========================================================================
class StructuredFilterPruner:
    """Prunes filters physically using Torch-Pruning."""
    def __init__(self, model_wrapper: BaseModel, pruning_ratio: float = 0.0):
        self.model_wrapper = model_wrapper
        self.pruning_ratio = pruning_ratio

    def prune(self):
        if self.pruning_ratio <= 0.0:
            return self.model_wrapper

        inner, model_type = self._get_inner_and_type()
        example_inputs = self._make_example_inputs(model_type)
        ignored_layers = self._collect_ignored(inner, model_type)
        custom_pruners = self._find_custom_pruners(inner)

        was_training = inner.training
        inner.train()  # require train mode for tracing

        try:
            imp = tp.importance.MagnitudeImportance(p=1)  # L1-norm pruner
            pruner = tp.pruner.MagnitudePruner(
                inner,
                example_inputs=example_inputs,
                importance=imp,
                pruning_ratio=self.pruning_ratio,
                ignored_layers=ignored_layers,
                customized_pruners=custom_pruners,
            )
            pruner.step()
        except Exception as e:
            print(f"  [ERROR] Torch-Pruning failed: {e}")
            raise e
        finally:
            inner.train(was_training)

        return self.model_wrapper

    def _get_inner_and_type(self) -> Tuple[nn.Module, str]:
        if isinstance(self.model_wrapper, YOLOv5Model):
            return self.model_wrapper.raw_model, "yolo"
        elif isinstance(self.model_wrapper, DETRModel):
            return self.model_wrapper.model, "detr"
        return self.model_wrapper, "raw"

    def _make_example_inputs(self, model_type: str) -> torch.Tensor:
        device = next(self.model_wrapper.parameters()).device
        size = 800 if model_type == "detr" else 640
        return torch.randn(1, 3, size, size, device=device)

    def _collect_ignored(self, inner: nn.Module, model_type: str) -> List[nn.Module]:
        ignored = []
        if model_type == "yolo":
            for m in inner.modules():
                if m.__class__.__name__ == "Detect":
                    ignored.append(m)
                    for child in m.modules():
                        if isinstance(child, nn.Conv2d):
                            ignored.append(child)
        elif model_type == "detr":
            for name, m in inner.named_modules():
                if not name.startswith("backbone") and not list(m.children()):
                    ignored.append(m)
        return ignored

    def _find_custom_pruners(self, inner: nn.Module) -> Dict:
        customs = {}
        for m in inner.modules():
            if m.__class__.__name__ == "FrozenBatchNorm2d":
                customs[m.__class__] = FrozenBatchNormPruner()
        return customs


# ===========================================================================
#  COCO VALIDATION DATASET
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
        if self.model_type == 'detr':
            tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, img_id, orig_w, orig_h


# ===========================================================================
#  COCO TRAINING DATASET
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
        pred = pred.get('output', pred.get('pred', list(pred.values())[0]))
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    
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


# ===========================================================================
#  RECOVERY TRAINING (No masking required since filters are deleted)
# ===========================================================================
def recover_model(model, train_loader, device, epochs=3, lr=1e-4,
                  report_every=50, loss_fn=None, num_classes=80, model_type='yolo'):
    model.train()
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
            optimizer.step()
            running_loss += loss.item()
            if step % report_every == 0 and step > 0:
                pbar.set_postfix({'loss': f'{running_loss/(step+1):.4f}'})
    return running_loss / max(len(train_loader), 1)


# ===========================================================================
#  BENCHMARKING HELPERS
# ===========================================================================
def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(model.get_raw_model().state_dict(), f.name)
        sz = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return sz


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
        from thop import profile
        m = model.get_raw_model().to(device).eval()
        x = torch.zeros(*input_size, device=device)
        macs, _ = profile(m, inputs=(x,), verbose=False)
        return float(macs * 2 / 1e9)
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            import logging
            logging.getLogger("fvcore").setLevel(logging.ERROR)
            m = model.get_raw_model().to(device).eval()
            dummy = torch.randn(*input_size, device=device)
            flops = FlopCountAnalysis(m, dummy)
            return flops.total() / 1e9
        except Exception:
            return float('nan')


def evaluate_coco(model, dataloader, coco_gt, device, desc='eval'):
    model.eval(); model.to(device)
    results = []
    
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
                        category_id = int(cls_id)
                    
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
#  MAIN EXECUTION BLOCK
# ===========================================================================
if __name__ == "__main__":
    # ===========================================================================
    #  COCO VAL2017 DOWNLOAD
    # ===========================================================================
    DATA_DIR = "/kaggle/working/coco" if os.path.exists("/kaggle") else "./data/coco"
    os.makedirs(DATA_DIR, exist_ok=True)

    VAL_ROOT = os.path.join(DATA_DIR, "val2017")
    VAL_ANN  = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")

    # Download if not present
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

    # ===========================================================================
    #  VALIDATION DATALOADERS
    # ===========================================================================
    VAL_MAX_SAMPLES = 100  # set small for quick testing, increase to 500 for full run

    val_ds_640 = CocoValDataset(VAL_ROOT, VAL_ANN, img_size=640, max_samples=VAL_MAX_SAMPLES, model_type='yolo')
    val_ds_800 = CocoValDataset(VAL_ROOT, VAL_ANN, img_size=800, max_samples=VAL_MAX_SAMPLES, model_type='detr')

    val_loader_640 = DataLoader(val_ds_640, batch_size=4, shuffle=False, num_workers=2)
    val_loader_800 = DataLoader(val_ds_800, batch_size=2, shuffle=False, num_workers=2)

    coco_gt = COCO(VAL_ANN)
    print(f"Val samples: {len(val_ds_640)}")
    print(f"YOLO loader: batch=4, img=640x640")
    print(f"DETR loader: batch=2, img=800x800")

    # Collate functions
    def collate_detr(batch):
        imgs = torch.stack([b[0] for b in batch])
        boxes = [b[1]['boxes'] for b in batch]
        labels = [b[1]['labels'] for b in batch]
        return imgs, {'boxes': boxes, 'labels': labels}

    def collate_yolo(batch):
        imgs = torch.stack([b[0] for b in batch])
        boxes = [b[1]['boxes'] for b in batch]
        labels = [b[1]['labels'] for b in batch]
        return imgs, {'boxes': boxes, 'labels': labels}

    # ===========================================================================
    #  MAIN EXPERIMENT LOOP
    # ===========================================================================
    PRUNING_RATIOS = [0.0, 0.3, 0.5, 0.7]
    REPORT = []

    RECOVERY_OFFSET = VAL_MAX_SAMPLES
    RECOVERY_SAMPLES = 100  # small for fast pipeline, increase to 500 for complete metrics

    MODEL_CONFIGS = [
        ("YOLOv5", yolo, val_loader_640),
        ("DETR",   detr, val_loader_800),
    ]

    for model_name, model_obj, val_loader in MODEL_CONFIGS:
        print(f"\n{'='*65}")
        print(f"MODEL: {model_name}  |  input: {model_obj.get_input_size()}")
        print(f"{'='*65}")

        baseline_params = model_obj.get_params_count()
        baseline_flops = compute_flops(model_obj, device=str(device))
        baseline_size = model_size_mb(model_obj)

        for ratio in PRUNING_RATIOS:
            print(f"\n--- ratio={ratio} ---")

            if ratio == 0.0:
                row = {"model": model_name, "pruning_ratio": ratio, "stage": "baseline"}
                row["params"] = baseline_params
                row["active_params"] = baseline_params
                row["sparsity"] = 0.0
                row["compression_ratio"] = 1.0
                row["model_size_mb"] = baseline_size
                row["flops_g"] = baseline_flops

                bench = benchmark_model(model_obj, val_loader, device,
                                        desc=f"{model_name} baseline bench", num_batches=200)
                row.update(bench)
                metrics = evaluate_coco(model_obj, val_loader, coco_gt, device,
                                         desc=f"{model_name} baseline eval")
                row.update(metrics)
                REPORT.append(row)
                print(f"  Baseline: mAP={metrics['mAP']:.3f}, mAP50={metrics['mAP50']:.3f}, "
                      f"FPS={row['fps']:.1f}, FLOPs={row['flops_g']:.2f}G, Size={row['model_size_mb']:.1f}MB")
            else:
                # Structured pruning modifies the model architecture. We create a copy of the original model.
                cloned = copy.deepcopy(model_obj)
                
                # Apply filter pruning
                print(f"  Performing structured filter pruning @ {ratio*100:.0%}...")
                pruner = StructuredFilterPruner(cloned, pruning_ratio=ratio)
                cloned = pruner.prune()
                
                pruned_params = cloned.get_params_count()
                pruned_flops = compute_flops(cloned, device=str(device))
                pruned_size = model_size_mb(cloned)
                actual_sparsity = 1.0 - (pruned_params / baseline_params)

                row = {"model": model_name, "pruning_ratio": ratio, "stage": "after_pruning"}
                row["params"] = baseline_params
                row["active_params"] = pruned_params
                row["sparsity"] = actual_sparsity
                row["compression_ratio"] = baseline_params / max(pruned_params, 1)
                row["model_size_mb"] = pruned_size
                row["flops_g"] = pruned_flops

                bench = benchmark_model(cloned, val_loader, device,
                                        desc=f"{model_name} pruned {ratio}", num_batches=200)
                row.update(bench)
                metrics = evaluate_coco(cloned, val_loader, coco_gt, device,
                                         desc=f"{model_name} pruned {ratio} eval")
                row.update(metrics)
                REPORT.append(row)
                print(f"  Pruned ({ratio:.0%}): mAP={metrics['mAP']:.3f}, mAP50={metrics['mAP50']:.3f}, "
                      f"FPS={row['fps']:.1f}, Latency={row['latency_ms']:.1f}ms, "
                      f"FLOPs={row['flops_g']:.2f}G (reduced from {baseline_flops:.2f}G!), Size={row['model_size_mb']:.1f}MB")

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
                    _ = recover_model(cloned, train_loader, device,
                                      epochs=3, lr=1e-4, num_classes=cloned.num_classes,
                                      model_type='yolo')
                else:
                    _ = recover_model(cloned, train_loader, device,
                                      epochs=3, lr=1e-4, num_classes=cloned.num_classes,
                                      model_type='detr',
                                      loss_fn=lambda p, t, nc: compute_detr_loss(*p, t, num_classes=nc))

                # evaluation after_recovery
                row = {"model": model_name, "pruning_ratio": ratio, "stage": "after_recovery"}
                row["params"] = baseline_params
                row["active_params"] = pruned_params
                row["sparsity"] = actual_sparsity
                row["compression_ratio"] = baseline_params / max(pruned_params, 1)
                row["model_size_mb"] = pruned_size
                row["flops_g"] = pruned_flops

                bench = benchmark_model(cloned, val_loader, device,
                                        desc=f"{model_name} recovered {ratio}", num_batches=200)
                row.update(bench)
                metrics = evaluate_coco(cloned, val_loader, coco_gt, device,
                                         desc=f"{model_name} recovered {ratio} eval")
                row.update(metrics)
                REPORT.append(row)
                print(f"  Recovered ({ratio:.0%}): mAP={metrics['mAP']:.3f}, "
                      f"mAP50={metrics['mAP50']:.3f}, FPS={row['fps']:.1f}, Latency={row['latency_ms']:.1f}ms")

    print(f"\n{'='*65}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*65}")

    # ===========================================================================
    #  RESULTS TABLE & EXPORT
    # ===========================================================================
    df = pd.DataFrame(REPORT)
    if not df.empty:
        stage_order = {"baseline": 0, "after_pruning": 1, "after_recovery": 2}
        df["stage_order"] = df["stage"].map(stage_order)
        df = df.sort_values(["model", "stage_order", "pruning_ratio"]).drop(columns="stage_order")

        display_cols = ["model", "stage", "pruning_ratio", "params", "active_params",
                        "sparsity", "compression_ratio", "model_size_mb",
                        "flops_g", "latency_ms", "fps", "mAP50", "mAP"]
        display_cols = [c for c in display_cols if c in df.columns]
        print("\n" + df[display_cols].to_string(index=False))

        csv_path = "./filter_pruning_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nReport saved to {csv_path}")

        # ===========================================================================
        #  VISUALISATION — 4 subplots: mAP, FPS, FLOPs, Latency
        # ===========================================================================
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Structured Filter Pruning: DETR-ResNet50 vs YOLOv5s | COCO val2017", fontsize=14)

        COLORS  = {"DETR": "#2196F3", "YOLOv5": "#FF5722"}
        METRICS = [
            ("mAP",             "mAP (IoU 0.50:0.95)",  axes[0, 0]),
            ("fps",             "FPS",                   axes[0, 1]),
            ("flops_g",         "Actual FLOPs (G)",      axes[1, 0]),
            ("latency_ms",      "Latency (ms/batch)",    axes[1, 1]),
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

            ax.set_title(title)
            ax.set_xlabel("Pruning ratio")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = "./filter_pruning_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to {plot_path}")

        print("\nAll done. Output files:")
        print(f"  ./filter_pruning_report.csv")
        print(f"  ./filter_pruning_plots.png")
    else:
        print("No data available to display or save.")
