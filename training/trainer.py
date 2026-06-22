import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
from models import NUM_UA_DETRAC_CLASSES
from .base import BaseTrainer

logger = logging.getLogger("TrafficTrainer")


def bbox_iou_pytorch(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Computes IoU between box1 [N, 4] and box2 [M, 4] in cxcywh format."""
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


class TrafficTrainer(BaseTrainer):
    """Trainer specializing in target classification and bounding box matching losses for highway traffic models."""

    def compute_loss(self, pred: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Unified matching loss computation for YOLOv5 and DETR outputs."""
        B = pred.shape[0]
        S = pred.shape[1]
        num_classes = self.model.num_classes
        img_size = self.model.img_size

        total_box_loss = torch.tensor(0.0, device=self.device)
        total_cls_loss = torch.tensor(0.0, device=self.device)
        total_obj_loss = torch.tensor(0.0, device=self.device)

        for b in range(B):
            p = pred[b]
            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]

            p_boxes = p[:, :4]
            p_conf = p[:, 4]
            p_cls = p[:, 5:]

            obj_targets = torch.zeros(S, device=self.device)

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

                matched_preds_t = torch.tensor(matched_preds, dtype=torch.long, device=self.device)
                matched_gts_t = torch.tensor(matched_gts, dtype=torch.long, device=self.device)

                pos_preds_box = p_boxes[matched_preds_t]
                pos_gts_box = gt_pixel_cxcywh[matched_gts_t]
                total_box_loss += F.l1_loss(pos_preds_box, pos_gts_box, reduction="mean")

                pos_preds_cls = p_cls[matched_preds_t]
                pos_gts_labels = gt_labels[matched_gts_t]
                target_one_hot = torch.zeros((len(matched_gts), num_classes), device=self.device)
                target_one_hot.scatter_(1, pos_gts_labels.unsqueeze(1), 1.0)
                total_cls_loss += F.binary_cross_entropy(pos_preds_cls, target_one_hot, reduction="mean")

            total_obj_loss += F.binary_cross_entropy(p_conf, obj_targets, reduction="mean")

        box_w, cls_w, obj_w = 0.05, 0.5, 1.0
        loss_dict = {
            "box_loss": (total_box_loss / B) * box_w,
            "cls_loss": (total_cls_loss / B) * cls_w,
            "obj_loss": (total_obj_loss / B) * obj_w,
        }
        loss_dict["total_loss"] = loss_dict["box_loss"] + loss_dict["cls_loss"] + loss_dict["obj_loss"]
        return loss_dict


# Backwards compatibility import alias
Trainer = TrafficTrainer
