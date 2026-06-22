import time
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

from models import BaseTrafficDetector
from pruning.benchmark_utils import benchmark_latency_fps

logger = logging.getLogger("Validator")


def bbox_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Computes IoU matrix between box1 [N, 4] and box2 [M, 4] in numpy."""
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


class Validator:
    """Class to manage validation calculations and performance profiling."""

    def __init__(self, model: BaseTrafficDetector, device: torch.device):
        self.model = model
        self.device = device

    def gather_predictions(self, dataloader: DataLoader) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
        """Runs model inference over the loader to collect predicted and ground truth annotations."""
        self.model.eval()
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for imgs, targets in dataloader:
                batch_imgs = torch.stack(imgs).to(self.device)
                outputs = self.model(batch_imgs)
                
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
        return all_preds, all_gts

    def calculate_accuracy_metrics(
        self, all_preds: List[Dict[str, np.ndarray]], all_gts: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Computes Precision, Recall, mAP50, and mAP50-95."""
        num_classes = self.model.num_classes
        iou_thresholds = sorted(list(set([0.5] + np.linspace(0.5, 0.95, 10).tolist())))
        aps = {iou: [] for iou in iou_thresholds}
        total_tp_50 = 0
        total_fp_50 = 0
        total_gts = 0

        for cls in range(num_classes):
            cls_preds = []
            num_cls_gts = 0
            for pred, gt in zip(all_preds, all_gts):
                cls_mask = pred["labels"] == cls
                p_boxes = pred["boxes"][cls_mask]
                p_scores = pred["scores"][cls_mask]
                for box, score in zip(p_boxes, p_scores):
                    cls_preds.append({
                        "image_id": pred["image_id"],
                        "box": box,
                        "score": score,
                    })
                gt_mask = gt["labels"] == cls
                num_cls_gts += gt_mask.sum()
                
            total_gts += num_cls_gts
            if num_cls_gts == 0:
                continue

            cls_preds = sorted(cls_preds, key=lambda x: x["score"], reverse=True)
            gt_tracked = {}
            for gt in all_gts:
                img_id = gt["image_id"]
                cls_mask = gt["labels"] == cls
                gt_tracked[img_id] = {
                    "boxes": gt["boxes"][cls_mask],
                    "matched": {iou: np.zeros(gt["boxes"][cls_mask].shape[0], dtype=bool) for iou in iou_thresholds},
                }

            tps = {iou: np.zeros(len(cls_preds)) for iou in iou_thresholds}
            fps = {iou: np.zeros(len(cls_preds)) for iou in iou_thresholds}

            for idx, pred in enumerate(cls_preds):
                img_id = pred["image_id"]
                p_box = pred["box"]
                gts_img = gt_tracked[img_id]["boxes"]
                
                if len(gts_img) == 0:
                    for iou in iou_thresholds:
                        fps[iou][idx] = 1
                    continue
                    
                ious = bbox_iou_numpy(p_box[None, :], gts_img)[0]
                
                for iou_thresh in iou_thresholds:
                    best_iou = -1
                    best_gt_idx = -1
                    for gt_idx, iou_val in enumerate(ious):
                        if iou_val >= iou_thresh and iou_val > best_iou:
                            if not gt_tracked[img_id]["matched"][iou_thresh][gt_idx]:
                                best_iou = iou_val
                                best_gt_idx = gt_idx
                                
                    if best_gt_idx >= 0:
                        tps[iou_thresh][idx] = 1
                        gt_tracked[img_id]["matched"][iou_thresh][best_gt_idx] = True
                        if iou_thresh == 0.5:
                            total_tp_50 += 1
                    else:
                        fps[iou_thresh][idx] = 1
                        if iou_thresh == 0.5:
                            total_fp_50 += 1

            for iou in iou_thresholds:
                tp_cum = np.cumsum(tps[iou])
                fp_cum = np.cumsum(fps[iou])
                recalls = tp_cum / np.clip(num_cls_gts, 1e-8, None)
                precisions = tp_cum / np.clip(tp_cum + fp_cum, 1e-8, None)
                
                # VOC 11-point interpolation
                ap = 0.0
                for r in np.linspace(0.0, 1.0, 11):
                    p = precisions[recalls >= r]
                    p_val = p.max() if len(p) > 0 else 0.0
                    ap += p_val / 11.0
                aps[iou].append(ap)

        mAP50 = float(np.mean(aps[0.5])) if len(aps[0.5]) > 0 else 0.0
        map50_95_list = [float(np.mean(aps[iou])) if len(aps[iou]) > 0 else 0.0 for iou in iou_thresholds]
        mAP50_95 = float(np.mean(map50_95_list))
        
        precision = float(total_tp_50 / max(total_tp_50 + total_fp_50, 1))
        recall = float(total_tp_50 / max(total_gts, 1))
        
        return {"mAP50": mAP50, "mAP50-95": mAP50_95, "precision": precision, "recall": recall}

    def measure_speed(self, img_size: Tuple[int, ...], num_runs: int = 50) -> Tuple[float, float]:
        """Measures execution latency (ms) and FPS throughput."""
        return benchmark_latency_fps(self.model, self.device, num_runs=num_runs, img_size=img_size)

    def run_evaluation(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Runs the validation pipeline and returns all performance & accuracy metrics."""
        # 1. Compute accuracy metrics
        logger.info("Computing accuracy metrics (mAP, Precision, Recall)...")
        preds, gts = self.gather_predictions(dataloader)
        metrics = self.calculate_accuracy_metrics(preds, gts)
        
        # 2. Get image size
        img_size_val = (3, self.model.img_size, self.model.img_size)
        
        # 3. Compute speed metrics
        logger.info("Profiling speed latency and FPS...")
        latency, fps = self.measure_speed(img_size_val, num_runs=50)
        
        # 4. Compute structural metrics
        params = self.model.get_params_count()
        size_mb = self.model.get_model_size_mb()
        flops = self.model.calculate_flops(img_size_val)
        
        metrics.update({
            "params": params,
            "flops": flops,
            "size_mb": size_mb,
            "latency_ms": latency,
            "fps": fps
        })
        
        return metrics
