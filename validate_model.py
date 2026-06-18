import os
import time
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import argparse
import logging
from models.models import get_model, BaseTrafficDetector
from datasets.dataset import get_data_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ValidationPipeline")


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
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


def get_predictions_and_gts(
    model: BaseTrafficDetector, dataloader: Any, device: torch.device
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for imgs, targets in dataloader:
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
    return all_preds, all_gts


def evaluate_detection_metrics(
    all_preds: List[Dict[str, np.ndarray]],
    all_gts: List[Dict[str, np.ndarray]],
    iou_thresholds: List[float] = [0.5],
) -> Dict[str, float]:
    num_classes = 4
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
            ious = bbox_iou(p_box[None, :], gts_img)[0]
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
            ap = 0.0
            for r in np.linspace(0.0, 1.0, 11):
                p = precisions[recalls >= r]
                p_val = p.max() if len(p) > 0 else 0.0
                ap += p_val / 11.0
            aps[iou].append(ap)

    mAP50 = np.mean(aps[0.5]) if len(aps[0.5]) > 0 else 0.0
    map50_95_list = [np.mean(aps[iou]) if len(aps[iou]) > 0 else 0.0 for iou in iou_thresholds]
    mAP50_95 = np.mean(map50_95_list)
    precision = total_tp_50 / max(total_tp_50 + total_fp_50, 1)
    recall = total_tp_50 / max(total_gts, 1)
    return {"mAP50": mAP50, "mAP50-95": mAP50_95, "precision": precision, "recall": recall}


def evaluate_map(
    model: BaseTrafficDetector, dataloader: Any, device: torch.device
) -> Tuple[float, float]:
    preds, gts = get_predictions_and_gts(model, dataloader, device)
    iou_thresholds = sorted(list(set([0.5] + np.linspace(0.5, 0.95, 10).tolist())))
    res = evaluate_detection_metrics(preds, gts, iou_thresholds)
    return res["mAP50"], res["mAP50-95"]


def evaluate_precision(model: BaseTrafficDetector, dataloader: Any, device: torch.device) -> float:
    preds, gts = get_predictions_and_gts(model, dataloader, device)
    return evaluate_detection_metrics(preds, gts, [0.5])["precision"]


def evaluate_recall(model: BaseTrafficDetector, dataloader: Any, device: torch.device) -> float:
    preds, gts = get_predictions_and_gts(model, dataloader, device)
    return evaluate_detection_metrics(preds, gts, [0.5])["recall"]


def measure_fps(
    model: BaseTrafficDetector,
    device: torch.device,
    num_runs: int = 100,
    img_size: Tuple[int, int, int, int] = (1, 3, 640, 640),
) -> float:
    model.eval()
    x = torch.zeros(img_size, device=device)
    for _ in range(10):
        with torch.no_grad():
            model(x)
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(x)
    return num_runs / (time.perf_counter() - start_time)


def measure_latency(
    model: BaseTrafficDetector,
    device: torch.device,
    num_runs: int = 100,
    img_size: Tuple[int, int, int, int] = (1, 3, 640, 640),
) -> float:
    model.eval()
    x = torch.zeros(img_size, device=device)
    for _ in range(10):
        with torch.no_grad():
            model(x)
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(x)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(latencies))


def calculate_flops(
    model: BaseTrafficDetector, img_size: Tuple[int, int, int, int] = (1, 3, 640, 640)
) -> int:
    _, flops = model.calculate_flops(img_size)
    return flops


def calculate_params(model: BaseTrafficDetector) -> int:
    return model.get_params_count()


def save_results(results: List[Dict[str, Any]], base_path: str):
    df = pd.DataFrame(results)
    df.to_csv(f"{base_path}.csv", index=False)
    df.to_json(f"{base_path}.json", orient="records", indent=4)
    df.to_excel(f"{base_path}.xlsx", index=False, engine="openpyxl")
    logger.info(f"Saved validation reports to: {base_path}.[csv/json/xlsx]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Validation Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov8n", "yolov8s", "detr", "yolov5", "yolov8"],
        help="Model name",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file (optional)")
    parser.add_argument(
        "--img-dir",
        type=str,
        default=os.environ.get("DETRAC_IMG_DIR", "data/UA-DETRAC/train"),
        help="UA-DETRAC images root directory",
    )
    parser.add_argument(
        "--anno-dir",
        type=str,
        default=os.environ.get("DETRAC_ANNO_DIR", "data/UA-DETRAC/annotations"),
        help="UA-DETRAC XML annotations directory",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Validating {args.model} on {device}...")

    model = get_model(args.model)
    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state, strict=False)
    model = model.to(device)

    img_size_val = (640, 640) if "yolo" in args.model else (800, 800)
    dataloader = get_data_loader(
        img_dir=args.img_dir, anno_dir=args.anno_dir, batch_size=2, img_size=img_size_val
    )

    mAP50, mAP50_95 = evaluate_map(model, dataloader, device)
    precision = evaluate_precision(model, dataloader, device)
    recall = evaluate_recall(model, dataloader, device)

    flops = calculate_flops(model, (1, 3) + img_size_val)
    params = calculate_params(model)
    size_mb = model.get_model_size_mb()

    fps = measure_fps(model, device, num_runs=50, img_size=(1, 3) + img_size_val)
    latency = measure_latency(model, device, num_runs=50, img_size=(1, 3) + img_size_val)

    logger.info(f"--- Results for {args.model} ---")
    logger.info(f"Params: {params:,}")
    logger.info(f"FLOPs: {flops:,}")
    logger.info(f"Size: {size_mb:.2f} MB")
    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"Latency: {latency:.2f} ms")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"mAP50: {mAP50:.4f}")
    logger.info(f"mAP50-95: {mAP50_95:.4f}")
