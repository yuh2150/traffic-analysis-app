#!/usr/bin/env python3
import argparse
import os
import sys
from itertools import islice

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.factory import DatasetFactory
from models.factory import ModelFactory


def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _get_sample(loader, image_index: int):
    for current_index, batch in enumerate(loader):
        if current_index == image_index:
            imgs, targets = batch
            return imgs[0], targets[0]
    raise IndexError(f"Image index {image_index} is out of range")


def _to_pixel_box(box, width, height):
    x1, y1, x2, y2 = box
    return [
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    ]


def _draw_boxes(image, boxes, labels, scores=None, color=(0, 0, 255), prefix="pred"):
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{prefix}:{int(labels[index])}"
        if scores is not None:
            label_text += f" {float(scores[index]):.2f}"
        cv2.putText(
            image,
            label_text,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Run one-image inference and save visualized detections.")
    parser.add_argument("--model", choices=["yolov5s", "detr"], default="detr")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["coco", "detrac"], default="coco")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", default="reports/visualizations/single_image.png")
    parser.add_argument("--coco-val-img", type=str, default="data/coco/val2017")
    parser.add_argument("--coco-val-anno", type=str, default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--img-dir", type=str, default="data/DETRAC-Images/DETRAC-Images")
    parser.add_argument("--val-anno-dir", type=str, default="data/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    img_size = (640, 640) if args.model == "yolov5s" else (800, 800)
    if args.dataset == "coco":
        val_img = args.coco_val_img
        val_anno = args.coco_val_anno
    else:
        val_img = args.img_dir
        val_anno = args.val_anno_dir

    dataloader = DatasetFactory.get_dataloader(
        img_dir=val_img,
        anno_dir=val_anno,
        batch_size=1,
        img_size=img_size,
        shuffle=False,
        max_samples=None,
        dataset_type=args.dataset,
    )

    dataset = _unwrap_dataset(dataloader.dataset)
    image_tensor, target = _get_sample(dataloader, args.image_index)
    model = ModelFactory.load(args.model, weights_path=args.checkpoint, device=device, num_classes=80 if args.dataset == "coco" else 4)
    model.eval()

    with torch.no_grad():
        batch = image_tensor.unsqueeze(0).to(device)
        outputs = model(batch)

    image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    pred_boxes = outputs["boxes"][0].detach().cpu().numpy()
    pred_scores = outputs["scores"][0].detach().cpu().numpy()
    pred_labels = outputs["class_ids"][0].detach().cpu().numpy()
    keep = pred_scores >= args.score_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    if hasattr(dataset, "coco_cat_to_idx"):
        inverse_map = {idx: cat_id for cat_id, idx in dataset.coco_cat_to_idx.items()}
        pred_labels = np.array([inverse_map.get(int(label), int(label)) for label in pred_labels], dtype=np.int64)

    pred_boxes_px = [_to_pixel_box(box, width, height) for box in pred_boxes]
    _draw_boxes(image, pred_boxes_px, pred_labels, pred_scores, color=(0, 0, 255), prefix="pred")

    gt_boxes = target["boxes"].cpu().numpy()
    gt_labels = target["labels"].cpu().numpy()
    gt_boxes_px = [_to_pixel_box(box, width, height) for box in gt_boxes]
    _draw_boxes(image, gt_boxes_px, gt_labels, color=(0, 200, 0), prefix="gt")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, image)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()