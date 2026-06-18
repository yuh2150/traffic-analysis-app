#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset import TrafficDataset, CLASS_MAP

def get_class_names() -> Dict[int, str]:
    return {v: k for k, v in CLASS_MAP.items()}

def analyze_dataset(dataset: TrafficDataset, limit: int = 100) -> Tuple[Dict[str, int], List[int]]:
    """Analyzes the dataset and prints basic stats."""
    class_names = get_class_names()
    class_counts = {name: 0 for name in class_names.values()}
    boxes_per_img = []
    
    total_imgs = len(dataset)
    print(f"\n[INFO] Total images in dataset: {total_imgs}")
    
    # We scan up to limit images for detailed stats to be fast
    scan_limit = min(total_imgs, limit)
    print(f"[INFO] Scanning first {scan_limit} images for detailed statistics...")
    
    for i in range(scan_limit):
        _, target = dataset[i]
        boxes = target["boxes"]
        labels = target["labels"]
        
        boxes_per_img.append(len(boxes))
        for label in labels:
            cls_name = class_names.get(label.item(), "others")
            class_counts[cls_name] += 1
            
    return class_counts, boxes_per_img

def visualize_sample(dataset: TrafficDataset, output_path: str = "reports/dataset_sample.png"):
    """Visualizes a sample image with bounding boxes and saves it."""
    if len(dataset) == 0:
        print("[WARNING] Dataset is empty, cannot visualize.")
        return
        
    # Find a sample with some bounding boxes
    sample_idx = 0
    for i in range(len(dataset)):
        img_tensor, target = dataset[i]
        if len(target["boxes"]) > 0:
            sample_idx = i
            break
            
    img_tensor, target = dataset[sample_idx]
    file_name = target["file_name"]
    boxes = target["boxes"]
    labels = target["labels"]
    
    # Convert tensor back to numpy image [H, W, C]
    img = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV expects BGR to draw, but matplotlib expects RGB. 
    # Actually cv2.imread loads BGR, TrafficDataset converts BGR->RGB, does resize, then we permuted.
    # So img is RGB. Let's make a copy for cv2 drawing.
    img_draw = img.copy()
    h, w, _ = img_draw.shape
    
    class_names = get_class_names()
    colors = {
        0: (255, 0, 0),    # Car - Red
        1: (0, 255, 0),    # Van - Green
        2: (0, 0, 255),    # Bus - Blue
        3: (255, 165, 0)   # Others - Orange
    }
    
    print(f"\n[INFO] Visualizing sample index: {sample_idx}")
    print(f"[INFO] File name: {file_name}")
    print(f"[INFO] Image size: {w}x{h}")
    print(f"[INFO] Number of objects: {len(boxes)}")
    
    for box, label in zip(boxes, labels):
        # Coordinates are normalized [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.numpy()
        ix1, iy1 = int(x1 * w), int(y1 * h)
        ix2, iy2 = int(x2 * w), int(y2 * h)
        
        cls_id = label.item()
        cls_name = class_names.get(cls_id, "others")
        color = colors.get(cls_id, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img_draw, (ix1, iy1), (ix2, iy2), color, 2)
        
        # Text label
        label_text = f"{cls_name}"
        cv2.putText(img_draw, label_text, (ix1, max(iy1 - 5, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_draw)
    plt.title(f"Sample: {file_name} ({len(boxes)} objects)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"[INFO] Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test and Verify UA-DETRAC Dataset loading")
    parser.add_argument(
        "--img-dir", 
        type=str, 
        default="data/UA-DETRAC/DETRAC-Images/DETRAC-Images", 
        help="UA-DETRAC images root directory"
    )
    parser.add_argument(
        "--anno-dir", 
        type=str, 
        default="data/UA-DETRAC/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML", 
        help="UA-DETRAC XML annotations directory"
    )
    parser.add_argument(
        "--limit", 
        type=str, 
        default="100", 
        help="Limit number of images to scan for statistics"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="reports/dataset_sample.png", 
        help="Path to save visualization sample"
    )
    args = parser.parse_args()
    
    print("====================================================")
    print("          UA-DETRAC DATASET VERIFIER & TESTER        ")
    print("====================================================")
    print(f"Image Directory:       {args.img_dir}")
    print(f"Annotation Directory:  {args.anno_dir}")
    print("====================================================")
    
    if not os.path.exists(args.img_dir):
        print(f"[ERROR] Image directory does not exist: {args.img_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.anno_dir):
        print(f"[ERROR] Annotation directory does not exist: {args.anno_dir}")
        sys.exit(1)
        
    # Try initializing the dataset
    try:
        print("[INFO] Initializing TrafficDataset...")
        dataset = TrafficDataset(img_dir=args.img_dir, anno_dir=args.anno_dir)
    except Exception as e:
        print(f"[ERROR] Failed to initialize TrafficDataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    if len(dataset) == 0:
        print("[WARNING] Dataset has 0 images. Check if your directories contain appropriate DETRAC data structure.")
        sys.exit(0)
        
    # Analyze dataset
    try:
        class_counts, boxes_per_img = analyze_dataset(dataset, limit=int(args.limit))
        
        # Print Stats
        print("\n================ DATASET STATISTICS ================")
        print(f"Class-wise Object counts (in first {args.limit} images):")
        for cls_name, count in class_counts.items():
            print(f"  - {cls_name:10s}: {count}")
            
        print("\nBounding Boxes per Image statistics:")
        print(f"  - Min bboxes:   {np.min(boxes_per_img)}")
        print(f"  - Max bboxes:   {np.max(boxes_per_img)}")
        print(f"  - Mean bboxes:  {np.mean(boxes_per_img):.2f}")
        print(f"  - Median bboxes:{np.median(boxes_per_img):.2f}")
        print("====================================================")
    except Exception as e:
        print(f"[ERROR] Error during dataset analysis: {e}")
        
    # Visualize sample
    try:
        visualize_sample(dataset, args.output)
    except Exception as e:
        print(f"[ERROR] Error during visualization: {e}")

if __name__ == "__main__":
    main()
