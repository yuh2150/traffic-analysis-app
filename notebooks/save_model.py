# -*- coding: utf-8 -*-
"""
Helper script to visualize and save YOLOv5s models (COCO and adapted UA-DETRAC).
You can copy this code block directly into a Jupyter Notebook cell.
"""

import os
import sys
import torch
import torch.nn as nn
import copy

# Ensure project root is in the python path (so models/ wrappers can be imported)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.yolov5_wrapper import YOLOv5Wrapper

def main():
    # Create output directory for saving checkpoints
    output_dir = os.path.join(project_root, "checkpoints", "yolov5s")
    os.makedirs(output_dir, exist_ok=True)
    print(f"1. Target save directory: {output_dir}\n")

    # ----------------------------------------------------
    # GIAI ĐOẠN 1: Tải YOLOv5s COCO gốc (80 classes)
    # ----------------------------------------------------
    print("2. Loading pretrained YOLOv5s COCO from Ultralytics...")
    yolo_coco = YOLOv5Wrapper(num_classes=80)
    
    # Lưu trọng số COCO gốc dạng raw state dict
    coco_save_path = os.path.join(output_dir, "coco_baseline_raw.pt")
    torch.save(yolo_coco.state_dict(), coco_save_path)
    print(f"-> Saved YOLOv5s COCO pretrained to: {coco_save_path}")
    print(f"   (Total active parameters: {yolo_coco.get_params_count():,} params)\n")

    # ----------------------------------------------------
    # GIAI ĐOẠN 2: Khởi tạo mô hình adapted cho UA-DETRAC (4 classes)
    # ----------------------------------------------------
    print("3. Initializing adapted YOLOv5s for UA-DETRAC (4 classes)...")
    # Sử dụng Wrapper của dự án để tự động thích ứng head
    yolo_detrac = YOLOv5Wrapper(num_classes=4)
    
    # Lưu trọng số adapted chưa qua huấn luyện (backbone COCO + head ngẫu nhiên)
    detrac_save_path = os.path.join(output_dir, "yolov5s_detrac_adapted.pt")
    torch.save(yolo_detrac.state_dict(), detrac_save_path)
    print(f"-> Saved adapted YOLOv5s to: {detrac_save_path}")
    print(f"   (Total active parameters: {yolo_detrac.get_params_count():,} params)\n")

    # ----------------------------------------------------
    # GIAI ĐOẠN 3: Hiển thị danh sách file
    # ----------------------------------------------------
    print("4. Saved weight files list in checkpoints/yolov5s/:")
    for f in os.listdir(output_dir):
        f_path = os.path.join(output_dir, f)
        size_mb = os.path.getsize(f_path) / (1024 ** 2)
        print(f"   - {f:<35} | Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
