import cv2 as cv
import numpy as np
import supervision as sv
import time
import argparse
import sys
import os
import torch
import torch.nn as nn

# Ensure project root is in python path to load models and utils
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.factory import ModelFactory, extract_model_state_dict
from pruning.base import PRUNER_REGISTRY
from utils.pipeline_utils import load_checkpoint_weights, bake_pruned_weights

# 1. Define command line arguments
parser = argparse.ArgumentParser(description="Test pruned/recovered custom models on traffic counting.")
parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr"], help="Model architecture")
parser.add_argument("--checkpoint", type=str, default="checkpoints/yolov5s/pruned/magnitude_0.3.pt", help="Path to checkpoint .pt file")
parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "detrac"], help="Dataset format/classes")
parser.add_argument("--prune-type", type=str, default="None", help="Fallback pruning method if not in checkpoint")
parser.add_argument("--sparsity", type=float, default=0.0, help="Fallback sparsity ratio if not in checkpoint")
parser.add_argument("--video", type=str, default="DATA/INPUTS/cars_on_highway_5.mp4", help="Path to input video")
parser.add_argument("--output", type=str, default="", help="Path to output video (default auto-build)")
parser.add_argument("--headless", action="store_true", help="Run without showing GUI window (no cv.imshow)")
parser.add_argument("--max-frames", type=int, default=300, help="Maximum number of frames to process (default 300 for a shorter run)")
args = parser.parse_args()

# 2. Checkpaths
video_path = args.video
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Input video not found: {video_path}")

# Initialize video info
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = video_info.width, video_info.height, video_info.fps

# 3. Setup output path
output_path = args.output
if not output_path:
    # Build a clean output name based on args
    filename = f"car_counter_{args.model}_{os.path.basename(args.checkpoint)}"
    if filename.endswith(".pt"):
        filename = filename[:-3] + ".mp4"
    os.makedirs("DATA/OUTPUTS", exist_ok=True)
    output_path = os.path.join("DATA/OUTPUTS", filename)

# 4. Resolve Device and Model checkpoint details
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Loading checkpoint: {args.checkpoint}")

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state_dict = extract_model_state_dict(checkpoint)

# Detect if checkpoint has weight masks (indicates magnitude pruning)
has_masks = any(k.endswith(".weight_mask") for k in state_dict.keys())

config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
prune_type = config.get("prune_type", config.get("pruning_method", args.prune_type))
sparsity = config.get("sparsity", args.sparsity)

prune_type_str = str(prune_type)
sparsity_val = float(sparsity)

print(f"Detected pruning config -> Method: {prune_type_str}, Sparsity: {sparsity_val}")
print(f"Dynamic masks present: {has_masks}")

# Initialize model structure
num_classes = 80 if args.dataset == "coco" else 4
model = ModelFactory.load(args.model, num_classes=num_classes, device=device)

# Register pruning masks if required
should_prune = False
if prune_type_str.lower() != "none" and sparsity_val > 0.0:
    if prune_type_str.lower() == "magnitude":
        should_prune = has_masks
    else:
        should_prune = True

if should_prune:
    print(f"Registering pruning structure for type '{prune_type_str}'...")
    prune_type_lower = prune_type_str.lower()
    if prune_type_lower in PRUNER_REGISTRY:
        pruner_cls = PRUNER_REGISTRY[prune_type_lower]
        pruner = pruner_cls(model, sparsity_val)
        model = pruner.prune()

# Load weights
load_checkpoint_weights(model, args.checkpoint, device)

# Bake weights (remove hooks and finalize weights to boost inference speed)
if has_masks:
    print("Baking pruning masks into weights...")
    bake_pruned_weights(model)

model.eval()

# 5. Define Class lists and filters
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
DETRAC_CLASSES = ["car", "van", "bus", "others"]

class_names = COCO_CLASSES if num_classes == 80 else DETRAC_CLASSES
vehicle_classes = {'car', 'motorbike', 'motorcycle', 'bus', 'truck', 'van'}
selected_classes = [cls_id for cls_id, class_name in enumerate(class_names) if class_name in vehicle_classes]

# Setup tracker, smoother
tracker = sv.ByteTrack(frame_rate=fps)
smoother = sv.DetectionsSmoother()

# Define counting zones (scaled)
arr1 = np.array([[761, 642], [1073, 642], [1070, 732], [968, 776], [872, 1038], [97, 1049]], dtype=np.int32)
arr2 = np.array([[1105, 639], [1402, 645], [1920, 959], [1920, 1080], [930, 1073], [991, 811], [1105, 755]],
                dtype=np.int32)
zone_points = [np.floor(arr * 0.66).astype(np.int32) for arr in [arr1, arr2]]
zones = [sv.PolygonZone(points) for points in zone_points]
colors = sv.ColorPalette.from_hex(['#ef260e', '#07f921'])  # Colors for zones

# Initialize count storage and ID tracking
total_counts, crossed_ids = [], set()
counts_up, ids_up = [], set()
counts_down, ids_down = [], set()

thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)


# Overlay helper function
def draw_overlay(frame, points, color, alpha=0.25):
    overlay = frame.copy()
    cv.fillPoly(overlay, [points], color)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# Counting functions for zones
def count_vehicle_in_zone(ID, cx, cy, zone_idx):
    if cv.pointPolygonTest(zone_points[zone_idx], (cx, cy), False) >= 0:
        if ID not in crossed_ids:
            total_counts.append(ID)
            crossed_ids.add(ID)
        if zone_idx == 0 and ID not in ids_down:
            counts_down.append(ID)
            ids_down.add(ID)
        elif zone_idx == 1 and ID not in ids_up:
            counts_up.append(ID)
            ids_up.add(ID)


# Annotation function for zones and detections
def annotate_frame(frame, detections):
    # Filter detections to selected vehicle classes
    detections = detections[np.isin(detections.class_id, selected_classes)]

    for points, color in zip(zone_points, [(88, 117, 234), (11, 244, 113)]):
        draw_overlay(frame, points, color=color, alpha=0.25)

    for idx, zone in enumerate(zones):
        zone_annotator = sv.PolygonZoneAnnotator(zone, thickness=4, color=colors.by_idx(idx), text_scale=2,
                                                 text_thickness=2)
        mask = zone.trigger(detections)
        filtered_detections = detections[mask]

        if len(filtered_detections) > 0 and filtered_detections.tracker_id is not None:
            # Draw boxes and labels for filtered detections
            box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
            label_annotator = sv.LabelAnnotator(text_scale=text_scale,
                                                text_thickness=thickness,
                                                text_position=sv.Position.TOP_CENTER,
                                                color_lookup=sv.ColorLookup.TRACK)

            box_annotator.annotate(frame, filtered_detections)
            labels = []
            for class_id, trk_id in zip(filtered_detections.class_id, filtered_detections.tracker_id):
                cls_name = class_names[class_id] if class_id < len(class_names) else f"cls_{class_id}"
                labels.append(f"{cls_name} #{trk_id}")

            label_annotator.annotate(
                frame, detections=filtered_detections,
                labels=labels
            )
        zone_annotator.annotate(frame)

    # Count vehicles based on their bottom center coordinates
    if detections.tracker_id is not None:
        for track_id, bottom_center in zip(detections.tracker_id,
                                           detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
            cx, cy = map(int, bottom_center)
            cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)
            count_vehicle_in_zone(track_id, cx, cy, 0)
            count_vehicle_in_zone(track_id, cx, cy, 1)


# Process video
cap = cv.VideoCapture(video_path)
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
prev_time = time.time()
display_fps = 0.0

if not cap.isOpened():
    raise Exception(f"Error: couldn't open the video: {video_path}")

print(f"Processing video. Output will be saved to: {output_path}")

frame_idx = 0
while cap.isOpened():
    if args.max_frames > 0 and frame_idx >= args.max_frames:
        print(f"Reached max-frames limit of {args.max_frames}. Stopping.")
        break

    frame_start_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # 6. Preprocessing for custom wrappers
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img_rgb, (model.img_size, model.img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

    # Apply ImageNet normalization for DETR
    if args.model == "detr":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

    input_tensor = img_tensor.unsqueeze(0).to(device)

    # 7. Model Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # 8. Extract predictions
    pred_boxes = outputs["boxes"][0]
    pred_scores = outputs["scores"][0]
    pred_classes = outputs["class_ids"][0]

    # Filter detections (confidence >= 0.05)
    keep = pred_scores >= 0.05
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_classes = pred_classes[keep]

    if len(pred_boxes) > 0:
        # Scale coordinates from relative [0, 1] to absolute pixels [w, h]
        scaled_boxes = pred_boxes * torch.tensor([w, h, w, h], device=device)
        xyxy_ndarray = scaled_boxes.cpu().numpy()
        confidence_ndarray = pred_scores.cpu().numpy()
        class_id_ndarray = pred_classes.cpu().numpy()
    else:
        xyxy_ndarray = np.zeros((0, 4), dtype=np.float32)
        confidence_ndarray = np.zeros((0,), dtype=np.float32)
        class_id_ndarray = np.zeros((0,), dtype=np.int32)

    # Construct supervision detections
    detections = sv.Detections(
        xyxy=xyxy_ndarray,
        confidence=confidence_ndarray,
        class_id=class_id_ndarray
    )

    # 9. Update tracking and smoothing
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    # Annotate frame details
    if detections.tracker_id is not None and len(detections) > 0:
        annotate_frame(frame, detections)
    else:
        # Drawing default overlays and zones if no active tracking
        for points, color in zip(zone_points, [(88, 117, 234), (11, 244, 113)]):
            draw_overlay(frame, points, color=color, alpha=0.25)
        for idx, zone in enumerate(zones):
            zone_annotator = sv.PolygonZoneAnnotator(zone, thickness=4, color=colors.by_idx(idx), text_scale=2,
                                                     text_thickness=2)
            zone_annotator.annotate(frame)

    # Draw persistent Counters
    counter_labels = [f"COUNTS: {len(total_counts)}", f"UP: {len(counts_up)}", f"DOWN: {len(counts_down)}"]
    count_colors = [(0, 0, 0), (6, 104, 2), (0, 0, 255)]
    cv.rectangle(frame, (0, 0), (300, 150), (255, 255, 255), cv.FILLED)
    for i, (label, color) in enumerate(zip(counter_labels, count_colors)):
        cv.putText(frame, label, (20, 50 + i * 40), cv.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)

    # Calculate real-time FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    if elapsed_time > 0:
        display_fps = 0.9 * display_fps + 0.1 * (1 / elapsed_time)

    fps_label = f"FPS: {display_fps:.1f}"
    cv.rectangle(frame, (w - 190, 10), (w - 15, 60), (255, 255, 255), cv.FILLED)
    cv.putText(frame, fps_label, (w - 175, 45), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Output writing
    out.write(frame)

    if not args.headless:
        # Calculate dynamic delay to play at normal speed
        process_time_ms = (time.perf_counter() - frame_start_time) * 1000
        target_delay_ms = 1000.0 / fps
        wait_delay = int(max(1.0, target_delay_ms - process_time_ms))
        try:
            cv.imshow("Video", frame)
            # Close the window by pressing 'p'
            if cv.waitKey(wait_delay) & 0xff == ord('p'):
                break
        except Exception:
            # Fallback quietly if headless display is not available
            pass

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()
print(f"Finished processing. Output video saved to: {output_path}")
