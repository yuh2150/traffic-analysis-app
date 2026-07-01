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
parser = argparse.ArgumentParser(description="Test pruned/recovered custom models on traffic counting (Counter 4).")
parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr"], help="Model architecture")
parser.add_argument("--checkpoint", type=str, default="checkpoints/yolov5s/pruned/magnitude_0.3.pt", help="Path to checkpoint .pt file")
parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "detrac"], help="Dataset format/classes")
parser.add_argument("--prune-type", type=str, default="None", help="Fallback pruning method if not in checkpoint")
parser.add_argument("--sparsity", type=float, default=0.0, help="Fallback sparsity ratio if not in checkpoint")
parser.add_argument("--video", type=str, default="DATA/INPUTS/cars_on_highway_2.mp4", help="Path to input video")
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
    filename = f"car_counter_4_{args.model}_{os.path.basename(args.checkpoint)}"
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

# Detect if checkpoint has weight masks
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

# Bake weights
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

# Setup annotators
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                                    text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps * 2,
                                    position=sv.Position.CENTER, color_lookup=sv.ColorLookup.TRACK)

# Tracker and vehicle class setup
tracker = sv.ByteTrack(frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother()

# Initialize counters
limits = [0, 300, w, 300]  # Line for vehicle counting, dynamically set end to w
partition_limit = int(w * 0.43) # equivalent to 550 for 1280 width
total_counts, crossed_ids = [], set()

total_counts_up, crossed_ids_up = [], set()
total_counts_down, crossed_ids_down = [], set()


def draw_overlay(frame, pt1, pt2, alpha=0.25, color=(51, 68, 255), filled=True):
    """Draws a semi-transparent overlay rectangle."""
    overlay = frame.copy()
    rect_color = color if filled else (0, 0, 0)
    cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def count_vehicles(track_id, cx, cy, limits, crossed_ids):
    """Counts vehicles crossing the line."""
    if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10 and track_id not in crossed_ids:
        crossed_ids.add(track_id)
        return True
    return False


def count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up):
    """Counts vehicles crossing the line."""
    if limits[0] < cx < partition_limit and limits[1] - 10 < cy < limits[1] + 10 and track_id not in crossed_ids_up:
        crossed_ids_up.add(track_id)
        return True
    return False


def count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down):
    """Counts vehicles crossing the line."""
    if partition_limit < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and track_id not in crossed_ids_down:
        crossed_ids_down.add(track_id)
        return True
    return False


def draw_tracks_and_count(frame, detections, total_counts, limits):
    """Annotates the frame with detected tracks and counts vehicles."""
    detections = detections[np.isin(detections.class_id, selected_classes)]  # Filter by vehicle classes
    
    labels = []
    for track_id, cls_id in zip(detections.tracker_id, detections.class_id):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        labels.append(f"#{track_id} {cls_name}")

    label_annotator.annotate(frame, detections=detections, labels=labels)
    box_annotator.annotate(frame, detections=detections)
    trace_annotator.annotate(frame, detections=detections)

    for track_id, center_point in zip(detections.tracker_id,
                                      detections.get_anchors_coordinates(anchor=sv.Position.CENTER)):
        cx, cy = map(int, center_point)
        cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # Draw vehicle center point

        # Storing the counts
        if count_vehicles(track_id, cx, cy, limits, crossed_ids):
            total_counts.append(track_id)
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.ROBOFLOW, thickness=4)
            draw_overlay(frame, (0, 200), (w, 400), alpha=0.25, color=(10, 255, 50))

        if count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up):
            total_counts_up.append(track_id)
        if count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down):
            total_counts_down.append(track_id)

    # Annotating the total counts
    sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30), sv.Color.ROBOFLOW, 1.25,
                 2, background_color=sv.Color.WHITE)
    sv.draw_text(frame, f"UP: {len(total_counts_up)}", sv.Point(x=int(w * 0.44), y=280), sv.Color.WHITE, 1,
                 2)
    sv.draw_text(frame, f"DOWN: {len(total_counts_down)}", sv.Point(x=int(w * 0.44), y=320), sv.Color.WHITE, 1,
                 2)


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

    crop = frame[150:, :]
    mask_b = np.zeros_like(frame, dtype=np.uint8)
    mask_w = np.ones_like(frame[150:, :], dtype=np.uint8) * 255
    mask_b[150:, :] = mask_w

    # Apply the mask to the original frame
    ROI = cv.bitwise_and(frame, mask_b)

    # Preprocessing for custom wrappers
    img_rgb = cv.cvtColor(ROI, cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img_rgb, (model.img_size, model.img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

    # Apply ImageNet normalization for DETR
    if args.model == "detr":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

    input_tensor = img_tensor.unsqueeze(0).to(device)

    # Model Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Extract predictions
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

    # Update tracking and smoothing
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    if detections.tracker_id is not None and len(detections) > 0:
        # Draw counting line and process vehicle tracks
        sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                     color=sv.Color.RED, thickness=4)
        draw_overlay(frame, (0, 200), (w, 400), alpha=0.2)
        draw_tracks_and_count(frame, detections, total_counts, limits)
    else:
        # Draw default counting line and overlay
        sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                     color=sv.Color.RED, thickness=4)
        draw_overlay(frame, (0, 200), (w, 400), alpha=0.2)

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
            cv.imshow("Camera", frame)
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
