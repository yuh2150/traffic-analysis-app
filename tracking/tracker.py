import cv2
import numpy as np
import torch
import supervision as sv
from typing import Tuple, List, Dict, Any, Optional

class TrafficTracker:
    """
    Traffic Tracker using ByteTrack and Supervision.
    Handles bounding box tracking, counting vehicles across a virtual line,
    and estimating traffic density.
    """
    def __init__(self, line_start: Tuple[int, int] = (160, 320), line_end: Tuple[int, int] = (480, 320)):
        # Initialize ByteTrack tracker
        # track_activation_threshold is default 0.25
        self.tracker = sv.ByteTrack()
        
        # Define line zone for counting (e.g. crossing a horizontal line in the middle of frame)
        self.start_point = sv.Point(line_start[0], line_start[1])
        self.end_point = sv.Point(line_end[0], line_end[1])
        self.line_zone = sv.LineZone(start=self.start_point, end=self.end_point)
        
        # Visual annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
            text_scale=0.5,
            text_thickness=1
        )
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=30,
            position=sv.Position.CENTER
        )
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_scale=0.6,
            text_thickness=2
        )
        
        self.classes = ["car", "truck", "bus", "motorcycle"]

    def process_frame(self, frame: np.ndarray, model_outputs: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single frame:
        1. Prepares detections from model outputs.
        2. Updates tracking IDs using ByteTrack.
        3. Checks line crossing counts.
        4. Estimates density.
        5. Returns annotated frame and tracking statistics.
        """
        # Get frame size
        h, w = frame.shape[:2]
        
        # Convert predictions to Supervision Detections
        boxes = model_outputs["boxes"][0].cpu().numpy()     # [N, 4] in normalized coordinates
        scores = model_outputs["scores"][0].cpu().numpy()   # [N]
        class_ids = model_outputs["class_ids"][0].cpu().numpy() # [N]
        
        # Scale normalized boxes to pixel coordinates
        pixel_boxes = boxes.copy()
        pixel_boxes[:, [0, 2]] *= w
        pixel_boxes[:, [1, 3]] *= h
        
        # Filter detections with threshold
        mask = scores >= 0.25
        pixel_boxes = pixel_boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask].astype(int)
        
        if len(pixel_boxes) > 0:
            detections = sv.Detections(
                xyxy=pixel_boxes,
                confidence=scores,
                class_id=class_ids
            )
            
            # Update tracker
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()
            
        # Trigger line zone counter
        self.line_zone.trigger(detections)
        
        # Draw annotations
        annotated_frame = frame.copy()
        
        if len(detections) > 0:
            # Annotate traces (paths)
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
            
            # Annotate boxes
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
            
            # Draw labels with tracking IDs
            labels = []
            for class_id, tracker_id, conf in zip(detections.class_id, detections.tracker_id, detections.confidence):
                class_name = self.classes[class_id] if class_id < len(self.classes) else "vehicle"
                labels.append(f"#{tracker_id} {class_name} ({conf:.2f})")
                
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
            
        # Annotate line zone
        annotated_frame = self.line_zone_annotator.annotate(
            frame=annotated_frame,
            line_counter=self.line_zone
        )
        
        # Density estimation: based on current number of active tracks in frame
        active_count = len(detections)
        if active_count <= 2:
            density = "Low"
            density_color = (0, 255, 0) # Green
        elif active_count <= 5:
            density = "Medium"
            density_color = (0, 255, 255) # Yellow
        else:
            density = "High"
            density_color = (0, 0, 255) # Red
            
        # Draw density on screen
        cv2.putText(
            annotated_frame,
            f"Density: {density} ({active_count} active)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            density_color,
            2,
            cv2.LINE_AA
        )
        
        stats = {
            "active_vehicles": active_count,
            "in_count": self.line_zone.in_count,
            "out_count": self.line_zone.out_count,
            "total_count": self.line_zone.in_count + self.line_zone.out_count,
            "density": density
        }
        
        return annotated_frame, stats
