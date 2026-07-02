import sys
import os
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Union

from .yolov5_wrapper import BaseTrafficDetector, NUM_UA_DETRAC_CLASSES

logger = logging.getLogger("Models")


class DETRWrapper(BaseTrafficDetector):
    """Wrapper around DETR-ResNet-50 for UA-DETRAC dataset.

    Inherits from BaseTrafficDetector and supports training and evaluation modes
    aligned with the YOLOv5 wrapper.
    """

    def __init__(self, num_classes: int = 80):
        super().__init__(num_classes)
        self.img_size = 800

        # Save and restore path to prevent import issues in Facebook DETR Hub loading
        saved_path = sys.path.copy()
        saved_models_module = sys.modules.get('models')
        if 'models' in sys.modules:
            del sys.modules['models']

        try:
            sys.path = [p for p in sys.path if p not in ('', os.getcwd(), os.path.abspath('.'))]
            logger.info("Loading pretrained DETR-ResNet-50 from Facebook Research via torch.hub...")
            # Load detr from official github repository
            self.model = torch.hub.load("facebookresearch/detr:main", "detr_resnet50", pretrained=True)
            self._adapt_head(num_classes)
            logger.info(f"Successfully loaded and adapted DETR-ResNet-50 ({self.get_params_count():,} params)")
        except Exception as e:
            raise RuntimeError(f"Failed to load DETR-ResNet-50: {e}")
        finally:
            sys.path = saved_path
            if saved_models_module is not None:
                sys.modules['models'] = saved_models_module

    def _adapt_head(self, num_classes: int):
        """Replace DETR class projection head."""
        in_feat = self.model.class_embed.in_features
        # If num_classes is 80 (COCO dataset) and the model already has 91 classes (which is 92 out_features),
        # preserve the pretrained head and handle mapping in decode/forward.
        if num_classes == 80 and self.model.class_embed.out_features == 92:
            logger.info("Preserving pretrained 91-class DETR head for COCO (80 classes evaluation with mapping).")
            self.model.num_classes = 91
            return
            
        # DETR class projection outputs class scores + 1 (background is the last index)
        if hasattr(self.model, "num_classes") and self.model.num_classes == num_classes + 1:
            logger.info(f"DETR head classes already matches {num_classes + 1}. Preserving pretrained weights.")
            return
            
        self.model.class_embed = nn.Linear(in_feat, num_classes + 1)
        self.model.num_classes = num_classes + 1
        
        # Re-initialize the linear layer weights
        nn.init.kaiming_normal_(self.model.class_embed.weight, mode="fan_out", nonlinearity="relu")
        if self.model.class_embed.bias is not None:
            self.model.class_embed.bias.data.zero_()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Override load_state_dict to dynamically adjust the classification head if there's a shape mismatch."""
        key = "model.class_embed.weight"
        if key in state_dict:
            checkpoint_shape = state_dict[key].shape
            current_shape = self.model.class_embed.weight.shape
            if checkpoint_shape != current_shape:
                logger.info(f"Dynamically adapting class_embed head shape to match checkpoint: {current_shape} -> {checkpoint_shape}")
                in_features = self.model.class_embed.in_features
                out_features = checkpoint_shape[0]
                self.model.class_embed = nn.Linear(in_features, out_features)
                self.model.num_classes = out_features - 1
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        During training, returns decoded predictions aligned with YOLOv5 shape: [B, 100, 5 + num_classes].
        During evaluation, returns boxes, scores, and class_ids.
        """
        if self.training:
            out = self.model(x)  # out contains 'pred_logits' [B, 100, num_classes+1], 'pred_boxes' [B, 100, 4]
            
            # Extract boxes [B, 100, 4] in absolute pixel coordinates for the loss function
            pred_boxes_pixels = out["pred_boxes"] * self.img_size
            
            # Compute class probabilities (excluding the background class at index -1)
            cls_prob = F.softmax(out["pred_logits"], dim=-1)[:, :, :-1]
            
            if getattr(self.model, "num_classes", 0) == 91:
                # Map 91 class probabilities to 80 classes
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
                indices = [0] * 80
                for cat_id, idx in coco_cat_to_idx.items():
                    indices[idx] = cat_id
                indices_tensor = torch.tensor(indices, dtype=torch.long, device=cls_prob.device)
                cls_prob = cls_prob.index_select(-1, indices_tensor)
            
            # Compute confidence score as maximum class probability
            conf = cls_prob.max(-1, keepdim=True).values
            
            # Concatenate coordinates, confidence, and class probabilities
            pred = torch.cat((pred_boxes_pixels, conf, cls_prob), dim=-1)
            return pred
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x)
            boxes, scores, class_ids = self._decode(out)
            return {"boxes": boxes, "scores": scores, "class_ids": class_ids}

    def _decode(self, out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode predictions and filter by score threshold."""
        device = out["pred_boxes"].device
        B = out["pred_boxes"].shape[0]

        # Calculate class probabilities
        probs = out["pred_logits"].softmax(-1)[:, :, :-1]
        scores, class_ids = probs.max(-1)

        # If using the 91-class pretrained head for COCO, map classes to 0-79
        if getattr(self.model, "num_classes", 0) == 91:
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
            mapping_arr = [-1] * 92
            for cat_id, idx in coco_cat_to_idx.items():
                mapping_arr[cat_id] = idx
            mapping_tensor = torch.tensor(mapping_arr, dtype=torch.long, device=device)
            mapped_class_ids = mapping_tensor[class_ids]
        else:
            mapped_class_ids = class_ids

        # Convert relative cxcywh coordinates to relative x1y1x2y2
        cx, cy, w, h = out["pred_boxes"].unbind(-1)
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        out_boxes, out_scores, out_ids = [], [], []
        for b in range(B):
            if getattr(self.model, "num_classes", 0) == 91:
                keep = (scores[b] >= 0.05) & (mapped_class_ids[b] >= 0)
            else:
                keep = scores[b] >= 0.05
            if keep.any():
                out_boxes.append(boxes[b, keep])
                out_scores.append(scores[b, keep])
                out_ids.append(mapped_class_ids[b, keep])
            else:
                out_boxes.append(torch.zeros(0, 4, device=device))
                out_scores.append(torch.zeros(0, device=device))
                out_ids.append(torch.zeros(0, dtype=torch.long, device=device))

        max_dets = max(b.shape[0] for b in out_boxes)
        if max_dets == 0:
            return (
                torch.zeros(B, 0, 4, device=device),
                torch.zeros(B, 0, device=device),
                torch.zeros(B, 0, dtype=torch.long, device=device),
            )

        pb = torch.zeros(B, max_dets, 4, device=device)
        ps = torch.zeros(B, max_dets, device=device)
        pi = torch.zeros(B, max_dets, dtype=torch.long, device=device)
        for i in range(B):
            n = out_boxes[i].shape[0]
            pb[i, :n] = out_boxes[i]
            ps[i, :n] = out_scores[i]
            pi[i, :n] = out_ids[i]

        return pb, ps, pi
