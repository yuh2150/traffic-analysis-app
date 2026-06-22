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

        # Convert relative cxcywh coordinates to relative x1y1x2y2
        cx, cy, w, h = out["pred_boxes"].unbind(-1)
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        out_boxes, out_scores, out_ids = [], [], []
        for b in range(B):
            keep = scores[b] >= 0.05
            if keep.any():
                out_boxes.append(boxes[b, keep])
                out_scores.append(scores[b, keep])
                out_ids.append(class_ids[b, keep])
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
