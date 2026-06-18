import math
import copy
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger("Models")

UA_DETRAC_CLASSES = ["car", "van", "bus", "others"]
NUM_UA_DETRAC_CLASSES = 4


class BaseTrafficDetector(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

    def get_params_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def calculate_flops(self, input_size: Tuple[int, int, int] = (3, 640, 640)) -> int:
        try:
            from thop import profile
            x = torch.zeros(1, *input_size)
            macs, _ = profile(self, inputs=(x,), verbose=False)
            return int(macs * 2)
        except ImportError:
            try:
                from fvcore.nn import FlopCountAnalysis
                x = torch.zeros(1, *input_size)
                flops = FlopCountAnalysis(self, x)
                return flops.total()
            except ImportError:
                logger.warning("Neither thop nor fvcore installed. Install one: pip install thop")
                return 0


class YOLOWrapper(BaseTrafficDetector):
    """Wrapper around YOLOv5 with custom number of classes.

    Loads pretrained COCO weights and adapts the detection head for
    UA-DETRAC classes so the model is ready for pruning experiments.
    """

    def __init__(self, num_classes: int = NUM_UA_DETRAC_CLASSES, model_size: str = "s"):
        super().__init__(num_classes)
        self.model_size = model_size
        self.img_size = 640

        import sys
        import os
        saved_path = sys.path.copy()
        saved_models_module = sys.modules.get('models')
        if 'models' in sys.modules:
            del sys.modules['models']
        try:
            sys.path = [p for p in sys.path if p not in ('', os.getcwd(), os.path.abspath('.'))]
            ckpt_name = f"yolov5{model_size}"
            logger.info(f"Loading pretrained {ckpt_name} ...")
            yolo = torch.hub.load("ultralytics/yolov5", ckpt_name, pretrained=True)
            self.model = copy.deepcopy(yolo.model)
            self._adapt_head(num_classes)
            logger.info(f"YOLOv5{model_size} loaded ({self.get_params_count():,} params)")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5{model_size}: {e}")
        finally:
            sys.path = saved_path
            if saved_models_module is not None:
                sys.modules['models'] = saved_models_module

    def _get_detect(self):
        m = self.model
        if hasattr(m, "model"):
            inner = m.model
            if hasattr(inner, "model") and isinstance(inner.model, nn.Sequential):
                return inner.model[-1]
            if isinstance(inner, nn.Sequential):
                return inner[-1]
        if isinstance(m, nn.Sequential):
            return m[-1]
        raise AttributeError("Cannot locate Detect module")

    def _adapt_head(self, num_classes: int):
        """Replace detection head convs to output num_classes instead of 80."""
        detect = self._get_detect()
        if not hasattr(detect, "m"):
            logger.warning("Detect module has no 'm' attribute; head adaptation skipped")
            return
        detect.nc = num_classes
        detect.no = 5 + num_classes
        for i in range(len(detect.m)):
            old_conv = detect.m[i]
            na = old_conv.out_channels // (5 + 80)
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                na * (5 + num_classes),
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=old_conv.bias is not None,
            )
            detect.m[i] = new_conv

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, tuple):
            decoded = out[0]
        else:
            decoded = out
        boxes, scores, class_ids = self._decode(decoded, x.shape[2:])
        return {"boxes": boxes, "scores": scores, "class_ids": class_ids}

    def _decode(self, decoded: torch.Tensor, img_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode YOLOv5 concatenated eval output into batched detections."""
        device = decoded.device
        B = decoded.shape[0]
        img_h, img_w = img_shape
        dets_list = [[] for _ in range(B)]

        for b in range(B):
            preds = decoded[b]
            obj = preds[:, 4:5]
            cls = preds[:, 5:]
            scores_i, ids_i = (obj * cls).max(dim=1)
            xc, yc, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            dets_list[b].append(torch.stack([x1, y1, x2, y2, scores_i, ids_i.float()], dim=1))

        final_boxes, final_scores, final_ids = [], [], []
        for b in range(B):
            cat = torch.cat(dets_list[b], dim=0)
            if cat.numel() == 0:
                final_boxes.append(torch.zeros(0, 4, device=device))
                final_scores.append(torch.zeros(0, device=device))
                final_ids.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            keep = self._nms(cat[:, :4], cat[:, 4], 0.5)
            final_boxes.append(cat[keep, :4] / torch.tensor([img_w, img_h] * 2, device=device))
            final_scores.append(cat[keep, 4])
            final_ids.append(cat[keep, 5].long())

        max_dets = max(b.shape[0] for b in final_boxes)
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
            n = final_boxes[i].shape[0]
            pb[i, :n] = final_boxes[i]
            ps[i, :n] = final_scores[i]
            pi[i, :n] = final_ids[i]
        return pb, ps, pi

    @staticmethod
    def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(-1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            order = order[1:][iou <= iou_thresh]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def get_all_convs(self) -> List[Tuple[str, nn.Conv2d]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, nn.Conv2d)]

    def get_all_linears(self) -> List[Tuple[str, nn.Linear]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, nn.Linear)]


class DETRWrapper(BaseTrafficDetector):
    """Wrapper around torchvision DETR-ResNet-50 with custom number of classes.

    Loads COCO pretrained weights and replaces the classification head for
    UA-DETRAC classes.  The backbone and transformer are left intact,
    making the model suitable for pruning experiments.
    """

    COCO_VEHICLE_MAP = {
        3: 0,   # car (COCO 91-class index)
        6: 1,   # bus
        8: 2,   # truck → van
        4: 3,   # motorcycle → others
    }

    def __init__(self, num_classes: int = NUM_UA_DETRAC_CLASSES):
        super().__init__(num_classes)
        self.img_size = 800

        try:
            import torchvision
            logger.info("Loading pretrained DETR-ResNet-50 ...")
            self.model = torchvision.models.detection.detr_resnet50(pretrained=True)
            self._adapt_head(num_classes)
            logger.info(f"DETR-ResNet-50 loaded ({self.get_params_count():,} params)")
        except Exception as e:
            raise RuntimeError(f"Failed to load DETR-ResNet-50: {e}")

    def _adapt_head(self, num_classes: int):
        in_feat = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_feat, num_classes + 1)  # +1 background
        self.model.num_classes = num_classes + 1

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig = self.model(x)
        B = x.shape[0]
        pb = torch.zeros(B, 0, 4, device=x.device)
        ps = torch.zeros(B, 0, device=x.device)
        pi = torch.zeros(B, 0, dtype=torch.long, device=x.device)
        out_boxes, out_scores, out_ids = [], [], []
        for b in range(B):
            logits = orig[b]["pred_logits"]     # [N_q, C+1]
            boxes = orig[b]["pred_boxes"]        # [N_q, 4]  cxcywh [0,1]
            prob = logits.softmax(-1)
            scores, ids = prob[:, :-1].max(-1)
            keep = scores > 0.05
            if keep.any():
                cx, cy, w, h = boxes[keep].unbind(-1)
                x1 = (cx - w / 2).clamp(0, 1)
                y1 = (cy - h / 2).clamp(0, 1)
                x2 = (cx + w / 2).clamp(0, 1)
                y2 = (cy + h / 2).clamp(0, 1)
                out_boxes.append(torch.stack([x1, y1, x2, y2], dim=-1))
                out_scores.append(scores[keep])
                out_ids.append(ids[keep])
            else:
                out_boxes.append(torch.zeros(0, 4, device=x.device))
                out_scores.append(torch.zeros(0, device=x.device))
                out_ids.append(torch.zeros(0, dtype=torch.long, device=x.device))
        max_dets = max(b.shape[0] for b in out_boxes)
        if max_dets > 0:
            pb = torch.zeros(B, max_dets, 4, device=x.device)
            ps = torch.zeros(B, max_dets, device=x.device)
            pi = torch.zeros(B, max_dets, dtype=torch.long, device=x.device)
            for i in range(B):
                n = out_boxes[i].shape[0]
                pb[i, :n] = out_boxes[i]
                ps[i, :n] = out_scores[i]
                pi[i, :n] = out_ids[i]
        return {"boxes": pb, "scores": ps, "class_ids": pi}

    def get_all_convs(self) -> List[Tuple[str, nn.Conv2d]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, nn.Conv2d)]

    def get_all_linears(self) -> List[Tuple[str, nn.Linear]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, nn.Linear)]


def get_model(model_name: str, num_classes: int = NUM_UA_DETRAC_CLASSES, **kwargs) -> BaseTrafficDetector:
    name = model_name.lower().replace("_", "").replace("-", "")
    if "yolov5" in name or "yolo" in name:
        size = kwargs.get("model_size", None)
        if size is None:
            for s in ("x", "l", "m", "s", "n"):
                if s in name:
                    size = s
                    break
        if size is None:
            size = "s"
        return YOLOWrapper(num_classes=num_classes, model_size=size)
    if "detr" in name:
        return DETRWrapper(num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}. Options: yolov5, detr")


YOLOv5Detector = YOLOWrapper
DETRDetector = DETRWrapper
