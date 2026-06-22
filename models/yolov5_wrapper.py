import sys
import os
import copy
import logging
import torch
import torch.nn as nn
import types
from importlib.machinery import ModuleSpec
from typing import Tuple, Dict, Any, Union

logger = logging.getLogger("Models")

UA_DETRAC_CLASSES = ["car", "van", "bus", "others"]
NUM_UA_DETRAC_CLASSES = len(UA_DETRAC_CLASSES)


def _install_yolov5_import_shims() -> list:
    """Install minimal optional-dependency shims needed by YOLOv5 hub imports."""
    created_modules = []
    force_plot_shims = False
    try:
        import numpy as _np
        force_plot_shims = int(_np.__version__.split(".", 1)[0]) >= 2
    except Exception:
        force_plot_shims = False

    def _clear_modules(prefix: str):
        for module_name in list(sys.modules):
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                sys.modules.pop(module_name, None)

    try:
        if force_plot_shims:
            raise ImportError("using pandas shim with NumPy 2.x")
        import pandas  # noqa: F401
    except Exception:
        _clear_modules("pandas")
        pandas_module = types.ModuleType("pandas")
        pandas_module.__file__ = "<yolov5_pandas_shim>"
        pandas_module.__spec__ = ModuleSpec("pandas", loader=None)

        class _FakeDataFrame:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        def _read_csv(*args, **kwargs):
            raise ImportError("pandas is unavailable in this environment")

        pandas_module.options = types.SimpleNamespace(display=types.SimpleNamespace(max_columns=None))
        pandas_module.DataFrame = _FakeDataFrame
        pandas_module.read_csv = _read_csv
        sys.modules["pandas"] = pandas_module
        created_modules.append("pandas")

    try:
        if force_plot_shims:
            raise ImportError("using matplotlib shim with NumPy 2.x")
        import matplotlib  # noqa: F401
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        _clear_modules("matplotlib")
        matplotlib_module = types.ModuleType("matplotlib")
        pyplot_module = types.ModuleType("matplotlib.pyplot")
        matplotlib_module.__file__ = "<yolov5_matplotlib_shim>"
        pyplot_module.__file__ = "<yolov5_pyplot_shim>"
        matplotlib_module.__spec__ = ModuleSpec("matplotlib", loader=None)
        pyplot_module.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)
        matplotlib_module.rc = lambda *args, **kwargs: None
        matplotlib_module.use = lambda *args, **kwargs: None
        matplotlib_module.colors = types.SimpleNamespace(TABLEAU_COLORS={})

        def _noop(*args, **kwargs):
            return None

        def _pyplot_getattr(name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _noop

        pyplot_module.subplots = lambda *args, **kwargs: (None, types.SimpleNamespace(ravel=lambda: []))
        pyplot_module.__getattr__ = _pyplot_getattr
        sys.modules["matplotlib"] = matplotlib_module
        sys.modules["matplotlib.pyplot"] = pyplot_module
        created_modules.extend(["matplotlib.pyplot", "matplotlib"])

    try:
        if force_plot_shims:
            raise ImportError("using seaborn shim with NumPy 2.x")
        import seaborn  # noqa: F401
    except Exception:
        _clear_modules("seaborn")
        seaborn_module = types.ModuleType("seaborn")
        seaborn_module.__file__ = "<yolov5_seaborn_shim>"
        seaborn_module.__spec__ = ModuleSpec("seaborn", loader=None)
        seaborn_module.histplot = lambda *args, **kwargs: None
        seaborn_module.pairplot = lambda *args, **kwargs: None
        sys.modules["seaborn"] = seaborn_module
        created_modules.append("seaborn")

    return created_modules


class BaseTrafficDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_UA_DETRAC_CLASSES):
        super().__init__()
        self.num_classes = num_classes

    def get_params_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def calculate_flops(self, input_size: Tuple[int, int, int] = (3, 640, 640)) -> int:
        """Estimates model FLOPs using thop or fvcore."""
        try:
            from thop import profile
            x = torch.zeros(1, *input_size, device=next(self.parameters()).device)
            macs, _ = profile(self, inputs=(x,), verbose=False)
            return int(macs * 2)
        except Exception:
            try:
                from fvcore.nn import FlopCountAnalysis
                x = torch.zeros(1, *input_size, device=next(self.parameters()).device)
                flops = FlopCountAnalysis(self, x)
                return flops.total()
            except Exception:
                # Direct analytical approximation fallback
                logger.warning("Neither thop nor fvcore profiled successfully. Using fallback estimation.")
                active_params = sum(p.numel() for p in self.parameters() if p.requires_grad and (p != 0).any())
                return int(active_params * 2 * 10)


class YOLOv5Wrapper(BaseTrafficDetector):
    """Wrapper around YOLOv5s for UA-DETRAC dataset with support for magnitude pruning.

    This class supports training mode with proper gradient propagation and decoding.
    """

    def __init__(self, num_classes: int = 80, model_size: str = "s"):
        super().__init__(num_classes)
        self.model_size = model_size
        self.img_size = 640

        # Save and restore path to prevent import issues in Ultralytics YOLOv5 Hub loading
        saved_path = sys.path.copy()
        saved_models_module = sys.modules.get('models')
        saved_utils_module = sys.modules.get('utils')
        saved_torch_load = torch.load
        created_shim_modules = _install_yolov5_import_shims()
        if 'models' in sys.modules:
            del sys.modules['models']
        if 'utils' in sys.modules:
            del sys.modules['utils']

        try:
            try:
                import ultralytics.utils.patches  # noqa: F401
            except ModuleNotFoundError:
                ultralytics_module = types.ModuleType("ultralytics")
                ultralytics_utils_module = types.ModuleType("ultralytics.utils")
                ultralytics_patches_module = types.ModuleType("ultralytics.utils.patches")
                ultralytics_patches_module.torch_load = torch.load
                sys.modules["ultralytics"] = ultralytics_module
                sys.modules["ultralytics.utils"] = ultralytics_utils_module
                sys.modules["ultralytics.utils.patches"] = ultralytics_patches_module
                created_shim_modules.extend([
                    "ultralytics.utils.patches",
                    "ultralytics.utils",
                    "ultralytics",
                ])

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            sys.path = [
                p for p in sys.path
                if p not in ('', os.getcwd(), os.path.abspath('.'), project_root, os.path.dirname(os.path.abspath(__file__)))
                and not p.startswith(project_root)
            ]
            ckpt_name = f"yolov5{model_size}"
            logger.info(f"Loading pretrained {ckpt_name} via torch.hub (YOLOv5 v7.0)...")

            def _torch_load_compat(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return saved_torch_load(*args, **kwargs)

            torch.load = _torch_load_compat
            yolo = torch.hub.load("ultralytics/yolov5:v7.0", ckpt_name, pretrained=True, autoshape=False)
            self.model = copy.deepcopy(yolo)
            
            # Unfreeze all parameters (backbone and neck) since torch.hub loads them frozen
            for param in self.model.parameters():
                param.requires_grad = True
                
            self._adapt_head(num_classes)
            logger.info(f"Successfully loaded and adapted YOLOv5{model_size} ({self.get_params_count():,} params)")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5: {e}")
        finally:
            torch.load = saved_torch_load
            sys.path = saved_path
            if saved_models_module is not None:
                sys.modules['models'] = saved_models_module
            if saved_utils_module is not None:
                sys.modules['utils'] = saved_utils_module
            for module_name in created_shim_modules:
                sys.modules.pop(module_name, None)

    def _get_detect(self):
        """Locates the final Detect module in YOLOv5."""
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
        """Replace detection head convs to output class prediction shapes."""
        detect = self._get_detect()
        if not hasattr(detect, "m"):
            logger.warning("Detect module has no 'm' attribute; head adaptation skipped")
            return
        
        # Preserving pretrained COCO weights if class count matches
        if detect.nc == num_classes:
            logger.info(f"Detect head classes already matches {num_classes}. Preserving pretrained head weights.")
            return
            
        detect.nc = num_classes
        detect.no = 5 + num_classes
        for i in range(len(detect.m)):
            old_conv = detect.m[i]
            na = old_conv.out_channels // (5 + 80)  # COCO na is typically 3
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                na * (5 + num_classes),
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Copy existing weights for matching classes if possible, or initialize
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                new_conv.bias.data.zero_()
            detect.m[i] = new_conv

    def decode_predictions(self, raw_outputs: list) -> torch.Tensor:
        """Decodes multi-scale raw outputs into absolute coordinate predictions.

        Returns a single tensor of shape [B, 25200, 5 + num_classes].
        """
        detect = self._get_detect()
        strides = detect.stride
        anchors = detect.anchors * strides.view(-1, 1, 1)  # pixel space anchors [3, 3, 2]

        decoded_list = []
        for i, x in enumerate(raw_outputs):
            # x has shape [B, na, ny, nx, no]
            B, na, ny, nx, no = x.shape
            device = x.device

            # Create grid using meshgrid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(ny, device=device),
                torch.arange(nx, device=device),
                indexing="ij"
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, ny, nx, 2).float()

            # Sigmoid decoding matching YOLOv5 formula
            xy = (torch.sigmoid(x[..., 0:2]) * 2.0 - 0.5 + grid) * strides[i]
            wh = (torch.sigmoid(x[..., 2:4]) * 2.0) ** 2 * anchors[i].view(1, na, 1, 1, 2)

            conf = torch.sigmoid(x[..., 4:5])
            cls_prob = torch.sigmoid(x[..., 5:])

            decoded_scale = torch.cat((xy, wh, conf, cls_prob), dim=-1)
            decoded_list.append(decoded_scale.view(B, -1, no))

        return torch.cat(decoded_list, dim=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        During training, returns decoded predictions with gradients.
        During evaluation, performs NMS and returns boxes, scores, and class_ids.
        """
        if self.training:
            # Under train() mode, self.model(x) returns a list of scale predictions [p0, p1, p2]
            raw = self.model(x)
            return self.decode_predictions(raw)
        else:
            # Under eval() mode, self.model(x) returns (decoded_preds, raw_preds)
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
        """Decode YOLOv5 outputs and apply Non-Maximum Suppression (NMS)."""
        device = decoded.device
        B = decoded.shape[0]
        img_h, img_w = img_shape
        dets_list = [[] for _ in range(B)]

        for b in range(B):
            preds = decoded[b]
            obj = preds[:, 4:5]
            cls = preds[:, 5:]
            scores_i, ids_i = (obj * cls).max(dim=1)
            
            # Pre-filter by confidence score threshold
            conf_mask = scores_i > 0.05
            if not conf_mask.any():
                dets_list[b].append(torch.zeros(0, 6, device=device))
                continue
                
            xc = preds[conf_mask, 0]
            yc = preds[conf_mask, 1]
            w = preds[conf_mask, 2]
            h = preds[conf_mask, 3]
            scores_i = scores_i[conf_mask]
            ids_i = ids_i[conf_mask]
            
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
            # NMS IoU threshold 0.5, score threshold 0.05
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
