"""
Hard Structured L1-Norm Filter Pruner using Torch-Pruning.

Instead of masking weights (soft pruning), this pruner actually removes
Conv2d filters from the architecture, updates downstream layers accordingly,
and produces a truly smaller model with fewer parameters and FLOPs.

Algorithm
---------
1. For each Conv2d layer, compute L1-norm per filter:  score_i = ||W_i||_1
2. Select the filters with the smallest scores at the given pruning ratio
3. Remove those filters from the Conv2d (out_channels reduction)
4. Automatically update:
   - BatchNorm channels after the Conv2d
   - Input channels of the next Conv2d layer
   - Any other downstream layers connected via the computation graph
5. Handle residual / skip connections via Torch-Pruning's dependency graph

Layer protection
----------------
- YOLOv5: Detect head layers are excluded from pruning
- DETR:   Only the backbone (ResNet) is pruned; transformer, heads, input_proj
          are preserved
- Custom: Users can pass ignored_layers to exclude any modules

No masks, no hooks, no weight-zeroing. The model architecture is truly modified.
ONNX export works out of the box since there are no pruning-specific artifacts.
"""

import torch
import torch.nn as nn
import torch_pruning as tp
import logging
from typing import Dict, Any, Sequence, List, Optional, Tuple

from .base import BasePruner, register_pruner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom pruner for FrozenBatchNorm2d (used in DETR's ResNet backbone)
# ---------------------------------------------------------------------------

class FrozenBatchNormPruner(tp.pruner.BasePruningFunc):
    """Torch-Pruning does not natively support FrozenBatchNorm2d.

    This custom pruner handles channel pruning for DETR's frozen batch norm
    layers by trimming weight, bias, running_mean, and running_var.
    """

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep = sorted(set(range(layer.weight.shape[0])) - set(idxs))
        for attr in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(layer, attr, None)
            if t is not None:
                new = t.data[keep].clone()
                if isinstance(t, nn.Parameter):
                    new = nn.Parameter(new)
                setattr(layer, attr, new)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.weight.shape[0]

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.weight.shape[0]


# ---------------------------------------------------------------------------
# FLOPs estimation (standalone, no dependency on BaseTrafficDetector)
# ---------------------------------------------------------------------------

def _estimate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """Return total FLOPs using thop (preferred) or fvcore (fallback).

    Cleans up any profiling buffers (``total_ops``, ``total_params``) that
    thop may leave behind on the model.
    """
    device = next(model.parameters()).device
    flops = 0
    try:
        from thop import profile
        x = torch.zeros(1, *input_size, device=device)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        flops = int(macs * 2)
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            x = torch.zeros(1, *input_size, device=device)
            flops = FlopCountAnalysis(model, x).total()
        except Exception:
            pass
    finally:
        # thop registers total_ops / total_params as buffers; remove them
        for hook in list(model._forward_pre_hooks.values()):
            if hasattr(hook, "fh"):
                model._forward_pre_hooks = {
                    k: v for k, v in model._forward_pre_hooks.items() if v is not hook
                }
        for key in ("total_ops", "total_params"):
            if hasattr(model, key):
                delattr(model, key)
    return flops


# ---------------------------------------------------------------------------
# Main pruner
# ---------------------------------------------------------------------------

@register_pruner("filter")
class FilterPruner(BasePruner):
    """Hard structured L1-norm filter pruner backed by Torch-Pruning.

    Parameters
    ----------
    model : nn.Module
        The model to prune.
    pruning_ratio : float
        Fraction of filters to remove globally (0.0 = no pruning).
    example_inputs : torch.Tensor, optional
        Example input for Torch-Pruning's graph tracing.  Auto-created if
        ``None`` (size 640 for YOLO / generic, 800 for DETR).
    ignored_layers : list[nn.Module], optional
        Additional modules that must **not** be pruned.
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.0,
        example_inputs: Optional[torch.Tensor] = None,
        ignored_layers: Optional[List[nn.Module]] = None,
    ):
        super().__init__(model, pruning_ratio)
        self._example_inputs = example_inputs
        self._ignored_layers = list(ignored_layers or [])
        self.base_params = sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inner_model(self) -> Tuple[nn.Module, str]:
        """Unwrap the model that Torch-Pruning should trace.

        Returns
        -------
        (inner_module, model_type)
            ``model_type`` is one of ``'yolo'``, ``'detr'``, or ``'raw'``.
        """
        name = self.model.__class__.__name__.lower()
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Module):
            inner = self.model.model
            if "yolo" in name:
                return inner, "yolo"
            elif "detr" in name:
                return inner, "detr"
            else:
                return inner, "wrapped"
        return self.model, "raw"

    def _make_example_inputs(self, model_type: str) -> torch.Tensor:
        if self._example_inputs is not None:
            return self._example_inputs
        device = next(self.model.parameters()).device
        size = 800 if model_type == "detr" else 640
        return torch.randn(1, 3, size, size, device=device)

    def _collect_ignored(self, inner: nn.Module, model_type: str) -> List[nn.Module]:
        """Build the full ignored-layers list.

        Starts from user-supplied layers, then adds model-specific protection.
        """
        ignored = list(self._ignored_layers)

        if model_type == "yolo":
            # Protect all children of the Detect head
            for m in inner.modules():
                if m.__class__.__name__ == "Detect":
                    ignored.append(m)
                    for child in m.modules():
                        if isinstance(child, nn.Conv2d):
                            ignored.append(child)

        elif model_type == "detr":
            # Protect everything outside the backbone (transformer, heads, …)
            for name, m in inner.named_modules():
                if not name.startswith("backbone") and not list(m.children()):
                    ignored.append(m)

        return ignored

    def _find_custom_pruners(self, inner: nn.Module) -> Dict:
        """Register custom pruners for non-standard layer types."""
        for m in inner.modules():
            if m.__class__.__name__ == "FrozenBatchNorm2d":
                return {m.__class__: FrozenBatchNormPruner()}
        return {}

    def _layer_log(self, inner: nn.Module, before: Dict[str, int]):
        """Log per-layer filter changes."""
        rows = []
        for name, m in inner.named_modules():
            if isinstance(m, nn.Conv2d) and name in before:
                orig = before[name]
                curr = m.out_channels
                if curr != orig:
                    rows.append((name, orig, curr, orig - curr, (orig - curr) / orig * 100))
        if not rows:
            return
        sep = "=" * 68
        logger.info(sep)
        logger.info(f"{'Layer':<44} {'Orig':>5} {'Now':>5} {'Δ':>5} {'Ratio':>6}")
        logger.info("-" * 68)
        for name, o, n, d, r in rows:
            logger.info(f"{name:<44} {o:>5} {n:>5} {d:>5} {r:>5.1f}%")
        logger.info(sep)

    def _summary_log(self, pb, pa, fb, fa, sb, sa, sp):
        """Log pruning summary."""
        sep = "=" * 56
        logger.info(sep)
        logger.info("  HARD STRUCTURED PRUNING COMPLETED")
        logger.info(sep)
        logger.info(f"  Parameters   {pb:>12,} -> {pa:<12,}  ({sp*100:.2f}% reduction)")
        if fb:
            logger.info(f"  FLOPs        {fb:>12,} -> {fa:<12,}  ({(fb - fa) / fb * 100:.2f}% reduction)")
        logger.info(f"  Model Size   {sb:>9.2f} MB -> {sa:<9.2f} MB")
        logger.info(sep)

    def _model_size_mb(self) -> float:
        p = sum(p.numel() * p.element_size() for p in self.model.parameters())
        b = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (p + b) / (1024 ** 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self) -> nn.Module:
        """Execute hard structured filter pruning.

        Returns the pruned model (modified **in-place**).
        """
        if self.pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0.0; returning model unchanged.")
            return self.model

        if getattr(self.model, "is_structured_pruned", False):
            logger.info("Model already structured-pruned; skipping.")
            return self.model

        logger.info(
            f"Hard structured filter pruning @ {self.pruning_ratio * 100:.1f}%"
        )

        # 1 -------- prepare -------------------------------------------------
        inner, model_type = self._inner_model()
        example_inputs = self._make_example_inputs(model_type)
        ignored_layers = self._collect_ignored(inner, model_type)
        custom_pruners = self._find_custom_pruners(inner)

        was_training = inner.training
        inner.train()  # Torch-Pruning requires train mode for autograd tracing

        # 2 -------- before stats --------------------------------------------
        params_before = sum(p.numel() for p in self.model.parameters())
        flops_before = _estimate_flops(self.model, tuple(example_inputs.shape[1:]))
        size_before = self._model_size_mb()

        conv_channels_before = {
            n: m.out_channels
            for n, m in inner.named_modules()
            if isinstance(m, nn.Conv2d)
        }

        # 3 -------- prune ---------------------------------------------------
        try:
            imp = tp.importance.MagnitudeImportance(p=1)  # L1-norm per filter
            pruner = tp.pruner.MagnitudePruner(
                inner,
                example_inputs=example_inputs,
                importance=imp,
                pruning_ratio=self.pruning_ratio,
                ignored_layers=ignored_layers,
                customized_pruners=custom_pruners,
            )
            pruner.step()
        except Exception as exc:
            logger.error(f"Torch-Pruning step failed: {exc}")
            raise
        finally:
            inner.train(was_training)

        # 4 -------- after stats ---------------------------------------------
        params_after = sum(p.numel() for p in self.model.parameters())
        flops_after = _estimate_flops(self.model, tuple(example_inputs.shape[1:]))
        size_after = self._model_size_mb()
        sparsity = 1.0 - (params_after / params_before) if params_before else 0.0

        # 5 -------- logging -------------------------------------------------
        self._layer_log(inner, conv_channels_before)
        self._summary_log(
            params_before, params_after,
            flops_before, flops_after,
            size_before, size_after,
            sparsity,
        )

        self.model.is_structured_pruned = True
        return self.model

    def collect_statistics(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.model.parameters())
        sp = 1.0 - (total / self.base_params) if self.base_params else 0.0
        return {
            "total_params": self.base_params,
            "active_params": total,
            "sparsity": sp,
            "size_mb": self._model_size_mb(),
        }
