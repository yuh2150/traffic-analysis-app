import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
from .base import BasePruner, register_pruner

logger = logging.getLogger("MagnitudePruner")


def get_detr_layer_sparsity(name: str, S: float) -> float:
    """Determines the custom sparsity ratio for DETR layers to prevent model collapse."""
    # 1. Protect prediction heads completely
    if "class_embed" in name or "bbox_embed" in name:
        return 0.0
        
    # 2. Protect critical backbone layers
    if "backbone.0.body.conv1" in name or "downsample" in name:
        return 0.0
        
    # 3. Apply block-wise scaling
    if "backbone" in name:
        return min(0.65, S * 0.8)
    elif "transformer.encoder" in name:
        return min(0.75, S * 0.9)
    elif "transformer.decoder" in name:
        return min(0.60, S * 0.7)
        
    # 4. Fallback/other layers (e.g. input_proj)
    return min(0.65, S * 0.8)


def get_yolov5_layer_sparsity(name: str, S: float) -> float:
    """Determines the custom sparsity ratio for YOLOv5 layers, protecting prediction heads."""
    # Protect detection head (Detect module, layer 24)
    if "model.24" in name or "model.model.24" in name:
        return 0.0
    return S


@register_pruner("magnitude")
class MagnitudePruner(BasePruner):
    """Layer-wise unstructured magnitude pruner using torch.nn.utils.prune.
    
    This implementation applies custom, block-wise sparsity ratios for DETR and YOLOv5s
    to prevent model collapse at high target sparsities, while protecting critical prediction heads.
    """

    def prune(self) -> nn.Module:
        if self.pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0.0; skipping magnitude pruning.")
            return self.model

        logger.info(f"Applying custom layer-wise Magnitude Pruning with target sparsity = {self.pruning_ratio*100:.1f}%...")

        # Remove any existing pruning masks to avoid double masking
        self._remove_all_pruning_masks()

        layers = self.discover_prunable_layers()
        if not layers:
            logger.warning("No prunable layers found.")
            return self.model

        # Detect model architecture type
        raw_model = self.model.get_raw_model() if hasattr(self.model, "get_raw_model") else self.model
        model_name = type(raw_model).__name__.lower()
        is_detr = "detr" in model_name or hasattr(raw_model, "transformer")
        is_yolo = "yolo" in model_name or "detectionmodel" in model_name or hasattr(raw_model, "model")

        logger.info(f"Model signature detected: {'DETR' if is_detr else 'YOLOv5' if is_yolo else 'Generic'}")

        pruned_count = 0
        for name, module in layers:
            if is_detr:
                ratio = get_detr_layer_sparsity(name, self.pruning_ratio)
            elif is_yolo:
                ratio = get_yolov5_layer_sparsity(name, self.pruning_ratio)
            else:
                # Generic model: protect the last classification layer
                if name == layers[-1][0]:
                    ratio = 0.0
                else:
                    ratio = self.pruning_ratio

            if ratio > 0.0:
                prune.l1_unstructured(module, name='weight', amount=ratio)
                # Enforce the mask immediately in the weights tensor
                if hasattr(module, 'weight_mask'):
                    module.weight.data.mul_(module.weight_mask)
                pruned_count += 1
                logger.debug(f"Pruned layer '{name}' to sparsity: {ratio*100:.1f}%")
            else:
                logger.info(f"Layer '{name}' is protected from pruning (0.0% sparsity).")

        # Log the achieved sparsity
        actual_sparsity = self.calculate_sparsity()
        logger.info(f"Layer-wise pruning applied to {pruned_count}/{len(layers)} Conv/Linear layers. Realized Model Sparsity: {actual_sparsity*100:.2f}%")
        return self.model

    def _remove_all_pruning_masks(self):
        """Remove previously applied pruning masks from all modules."""
        for module in self.model.modules():
            if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                try:
                    prune.remove(module, 'weight')
                except Exception as e:
                    logger.warning(f"Could not remove pruning from {module}: {e}")

    def calculate_sparsity(self) -> float:
        """Compute the fraction of zero weights in Conv2d and Linear layers of the model."""
        total_elements = 0
        zero_elements = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # If weights are pruned via PyTorch hooks, module.weight already reflects the mask
                w = module.weight
                total_elements += w.numel()
                zero_elements += (w == 0.0).sum().item()
        if total_elements == 0:
            return 0.0
        return zero_elements / total_elements

    def apply_mask(self, module):
        pass

    def register_mask(self, module, mask):
        pass