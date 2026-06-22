import torch
import torch.nn as nn
import logging
import copy
from .base import BasePruner, register_pruner

logger = logging.getLogger("LayerPruner")


def get_c3_bottlenecks(model: nn.Module):
    """Locates C3 blocks and returns metadata and calculated bottleneck importance scores."""
    bottlenecks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'C3':
            if hasattr(module, 'm') and isinstance(module.m, nn.Sequential):
                for idx, bottleneck in enumerate(module.m):
                    if bottleneck.__class__.__name__ == 'Bottleneck':
                        if hasattr(bottleneck, 'cv2'):
                            conv = bottleneck.cv2.conv
                            # Calculate importance score using Conv weight magnitude
                            conv_weight = conv.weight.data.abs().cpu()
                            weight_alpha = torch.mean(conv_weight.view(conv_weight.size(0), -1), dim=1)
                            
                            # Combine with BN weights if BatchNorm is present (not fused)
                            if hasattr(bottleneck.cv2, 'bn') and bottleneck.cv2.bn is not None:
                                bn = bottleneck.cv2.bn
                                bn_weight = bn.weight.data.abs().cpu()
                                importance_tensor = 10 * weight_alpha * bn_weight
                            else:
                                importance_tensor = weight_alpha
                                
                            mean_importance = torch.mean(importance_tensor).item()
                            bottlenecks.append({
                                'c3_name': name,
                                'c3_module': module,
                                'bottleneck_idx': idx,
                                'bottleneck_module': bottleneck,
                                'importance': mean_importance
                            })
    return bottlenecks


def prune_yolov5_c3_layers(model: nn.Module, num_to_prune: int) -> nn.Module:
    """Prunes C3 bottleneck blocks dynamically to reduce depth safely."""
    model = copy.deepcopy(model)
    bottlenecks = get_c3_bottlenecks(model)
    total_bottlenecks = len(bottlenecks)
    
    if total_bottlenecks == 0:
        logger.warning("No C3 bottlenecks found for layer pruning.")
        return model

    if num_to_prune >= total_bottlenecks:
        num_to_prune = total_bottlenecks - 1

    # Sort bottlenecks ascending (least important first)
    sorted_bottlenecks = sorted(bottlenecks, key=lambda x: x['importance'])
    pruned_info = sorted_bottlenecks[:num_to_prune]

    for item in pruned_info:
        logger.info(f"  Selecting C3 Block '{item['c3_name']}' Bottleneck {item['bottleneck_idx']} (Importance: {item['importance']:.6f}) for removal")

    # Map kept bottlenecks to their parent modules
    c3_kept_map = {}
    for item in bottlenecks:
        is_pruned = any(
            p['c3_name'] == item['c3_name'] and p['bottleneck_idx'] == item['bottleneck_idx'] 
            for p in pruned_info
        )
        if not is_pruned:
            if item['c3_name'] not in c3_kept_map:
                c3_kept_map[item['c3_name']] = []
            c3_kept_map[item['c3_name']].append(item['bottleneck_module'])

    # Apply structural updates in-place on copies
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'C3':
            kept_modules = c3_kept_map.get(name, [])
            module.m = nn.Sequential(*kept_modules)
            logger.info(f"  Layer Pruning: Updated C3 Block '{name}' | Remaining Bottlenecks: {len(kept_modules)}")
            
    return model


def prune_detr_encoder_layers(model: nn.Module, num_to_prune: int) -> nn.Module:
    """Prunes DETR encoder layers dynamically to reduce model depth safely."""
    model = copy.deepcopy(model)
    
    # Locate transformer encoder
    transformer = None
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Transformer':
            transformer = module
            break
            
    if transformer is None or not hasattr(transformer, "encoder"):
        logger.warning("DETR Transformer encoder module not found.")
        return model
        
    encoder = transformer.encoder
    layers = encoder.layers
    total_layers = len(layers)
    
    if num_to_prune >= total_layers:
        num_to_prune = total_layers - 1
        
    # Remove the last num_to_prune layers (since dimension shapes match, this is shape-safe)
    kept_layers = layers[:total_layers - num_to_prune]
    encoder.layers = nn.ModuleList(kept_layers)
    logger.info(f"  Layer Pruning: Updated DETR Encoder | Remaining Layers: {len(kept_layers)} (Pruned: {num_to_prune})")
    
    return model


def replace_layer_with_identity(model: nn.Module, layer_path: str) -> nn.Module:
    """Replaces a specific named layer in the model with nn.Identity.

    WARNING: May cause channel dimension mismatches unless input/output dimensions align.
    """
    model = copy.deepcopy(model)
    parts = layer_path.split(".")
    curr_mod = model
    for part in parts[:-1]:
        if part.isdigit():
            curr_mod = curr_mod[int(part)]
        else:
            curr_mod = getattr(curr_mod, part)
            
    target_part = parts[-1]
    if target_part.isdigit():
        curr_mod[int(target_part)] = nn.Identity()
    else:
        setattr(curr_mod, target_part, nn.Identity())
        
    logger.info(f"  Layer Pruning: Layer '{layer_path}' replaced with nn.Identity")
    return model


@register_pruner("layer")
class LayerPruner(BasePruner):
    """Depth-wise structured layer pruner."""

    def __init__(self, model: nn.Module, pruning_ratio: float = 0.0, layer_path: str = ""):
        super().__init__(model, pruning_ratio)
        self.layer_path = layer_path

    def prune(self) -> nn.Module:
        if self.layer_path:
            logger.info(f"Replacing layer '{self.layer_path}' with nn.Identity...")
            self.model = replace_layer_with_identity(self.model, self.layer_path)
            return self.model

        # Otherwise, dynamically remove C3 blocks or Transformer layers based on ratio
        num_layers_to_prune = max(1, int(self.pruning_ratio * 10.0))
        logger.info(f"Applying Depth-wise Layer Pruning (removing {num_layers_to_prune} layers)...")

        model_class_name = self.model.__class__.__name__
        if "YOLO" in model_class_name:
            self.model = prune_yolov5_c3_layers(self.model, num_layers_to_prune)
        elif "DETR" in model_class_name:
            self.model = prune_detr_encoder_layers(self.model, num_layers_to_prune)
        else:
            logger.warning(f"Unknown model class {model_class_name} for dynamic layer removal.")

        return self.model
