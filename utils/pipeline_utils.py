import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

logger = logging.getLogger("PipelineUtils")


def calculate_actual_sparsity(model: nn.Module) -> float:
    """Calculates actual sparsity of Conv2d and Linear layers, supporting structured pruning."""
    is_structured = getattr(model, "is_structured_pruned", False)
    model_name = model.__class__.__name__.lower()
    total_params = sum(p.numel() for p in model.parameters())
    
    if "yolo" in model_name:
        base_params = 7225880  # YOLOv5s baseline params
        if total_params < 7000000 or is_structured:
            return 1.0 - (total_params / base_params)
    elif "detr" in model_name:
        base_params = 41524768  # DETR baseline params
        if total_params < 40000000 or is_structured:
            return 1.0 - (total_params / base_params)

    total_weights = 0
    zero_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total_weights += w.numel()
            zero_weights += (w == 0.0).sum().item()
    return zero_weights / total_weights if total_weights > 0 else 0.0


def load_checkpoint_weights(model: nn.Module, checkpoint_path: str, device: torch.device, logger=logger) -> Tuple[list, list]:
    """Loads weights from checkpoint into the model, logging matched/missing/unexpected keys."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Import locally to avoid circular import issues
    from models.factory import extract_model_state_dict
    state_dict = extract_model_state_dict(checkpoint)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    logger.info(f"Loaded weights from {checkpoint_path}:")
    logger.info(f" - Matched keys: {len(state_dict) - len(unexpected_keys)}")
    logger.info(f" - Missing keys: {len(missing_keys)}")
    logger.info(f" - Unexpected keys: {len(unexpected_keys)}")
    
    return missing_keys, unexpected_keys


def bake_pruned_weights(model: nn.Module) -> int:
    """Fuses the pruning masks permanently into the weights and removes forward hooks.
    
    Returns the number of modules updated.
    """
    import torch.nn.utils.prune as prune
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            try:
                prune.remove(module, 'weight')
                count += 1
            except Exception as e:
                logger.warning(f"Could not remove pruning hook from weight of module {name}: {e}")
        if hasattr(module, 'bias_orig'):
            try:
                prune.remove(module, 'bias')
                count += 1
            except Exception as e:
                logger.warning(f"Could not remove pruning hook from bias of module {name}: {e}")
    return count


def get_clean_state_dict(model: nn.Module) -> dict:
    """Returns a state_dict of the model where all pruning masks are baked, without modifying the input model."""
    import copy
    
    # Try deep copying the model to bake weights without modifying the active training model
    try:
        model_copy = copy.deepcopy(model)
        bake_pruned_weights(model_copy)
        state_dict = model_copy.state_dict()
    except Exception as e:
        logger.warning(f"Failed to deepcopy model for state_dict baking: {e}. Falling back to manual state_dict cleaning.")
        state_dict = model.state_dict()
        clean_sd = {}
        # Find all keys that end with .weight_orig or .bias_orig
        orig_keys = [k for k in state_dict.keys() if k.endswith(".weight_orig") or k.endswith(".bias_orig")]
        if not orig_keys:
            return {k: v for k, v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
            
        for k, v in state_dict.items():
            if k.endswith("_orig"):
                prefix = k[:-5] # remove '_orig'
                mask_key = prefix + "_mask"
                if mask_key in state_dict:
                    clean_sd[prefix] = v * state_dict[mask_key]
                else:
                    clean_sd[prefix] = v
            elif k.endswith("_mask"):
                continue
            else:
                clean_sd[k] = v
        return {k: v for k, v in clean_sd.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
        
    # Strip thop profile keys
    clean_state_dict = {k: v for k, v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
    return clean_state_dict


def clean_state_dict_on_the_fly(state_dict: dict) -> dict:
    """Converts a state_dict containing weight_orig and weight_mask into a clean state_dict with normal weight parameters."""
    clean_sd = {}
    orig_keys = [k for k in state_dict.keys() if k.endswith(".weight_orig") or k.endswith(".bias_orig")]
    if not orig_keys:
        return state_dict

    # We need to construct the baked weights
    for k, v in state_dict.items():
        if k.endswith("_orig"):
            prefix = k[:-5] # remove '_orig'
            mask_key = prefix + "_mask"
            if mask_key in state_dict:
                clean_sd[prefix] = v * state_dict[mask_key]
            else:
                clean_sd[prefix] = v
        elif k.endswith("_mask"):
            continue
        else:
            clean_sd[k] = v
            
    # Strip thop profile keys if present
    clean_sd = {k: v for k, v in clean_sd.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
    return clean_sd
