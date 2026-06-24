import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from models.factory import extract_model_state_dict

logger = logging.getLogger("PipelineUtils")


def calculate_actual_sparsity(model: nn.Module) -> float:
    """Calculates actual sparsity of Conv2d and Linear layers."""
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

