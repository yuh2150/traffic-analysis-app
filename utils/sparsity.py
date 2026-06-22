import torch
import torch.nn as nn
import logging

logger = logging.getLogger("Sparsity")


def calculate_model_sparsity(model: nn.Module) -> float:
    """Calculates the global weight sparsity of the Conv2d and Linear layers in the model."""
    total_weights = 0
    zero_weights = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total_weights += w.numel()
            zero_weights += (w == 0.0).sum().item()

    return zero_weights / total_weights if total_weights > 0 else 0.0


def print_layer_sparsity(model: nn.Module):
    """Logs the sparsity of each Conv2d and Linear layer in the model."""
    logger.info("=== Layer-wise Sparsity Report ===")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total = w.numel()
            zeros = (w == 0.0).sum().item()
            sparsity = zeros / total if total > 0 else 0.0
            logger.info(f"Layer: {name:40s} | Shape: {str(list(w.shape)):20s} | Sparsity: {sparsity*100:6.2f}%")
    logger.info("==================================")
