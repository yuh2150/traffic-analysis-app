import torch
import torch.nn as nn
import logging
from .base import BasePruner, register_pruner

logger = logging.getLogger("MagnitudePruner")


@register_pruner("magnitude")
class MagnitudePruner(BasePruner):
    """Global unstructured magnitude pruner."""

    def prune(self) -> nn.Module:
        if self.pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0.0; skipping magnitude pruning.")
            return self.model

        logger.info(f"Applying Magnitude Pruning with ratio = {self.pruning_ratio*100:.1f}%...")

        all_weights = []
        layers = self.discover_prunable_layers()
        
        for name, module in layers:
            all_weights.append(module.weight.data.abs().view(-1))

        if not all_weights:
            logger.warning("No prunable layers found.")
            return self.model

        flat_weights = torch.cat(all_weights)
        threshold = torch.quantile(flat_weights, self.pruning_ratio).item()
        logger.info(f"Magnitude threshold: {threshold:.6f}")

        for name, module in layers:
            weight = module.weight.data
            mask = (weight.abs() > threshold).float()
            self.register_mask(module, mask)
            self.apply_mask(module)

        logger.info(f"Magnitude pruning applied. Current model weight sparsity: {self.calculate_sparsity()*100:.2f}%")
        return self.model
