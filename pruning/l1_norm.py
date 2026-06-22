import torch
import torch.nn as nn
import logging
from .base import BasePruner, register_pruner

logger = logging.getLogger("L1NormPruner")


@register_pruner("l1_norm")
class L1NormPruner(BasePruner):
    """Structured L1-norm filter pruner for Conv2d layers."""

    def prune(self) -> nn.Module:
        if self.pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0.0; skipping L1-norm filter pruning.")
            return self.model

        logger.info(f"Applying L1-Norm Filter Pruning with ratio = {self.pruning_ratio*100:.1f}%...")

        for name, module in self.discover_prunable_layers():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                num_filters = w.size(0)
                
                # Compute L1-norm across dimensions [1, 2, 3]
                l1_norms = w.abs().sum(dim=[1, 2, 3])
                
                num_to_prune = int(num_filters * self.pruning_ratio)
                if num_to_prune == 0:
                    continue

                threshold = torch.topk(l1_norms, num_to_prune, largest=False).values[-1].item()
                mask = l1_norms >= threshold
                mask_expanded = mask.view(-1, 1, 1, 1).to(w.device).float()
                
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
                logger.info(f"  Layer {name:40s} | Pruned {num_to_prune}/{num_filters} filters")

        logger.info(f"L1-Norm pruning applied. Current model weight sparsity: {self.calculate_sparsity()*100:.2f}%")
        return self.model
