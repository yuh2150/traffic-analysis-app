import torch
import torch.nn as nn
import logging
from .base import BasePruner, register_pruner

logger = logging.getLogger("FilterPruner")


@register_pruner("filter")
class FilterPruner(BasePruner):
    """Structured mean absolute magnitude filter pruner for Conv2d layers."""

    def prune(self) -> nn.Module:
        if self.pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0.0; skipping filter pruning.")
            return self.model

        logger.info(f"Applying Filter Pruning with ratio = {self.pruning_ratio*100:.1f}%...")

        for name, module in self.discover_prunable_layers():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                num_filters = w.size(0)
                
                # Compute mean absolute weight of each filter
                filter_means = w.abs().mean(dim=[1, 2, 3])
                
                num_to_prune = int(num_filters * self.pruning_ratio)
                if num_to_prune == 0:
                    continue

                threshold = torch.topk(filter_means, num_to_prune, largest=False).values[-1].item()
                mask = filter_means >= threshold
                mask_expanded = mask.view(-1, 1, 1, 1).to(w.device).float()
                
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
                logger.info(f"  Layer {name:40s} | Pruned {num_to_prune}/{num_filters} filters by mean magnitude")

        logger.info(f"Filter pruning applied. Current model weight sparsity: {self.calculate_sparsity()*100:.2f}%")
        return self.model
