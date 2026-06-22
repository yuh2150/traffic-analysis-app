import torch
import torch.nn as nn
import logging
from .base import BasePruner, register_pruner

logger = logging.getLogger("ChannelPruner")


@register_pruner("channel")
class ChannelPruner(BasePruner):
    """Structured channel pruner for Conv2d and Linear layers."""

    def prune(self) -> nn.Module:
        if self.pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0.0; skipping channel pruning.")
            return self.model

        logger.info(f"Applying Channel Pruning with ratio = {self.pruning_ratio*100:.1f}%...")

        for name, module in self.discover_prunable_layers():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                num_channels = w.size(1)
                
                # Compute norm along output channels and spatial dimensions [0, 2, 3]
                channel_norms = w.abs().sum(dim=[0, 2, 3])
                
                num_to_prune = int(num_channels * self.pruning_ratio)
                if num_to_prune == 0:
                    continue

                threshold = torch.topk(channel_norms, num_to_prune, largest=False).values[-1].item()
                mask = channel_norms >= threshold
                mask_expanded = mask.view(1, -1, 1, 1).to(w.device).float()
                
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
                logger.info(f"  Conv2d Layer {name:40s} | Pruned {num_to_prune}/{num_channels} channels")
                
            elif isinstance(module, nn.Linear):
                w = module.weight.data
                num_channels = w.size(1)
                
                # Compute norm along output features [0]
                channel_norms = w.abs().sum(dim=0)
                
                num_to_prune = int(num_channels * self.pruning_ratio)
                if num_to_prune == 0:
                    continue

                threshold = torch.topk(channel_norms, num_to_prune, largest=False).values[-1].item()
                mask = channel_norms >= threshold
                mask_expanded = mask.view(1, -1).to(w.device).float()
                
                self.register_mask(module, mask_expanded)
                self.apply_mask(module)
                logger.info(f"  Linear Layer {name:40s} | Pruned {num_to_prune}/{num_channels} channels")

        logger.info(f"Channel pruning applied. Current model weight sparsity: {self.calculate_sparsity()*100:.2f}%")
        return self.model
