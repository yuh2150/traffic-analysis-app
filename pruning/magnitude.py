import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
from .base import BasePruner, register_pruner

logger = logging.getLogger("MagnitudePruner")


@register_pruner("magnitude")
class MagnitudePruner(BasePruner):
    """Global unstructured magnitude pruner using torch.nn.utils.prune.

    This implementation ensures that sparsity is maintained during subsequent
    training (fine‑tuning) because the mask is applied via forward hooks.
    """

    def prune(self) -> nn.Module:
        if self.pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0.0; skipping magnitude pruning.")
            return self.model

        logger.info(f"Applying Magnitude Pruning with ratio = {self.pruning_ratio*100:.1f}%...")

        # Remove any existing pruning masks to avoid double masking
        self._remove_all_pruning_masks()

        layers = self.discover_prunable_layers()
        if not layers:
            logger.warning("No prunable layers found.")
            return self.model

        # Prepare the list of (module, param_name) for global pruning
        parameters_to_prune = [(module, 'weight') for _, module in layers]

        # Perform global magnitude pruning (L1Unstructured is equivalent to magnitude)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_ratio,
        )

        # Log the achieved sparsity
        sparsity = self.calculate_sparsity()
        logger.info(f"Magnitude pruning applied. Model sparsity: {sparsity*100:.2f}%")
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
        """Compute the fraction of zero weights in the entire model."""
        total_elements = 0
        zero_elements = 0
        for module in self.model.modules():
            if hasattr(module, 'weight'):
                w = module.weight
                total_elements += w.numel()
                zero_elements += (w == 0).sum().item()
        if total_elements == 0:
            return 0.0
        return zero_elements / total_elements

    # The following two methods are stubs to satisfy the base class interface.
    # They are not needed because prune module manages masks automatically.
    def apply_mask(self, module):
        pass

    def register_mask(self, module, mask):
        pass