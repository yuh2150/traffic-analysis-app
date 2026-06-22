import os
import torch
import torch.nn as nn
import json
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("PrunerBase")


PRUNER_REGISTRY = {}


def register_pruner(name: str):
    """Decorator to register new pruning strategies in the global registry."""
    def decorator(cls):
        PRUNER_REGISTRY[name] = cls
        return cls
    return decorator


class BasePruner:
    """Abstract Base Class for all pruning methodologies.

    Encapsulates shared functionality like layer discovery, mask registration,
    sparsity enforcement, gradient zeroing, and statistics collection.
    """

    def __init__(self, model: nn.Module, pruning_ratio: float = 0.0):
        self.model = model
        self.pruning_ratio = pruning_ratio

    def discover_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Identifies and returns all prunable Conv2d and Linear layers in the model."""
        prunable = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable.append((name, module))
        return prunable

    def calculate_sparsity(self) -> float:
        """Calculates the global sparsity of the prunable layers in the model."""
        total_weights = 0
        zero_weights = 0

        for name, module in self.discover_prunable_layers():
            w = module.weight.data
            total_weights += w.numel()
            zero_weights += (w == 0.0).sum().item()

        return zero_weights / total_weights if total_weights > 0 else 0.0

    def register_mask(self, module: nn.Module, mask: torch.Tensor):
        """Registers a binary pruning mask as a module buffer."""
        # Clean existing mask if present
        if hasattr(module, "pruning_mask"):
            delattr(module, "pruning_mask")
        module.register_buffer("pruning_mask", mask.float())

    def apply_mask(self, module: nn.Module):
        """Applies the registered pruning mask to the module's weights and bias."""
        if hasattr(module, "pruning_mask"):
            module.weight.data.mul_(module.pruning_mask)
            # If mask is filter-level (shape matching output channels), mask bias too
            if module.bias is not None and module.pruning_mask.shape[0] == module.bias.shape[0]:
                if module.pruning_mask.ndim == 1:
                    module.bias.data.mul_(module.pruning_mask)
                elif all(d == 1 for d in module.pruning_mask.shape[1:]):
                    module.bias.data.mul_(module.pruning_mask.view(-1))

    def enforce_sparsity(self):
        """Enforces sparsity by multiplying all pruned modules by their registered masks in-place."""
        for name, module in self.discover_prunable_layers():
            if hasattr(module, "pruning_mask"):
                module.weight.data.mul_(module.pruning_mask)

    def zero_gradients(self):
        """Zeros out gradients for all pruned weights to prevent optimizer tracking updates."""
        for name, module in self.discover_prunable_layers():
            if hasattr(module, "pruning_mask") and module.weight.grad is not None:
                module.weight.grad.data.mul_(module.pruning_mask)

    def collect_statistics(self) -> Dict[str, Any]:
        """Compiles model structural statistics."""
        active_params = 0
        total_params = 0
        for p in self.model.parameters():
            total_params += p.numel()
            active_params += (p.data != 0).sum().item()

        size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)

        return {
            "total_params": total_params,
            "active_params": int(active_params),
            "sparsity": 1.0 - (active_params / total_params) if total_params > 0 else 0.0,
            "size_mb": size_mb,
        }

    def save_metadata(self, save_path: str):
        """Saves pruning masks and metadata to a JSON file."""
        metadata = {
            "pruning_ratio": self.pruning_ratio,
            "global_sparsity": self.calculate_sparsity(),
            "masks": {}
        }
        for name, module in self.discover_prunable_layers():
            if hasattr(module, "pruning_mask"):
                # Convert mask to list for JSON serialization
                metadata["masks"][name] = module.pruning_mask.cpu().numpy().tolist()

        with open(save_path, "w") as f:
            json.dump(metadata, f)
        logger.info(f"Saved pruning metadata to: {save_path}")

    def load_metadata(self, load_path: str):
        """Loads and applies pruning masks and metadata from a JSON file."""
        if not os.path.exists(load_path):
            logger.warning(f"Metadata file not found: {load_path}")
            return

        with open(load_path, "r") as f:
            metadata = json.load(f)

        self.pruning_ratio = metadata.get("pruning_ratio", self.pruning_ratio)
        masks = metadata.get("masks", {})

        for name, module in self.discover_prunable_layers():
            if name in masks:
                mask_tensor = torch.tensor(masks[name], dtype=torch.float32, device=module.weight.device)
                self.register_mask(module, mask_tensor)
                self.apply_mask(module)

        logger.info(f"Loaded and applied pruning metadata from: {load_path}")

    def prune(self) -> nn.Module:
        """Abstract method to implement pruning logic. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the prune() method.")


@torch.no_grad()
def enforce_sparsity(model: nn.Module):
    """Enforces sparsity on all modules with a registered pruning_mask."""
    for module in model.modules():
        if hasattr(module, "pruning_mask"):
            module.weight.data.mul_(module.pruning_mask)


def zero_pruned_gradients(model: nn.Module):
    """Zeros gradients on all modules with a registered pruning_mask."""
    for module in model.modules():
        if hasattr(module, "pruning_mask") and module.weight.grad is not None:
            module.weight.grad.data.mul_(module.pruning_mask)

