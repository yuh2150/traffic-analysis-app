import os
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

logger = logging.getLogger("ArtifactManager")


class ArtifactManager:
    """Manages model checkpoints and metadata for reproducibility across stages."""

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = base_dir

    def get_model_dir(self, model_name: str) -> str:
        """Returns the directory path for the given model, ensuring it exists."""
        cleaned_name = model_name.lower().replace("-", "").replace("_", "")
        model_dir = os.path.join(self.base_dir, cleaned_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def get_checkpoint_path(self, model_name: str, stage_filename: str) -> str:
        """Returns the absolute or relative path for a checkpoint file."""
        model_dir = self.get_model_dir(model_name)
        return os.path.join(model_dir, stage_filename)

    def get_baseline_dir(self, model_name: str) -> str:
        """Returns the baseline directory path for the given model, ensuring it exists."""
        model_dir = self.get_model_dir(model_name)
        d = os.path.join(model_dir, "baseline")
        os.makedirs(d, exist_ok=True)
        return d

    def get_pruned_dir(self, model_name: str) -> str:
        """Returns the pruned directory path for the given model, ensuring it exists."""
        model_dir = self.get_model_dir(model_name)
        d = os.path.join(model_dir, "pruned")
        os.makedirs(d, exist_ok=True)
        return d

    def get_recovered_dir(self, model_name: str) -> str:
        """Returns the recovered directory path for the given model, ensuring it exists."""
        model_dir = self.get_model_dir(model_name)
        d = os.path.join(model_dir, "recovered")
        os.makedirs(d, exist_ok=True)
        return d

    def get_baseline_checkpoint(self, model_name: str, suffix: str = 'best') -> str:
        """Returns baseline checkpoint path: .../baseline/{suffix}.pt"""
        d = self.get_baseline_dir(model_name)
        return os.path.join(d, f"{suffix}.pt")

    def get_pruned_checkpoint(self, model_name: str, method: str, sparsity: Union[float, str]) -> str:
        """Returns pruned checkpoint path: .../pruned/{method}_{sparsity}.pt"""
        d = self.get_pruned_dir(model_name)
        return os.path.join(d, f"{method}_{sparsity}.pt")

    def get_recovered_checkpoint(self, model_name: str, method: str, sparsity: Union[float, str], suffix: str = 'best') -> str:
        """Returns recovered checkpoint path: .../recovered/{method}_{sparsity}_{suffix}.pt"""
        d = self.get_recovered_dir(model_name)
        return os.path.join(d, f"{method}_{sparsity}_{suffix}.pt")

    def get_metadata_path(self, checkpoint_path: str) -> str:
        """Returns metadata JSON file path corresponding to the checkpoint (.pt -> _metadata.json)."""
        if checkpoint_path.endswith(".pt"):
            return checkpoint_path[:-3] + "_metadata.json"
        return checkpoint_path + "_metadata.json"

    def save_checkpoint(
        self,
        model: nn.Module,
        save_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        loss: float = 0.0,
        best_map: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves a model checkpoint including optimizer, scheduler, epoch, best_map, and config info."""
        # Strip thop profile keys if present in weights
        state_dict = model.state_dict()
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
        
        state = {
            "model_state_dict": clean_state_dict,
            "epoch": epoch,
            "loss": loss,
            "best_map": best_map,
            "config": config
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
            
        torch.save(state, save_path)
        logger.info(f"Saved model checkpoint to: {save_path}")

    def load_checkpoint(
        self,
        model: nn.Module,
        load_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Dict[str, Any]:
        """Loads weights into model and optional optimizer/scheduler, returning loaded state info."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found at: {load_path}")
            
        logger.info(f"Loading checkpoint: {load_path}")
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        # Check if wrapped in trainer dict vs raw weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            return {
                "epoch": checkpoint.get("epoch", 0),
                "loss": checkpoint.get("loss", 0.0),
                "best_map": checkpoint.get("best_map", 0.0),
                "config": checkpoint.get("config", None)
            }
        else:
            # Raw state dict loading
            model.load_state_dict(checkpoint, strict=False)
            return {"epoch": 0, "loss": 0.0, "best_map": 0.0, "config": None}

    def save_metadata(self, metadata: Dict[str, Any], save_path: str) -> None:
        """Saves run or architecture metadata to a JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved metadata to: {save_path}")

    def load_metadata(self, load_path: str) -> Dict[str, Any]:
        """Loads and returns metadata from a JSON file."""
        if not os.path.exists(load_path):
            logger.warning(f"Metadata file not found: {load_path}")
            return {}
        with open(load_path, "r") as f:
            return json.load(f)
