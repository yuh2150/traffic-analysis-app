import os
import logging
import torch
import torch.nn as nn
from typing import Any, Union
from .yolov5_wrapper import YOLOv5Wrapper
from .detr_wrapper import DETRWrapper

logger = logging.getLogger("ModelFactory")


def extract_model_state_dict(state: Any) -> dict:
    """Extracts the model state dictionary from various checkpoint formats."""
    if not isinstance(state, dict):
        return state

    # Local import to prevent circular dependency
    from utils.pipeline_utils import clean_state_dict_on_the_fly
    from utils.sparsity import state_dict_to_dense
    
    # Case 1: Our trainer's checkpoint
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
        
    # Convert sparse state dict back to dense if sparse representation is used
    state_dict = state_dict_to_dense(state_dict)
        
    state_dict = clean_state_dict_on_the_fly(state_dict)
        
    # Case 2: Ultralytics/torch.hub checkpoint
    if "model" in state_dict:
        model_obj = state_dict["model"]
        if hasattr(model_obj, "state_dict"):
            raw_sd = model_obj.state_dict()
        else:
            raw_sd = model_obj
            
        if isinstance(raw_sd, dict):
            # Prepend 'model.' to all keys of the inner model
            wrapped_sd = {}
            for k, v in raw_sd.items():
                wrapped_sd[f"model.{k}"] = v
            state_dict = wrapped_sd

    # Case 3: Raw state dict
    # Detect existing wrapper prefix and add if missing
    is_weight_dict = any(isinstance(v, torch.Tensor) for v in state_dict.values())
    if is_weight_dict:
        has_yolo_wrapper = any(k.startswith("model.model.") for k in state_dict.keys())
        has_detr_wrapper = any(k.startswith("model.class_embed.") or k.startswith("model.query_embed.") for k in state_dict.keys())
        
        # Fix accidental triple prefix from older versions
        has_triple_prefix = any(k.startswith("model.model.model.") for k in state_dict.keys())
        if has_triple_prefix:
            wrapped_sd = {}
            for k, v in state_dict.items():
                wrapped_sd[k.replace("model.model.model.", "model.model.", 1)] = v
            state_dict = wrapped_sd
        
        if not (has_yolo_wrapper or has_detr_wrapper):
            wrapped_sd = {}
            for k, v in state_dict.items():
                wrapped_sd[f"model.{k}"] = v
            state_dict = wrapped_sd
            
    # Clean again in case any mapping additions created mask keys
    state_dict = clean_state_dict_on_the_fly(state_dict)
    return state_dict


class ModelFactory:
    """Factory to instantiate and load supported machine learning architectures."""

    @staticmethod
    def load(model_name: str, weights_path: str = "", device: Union[str, torch.device] = "cpu", num_classes: int = 80) -> nn.Module:
        """Loads and returns a model wrapper based on the name.

        Args:
            model_name: Name of the model ('yolov5s' or 'detr').
            weights_path: Path to checkpoint weight file (.pt).
            device: Computing device ('cpu' or 'cuda').
            num_classes: Number of target dataset classes.
        """
        # Auto-detect class count from checkpoint weights to prevent size mismatch
        if num_classes == 80 and weights_path and os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = extract_model_state_dict(state)
                for k, v in state_dict.items():
                    if "24.m.0.weight" in k:
                        num_classes = (v.shape[0] // 3) - 5
                        break
                    elif "class_embed.weight" in k:
                        num_classes = v.shape[0] - 1
                        break
                logger.info(f"Auto-detected num_classes={num_classes} from checkpoint: {weights_path}")
            except Exception as e:
                logger.warning(f"Could not auto-detect classes from checkpoint: {e}")

        name = model_name.lower().replace("-", "").replace("_", "")
        
        if "yolov5" in name or "yolo" in name:
            logger.info(f"Instantiating YOLOv5Wrapper with num_classes={num_classes}...")
            model = YOLOv5Wrapper(num_classes=num_classes)
        elif "detr" in name:
            logger.info(f"Instantiating DETRWrapper with num_classes={num_classes}...")
            model = DETRWrapper(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}. Options: yolov5s, detr")

        # Load weights if checkpoint is provided
        if weights_path:
            if os.path.exists(weights_path):
                logger.info(f"Loading checkpoint weights from: {weights_path}")
                state = torch.load(weights_path, map_location=device, weights_only=False)
                state_dict = extract_model_state_dict(state)
                
                # Check if there is an associated metadata file to detect structured filter pruning
                metadata_path = weights_path[:-3] + "_metadata.json" if weights_path.endswith(".pt") else weights_path + "_metadata.json"
                if os.path.exists(metadata_path):
                    try:
                        import json
                        with open(metadata_path, "r") as f:
                            meta = json.load(f)
                        pruning_method = meta.get("pruning_method")
                        sparsity = meta.get("sparsity", 0.0)
                        if pruning_method == "filter" and sparsity > 0.0:
                            logger.info(f"Detected structured filter pruning with sparsity {sparsity} in metadata. Pruning model structure before loading weights...")
                            from pruning.filter import FilterPruner
                            pruner = FilterPruner(model, sparsity)
                            model = pruner.prune()
                    except Exception as e:
                        logger.warning(f"Could not pre-prune model based on metadata: {e}")
                
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning(f"Checkpoint loading: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
            else:
                logger.warning(f"Weights file not found at: {weights_path}. Model initialized with default weights.")

        return model.to(device)
