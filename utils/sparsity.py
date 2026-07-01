import torch
import torch.nn as nn
import logging
import numpy as np

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


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Helper to convert string representations of dtypes back to torch.dtype."""
    dtype_str = dtype_str.lower()
    if "float16" in dtype_str or "half" in dtype_str:
        return torch.float16
    elif "float64" in dtype_str or "double" in dtype_str:
        return torch.float64
    elif "int8" in dtype_str:
        return torch.int8
    elif "int16" in dtype_str:
        return torch.int16
    elif "int32" in dtype_str:
        return torch.int32
    elif "int64" in dtype_str:
        return torch.int64
    elif "bool" in dtype_str:
        return torch.bool
    return torch.float32


def state_dict_to_sparse(state_dict: dict, threshold: float = 0.1) -> dict:
    """Converts dense weights in a state_dict to a bit-packed sparse representation.
    
    Only applies to tensor weights with sparsity >= threshold and dimensionality >= 2.
    """
    sparse_sd = {}
    converted_count = 0
    total_count = 0
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            total_count += 1
            numel = v.numel()
            zeros = (v == 0.0).sum().item()
            sparsity = zeros / numel if numel > 0 else 0.0
            
            if sparsity >= threshold and zeros > 0:
                mask = (v != 0.0).cpu().numpy()
                packed_mask = np.packbits(mask.flatten())
                values = v[v != 0.0]
                
                sparse_sd[k] = {
                    "__sparse__": True,
                    "packed_mask": packed_mask,
                    "values": values,
                    "shape": list(v.shape),
                    "numel": numel,
                    "dtype": str(v.dtype)
                }
                converted_count += 1
            else:
                sparse_sd[k] = v
        else:
            sparse_sd[k] = v
            
    if converted_count > 0:
        logger.info(f"Compressed {converted_count}/{total_count} layers to bit-packed sparse representation.")
        
    return sparse_sd


def state_dict_to_dense(sparse_state_dict: dict) -> dict:
    """Converts a bit-packed sparse state_dict back to a standard dense state_dict."""
    dense_sd = {}
    inflated_count = 0
    
    for k, v in sparse_state_dict.items():
        if isinstance(v, dict) and v.get("__sparse__", False):
            # Unpack bit mask
            unpacked = np.unpackbits(v["packed_mask"])[:v["numel"]]
            mask = torch.from_numpy(unpacked.astype(bool)).view(v["shape"])
            
            # Restore dtype
            dtype = _str_to_dtype(v["dtype"])
            
            # Restore values to dense tensor
            dense_v = torch.zeros(v["shape"], dtype=dtype, device=v["values"].device)
            dense_v[mask] = v["values"]
            
            dense_sd[k] = dense_v
            inflated_count += 1
        else:
            dense_sd[k] = v
            
    if inflated_count > 0:
        logger.debug(f"Inflated {inflated_count} sparse layers back to dense tensors.")
        
    return dense_sd

