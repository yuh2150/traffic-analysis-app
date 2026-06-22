import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

def analyze_weight_distribution(model: nn.Module, save_path: str = "weight_analysis.json") -> Dict[str, Any]:
    """Analyzes the weight distribution of the model and saves stats to weight_analysis.json."""
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            all_weights.append(module.weight.data.cpu().view(-1))
            
    if not all_weights:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "global_sparsity": 0.0,
            "histogram": {"counts": [], "bin_edges": []}
        }
        
    flat_weights = torch.cat(all_weights)
    flat_weights_np = flat_weights.numpy()
    
    mean_val = float(flat_weights.mean().item())
    std_val = float(flat_weights.std().item())
    min_val = float(flat_weights.min().item())
    max_val = float(flat_weights.max().item())
    
    total_elements = flat_weights.numel()
    zero_elements = (flat_weights == 0.0).sum().item()
    global_sparsity = float(zero_elements / total_elements)
    
    counts, bin_edges = np.histogram(flat_weights_np, bins=20)
    
    analysis_report = {
        "mean": mean_val, "std": std_val, "min": min_val, "max": max_val,
        "global_sparsity": global_sparsity, "total_weights": int(total_elements),
        "zero_weights": int(zero_elements),
        "histogram": {
            "counts": counts.tolist(), "bin_edges": bin_edges.tolist()
        }
    }
    
    with open(save_path, "w") as f:
        json.dump(analysis_report, f, indent=4)
        
    return analysis_report
