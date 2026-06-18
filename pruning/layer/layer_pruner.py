import os
import argparse
import copy
import torch
import torch.nn as nn
from models.models import get_model
from pruning.pruner import get_effective_metrics

def get_c3_bottlenecks(model: nn.Module):
    """Dynamically finds all C3 modules and their bottlenecks in the model.
    
    Returns:
        bottlenecks: list of dicts with keys:
            - 'c3_name': name of the C3 module
            - 'c3_module': reference to the C3 module
            - 'bottleneck_idx': index of the bottleneck within C3.m
            - 'bottleneck_module': reference to the Bottleneck module
            - 'importance': calculated importance score
    """
    bottlenecks = []
    
    for name, module in model.named_modules():
        # Identify C3 modules by class name dynamically (prevents import path issues)
        if module.__class__.__name__ == 'C3':
            if hasattr(module, 'm') and isinstance(module.m, nn.Sequential):
                for idx, bottleneck in enumerate(module.m):
                    if bottleneck.__class__.__name__ == 'Bottleneck':
                        # cv2 is the second conv block in the Bottleneck
                        if hasattr(bottleneck, 'cv2'):
                            conv = bottleneck.cv2.conv
                            
                            # Calculate importance score using weight magnitude
                            conv_weight = conv.weight.data.abs().cpu()
                            weight_alpha = torch.mean(conv_weight.view(conv_weight.size(0), -1), dim=1)
                            
                            # Check if BatchNorm exists (not fused)
                            if hasattr(bottleneck.cv2, 'bn') and bottleneck.cv2.bn is not None:
                                bn = bottleneck.cv2.bn
                                bn_weight = bn.weight.data.abs().cpu()
                                # Combine both metrics
                                importance_tensor = 10 * weight_alpha * bn_weight
                            else:
                                # Fused model: use Conv weight magnitude directly
                                importance_tensor = weight_alpha
                                
                            mean_importance = torch.mean(importance_tensor).item()
                            
                            bottlenecks.append({
                                'c3_name': name,
                                'c3_module': module,
                                'bottleneck_idx': idx,
                                'bottleneck_module': bottleneck,
                                'importance': mean_importance
                            })
                            
    return bottlenecks

def prune_layers(model: nn.Module, num_to_prune: int) -> nn.Module:
    """Prunes the least important C3 Bottlenecks dynamically in-place (Layer Pruning)."""
    model = copy.deepcopy(model)
    bottlenecks = get_c3_bottlenecks(model)
    
    total_bottlenecks = len(bottlenecks)
    print(f"Found {total_bottlenecks} C3 layers (bottlenecks) in the model.")
    
    if num_to_prune >= total_bottlenecks:
        print(f"Warning: requested to prune {num_to_prune} layers but only {total_bottlenecks} exist. Pruning {total_bottlenecks - 1} instead.")
        num_to_prune = total_bottlenecks - 1
        
    if num_to_prune <= 0:
        print("No layers to prune.")
        return model

    # Sort bottlenecks by importance (ascending: least important first)
    sorted_bottlenecks = sorted(bottlenecks, key=lambda x: x['importance'])
    pruned_info = sorted_bottlenecks[:num_to_prune]
    
    print("\n--- Layers selected for pruning (least important first) ---")
    for i, item in enumerate(pruned_info):
        print(f"{i+1}. C3 Block: '{item['c3_name']}' | Layer Index: {item['bottleneck_idx']} | Importance: {item['importance']:.6f}")
        
    # Group kept bottlenecks by their C3 parent module
    c3_kept_map = {}
    for item in bottlenecks:
        is_pruned = any(
            p['c3_name'] == item['c3_name'] and p['bottleneck_idx'] == item['bottleneck_idx'] 
            for p in pruned_info
        )
        if not is_pruned:
            if item['c3_name'] not in c3_kept_map:
                c3_kept_map[item['c3_name']] = []
            c3_kept_map[item['c3_name']].append(item['bottleneck_module'])
            
    # Apply pruning in-place on C3 modules
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'C3':
            kept_modules = c3_kept_map.get(name, [])
            # Re-instantiate the sequential block with kept bottleneck modules
            module.m = nn.Sequential(*kept_modules)
            print(f"Updated C3 Block '{name}': remaining layers = {len(kept_modules)}")
            
    return model

def main():
    parser = argparse.ArgumentParser(description="Structured Layer Pruning (Depthwise) for YOLOv5")
    parser.add_argument("--weights", type=str, default="weights/baseline/yolov5s.pt", help="Path to input weights")
    parser.add_argument("--save", type=str, default="weights/layer/yolov5_layer_pruned.pt", help="Path to save pruned weights")
    parser.add_argument("--num_to_prune", type=int, default=3, help="Number of bottleneck layers to prune")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {opt.weights}")
    
    # Initialize YOLOv5s (Nano size used by project)
    model = get_model("yolov5", model_size="n").to(device)
    
    # Load state dict if weight file exists
    if os.path.exists(opt.weights):
        model.load_state_dict(torch.load(opt.weights, map_location=device), strict=False)
        print("Successfully loaded model state dict.")
    else:
        print(f"Warning: weights not found at {opt.weights}. Pruning baseline initialized weights.")

    # Get baseline metrics
    metrics_before = get_effective_metrics(model, (1, 3, 640, 640))
    print(f"\nBefore Pruning: Active Params = {metrics_before['active_params']:,} | FLOPs = {metrics_before['flops']:,}")
    
    # Prune C3 bottlenecks
    pruned_model = prune_layers(model, opt.num_to_prune)
    
    # Get pruned metrics
    metrics_after = get_effective_metrics(pruned_model, (1, 3, 640, 640))
    print(f"\nAfter Pruning: Active Params = {metrics_after['active_params']:,} | FLOPs = {metrics_after['flops']:,}")
    
    # Save pruned state dict
    os.makedirs(os.path.dirname(opt.save), exist_ok=True)
    torch.save(pruned_model.state_dict(), opt.save)
    print(f"\nSuccessfully saved pruned weights to: {opt.save}")

if __name__ == "__main__":
    main()
