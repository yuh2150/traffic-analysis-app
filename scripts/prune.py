#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import ModelFactory
from pruning.base import PRUNER_REGISTRY
from utils.artifact_manager import ArtifactManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PruneStage")


def main():
    parser = argparse.ArgumentParser(description="Stage B: Prune Baseline Model")
    parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr"], help="Model architecture")
    parser.add_argument("--prune-type", type=str, default="magnitude", choices=list(PRUNER_REGISTRY.keys()), help="Pruning method")
    parser.add_argument("--sparsity", type=float, default=0.3, help="Pruning sparsity ratio")
    parser.add_argument("--dataset", type=str, default="coco", choices=["detrac", "coco"], help="Dataset baseline to load")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible tie-breaking in pruning")
    args = parser.parse_args()

    # Enforce reproducibility
    from utils.reproducibility import set_seed
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    artifact_manager = ArtifactManager()
    model_name_base = "coco_baseline" if args.dataset == "coco" else "baseline"
    baseline_pt = artifact_manager.get_checkpoint_path(args.model, f"{model_name_base}.pt")
    
    if not os.path.exists(baseline_pt):
        logger.error(f"Baseline checkpoint not found at: {baseline_pt}. Please run scripts/train.py first.")
        sys.exit(1)

    # Load model and weights
    logger.info(f"Loading baseline model from: {baseline_pt} (Dataset: {args.dataset.upper()})")
    num_classes = 80 if args.dataset == "coco" else 4
    model = ModelFactory.load(args.model, weights_path=baseline_pt, device=device, num_classes=num_classes)

    # Calculate baseline parameters and FLOPs
    img_size = (640, 640) if args.model == "yolov5s" else (800, 800)
    base_params = sum(p.numel() for p in model.parameters())
    base_flops = model.calculate_flops((3,) + img_size)

    # Apply pruning
    logger.info(f"Resolving pruner: '{args.prune_type}'...")
    pruner_cls = PRUNER_REGISTRY[args.prune_type]
    pruner = pruner_cls(model, args.sparsity)
    pruned_model = pruner.prune()

    # Collect pruning statistics
    stats = pruner.collect_statistics()
    active_params = stats["active_params"]
    actual_sparsity = stats["sparsity"]
    compression_ratio = stats["total_params"] / max(active_params, 1)
    
    # Calculate pruned FLOPs (physically reduced for Layer pruner, theoretically reduced for others)
    if args.prune_type == "layer":
        pruned_flops = pruned_model.calculate_flops((3,) + img_size)
    else:
        pruned_flops = int(base_flops * (1.0 - actual_sparsity))
        
    removed_flops = base_flops - pruned_flops
    size_reduction_mb = (base_params - active_params) * 4 / (1024 ** 2)

    # Logging statistics
    logger.info("==================================================")
    logger.info(f"PRUNING STATISTICS REPORT: {args.model.upper()} ({args.prune_type.upper()})")
    logger.info("==================================================")
    logger.info(f"Original Parameters:        {base_params:,}")
    logger.info(f"Active Parameters:          {active_params:,}")
    logger.info(f"Realized Sparsity:          {actual_sparsity*100:.2f}%")
    logger.info(f"Compression Ratio:          {compression_ratio:.2f}x")
    logger.info(f"Original FLOPs:             {base_flops:,}")
    logger.info(f"Pruned/Theoretical FLOPs:   {pruned_flops:,}")
    logger.info(f"Removed FLOPs:              {removed_flops:,} ({removed_flops/max(base_flops,1)*100:.2f}%)")
    logger.info(f"Est. Model Size Reduction:  {size_reduction_mb:.2f} MB")
    logger.info("==================================================")

    # Define paths and save
    pruned_pt = artifact_manager.get_checkpoint_path(args.model, f"{args.prune_type}_{args.sparsity}.pt")
    pruned_meta = artifact_manager.get_checkpoint_path(args.model, f"{args.prune_type}_{args.sparsity}_metadata.json")

    torch.save(pruned_model.state_dict(), pruned_pt)
    logger.info(f"Saved pruned checkpoint to: {pruned_pt}")

    # Save metadata
    artifact_manager.save_metadata({
        "pruning_method": args.prune_type,
        "sparsity": args.sparsity,
        "actual_sparsity": actual_sparsity,
        "compression_ratio": compression_ratio,
        "base_params": base_params,
        "pruned_params": active_params,
        "removed_params": base_params - active_params,
        "base_flops": base_flops,
        "pruned_flops": pruned_flops,
        "removed_flops": removed_flops,
        "size_reduction_mb": size_reduction_mb
    }, pruned_meta)

    logger.info("Stage B: Pruning completed successfully!")


if __name__ == "__main__":
    main()
