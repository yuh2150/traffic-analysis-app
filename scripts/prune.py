#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import datetime

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
    parser.add_argument("--pretrained", action="store_true", help="Prune directly from pretrained weights (torch.hub)")
    parser.add_argument("--force", action="store_true", help="Force overwrite pruned checkpoint if it already exists")
    args = parser.parse_args()

    # Enforce reproducibility
    from utils.reproducibility import set_seed
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    artifact_manager = ArtifactManager()
    baseline_pt = artifact_manager.get_baseline_checkpoint(args.model, suffix='best')
    
    if not args.pretrained and not os.path.exists(baseline_pt):
        logger.error(f"Baseline checkpoint not found at: {baseline_pt}. Please run scripts/train.py first or use --pretrained.")
        sys.exit(1)

    # Load model and weights
    num_classes = 80 if args.dataset == "coco" else 4
    if args.pretrained:
        logger.info(f"Loading pretrained model directly from factory...")
        model = ModelFactory.load(args.model, weights_path="", device=device, num_classes=num_classes)
    else:
        logger.info(f"Loading baseline model from: {baseline_pt} (Dataset: {args.dataset.upper()})")
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
    
    # Calculate FLOPs
    actual_flops = pruned_model.calculate_flops((3,) + img_size)
    if args.prune_type == "layer":
        theoretical_flops = actual_flops
    else:
        theoretical_flops = int(base_flops * (1.0 - actual_sparsity))
        
    removed_flops_theoretical = base_flops - theoretical_flops
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
    logger.info(f"Actual FLOPs (dense HW):     {actual_flops:,}")
    logger.info(f"Theoretical FLOPs (sparse):  {theoretical_flops:,}")
    logger.info(f"Removed FLOPs (theoretical): {removed_flops_theoretical:,} ({removed_flops_theoretical/max(base_flops,1)*100:.2f}%)")
    logger.info(f"Est. Model Size Reduction:  {size_reduction_mb:.2f} MB")
    logger.info("==================================================")

    # Verification forward pass
    logger.info("Performing verification forward pass with dummy input...")
    try:
        pruned_model.eval()
        dummy_input = torch.zeros((1, 3, *img_size), device=device)
        with torch.no_grad():
            _ = pruned_model(dummy_input)
        logger.info("Verification forward pass completed successfully! Model is fully functional.")
    except Exception as e:
        logger.error(f"Verification forward pass FAILED: {e}")
        sys.exit(1)

    # Define paths and save
    pruned_pt = artifact_manager.get_pruned_checkpoint(args.model, args.prune_type, args.sparsity)
    pruned_meta = artifact_manager.get_metadata_path(pruned_pt)

    # Check for existing checkpoint and prompt
    if os.path.exists(pruned_pt) and not args.force:
        logger.warning(f"Pruned checkpoint already exists at: {pruned_pt}")
        try:
            choice = input("Overwrite? [y/N]: ").strip().lower()
            if choice not in ("y", "yes"):
                logger.info("Pruning aborted by user.")
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            logger.info("Non-interactive mode or interrupt. Pruning aborted.")
            sys.exit(0)

    torch.save(pruned_model.state_dict(), pruned_pt)
    logger.info(f"Saved pruned checkpoint to: {pruned_pt}")

    if args.prune_type == "magnitude":
        logger.info(
            "INFO: Magnitude pruning uses PyTorch's dynamic pruning hooks (torch.nn.utils.prune). "
            "The saved state_dict contains weight masks ('weight_mask') and original parameters ('weight_orig'). "
            "To successfully reload this checkpoint, you must: "
            "1) Initialize a fresh model, "
            "2) Instantiate and call MagnitudePruner on the model to register pruning buffers, "
            "3) Load the state_dict (as done in recover.py / experiment_manager.py)."
        )

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
        "actual_flops": actual_flops,
        "theoretical_flops": theoretical_flops,
        "removed_flops_theoretical": removed_flops_theoretical,
        "size_reduction_mb": size_reduction_mb,
        "config": vars(args),
        "timestamp": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "checkpoint_format_info": (
            "Since magnitude pruning uses PyTorch's dynamic pruning hooks, the checkpoint contains weight_orig and weight_mask. "
            "To reload, register pruning buffers first." if args.prune_type == "magnitude" else "Standard parameters checkpoint."
        )
    }, pruned_meta)

    logger.info("Stage B: Pruning completed successfully!")


if __name__ == "__main__":
    main()
