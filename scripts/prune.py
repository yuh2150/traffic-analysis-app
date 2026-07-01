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
    theoretical_size_reduction_mb = (base_params - active_params) * 4 / (1024 ** 2)

    # Logging statistics (size_reduction_mb được tính sau khi save sparse)
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
    logger.info(f"Theoretical Size Reduction: {theoretical_size_reduction_mb:.2f} MB (zero weights * 4B)")
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

    from utils.pipeline_utils import get_clean_state_dict
    from utils.sparsity import state_dict_to_sparse, state_dict_to_dense

    # Lấy state dict đã bake (không còn pruning hooks)
    clean_sd = get_clean_state_dict(pruned_model)

    # Tính kích thước dense ước lượng
    dense_bytes = sum(v.numel() * v.element_size() for v in clean_sd.values() if isinstance(v, torch.Tensor))
    dense_size_mb = dense_bytes / (1024 * 1024)

    # Threshold động: layer nào có sparsity > 5% mới compress
    sparse_threshold = 0.05
    sparse_sd = state_dict_to_sparse(clean_sd, threshold=sparse_threshold)
    torch.save(sparse_sd, pruned_pt)

    # Đo kích thước file thực tế trên đĩa
    file_size_mb = os.path.getsize(pruned_pt) / (1024 * 1024)
    actual_compression_ratio = dense_size_mb / file_size_mb if file_size_mb > 0 else 1.0
    logger.info(f"Saved pruned checkpoint to: {pruned_pt}")
    logger.info(f"Storage: {dense_size_mb:.2f}MB (dense) → {file_size_mb:.2f}MB (sparse file) = {actual_compression_ratio:.2f}x compression")

    # Update size_reduction_mb trong log sau
    size_reduction_mb = dense_size_mb - file_size_mb

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
        "dense_size_mb": dense_size_mb,
        "sparse_file_size_mb": file_size_mb,
        "actual_compression_ratio": actual_compression_ratio,
        "config": vars(args),
        "timestamp": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "checkpoint_format_info": (
            "Sparse bit-packed checkpoint. Weights are baked (no pruning hooks). "
            "Use state_dict_to_dense() before loading or rely on extract_model_state_dict() which calls it automatically."
        )
    }, pruned_meta)

    logger.info("Stage B: Pruning completed successfully!")


if __name__ == "__main__":
    main()
