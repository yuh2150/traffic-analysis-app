#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import ModelFactory, extract_model_state_dict
from datasets.factory import DatasetFactory
from benchmarking.benchmark import TrafficBenchmark
from pruning.base import PRUNER_REGISTRY
from utils.pipeline_utils import calculate_actual_sparsity, load_checkpoint_weights
from utils.artifact_manager import ArtifactManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BenchmarkScript")


def main():
    parser = argparse.ArgumentParser(description="Stage D: Benchmark a specific model checkpoint")
    parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr"], help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--prune-type", type=str, default="None", help="Fallback pruning type label if not found in checkpoint")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Fallback sparsity ratio if not found in checkpoint")
    parser.add_argument("--dataset", type=str, default="coco", choices=["detrac", "coco"], help="Dataset to use")
    parser.add_argument("--img-dir", type=str, default="data/DETRAC-Images/DETRAC-Images", help="DETRAC dataset images path")
    parser.add_argument("--val-anno-dir", type=str, default="data/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML", help="DETRAC validation annotations path")
    parser.add_argument("--coco-val-img", type=str, default="data/coco/val2017", help="COCO val images path")
    parser.add_argument("--coco-val-anno", type=str, default="data/coco/annotations/instances_val2017.json", help="COCO val annotations JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save benchmark JSON result")
    parser.add_argument("--no-save", action="store_true", help="Do not save results to file, only print to console")
    args = parser.parse_args()

    # Enforce reproducibility
    from utils.reproducibility import set_seed
    set_seed(args.seed)

    device_name = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # Load checkpoint dictionary first to detect metadata/config and masks
    logger.info(f"Pre-loading checkpoint to detect metadata and structure: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)

    # Detect if checkpoint has dynamic masks (for unstructured pruning)
    has_masks = any(k.endswith(".weight_mask") for k in state_dict.keys())

    # Detect pruning configuration from checkpoint config
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    prune_type = config.get("prune_type", config.get("pruning_method", args.prune_type))
    sparsity = config.get("sparsity", args.sparsity)

    # Read parallel metadata if exists
    artifact_manager = ArtifactManager()
    metadata_path = artifact_manager.get_metadata_path(args.checkpoint)
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            prune_type = meta.get("pruning_method", prune_type)
            sparsity = meta.get("sparsity", sparsity)
        except Exception:
            pass

    # Clean strings
    prune_type_str = str(prune_type)
    sparsity_val = float(sparsity)

    logger.info(f"Detected/Resolved pruning config -> Type: {prune_type_str}, Sparsity: {sparsity_val}")
    logger.info(f"Dynamic masks detected in checkpoint: {has_masks}")

    # Determine paths and image size
    img_size = (640, 640) if args.model == "yolov5s" else (800, 800)
    if args.dataset == "coco":
        val_img = args.coco_val_img
        val_anno = args.coco_val_anno
    else:
        val_img = args.img_dir
        val_anno = args.val_anno_dir

    # Validation loader
    val_loader = DatasetFactory.get_dataloader(
        img_dir=val_img,
        anno_dir=val_anno,
        batch_size=4,
        img_size=img_size,
        shuffle=False,
        max_samples=args.max_samples,
        dataset_type=args.dataset,
    )

    # 1. Load model structure
    logger.info(f"Initializing model structure for {args.model}...")
    num_classes = 80 if args.dataset == "coco" else 4
    model = ModelFactory.load(args.model, num_classes=num_classes, device=device)

    # 2. Register pruning masks/structures if needed
    should_prune = False
    if prune_type_str.lower() != "none" and sparsity_val > 0.0:
        if prune_type_str.lower() == "magnitude":
            # Only prune model to add masks if checkpoint has weight masks
            should_prune = has_masks
        else:
            # Structured/layer pruning always changes structure physically
            should_prune = True

    if should_prune:
        logger.info(f"Registering pruning structure for type '{prune_type_str}' with sparsity {sparsity_val}...")
        prune_type_lower = prune_type_str.lower()
        if prune_type_lower in PRUNER_REGISTRY:
            pruner_cls = PRUNER_REGISTRY[prune_type_lower]
            pruner = pruner_cls(model, sparsity_val)
            model = pruner.prune()
        else:
            logger.warning(f"Could not resolve pruner '{prune_type_str}' to register masks.")
    else:
        logger.info("No mask or structure pruning required before loading checkpoint weights (Full model or baked weights).")

    # 3. Load weights using pipeline_utils
    missing_keys, unexpected_keys = load_checkpoint_weights(model, args.checkpoint, device, logger)

    # Bake weights if dynamic hooks are present to improve benchmarking latency/FPS
    if has_masks:
        logger.info("Baking pruning masks into weights to eliminate forward hook overhead during benchmarking...")
        from utils.pipeline_utils import bake_pruned_weights
        num_baked = bake_pruned_weights(model)
        logger.info(f"Successfully baked weights and removed dynamic hooks for {num_baked} modules.")

    # 4. Calculate parameters, FLOPs, and actual sparsity after loading
    actual_sparsity = calculate_actual_sparsity(model)
    logger.info(f"Post-load verification -> Sparsity: {actual_sparsity*100:.2f}% | Params: {model.get_params_count():,}")

    # Run benchmark evaluation
    logger.info("Starting benchmark evaluation...")
    benchmark = TrafficBenchmark(args.model, device, (3,) + img_size)
    results = benchmark.evaluate_checkpoint(model, val_loader)

    # Update/Add custom metadata to results
    results["Config"] = f"Pruned {prune_type_str} ({int(sparsity_val*100)}%)" if sparsity_val > 0 else "Baseline FP32"
    results["Pruning Type"] = prune_type_str
    results["Sparsity"] = sparsity_val
    results["Actual Sparsity"] = actual_sparsity
    results["Params"] = model.get_params_count()
    results["FLOPs"] = model.calculate_flops((3,) + img_size)
    results["Timestamp"] = datetime.datetime.now().isoformat()
    results["Torch Version"] = torch.__version__
    results["State Dict Match Info"] = {
        "matched_keys_count": len(state_dict) - len(unexpected_keys),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys)
    }

    # Print results summary
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS SUMMARY:")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:<25}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"{k:<25}:")
            for subk, subv in v.items():
                print(f"  {subk:<23}: {subv}")
        else:
            print(f"{k:<25}: {v}")
    print("=" * 50 + "\n")

    # Save results using ArtifactManager
    if not args.no_save:
        output_dir = args.output_dir if args.output_dir else "reports"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{args.model}_{prune_type_str.lower()}_{sparsity_val}.json"
        save_path = os.path.join(output_dir, filename)
        
        artifact_manager.save_metadata(results, save_path)
        logger.info(f"Saved benchmark results to: {save_path}")


if __name__ == "__main__":
    main()
