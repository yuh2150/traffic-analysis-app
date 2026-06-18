import os
import torch
import logging
from models.models import get_model
from pruning.pruner import (
    prune_by_magnitude,
    prune_by_l1_norm,
    prune_layer,
    network_slimming_channel_prune,
    get_effective_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WeightGeneration")


def main():
    logger.info("Starting Weight Generation Pipeline...")

    dirs = [
        "weights/baseline",
        "weights/magnitude/mag30",
        "weights/magnitude/mag50",
        "weights/magnitude/mag70",
        "weights/l1_norm/l1_30",
        "weights/l1_norm/l1_50",
        "weights/l1_norm/l1_70",
        "weights/channel/pruned_50",
        "weights/channel/pruned_70",
        "weights/layer",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    for model_name in ["yolov5s", "detr"]:
        logger.info(f"Processing {model_name}...")
        model = get_model(model_name)
        model.eval()

        # Baseline
        base_path = f"weights/baseline/{model_name}.pt"
        torch.save(model.state_dict(), base_path)
        logger.info(f"Saved baseline: {base_path}")

        params = model.get_params_count()
        logger.info(f"Baseline params: {params:,}")

        # Magnitude pruning at various sparsity levels
        for sparsity in [30, 50, 70]:
            ratio = sparsity / 100.0
            logger.info(f"  Magnitude {sparsity}%...")
            pruned = prune_by_magnitude(model, ratio)
            path = f"weights/magnitude/mag{sparsity}/{model_name}.pt"
            torch.save(pruned.state_dict(), path)
            eff = get_effective_metrics(pruned, (1, 3, 640, 640) if "yolo" in model_name else (1, 3, 800, 800))
            logger.info(f"    Active params: {eff['active_params']:,}  Sparsity: {eff['sparsity']:.3f}")

        # L1-norm filter pruning
        for sparsity in [30, 50, 70]:
            ratio = sparsity / 100.0
            logger.info(f"  L1-Norm {sparsity}%...")
            pruned = prune_by_l1_norm(model, ratio)
            path = f"weights/l1_norm/l1_{sparsity}/{model_name}.pt"
            torch.save(pruned.state_dict(), path)
            eff = get_effective_metrics(pruned, (1, 3, 640, 640) if "yolo" in model_name else (1, 3, 800, 800))
            logger.info(f"    Active params: {eff['active_params']:,}  Sparsity: {eff['sparsity']:.3f}")

        # Network Slimming Channel Pruning (BN-gamma-based)
        if "yolo" in model_name:
            for sparsity in [50, 70]:
                ratio = sparsity / 100.0
                logger.info(f"  Network Slimming Channel Prune {sparsity}%...")
                try:
                    pruned = network_slimming_channel_prune(model, sparsity=ratio, divisor=8)
                    path = f"weights/channel/pruned_{sparsity}/{model_name}.pt"
                    torch.save(pruned.state_dict(), path)
                    eff = get_effective_metrics(pruned, (1, 3, 640, 640))
                    pruned_params = sum(p.numel() for p in pruned.parameters() if p.requires_grad)
                    logger.info(
                        f"    Params: {pruned_params:,}  "
                        f"Active: {eff['active_params']:,}  "
                        f"Sparsity: {eff['sparsity']:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"    Channel pruning failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            logger.info(f"  Skipping channel pruning for {model_name} (not YOLO-based)")

        # Layer pruning
        logger.info(f"  Layer pruning...")
        if "yolo" in model_name:
            layer_path = "model.model.3"
        else:
            layer_path = "backbone.layer1.0"
        try:
            pruned = prune_layer(model, layer_path)
            path = f"weights/layer/{model_name}.pt"
            torch.save(pruned.state_dict(), path)
            eff = get_effective_metrics(pruned, (1, 3, 640, 640) if "yolo" in model_name else (1, 3, 800, 800))
            logger.info(f"    Active params: {eff['active_params']:,}  Sparsity: {eff['sparsity']:.3f}")
        except Exception as e:
            logger.warning(f"    Layer pruning failed: {e}")

    logger.info("Weight generation completed!")
    for root, dirs, files in os.walk("weights"):
        for file in files:
            p = os.path.join(root, file)
            size_mb = os.path.getsize(p) / (1024 * 1024)
            logger.info(f"  {p}: {size_mb:.3f} MB")


if __name__ == "__main__":
    main()
