#!/usr/bin/env python3
import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MigrateCheckpoints")


def migrate_model_checkpoints(model_name: str, base_dir: str = "checkpoints"):
    model_dir = os.path.join(base_dir, model_name.lower().replace("-", "").replace("_", ""))
    if not os.path.isdir(model_dir):
        logger.info(f"Directory {model_dir} does not exist. Skipping.")
        return

    logger.info(f"Processing migration for model: {model_name.upper()} in {model_dir}")

    # Create target subfolders
    baseline_dir = os.path.join(model_dir, "baseline")
    pruned_dir = os.path.join(model_dir, "pruned")
    recovered_dir = os.path.join(model_dir, "recovered")

    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(pruned_dir, exist_ok=True)
    os.makedirs(recovered_dir, exist_ok=True)

    # Legacy mapping patterns
    # (source_pattern, target_dir, target_name)
    mappings = [
        # Baseline best
        ("baseline.pt", baseline_dir, "best.pt"),
        ("coco_baseline.pt", baseline_dir, "best.pt"),
        ("baseline_metadata.json", baseline_dir, "best_metadata.json"),
        ("coco_baseline_metadata.json", baseline_dir, "best_metadata.json"),
        
        # Baseline last
        ("baseline_last.pt", baseline_dir, "last.pt"),
        ("coco_baseline_last.pt", baseline_dir, "last.pt"),
        
        # Baseline history
        ("baseline_history.json", baseline_dir, "history.json"),
        ("coco_baseline_history.json", baseline_dir, "history.json"),
    ]

    # Process static mappings
    for filename, target_dir, target_name in mappings:
        src_path = os.path.join(model_dir, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(target_dir, target_name)
            logger.info(f"Moving {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)

    # Dynamic files (pruning and recovery)
    for filename in os.listdir(model_dir):
        src_path = os.path.join(model_dir, filename)
        if not os.path.isfile(src_path):
            continue

        # Skip files already processed or hidden files
        if filename.startswith(".") or filename in [m[0] for m in mappings]:
            continue

        # Pruned models: e.g. magnitude_0.3.pt, magnitude_0.3_metadata.json
        # Recovered models: e.g. magnitude_0.3_recovered.pt, magnitude_0.3_recovered_metadata.json
        # Recovery training files: e.g. magnitude_0.3_recover_last.pt, magnitude_0.3_recover_history.json

        if "_recovered" in filename:
            # magnitude_0.3_recovered.pt -> recovered/magnitude_0.3_best.pt
            # magnitude_0.3_recovered_metadata.json -> recovered/magnitude_0.3_best_metadata.json
            new_name = filename.replace("_recovered.pt", "_best.pt").replace("_recovered_metadata.json", "_best_metadata.json")
            dst_path = os.path.join(recovered_dir, new_name)
            logger.info(f"Moving {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)
            
        elif "_recover_last" in filename:
            # magnitude_0.3_recover_last.pt -> recovered/magnitude_0.3_last.pt
            new_name = filename.replace("_recover_last.pt", "_last.pt")
            dst_path = os.path.join(recovered_dir, new_name)
            logger.info(f"Moving {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)
            
        elif "_recover_history" in filename:
            # magnitude_0.3_recover_history.json -> recovered/magnitude_0.3_history.json
            new_name = filename.replace("_recover_history.json", "_history.json")
            dst_path = os.path.join(recovered_dir, new_name)
            logger.info(f"Moving {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)
            
        elif any(filename.startswith(method) for method in ["magnitude", "l1_norm", "filter", "channel", "layer"]):
            # magnitude_0.3.pt -> pruned/magnitude_0.3.pt
            # magnitude_0.3_metadata.json -> pruned/magnitude_0.3_metadata.json
            dst_path = os.path.join(pruned_dir, filename)
            logger.info(f"Moving {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)


def main():
    base_dir = "checkpoints"
    if not os.path.isdir(base_dir):
        logger.error(f"Base checkpoints directory '{base_dir}' not found.")
        sys.exit(1)

    for model_name in ["yolov5s", "detr"]:
        migrate_model_checkpoints(model_name, base_dir)

    logger.info("Migration completed successfully!")


if __name__ == "__main__":
    main()
