#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarking.experiment_manager import ExperimentManager
from pruning.base import PRUNER_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ExperimentStage")


def main():
    parser = argparse.ArgumentParser(description="Stage E: Run Automated Experiments")
    parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr", "all"], help="Model architecture matrix (or 'all' for both)")
    parser.add_argument("--prune-types", nargs="+", default=["magnitude", "l1_norm", "filter", "channel", "layer"], help="Pruning types matrix")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7], help="Pruning sparsity ratios matrix")
    parser.add_argument("--epochs-train", type=int, default=10, help="Epochs for training baseline")
    parser.add_argument("--epochs-recover", type=int, default=5, help="Epochs for recovery training")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="coco", choices=["detrac", "coco"], help="Dataset to use")
    parser.add_argument("--img-dir", type=str, default="data/DETRAC-Images/DETRAC-Images", help="DETRAC dataset images path")
    parser.add_argument("--anno-dir", type=str, default="data/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML", help="DETRAC train annotations path")
    parser.add_argument("--val-anno-dir", type=str, default="data/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML", help="DETRAC validation annotations path")
    parser.add_argument("--coco-train-img", type=str, default="data/coco/train2017", help="COCO train images path")
    parser.add_argument("--coco-train-anno", type=str, default="data/coco/annotations/instances_train2017.json", help="COCO train annotations JSON")
    parser.add_argument("--coco-val-img", type=str, default="data/coco/val2017", help="COCO val images path")
    parser.add_argument("--coco-val-anno", type=str, default="data/coco/annotations/instances_val2017.json", help="COCO val annotations JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick experiments")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Folder to save artifacts")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-train", "--skip_train", action="store_true", help="Skip training baseline if checkpoint already exists")
    args = parser.parse_args()

    # Enforce reproducibility
    from utils.reproducibility import set_seed
    set_seed(args.seed)

    device_name = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # Resolve models list
    models_to_run = ["yolov5s", "detr"] if args.model == "all" else [args.model]
    
    # Resolve pruning types
    prune_types_to_run = []
    for pt in args.prune_types:
        if pt in PRUNER_REGISTRY:
            prune_types_to_run.append(pt)
        else:
            logger.warning(f"Pruning style '{pt}' is not registered. Skipping.")

    if not prune_types_to_run:
        logger.error("No valid pruning styles specified for matrix runs.")
        sys.exit(1)

    logger.info(f"Running experiments with Models: {models_to_run} | Methods: {prune_types_to_run} | Sparsities: {args.sparsities} | Dataset: {args.dataset.upper()}")

    manager = ExperimentManager(
        model_names=models_to_run,
        prune_types=prune_types_to_run,
        sparsities=args.sparsities,
        epochs_train=args.epochs_train,
        epochs_recover=args.epochs_recover,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        img_dir=args.img_dir,
        anno_dir=args.anno_dir,
        val_anno_dir=args.val_anno_dir,
        device=device,
        checkpoints_dir=args.checkpoints_dir,
        dataset=args.dataset,
        coco_train_img=args.coco_train_img,
        coco_train_anno=args.coco_train_anno,
        coco_val_img=args.coco_val_img,
        coco_val_anno=args.coco_val_anno,
        skip_train=args.skip_train
    )

    results = manager.run_all()
    logger.info(f"Experiments finished. Processed {len(results)} configurations successfully!")


if __name__ == "__main__":
    main()
