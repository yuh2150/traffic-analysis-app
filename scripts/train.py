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

from models.factory import ModelFactory
from datasets.factory import DatasetFactory
from training.trainer import TrafficTrainer
from utils.artifact_manager import ArtifactManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainStage")


def main():
    parser = argparse.ArgumentParser(description="Stage A: Train Baseline Model")
    parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--dataset", type=str, default="coco", choices=["detrac", "coco"], help="Dataset to use")
    parser.add_argument("--img-dir", type=str, default="data/DETRAC-Images/DETRAC-Images", help="DETRAC dataset images path")
    parser.add_argument("--anno-dir", type=str, default="data/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML", help="DETRAC dataset annotations path")
    parser.add_argument("--val-anno-dir", type=str, default="data/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML", help="DETRAC validation annotations path")
    parser.add_argument("--coco-train-img", type=str, default="data/coco/train2017", help="COCO train images path")
    parser.add_argument("--coco-train-anno", type=str, default="data/coco/annotations/instances_train2017.json", help="COCO train annotations JSON")
    parser.add_argument("--coco-val-img", type=str, default="data/coco/val2017", help="COCO val images path")
    parser.add_argument("--coco-val-anno", type=str, default="data/coco/annotations/instances_val2017.json", help="COCO val annotations JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of dataset samples")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume training from")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Enforce reproducibility
    from utils.reproducibility import set_seed
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    artifact_manager = ArtifactManager()
    model_dir = artifact_manager.get_model_dir(args.model)
    
    model_name_base = "coco_baseline" if args.dataset == "coco" else "baseline"
    baseline_pt = artifact_manager.get_checkpoint_path(args.model, f"{model_name_base}.pt")
    baseline_meta = artifact_manager.get_checkpoint_path(args.model, f"{model_name_base}_metadata.json")

    # Configure file logging to write epoch logs to train.log
    log_file = os.path.join(model_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode="a" if args.resume else "w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging training progress to file: {log_file}")
    logger.info(f"Dataset type selected: {args.dataset.upper()}")

    # Load model
    num_classes = 80 if args.dataset == "coco" else 4
    model = ModelFactory.load(args.model, num_classes=num_classes, device=device)

    # Determine paths based on dataset
    if args.dataset == "coco":
        train_img = args.coco_train_img
        train_anno = args.coco_train_anno
        val_img = args.coco_val_img
        val_anno = args.coco_val_anno
    else:
        train_img = args.img_dir
        train_anno = args.anno_dir
        val_img = args.img_dir
        val_anno = args.val_anno_dir

    # Dataloader
    img_size = (640, 640) if args.model == "yolov5s" else (800, 800)
    train_loader = DatasetFactory.get_dataloader(
        img_dir=train_img,
        anno_dir=train_anno,
        batch_size=args.batch_size,
        img_size=img_size,
        shuffle=True,
        max_samples=args.max_samples,
        dataset_type=args.dataset,
    )
    
    # Validation Dataloader
    val_loader = DatasetFactory.get_dataloader(
        img_dir=val_img,
        anno_dir=val_anno,
        batch_size=args.batch_size,
        img_size=img_size,
        shuffle=False,
        max_samples=args.max_samples,
        dataset_type=args.dataset,
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Trainer
    trainer = TrafficTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=model_dir,
        model_name=model_name_base,
        val_loader=val_loader,
        patience=args.patience
    )

    start_epoch = 1
    last_pt = os.path.join(model_dir, f"{model_name_base}_last.pt")
    
    # Auto-resume if last_pt exists, or use explicitly passed --resume path
    resume_path = args.resume if args.resume else (last_pt if os.path.exists(last_pt) else "")
    if resume_path:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        start_epoch = trainer.load_checkpoint(resume_path)

    # Run training
    history = trainer.train(args.epochs, start_epoch=start_epoch)

    # Save training history to a JSON file
    baseline_history = artifact_manager.get_checkpoint_path(args.model, f"{model_name_base}_history.json")
    artifact_manager.save_metadata(history, baseline_history)
    logger.info(f"Saved training epoch history to: {baseline_history}")

    # Copy the best model checkpoint to baseline_pt for backwards compatibility
    best_pt = os.path.join(model_dir, f"{model_name_base}_best.pt")
    if os.path.exists(best_pt):
        import shutil
        shutil.copy(best_pt, baseline_pt)
        logger.info(f"Copied best model {best_pt} to {baseline_pt} for compatibility.")
        
        # Load best weights back to model for metadata flops calculations
        checkpoint = torch.load(best_pt, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        torch.save(model.state_dict(), baseline_pt)

    # Save baseline metadata
    artifact_manager.save_metadata({
        "params": model.get_params_count(),
        "flops": model.calculate_flops((3,) + img_size),
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_map": history.get("best_map", 0.0)
    }, baseline_meta)
    
    logger.info("Stage A: Baseline training completed successfully!")


if __name__ == "__main__":
    main()
