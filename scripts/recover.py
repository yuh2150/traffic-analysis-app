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
from pruning.base import PRUNER_REGISTRY
from utils.artifact_manager import ArtifactManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RecoverStage")


def main():
    parser = argparse.ArgumentParser(description="Stage C: Run Recovery Fine-tuning")
    parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr"], help="Model architecture")
    parser.add_argument("--prune-type", type=str, default="magnitude", choices=list(PRUNER_REGISTRY.keys()), help="Pruning method")
    parser.add_argument("--sparsity", type=float, default=0.3, help="Pruning sparsity ratio")
    parser.add_argument("--epochs", type=int, default=5, help="Number of recovery epochs")
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
    pruned_pt = artifact_manager.get_checkpoint_path(args.model, f"{args.prune_type}_{args.sparsity}.pt")

    # Configure file logging to write epoch logs to train.log
    log_file = os.path.join(model_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging recovery training progress to file: {log_file}")
    logger.info(f"Dataset type selected: {args.dataset.upper()}")

    if not os.path.exists(pruned_pt):
        logger.error(f"Pruned checkpoint not found at: {pruned_pt}. Please run scripts/prune.py first.")
        sys.exit(1)

    # 1. Load model structure
    logger.info("Initializing model structure and registering pruning masks...")
    num_classes = 80 if args.dataset == "coco" else 4
    model = ModelFactory.load(args.model, num_classes=num_classes, device=device)
    
    # Register masks by applying same pruner to register buffers
    pruner_cls = PRUNER_REGISTRY[args.prune_type]
    pruner = pruner_cls(model, args.sparsity)
    model = pruner.prune()

    # Load weights
    logger.info(f"Loading pruned weight state from: {pruned_pt}")
    model.load_state_dict(torch.load(pruned_pt, map_location=device, weights_only=False), strict=False)

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

    # 2. Dataloaders
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
    val_loader = DatasetFactory.get_dataloader(
        img_dir=val_img,
        anno_dir=val_anno,
        batch_size=args.batch_size,
        img_size=img_size,
        shuffle=False,
        max_samples=args.max_samples,
        dataset_type=args.dataset,
    )

    # 3. Setup optimizer and scheduler (Optimizer only updates parameters that require gradients)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Traffic Trainer
    model_name_base = f"{args.prune_type}_{args.sparsity}_recover"
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
    
    # Auto-resume if last_pt exists
    if os.path.exists(last_pt):
        logger.info(f"Resuming recovery training from checkpoint: {last_pt}")
        start_epoch = trainer.load_checkpoint(last_pt)

    # Run training loop
    logger.info(f"Starting recovery training for {args.epochs} epochs...")
    history = trainer.train(args.epochs, start_epoch=start_epoch)

    # Save recovery history to a JSON file
    recovery_history = artifact_manager.get_checkpoint_path(args.model, f"{args.prune_type}_{args.sparsity}_recover_history.json")
    artifact_manager.save_metadata(history, recovery_history)
    logger.info(f"Saved recovery epoch history to: {recovery_history}")

    # Copy the best checkpoint to recovered_pt for downstream compatibility
    recovered_pt = artifact_manager.get_checkpoint_path(args.model, f"{args.prune_type}_{args.sparsity}_recovered.pt")
    recovered_meta = artifact_manager.get_checkpoint_path(args.model, f"{args.prune_type}_{args.sparsity}_recovered_metadata.json")
    
    best_pt = os.path.join(model_dir, f"{model_name_base}_best.pt")
    if os.path.exists(best_pt):
        import shutil
        shutil.copy(best_pt, recovered_pt)
        logger.info(f"Copied best recovered weights from {best_pt} to {recovered_pt}")
        
        # Load best weights back to model for metadata calculations
        checkpoint = torch.load(best_pt, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        torch.save(model.state_dict(), recovered_pt)
        logger.info(f"Saved final recovered weights directly to: {recovered_pt}")

    # Save recovery metadata
    artifact_manager.save_metadata({
        "pruning_method": args.prune_type,
        "sparsity": args.sparsity,
        "epochs_recover": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_map": history.get("best_map", 0.0),
        "params": model.get_params_count(),
        "flops": model.calculate_flops((3,) + img_size)
    }, recovered_meta)

    logger.info("Stage C: Recovery fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
