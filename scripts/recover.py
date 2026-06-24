#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import ModelFactory, extract_model_state_dict
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
    parser.add_argument("--force", action="store_true", help="Force re-train from scratch, ignoring resume checkpoints")
    parser.add_argument("--skip-resume", action="store_true", help="Skip resuming and start training from scratch")
    args = parser.parse_args()

    # Enforce reproducibility
    from utils.reproducibility import set_seed
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    artifact_manager = ArtifactManager()
    recovered_dir = artifact_manager.get_recovered_dir(args.model)
    pruned_pt = artifact_manager.get_pruned_checkpoint(args.model, args.prune_type, args.sparsity)

    # Configure file logging to write epoch logs to train.log
    log_file = os.path.join(recovered_dir, "train.log")
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
    model.load_state_dict(extract_model_state_dict(torch.load(pruned_pt, map_location=device, weights_only=False)), strict=False)

    # Verify loaded model sparsity
    total_weights = 0
    zero_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total_weights += w.numel()
            zero_weights += (w == 0.0).sum().item()
    loaded_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0
    logger.info(f"Verified loaded model sparsity: {loaded_sparsity*100:.2f}% (Expected target: {args.sparsity*100:.2f}%)")

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
    model_name_base = f"{args.prune_type}_{args.sparsity}"
    trainer = TrafficTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=recovered_dir,
        model_name=model_name_base,
        val_loader=val_loader,
        patience=args.patience,
        config=vars(args)
    )

    start_epoch = 1
    last_pt = artifact_manager.get_recovered_checkpoint(args.model, args.prune_type, args.sparsity, suffix='last')
    
    # Auto-resume if last_pt exists (skipped when --force or --skip-resume is used)
    if os.path.exists(last_pt) and not args.force and not args.skip_resume:
        logger.info(f"Resuming recovery training from checkpoint: {last_pt}")
        try:
            start_epoch = trainer.load_checkpoint(last_pt)
        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}. Starting from scratch.")

    # Run training loop
    logger.info(f"Starting recovery training for {args.epochs} epochs...")
    history = trainer.train(args.epochs, start_epoch=start_epoch)

    # Save recovery history to a JSON file
    recovery_history = os.path.join(recovered_dir, f"{model_name_base}_history.json")
    artifact_manager.save_metadata(history, recovery_history)
    logger.info(f"Saved recovery epoch history to: {recovery_history}")

    best_pt = artifact_manager.get_recovered_checkpoint(args.model, args.prune_type, args.sparsity, suffix='best')
    recovered_meta = artifact_manager.get_metadata_path(best_pt)
    
    if os.path.exists(best_pt):
        # Load best weights back to model for metadata calculations
        checkpoint = torch.load(best_pt, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        # Save a proper dictionary checkpoint as fallback
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "loss": history[-1].get("loss", 0.0) if (isinstance(history, list) and len(history) > 0) else 0.0,
            "best_map": trainer.best_map,
            "config": vars(args),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, best_pt)
        logger.info(f"Saved fallback recovered model dictionary to {best_pt}.")

    # Calculate actual sparsity after loading/finetuning
    total_weights = 0
    zero_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total_weights += w.numel()
            zero_weights += (w == 0.0).sum().item()
    actual_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0

    # Save recovery metadata
    best_map_val = trainer.best_map

    artifact_manager.save_metadata({
        "pruning_method": args.prune_type,
        "sparsity": args.sparsity,
        "actual_sparsity": actual_sparsity,
        "epochs_recover": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_map": best_map_val,
        "params": model.get_params_count(),
        "flops": model.calculate_flops((3,) + img_size),
        "config": vars(args),
        "timestamp": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__
    }, recovered_meta)

    logger.info("Stage C: Recovery fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
