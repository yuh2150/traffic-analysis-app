import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import logging

try:
    from torchvision.ops import box_iou as tv_box_iou, generalized_box_iou as tv_giou
except ImportError:
    tv_box_iou = None
    tv_giou = None

from models.models import get_model, BaseTrafficDetector
from datasets.dataset import get_data_loader
from pruning.pruner import add_sparse_regularization

logger = logging.getLogger("Training")


def compute_detection_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    num_classes: int,
    box_weight: float = 5.0,
    cls_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pred_boxes = pred["boxes"]
    pred_scores = pred["scores"]
    pred_ids = pred["class_ids"]
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    losses = {}
    total_loss = torch.tensor(0.0, device=pred_boxes.device)

    # Match predictions to ground truth via simple assignment
    # For fine-tuning a pretrained detector, a per-prediction assignment is sufficient
    for b in range(pred_boxes.shape[0]):
        pboxes = pred_boxes[b]
        gboxes = gt_boxes[b]
        pids = pred_ids[b]
        glabels = gt_labels[b]

        if gboxes.numel() == 0 or pboxes.numel() == 0:
            if pboxes.numel() > 0:
                total_loss = total_loss + cls_weight * nn.functional.cross_entropy(
                    pred_scores[b].unsqueeze(0),
                    torch.full((1,), num_classes, dtype=torch.long, device=pboxes.device),
                )
            continue

        # Compute IoU matrix and match each GT to the best prediction
        ious = tv_box_iou(gboxes, pboxes)
        matched_gt, matched_pred = [], []
        for gt_idx in range(gboxes.shape[0]):
            best_iou, best_pred = ious[gt_idx].max(0)
            if best_iou > 0.5:
                matched_gt.append(gt_idx)
                matched_pred.append(best_pred.item())

        if not matched_pred:
            continue

        matched_pred_t = torch.tensor(matched_pred, dtype=torch.long, device=pboxes.device)
        matched_gt_t = torch.tensor(matched_gt, dtype=torch.long, device=pboxes.device)

        tgt_boxes = gboxes[matched_gt_t]
        tgt_labels = glabels[matched_gt_t]

        # Box loss (L1 on normalized coords)
        mp_boxes = pboxes[matched_pred_t]
        box_loss = nn.functional.l1_loss(mp_boxes, tgt_boxes, reduction="sum")

        # GIoU loss
        giou_loss = 1 - torch.diag(
            tv_giou(mp_boxes, tgt_boxes)
        ).mean()

        # Class loss
        cls_loss = nn.functional.cross_entropy(
            pred_scores[b, matched_pred_t].unsqueeze(0),
            tgt_labels.unsqueeze(0),
        )

        total_loss = total_loss + box_weight * box_loss + box_weight * giou_loss + cls_weight * cls_loss

    losses["total_loss"] = total_loss / max(pred_boxes.shape[0], 1)
    return losses


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    sparse: bool = False,
    sr: float = 0.0001,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    start_time = time.time()

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        batch_imgs = torch.stack(imgs).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(batch_imgs)

        loss = 0.0
        for i in range(len(targets)):
            tgt = {
                "boxes": targets[i]["boxes"].unsqueeze(0),
                "labels": targets[i]["labels"].unsqueeze(0),
            }
            pred = {
                "boxes": outputs["boxes"][i].unsqueeze(0),
                "scores": outputs["scores"][i].unsqueeze(0),
                "class_ids": outputs["class_ids"][i].unsqueeze(0),
            }
            loss_dict = compute_detection_loss(pred, tgt, model.num_classes)
            loss = loss + loss_dict["total_loss"]

        # Sparse training: L1 regularization on BatchNorm gamma
        if sparse:
            loss = add_sparse_regularization(loss, model, sr)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                f"Loss: {loss.item():.4f}  "
                f"({(time.time() - start_time) / (batch_idx + 1):.2f}s/batch)"
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    from validate_model import evaluate_map, evaluate_precision, evaluate_recall

    model.eval()
    mAP50, mAP50_95 = evaluate_map(model, dataloader, device)
    precision = evaluate_precision(model, dataloader, device)
    recall = evaluate_recall(model, dataloader, device)

    return {"mAP50": mAP50, "mAP50_95": mAP50_95, "precision": precision, "recall": recall}


def fine_tune(
    model: BaseTrafficDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "weights/finetuned",
    patience: int = 3,
    model_name: str = "model",
    sparse: bool = False,
    sr: float = 0.0001,
) -> BaseTrafficDetector:
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Freeze backbone for first half of training
    freeze_backbone = True
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "backbone" in name or "model.0" in name or "model.1" in name or "model.2" in name:
                param.requires_grad = False

    # Count trainable params after freezing
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Fine-tuning {trainable:,}/{total:,} parameters ({100 * trainable / total:.1f}%)")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_map = 0.0
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Unfreeze backbone after 3 epochs
        if epoch == 4 and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = True
            logger.info("Backbone unfrozen")
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - 3)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, sparse=sparse, sr=sr)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{num_epochs}  "
            f"Train Loss: {train_loss:.4f}  "
            f"mAP50: {val_metrics['mAP50']:.4f}  "
            f"mAP50-95: {val_metrics['mAP50_95']:.4f}  "
            f"Prec: {val_metrics['precision']:.3f}  "
            f"Rec: {val_metrics['recall']:.3f}"
        )

        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        if val_metrics["mAP50"] > best_map:
            best_map = val_metrics["mAP50"]
            no_improve = 0
            best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model: {best_path} (mAP50={best_map:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break

    # Load best checkpoint
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger.info(f"Loaded best checkpoint: {best_path}")

    return model


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune pretrained detector on UA-DETRAC")
    parser.add_argument("--model", type=str, default="yolov8n", choices=["yolov8n", "yolov8s", "detr"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-dir", type=str, default="data/UA-DETRAC/train")
    parser.add_argument("--anno-dir", type=str, default="data/UA-DETRAC/annotations")
    parser.add_argument("--checkpoint-dir", type=str, default="weights/finetuned")
    parser.add_argument("--sparse", action="store_true", help="Enable sparse training (L1 on BN gamma)")
    parser.add_argument("--sr", type=float, default=0.0001, help="Sparsity ratio for BN gamma regularization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model with pretrained weights
    model = get_model(args.model)
    model = model.to(device)

    logger.info(f"Model: {args.model}  Params: {model.get_params_count():,}")

    # Data loaders
    img_size = (640, 640) if "yolo" in args.model else (800, 800)
    train_loader = get_data_loader(
        img_dir=args.img_dir,
        anno_dir=args.anno_dir,
        batch_size=args.batch_size,
        img_size=img_size,
        shuffle=True,
    )
    val_loader = get_data_loader(
        img_dir=args.img_dir,
        anno_dir=args.anno_dir,
        batch_size=args.batch_size,
        img_size=img_size,
        shuffle=False,
    )

    model = fine_tune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model,
        sparse=args.sparse,
        sr=args.sr,
    )

    logger.info("Fine-tuning complete!")
