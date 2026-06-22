import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from utils.artifact_manager import ArtifactManager

logger = logging.getLogger("BaseTrainer")


class BaseTrainer:
    """Abstract base trainer standardizing training, validation steps, and checkpoint recovery."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        checkpoint_dir: str = "checkpoints/finetuned",
        model_name: str = "model",
        val_loader: Optional[DataLoader] = None,
        patience: int = 10,
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.patience = patience
        self.use_amp = use_amp
        
        self.best_loss = float("inf")
        self.best_map = 0.0
        self.artifact_manager = ArtifactManager()
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

        # Check if the model contains pruning masks
        self.is_pruned = any(hasattr(m, "pruning_mask") for m in model.modules())
        if self.is_pruned:
            logger.info("Pruning masks detected. BaseTrainer will enforce sparsity on parameters and gradients.")

    def compute_loss(self, pred: torch.Tensor, targets: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Computes matching losses. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement compute_loss()")

    def train_epoch(self, epoch: int) -> float:
        """Trains the model for a single epoch using mixed precision (AMP) if enabled."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        start_time = time.time()

        for batch_idx, (imgs, targets) in enumerate(self.train_loader):
            batch_imgs = torch.stack(imgs).to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            
            # Autocast forward pass under AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp and self.device.type == "cuda"):
                outputs = self.model(batch_imgs)

            # Compute loss in FP32 to avoid BCE autocast safety exceptions
            with torch.amp.autocast("cuda", enabled=False):
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.float()
                elif isinstance(outputs, dict):
                    outputs = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                loss_dict = self.compute_loss(outputs, targets)
                loss = loss_dict["total_loss"]

            # Scale loss and backpropagate
            self.scaler.scale(loss).backward()

            # Apply gradient mask if model is pruned to freeze pruned connections
            if self.is_pruned:
                from pruning import zero_pruned_gradients
                zero_pruned_gradients(self.model)

            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Step optimizer and update scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Enforce exact zero weights in pruned slots
            if self.is_pruned:
                from pruning import enforce_sparsity
                enforce_sparsity(self.model)

            total_loss += loss.item()

            pass

        return total_loss / num_batches

    def train(self, epochs: int, start_epoch: int = 1) -> Dict[str, Any]:
        """Runs the complete training / fine-tuning iterations with epoch validation & early stopping."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        history = []
        patience_counter = 0
        
        last_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_last.pt")
        best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")

        for epoch in range(start_epoch, epochs + 1):
            logger.info(f"\n--- Starting Epoch {epoch}/{epochs} ---")
            avg_loss = self.train_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            logger.info(f"Epoch {epoch}/{epochs} Summary | Average Train Loss: {avg_loss:.4f}")
            epoch_history = {"epoch": epoch, "loss": avg_loss}

            # Save the current state as the last model checkpoint
            self.save_checkpoint(epoch, avg_loss, last_path)
            
            # Validation Step
            if self.val_loader is not None:
                from evaluation.validator import Validator
                logger.info("Running Epoch Validation...")
                validator = Validator(self.model, self.device)
                preds, gts = validator.gather_predictions(self.val_loader)
                val_metrics = validator.calculate_accuracy_metrics(preds, gts)
                
                val_map50 = val_metrics["mAP50"]
                val_map50_95 = val_metrics["mAP50-95"]
                val_precision = val_metrics["precision"]
                val_recall = val_metrics["recall"]
                
                logger.info(
                    f"Epoch {epoch} Validation | mAP50: {val_map50:.4f} | mAP50-95: {val_map50_95:.4f} | "
                    f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}"
                )
                
                epoch_history.update({
                    "val_mAP50": val_map50,
                    "val_mAP50_95": val_map50_95,
                    "val_precision": val_precision,
                    "val_recall": val_recall
                })
                
                # Check for metric improvement (Maximize mAP50)
                if val_map50 > self.best_map:
                    self.best_map = val_map50
                    self.save_checkpoint(epoch, avg_loss, best_path)
                    logger.info(f"🏆 New best checkpoint saved to: {best_path} (mAP50: {self.best_map:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Early stopping patience: {patience_counter}/{self.patience}")
            else:
                # Track best by minimizing training loss if val_loader is not provided
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint(epoch, avg_loss, best_path)
                    logger.info(f"🏆 New best checkpoint saved to: {best_path} (Loss: {self.best_loss:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Early stopping patience: {patience_counter}/{self.patience}")

            history.append(epoch_history)

            # Early stopping check
            if patience_counter >= self.patience:
                logger.warning(f"🛑 Early stopping triggered at epoch {epoch} (patience limit of {self.patience} reached)")
                break

        return {"best_loss": self.best_loss, "best_map": self.best_map, "history": history}

    def save_checkpoint(self, epoch: int, loss: float, save_path: str) -> None:
        """Saves current state to a checkpoint file."""
        self.artifact_manager.save_checkpoint(
            model=self.model,
            save_path=save_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            loss=loss
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Loads states from checkpoint and returns next epoch to resume training."""
        status = self.artifact_manager.load_checkpoint(
            model=self.model,
            load_path=checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        self.best_loss = status.get("loss", float("inf"))
        next_epoch = status.get("epoch", 0) + 1
        return next_epoch
