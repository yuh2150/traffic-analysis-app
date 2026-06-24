import os
import logging
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict, Any, Optional
from models.factory import ModelFactory, extract_model_state_dict
from datasets.factory import DatasetFactory
from training.trainer import TrafficTrainer
from pruning.base import PRUNER_REGISTRY
from benchmarking.benchmark import TrafficBenchmark
from utils.artifact_manager import ArtifactManager

logger = logging.getLogger("ExperimentManager")


class ExperimentManager:
    """Orchestrates the modular pipeline across multiple models, pruners, and sparsities."""

    def __init__(
        self,
        model_names: List[str],
        prune_types: List[str],
        sparsities: List[float],
        epochs_train: int = 10,
        epochs_recover: int = 5,
        batch_size: int = 4,
        lr: float = 1e-4,
        max_samples: Optional[int] = None,
        img_dir: str = "data/DETRAC-Images/DETRAC-Images",
        anno_dir: str = "data/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML",
        val_anno_dir: str = "data/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML",
        device: Optional[torch.device] = None,
        checkpoints_dir: str = "checkpoints",
        dataset: str = "coco",
        coco_train_img: str = "data/coco/train2017",
        coco_train_anno: str = "data/coco/annotations/instances_train2017.json",
        coco_val_img: str = "data/coco/val2017",
        coco_val_anno: str = "data/coco/annotations/instances_val2017.json",
        skip_train: bool = False
    ):
        self.model_names = model_names
        self.prune_types = prune_types
        self.sparsities = sparsities
        self.epochs_train = epochs_train
        self.epochs_recover = epochs_recover
        self.batch_size = batch_size
        self.lr = lr
        self.max_samples = max_samples
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.val_anno_dir = val_anno_dir
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints_dir = checkpoints_dir
        self.artifact_manager = ArtifactManager(checkpoints_dir)
        self.dataset = dataset
        self.coco_train_img = coco_train_img
        self.coco_train_anno = coco_train_anno
        self.coco_val_img = coco_val_img
        self.coco_val_anno = coco_val_anno
        self.skip_train = skip_train

    def run_all(self) -> List[Dict[str, Any]]:
        """Runs the matrix of models, pruning strategies, and sparsities, returning all evaluation stats."""
        all_results = []
        
        # Load any existing results to enable skipping completed experiments
        existing_csv = "benchmark_results.csv"
        existing_results = {}
        if os.path.exists(existing_csv):
            try:
                df_existing = pd.read_csv(existing_csv)
                for _, row in df_existing.iterrows():
                    key = (row["Model"].upper(), row["Config"])
                    existing_results[key] = row.to_dict()
                logger.info(f"Loaded {len(existing_results)} existing configuration records for resume-support.")
            except Exception as e:
                logger.warning(f"Could not load existing benchmark CSV for resuming: {e}")

        for model_name in self.model_names:
            logger.info(f"========== Starting Experiment for Model: {model_name.upper()} ==========")
            
            # Setup loaders
            img_size_tuple = (640, 640) if "yolov5" in model_name.lower() else (800, 800)
            
            if self.dataset == "coco":
                train_img = self.coco_train_img
                train_anno = self.coco_train_anno
                val_img = self.coco_val_img
                val_anno = self.coco_val_anno
            else:
                train_img = self.img_dir
                train_anno = self.anno_dir
                val_img = self.img_dir
                val_anno = self.val_anno_dir

            train_loader = DatasetFactory.get_dataloader(
                img_dir=train_img,
                anno_dir=train_anno,
                batch_size=self.batch_size,
                img_size=img_size_tuple,
                shuffle=True,
                max_samples=self.max_samples,
                dataset_type=self.dataset,
            )
            val_loader = DatasetFactory.get_dataloader(
                img_dir=val_img,
                anno_dir=val_anno,
                batch_size=self.batch_size,
                img_size=img_size_tuple,
                shuffle=False,
                max_samples=self.max_samples,
                dataset_type=self.dataset,
            )



            
            if len(val_loader) == 0:
                raise ValueError("val_loader is empty! Ensure validation dataset has valid images and annotations.")
            
            # Ensure baseline checkpoint exists
            baseline_pt = self.artifact_manager.get_baseline_checkpoint(model_name, suffix='best')
            baseline_meta = self.artifact_manager.get_metadata_path(baseline_pt)
            
            # Load model structure
            num_classes = 80 if self.dataset == "coco" else 4
            model = ModelFactory.load(model_name, device=self.device, num_classes=num_classes)
            
            # Check if baseline is already evaluated in CSV
            baseline_key = (model_name.upper(), "Baseline FP32")
            if baseline_key in existing_results:
                logger.info(f"Found completed baseline FP32 record in CSV. Loading and skipping training.")
                baseline_results = existing_results[baseline_key]
                all_results.append(baseline_results)
                
                # Make sure the baseline weights exist on disk for pruners to load
                if not os.path.exists(baseline_pt):
                    logger.warning("Baseline weight file missing. Rerunning baseline training...")
                    baseline_key = None # Clear key to force retrain
            
            if baseline_key not in existing_results:
                # 1. Train baseline if not found or skipped
                if not os.path.exists(baseline_pt):
                    if self.skip_train:
                        raise FileNotFoundError(f"Baseline checkpoint not found at {baseline_pt} and skip_train is True.")
                    logger.info(f"Baseline checkpoint not found. Training baseline for {self.epochs_train} epochs...")
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs_train)
                    
                    baseline_dir = self.artifact_manager.get_baseline_dir(model_name)
                    trainer = TrafficTrainer(
                        model=model,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=self.device,
                        checkpoint_dir=baseline_dir,
                        model_name="",
                        val_loader=val_loader
                    )
                    trainer.train(self.epochs_train)
                    
                    # Load best weights back if saved
                    if os.path.exists(baseline_pt):
                        checkpoint = torch.load(baseline_pt, map_location=self.device, weights_only=False)
                        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        checkpoint = {
                            "model_state_dict": model.state_dict(),
                            "epoch": self.epochs_train,
                            "loss": 0.0,
                            "best_map": trainer.best_map,
                            "config": {}
                        }
                        torch.save(checkpoint, baseline_pt)
                    
                    # Save metadata
                    self.artifact_manager.save_metadata({
                        "params": model.get_params_count(),
                        "flops": model.calculate_flops((3,) + img_size_tuple),
                        "dataset": self.dataset,
                        "epochs": self.epochs_train,
                        "lr": self.lr,
                        "batch_size": self.batch_size,
                        "best_map": trainer.best_map
                    }, baseline_meta)
                else:
                    logger.info(f"Baseline checkpoint already exists at: {baseline_pt}. Loading weights.")
                    checkpoint = torch.load(baseline_pt, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                    model.load_state_dict(state_dict, strict=False)
                
                # Run evaluation on baseline
                benchmark = TrafficBenchmark(model_name, self.device, (3,) + img_size_tuple)
                baseline_results = benchmark.evaluate_checkpoint(model, val_loader)
                baseline_results["Config"] = "Baseline FP32"
                baseline_results["Compression Ratio"] = 1.0
                baseline_results["Throughput (img/s)"] = baseline_results["FPS"]
                baseline_results["Sparsity"] = 0.0
                baseline_results["Pruning Type"] = "None"
                all_results.append(baseline_results)
            else:
                # Still need baseline_results for compression ratio comparisons
                checkpoint = torch.load(baseline_pt, map_location=self.device, weights_only=False)
                state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                model.load_state_dict(state_dict, strict=False)
                benchmark = TrafficBenchmark(model_name, self.device, (3,) + img_size_tuple)
                baseline_results = existing_results[baseline_key]

            import copy
            baseline_model_clone = copy.deepcopy(model)

            # 2. Iterate pruning matrix
            for prune_type in self.prune_types:
                if prune_type not in PRUNER_REGISTRY:
                    logger.warning(f"Unknown pruning strategy '{prune_type}'. Skipping.")
                    continue
                    
                for sparsity in self.sparsities:
                    config_label = f"{prune_type.replace('_', ' ').capitalize()} Pruned ({int(sparsity*100)}%)"
                    
                    # Handle layer/filter/channel name normalizations used by GUI
                    if prune_type == "layer":
                        config_label = "Layer Pruned"
                    elif prune_type == "filter" and sparsity == 0.4:
                        config_label = "Filter Pruned (40%)"
                    elif prune_type == "channel" and sparsity == 0.4:
                        config_label = "Channel Pruned (40%)"

                    # Check if this config is already completed in CSV
                    config_key = (model_name.upper(), config_label)
                    if config_key in existing_results:
                        logger.info(f"Configuration '{config_label}' already completed. Skipping.")
                        all_results.append(existing_results[config_key])
                        continue

                    logger.info(f"--- Running Pruning Configuration: {prune_type} | Sparsity: {sparsity} ---")
                    
                    pruned_pt = self.artifact_manager.get_pruned_checkpoint(model_name, prune_type, sparsity)
                    pruned_meta = self.artifact_manager.get_metadata_path(pruned_pt)
                    recovered_pt = self.artifact_manager.get_recovered_checkpoint(model_name, prune_type, sparsity, suffix='best')
                    recovered_meta = self.artifact_manager.get_metadata_path(recovered_pt)
                    
                    # Ensure pruned weight is generated by cloning baseline
                    pruned_model = copy.deepcopy(baseline_model_clone)
                    
                    if not os.path.exists(pruned_pt):
                        logger.info(f"Applying pruner: {prune_type}...")
                        pruner_cls = PRUNER_REGISTRY[prune_type]
                        pruner = pruner_cls(pruned_model, sparsity)
                        pruned_model = pruner.prune()
                        
                        torch.save(pruned_model.state_dict(), pruned_pt)
                        
                        # Collect and save pruning metadata
                        stats = pruner.collect_statistics()
                        actual_sparsity = stats["sparsity"]
                        active_params = stats["active_params"]
                        
                        if prune_type == "layer":
                            pruned_flops = pruned_model.calculate_flops((3,) + img_size_tuple)
                        else:
                            pruned_flops = int(baseline_results["FLOPs"] * (1.0 - actual_sparsity))
                            
                        self.artifact_manager.save_metadata({
                            "pruning_method": prune_type,
                            "sparsity": sparsity,
                            "actual_sparsity": actual_sparsity,
                            "params": active_params,
                            "flops": pruned_flops,
                            "compression_ratio": stats["total_params"] / max(active_params, 1),
                            "size_reduction_mb": (stats["total_params"] - active_params) * 4 / (1024 ** 2)
                        }, pruned_meta)
                    else:
                        logger.info(f"Pruned checkpoint already exists. Loading {pruned_pt}...")
                        pruner_cls = PRUNER_REGISTRY[prune_type]
                        pruner = pruner_cls(pruned_model, sparsity)
                        pruned_model = pruner.prune()
                        pruned_model.load_state_dict(extract_model_state_dict(torch.load(pruned_pt, map_location=self.device, weights_only=False)), strict=False)
                        
                        # Verify sparsity
                        total_weights = 0
                        zero_weights = 0
                        for name, module in pruned_model.named_modules():
                            if isinstance(module, (nn.Conv2d, nn.Linear)):
                                w = module.weight.data
                                total_weights += w.numel()
                                zero_weights += (w == 0.0).sum().item()
                        actual_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0
                        if abs(actual_sparsity - sparsity) > 0.05:
                            logger.warning(
                                f"Significant sparsity mismatch detected after loading {pruned_pt}: "
                                f"Actual Sparsity = {actual_sparsity:.4f}, Expected Sparsity = {sparsity:.4f}"
                            )

                    # Ensure recovered weight is generated
                    if not os.path.exists(recovered_pt):
                        logger.info(f"Recovered checkpoint not found. Fine-tuning recovery for {self.epochs_recover} epochs...")
                        # Optimizer only updates parameters that require gradients
                        optimizer_p = torch.optim.AdamW([p for p in pruned_model.parameters() if p.requires_grad], lr=self.lr, weight_decay=1e-4)
                        scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p, T_max=self.epochs_recover)
                        
                        recovered_dir = self.artifact_manager.get_recovered_dir(model_name)
                        trainer_p = TrafficTrainer(
                            model=pruned_model,
                            train_loader=train_loader,
                            optimizer=optimizer_p,
                            scheduler=scheduler_p,
                            device=self.device,
                            checkpoint_dir=recovered_dir,
                            model_name=f"{prune_type}_{sparsity}",
                            val_loader=val_loader
                        )
                        trainer_p.train(self.epochs_recover)
                        
                        # Load best recovered weights back if saved
                        if os.path.exists(recovered_pt):
                            checkpoint = torch.load(recovered_pt, map_location=self.device, weights_only=False)
                            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                            pruned_model.load_state_dict(state_dict, strict=False)
                        else:
                            checkpoint = {
                                "model_state_dict": pruned_model.state_dict(),
                                "epoch": self.epochs_recover,
                                "loss": 0.0,
                                "best_map": trainer_p.best_map,
                                "config": {}
                            }
                            torch.save(checkpoint, recovered_pt)
                            
                        # Save recovery metadata
                        self.artifact_manager.save_metadata({
                            "pruning_method": prune_type,
                            "sparsity": sparsity,
                            "epochs_recover": self.epochs_recover,
                            "best_map": trainer_p.best_map
                        }, recovered_meta)
                    else:
                        logger.info(f"Recovered checkpoint already exists. Loading {recovered_pt}...")
                        pruned_model.load_state_dict(extract_model_state_dict(torch.load(recovered_pt, map_location=self.device, weights_only=False)), strict=False)
                        
                        # Verify sparsity
                        total_weights = 0
                        zero_weights = 0
                        for name, module in pruned_model.named_modules():
                            if isinstance(module, (nn.Conv2d, nn.Linear)):
                                w = module.weight.data
                                total_weights += w.numel()
                                zero_weights += (w == 0.0).sum().item()
                        actual_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0
                        if abs(actual_sparsity - sparsity) > 0.05:
                            logger.warning(
                                f"Significant sparsity mismatch detected after loading {recovered_pt}: "
                                f"Actual Sparsity = {actual_sparsity:.4f}, Expected Sparsity = {sparsity:.4f}"
                            )

                    # Benchmark recovered model on Test set
                    rec_results = benchmark.evaluate_checkpoint(pruned_model, val_loader)
                    rec_results["Config"] = config_label
                    rec_results["Compression Ratio"] = baseline_results["Size (MB)"] / rec_results["Size (MB)"] if rec_results["Size (MB)"] > 0 else 1.0
                    rec_results["Throughput (img/s)"] = rec_results["FPS"]
                    rec_results["Sparsity"] = sparsity
                    rec_results["Pruning Type"] = prune_type.replace("_", " ").capitalize()
                    all_results.append(rec_results)

        # Export consolidated reports
        TrafficBenchmark.export_results(all_results, csv_path="reports/benchmark_results.csv", json_path="reports/benchmark_results.json")
        
        # Copy to root directory for GUI Dashboard
        import shutil
        try:
            shutil.copy("reports/benchmark_results.csv", "benchmark_results.csv")
            logger.info("Copied reports/benchmark_results.csv to root directory for GUI dashboard.")
        except Exception as e:
            logger.warning(f"Could not copy benchmark results CSV to root: {e}")
            
        df_all = pd.DataFrame(all_results)
        df_all.to_excel("benchmark_results.xlsx", index=False)
        logger.info("Saved benchmark_results.xlsx to the root directory.")
        
        # Phase E: Generate Matplotlib Visualizations automatically
        try:
            from utils.visualization import generate_visualizations
            generate_visualizations(csv_path="reports/benchmark_results.csv", output_dir="reports")
        except Exception as e:
            logger.error(f"Failed to generate Matplotlib visualizations: {e}")
        
        return all_results
