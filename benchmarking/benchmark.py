import os
import logging
import torch
import torch.nn as nn
import pandas as pd
import json
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader
from evaluation.validator import Validator
from .base import BaseBenchmark

logger = logging.getLogger("TrafficBenchmark")


class TrafficBenchmark(BaseBenchmark):
    """Benchmark class for evaluating computer vision models on the traffic analysis task."""

    def __init__(self, model_name: str, device: torch.device, img_size: Tuple[int, int, int] = (3, 640, 640)):
        super().__init__(model_name, device, img_size)
        self.validator = None

    def evaluate_checkpoint(self, model: nn.Module, loader: DataLoader, checkpoint_path: Optional[str] = None, dataset_type: Optional[str] = None) -> Dict[str, Any]:
        """Loads and completely evaluates a model, computing structural, speed, and accuracy metrics."""
        # Check if the model has dynamic pruning hooks and bake them to get accurate speed/size metrics
        has_masks = any(hasattr(module, 'weight_orig') for module in model.modules())
        if has_masks:
            logger.info("Baking pruning masks into weights to eliminate forward hook overhead during benchmarking...")
            from utils.pipeline_utils import bake_pruned_weights
            num_baked = bake_pruned_weights(model)
            logger.info(f"Successfully baked weights and removed dynamic hooks for {num_baked} modules.")

        model = model.to(self.device)

        self.validator = Validator(model, self.device)
        
        logger.info(f"Evaluating model structure...")
        structural_metrics = self.profile_model(model)
        
        # Lưu dense size (in-memory) riêng — không ghi đè
        dense_size_mb = structural_metrics["size_mb"]
        file_size_mb = dense_size_mb  # fallback nếu không có checkpoint path
        
        # Override model size with actual file size on disk if checkpoint_path is provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            file_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
            logger.info(f"Actual checkpoint file size: {file_size_mb:.2f} MB (dense in-memory: {dense_size_mb:.2f} MB)")
            structural_metrics["size_mb"] = file_size_mb
        
        compression_ratio = dense_size_mb / file_size_mb if file_size_mb > 0 else 1.0
        
        logger.info(f"Evaluating inference speed on device: {self.device}...")
        speed_metrics = self.profile_speed(model)
        
        logger.info(f"Evaluating accuracy metrics (mAP, Precision, Recall)...")
        try:
            if dataset_type and dataset_type.lower() == "coco":
                accuracy_metrics = self.validator.calculate_accuracy_metrics_from_dataloader(loader)
            else:
                preds, gts = self.validator.gather_predictions(loader)
                accuracy_metrics = self.validator.calculate_accuracy_metrics(preds, gts)
        except Exception as e:
            logger.error(f"Failed to evaluate accuracy metrics: {e}", exc_info=True)
            accuracy_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0
            }
        
        results = {
            "Model": self.model_name.upper(),
            "Params": structural_metrics["params"],
            "FLOPs": structural_metrics["flops"],
            "Size (MB)": structural_metrics["size_mb"],
            "Dense Size (MB)": dense_size_mb,
            "Compression Ratio": round(compression_ratio, 2),
            "Sparsity": structural_metrics["sparsity"],
            "Latency (ms)": speed_metrics["latency_ms"],
            "Latency Std (ms)": speed_metrics.get("latency_std_ms", 0.0),
            "Latency Median (ms)": speed_metrics.get("latency_median_ms", 0.0),
            "FPS": speed_metrics["fps"],
            "Precision": accuracy_metrics["precision"],
            "Recall": accuracy_metrics["recall"],
            "mAP50": accuracy_metrics["mAP50"],
            "mAP50-95": accuracy_metrics["mAP50-95"]
        }
        
        logger.info(
            f"Evaluation Finished | mAP50: {results['mAP50']:.4f} | FPS: {results['FPS']:.2f} | "
            f"Sparsity: {results['Sparsity'] * 100:.2f}% | Size: {results['Size (MB)']:.2f} MB"
        )
        return results

    @staticmethod
    def export_results(results: List[Dict[str, Any]], csv_path: str = "reports/benchmark.csv", json_path: str = "reports/benchmark.json"):
        """Exports a list of evaluation results to CSV, JSON, and Markdown format."""
        if not results:
            logger.warning("Results list is empty. Skipping file export.")
            return

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported benchmark results to CSV: {csv_path}")
        
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Exported benchmark results to JSON: {json_path}")
        
        # Build a Markdown table summary
        md_path = csv_path.replace(".csv", "_report.md")
        with open(md_path, "w") as f:
            f.write("# Benchmarking Comparative Report\n\n")
            f.write(df.to_markdown(index=False))
        logger.info(f"Exported benchmark report to Markdown: {md_path}")
