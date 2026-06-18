import os
import torch
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from models.models import get_model, BaseTrafficDetector
from datasets.dataset import get_data_loader
from pruning.pruner import (
    get_effective_metrics,
    prune_by_magnitude,
    prune_by_l1_norm,
    prune_layer,
    network_slimming_channel_prune,
)
import validate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Benchmark")


def benchmark_model(
    model_name: str,
    model_variant: str,
    device: torch.device,
    img_dir: str,
    anno_dir: str,
    dataloader=None,
) -> Tuple[pd.DataFrame, float, float, float, int, float]:
    """Run all pruning variants for one model, return metrics DataFrame + baseline metrics."""
    img_size = (640, 640)
    logger.info(f"=== Benchmarking {model_name} ({model_variant}) ===")

    # --- Build baseline model ---
    model = get_model(model_variant).to(device)

    if dataloader is None:
        dataloader = get_data_loader(
            img_dir=img_dir, anno_dir=anno_dir, batch_size=2, img_size=img_size
        )

    # --- Baseline metrics ---
    mAP50, mAP50_95 = validate_model.evaluate_map(model, dataloader, device)
    precision = validate_model.evaluate_precision(model, dataloader, device)
    recall = validate_model.evaluate_recall(model, dataloader, device)

    baseline_params = model.get_params_count()
    baseline_flops = model.calculate_flops((3,) + img_size)
    input_size = (1, 3, 640, 640)

    fps = validate_model.measure_fps(model, device, num_runs=30, img_size=input_size)
    latency = validate_model.measure_latency(model, device, num_runs=30, img_size=input_size)

    rows = [{
        "Model": model_name,
        "size (pixels)": "640",
        "mAPbox 50-95": round(mAP50_95, 3),
        "mAPmask 50-95": "N/A",
        "Train time 300 epochs A100 (h)": "N/A",
        "Speed ONNX CPU (ms)": "N/A",
        "Speed TRT A100 (ms)": "N/A",
        "params (M)": round(baseline_params / 1e6, 3),
        "FLOPs @640 (B)": round(baseline_flops / 1e9, 2),
        "Config": "Baseline FP32",
        "Sparsity": 0.0,
    }]

    # --- Pruned variants ---
    sparsity_configs = [
        ("Magnitude", [0.3, 0.5, 0.7], prune_by_magnitude),
        ("L1-Norm", [0.3, 0.5, 0.7], prune_by_l1_norm),
    ]

    for prune_name, sparsities, prune_fn in sparsity_configs:
        for sparsity in sparsities:
            logger.info(f"  {prune_name} {sparsity*100:.0f}% ...")
            pruned = prune_fn(model, sparsity).to(device)

            mAP50_p, mAP50_95_p = validate_model.evaluate_map(pruned, dataloader, device)

            eff = get_effective_metrics(pruned, input_size)
            flops = eff["flops"]
            params = eff["active_params"]
            sparsity_actual = eff["sparsity"]

            fps_p = validate_model.measure_fps(pruned, device, num_runs=30, img_size=input_size)
            latency_p = validate_model.measure_latency(pruned, device, num_runs=30, img_size=input_size)

            rows.append({
                "Model": model_name,
                "size (pixels)": "640",
                "mAPbox 50-95": round(mAP50_95_p, 3),
                "mAPmask 50-95": "N/A",
                "Train time 300 epochs A100 (h)": "N/A",
                "Speed ONNX CPU (ms)": "N/A",
                "Speed TRT A100 (ms)": "N/A",
                "params (M)": round(params / 1e6, 3),
                "FLOPs @640 (B)": round(flops / 1e9, 2),
                "Config": f"{prune_name} {int(sparsity*100)}%",
                "Sparsity": round(sparsity_actual, 3),
            })

    # --- Network Slimming Channel Pruning ---
    if "yolo" in model_variant:
        for sparsity in [0.4]:
            logger.info(f"  Channel Prune {sparsity*100:.0f}% ...")
            try:
                pruned = network_slimming_channel_prune(model, sparsity=sparsity, divisor=8).to(device)
                mAP50_p, mAP50_95_p = validate_model.evaluate_map(pruned, dataloader, device)
                eff = get_effective_metrics(pruned, input_size)
                flops = eff["flops"]
                params = eff["active_params"]
                sparsity_actual = eff["sparsity"]
                fps_p = validate_model.measure_fps(pruned, device, num_runs=30, img_size=input_size)
                latency_p = validate_model.measure_latency(pruned, device, num_runs=30, img_size=input_size)
                rows.append({
                    "Model": model_name,
                    "size (pixels)": "640",
                    "mAPbox 50-95": round(mAP50_95_p, 3),
                    "mAPmask 50-95": "N/A",
                    "Train time 300 epochs A100 (h)": "N/A",
                    "Speed ONNX CPU (ms)": "N/A",
                    "Speed TRT A100 (ms)": "N/A",
                    "params (M)": round(params / 1e6, 3),
                    "FLOPs @640 (B)": round(flops / 1e9, 2),
                    "Config": "Channel Pruned (40%)",
                    "Sparsity": round(sparsity_actual, 3),
                })
            except Exception as e:
                logger.warning(f"    Channel pruning failed: {e}")

    # --- Layer pruning ---
    logger.info(f"  Layer Pruning ...")
    layer_path = "model.model.3"
    try:
        pruned = prune_layer(model, layer_path).to(device)
        mAP50_p, mAP50_95_p = validate_model.evaluate_map(pruned, dataloader, device)
        eff = get_effective_metrics(pruned, input_size)
        flops = eff["flops"]
        params = eff["active_params"]
        sparsity_actual = eff["sparsity"]
        fps_p = validate_model.measure_fps(pruned, device, num_runs=30, img_size=input_size)
        latency_p = validate_model.measure_latency(pruned, device, num_runs=30, img_size=input_size)
        rows.append({
            "Model": model_name,
            "size (pixels)": "640",
            "mAPbox 50-95": round(mAP50_95_p, 3),
            "mAPmask 50-95": "N/A",
            "Train time 300 epochs A100 (h)": "N/A",
            "Speed ONNX CPU (ms)": "N/A",
            "Speed TRT A100 (ms)": "N/A",
            "params (M)": round(params / 1e6, 3),
            "FLOPs @640 (B)": round(flops / 1e9, 2),
            "Config": "Layer Pruned",
            "Sparsity": round(sparsity_actual, 3),
        })
    except Exception as e:
        logger.warning(f"    Layer pruning failed: {e}")

    return pd.DataFrame(rows)


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running benchmark on {device}")

    img_dir = os.environ.get("DETRAC_IMG_DIR", "data/UA-DETRAC/train")
    anno_dir = os.environ.get("DETRAC_ANNO_DIR", "data/UA-DETRAC/annotations")
    dataloader = get_data_loader(
        img_dir=img_dir, anno_dir=anno_dir, batch_size=2, img_size=(640, 640)
    )

    all_results = []

    # Always benchmark YOLOv5s (the default)
    df = benchmark_model("YOLOv5s", "yolov5s", device, img_dir, anno_dir, dataloader)
    all_results.append(df)

    # Merge all results
    result = pd.concat(all_results, ignore_index=True)

    # --- Print the table ---
    print("\n" + "=" * 140)
    print("BENCHMARK RESULTS — YOLOv5 Network Slimming Channel Pruning on UA-DETRAC")
    print("=" * 140)
    display_cols = [
        "Model", "size (pixels)", "mAPbox 50-95", "mAPmask 50-95",
        "params (M)", "FLOPs @640 (B)",
        "Train time 300 epochs A100 (h)", "Speed ONNX CPU (ms)", "Speed TRT A100 (ms)",
    ]
    table = result[display_cols].copy()
    # Indent config under model name
    table.insert(0, "Config", result["Config"])
    table["Sparsity"] = result["Sparsity"]
    print(table.to_string(index=False))
    print("=" * 140)
    print("  * mAPmask 50-95: N/A (object detection only, no segmentation)")
    print("  * Train time: N/A (pretrained COCO weights, fine-tuned on UA-DETRAC)")
    print("  * ONNX CPU / TRT A100: N/A (requires onnx + tensorrt installation)")
    print("=" * 140)

    # --- Save ---
    os.makedirs("benchmark", exist_ok=True)
    result.to_csv("benchmark_results.csv", index=False)
    result.to_excel("benchmark_results.xlsx", index=False, engine="openpyxl")
    logger.info("Saved benchmarks to benchmark_results.csv and benchmark_results.xlsx")


if __name__ == "__main__":
    run_benchmark()
