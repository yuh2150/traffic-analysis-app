import time
import gc
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger("BenchmarkUtils")


def benchmark_latency_fps(
    model: nn.Module,
    device: torch.device,
    num_runs: int = 200,
    img_size: Tuple[int, int, int] = (3, 640, 640),
) -> Tuple[float, float, float, float]:
    """Measures the average latency (in ms), std dev, median, and throughput (in FPS) of the model on CPU/GPU.
    
    Ensures deterministic CPU measurements by fixing threads, disabling GC, and doing sufficient warm-up.
    """
    model.eval()
    
    # 1. Thread settings for CPU determinism
    original_threads = torch.get_num_threads()
    if device.type == "cpu":
        logger.info("Setting PyTorch CPU threads to 1 for deterministic benchmarking...")
        torch.set_num_threads(1)
        
    # 2. Create dummy input matching shape (1, C, H, W)
    x = torch.zeros((1,) + img_size, device=device)

    # 3. Warm up for 50 runs
    logger.info("Warming up the model for 50 runs...")
    with torch.no_grad():
        for _ in range(50):
            model(x)

    # 4. Disable garbage collection during measurement
    gc.collect()
    gc.disable()
    
    # Ensure minimum 200 runs
    actual_runs = max(num_runs, 200)
    logger.info(f"Running benchmarking for {actual_runs} forward passes...")
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(actual_runs):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

    # Re-enable GC and restore threads
    gc.enable()
    if device.type == "cpu":
        torch.set_num_threads(original_threads)

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    median_latency = float(np.median(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    logger.info(
        f"Benchmark completed: Mean Latency = {mean_latency:.2f} ms | "
        f"Std Dev = {std_latency:.2f} ms | Median = {median_latency:.2f} ms | FPS = {fps:.2f}"
    )
    return mean_latency, fps, std_latency, median_latency


def export_benchmark_results(
    results: List[Dict[str, Any]],
    csv_path: str = "benchmark.csv",
    json_path: str = "benchmark.json",
):
    import pandas as pd

    """Exports benchmark results to CSV and JSON formats."""
    df = pd.DataFrame(results)
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark results to CSV: {csv_path}")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved benchmark results to JSON: {json_path}")
