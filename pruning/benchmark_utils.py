import time
import torch
import torch.nn as nn
import json
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger("BenchmarkUtils")


def benchmark_latency_fps(
    model: nn.Module,
    device: torch.device,
    num_runs: int = 100,
    img_size: Tuple[int, int, int] = (3, 640, 640),
) -> Tuple[float, float]:
    """Measures the average latency (in ms) and throughput (in FPS) of the model.

    Returns:
        Tuple[float, float]: (latency_ms, fps)
    """
    model.eval()
    # Create input matching shape (1, C, H, W)
    x = torch.zeros((1,) + img_size, device=device)

    # Warm up
    logger.info("Warming up the model for 10 runs...")
    with torch.no_grad():
        for _ in range(10):
            model(x)

    logger.info(f"Running benchmarking for {num_runs} forward passes...")
    # Synchronize before starting timers
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

    mean_latency = float(sum(latencies) / len(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    logger.info(f"Benchmark completed: Latency = {mean_latency:.2f} ms | FPS = {fps:.2f}")
    return mean_latency, fps


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
