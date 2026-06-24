#!/usr/bin/env python3
import os
import sys
import glob
import json
import shutil
import logging

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarking.benchmark import TrafficBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ConsolidateReports")

def main():
    reports_dir = "reports"
    if not os.path.isdir(reports_dir):
        logger.error(f"Reports directory '{reports_dir}' not found.")
        sys.exit(1)

    # Find all JSON files except the consolidated ones
    json_paths = glob.glob(os.path.join(reports_dir, "*.json"))
    exclude_files = {"benchmark_results.json", "magnitude_benchmark_results.json", "weight_analysis.json"}

    individual_results = []
    for path in json_paths:
        filename = os.path.basename(path)
        if filename in exclude_files:
            continue

        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Simple validation that it's a benchmark result dict
            if isinstance(data, dict) and "Model" in data and "Sparsity" in data:
                # Add extra fields if missing to match consolidated matrix schema
                if "Compression Ratio" not in data:
                    # Compression ratio can be estimated or calculated relative to baseline if we sort later,
                    # but for now we set it to default or based on sparsity
                    data["Compression Ratio"] = 1.0 / (1.0 - data["Sparsity"]) if data["Sparsity"] < 1.0 else 1.0
                if "Throughput (img/s)" not in data:
                    data["Throughput (img/s)"] = data.get("FPS", 0.0)
                
                individual_results.append(data)
                logger.info(f"Loaded individual report: {filename}")
        except Exception as e:
            logger.warning(f"Skipping {filename} due to load error: {e}")

    if not individual_results:
        logger.warning("No individual benchmark reports found to consolidate.")
        sys.exit(0)

    # Sort results for readability: Model, then Sparsity
    individual_results.sort(key=lambda x: (x.get("Model", ""), x.get("Sparsity", 0.0)))

    # Export using TrafficBenchmark.export_results
    csv_out = os.path.join(reports_dir, "benchmark_results.csv")
    json_out = os.path.join(reports_dir, "benchmark_results.json")
    
    TrafficBenchmark.export_results(individual_results, csv_path=csv_out, json_path=json_out)
    
    # Copy to root directory for GUI Dashboard or ease of access
    try:
        shutil.copy(csv_out, "benchmark_results.csv")
        logger.info("Copied consolidated CSV to root directory: ./benchmark_results.csv")
        
        # Export as xlsx if pandas/openpyxl is installed
        try:
            import pandas as pd
            df = pd.DataFrame(individual_results)
            df.to_excel("benchmark_results.xlsx", index=False)
            logger.info("Saved benchmark_results.xlsx to the root directory.")
        except ImportError:
            pass
            
    except Exception as e:
        logger.warning(f"Could not copy benchmark results CSV to root: {e}")

    logger.info(f"Successfully consolidated {len(individual_results)} benchmark reports!")

if __name__ == "__main__":
    main()
