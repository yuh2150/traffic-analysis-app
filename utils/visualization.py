import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations(csv_path: str = "reports/benchmark_results.csv", output_dir: str = "reports"):
    """Generates five comparative research charts from the benchmark results CSV using Matplotlib."""
    if not os.path.exists(csv_path):
        print(f"Benchmark results file not found at: {csv_path}. Skipping visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Filter out INT8 simulated entries if they skew curves, or include them as distinct points
    # Let's group by model
    for model_name in df["Model"].unique():
        df_model = df[df["Model"] == model_name].copy()
        
        # Sort values by Sparsity for plotting curves
        df_curves = df_model[df_model["Config"] != "INT8 Quantized"].sort_values(by="Sparsity")
        df_quant = df_model[df_model["Config"] == "INT8 Quantized"]

        # Color and marker definitions
        colors = {"Magnitude": "#E91E63", "L1 norm": "#9C27B0", "Filter": "#2196F3", "Channel": "#4CAF50", "Layer": "#FF9800", "None": "#9E9E9E"}
        
        def get_color_and_marker(row):
            pt = row["Pruning Type"]
            return colors.get(pt, "#3F51B5"), "o"

        # --- Chart 1: Accuracy vs Compression (mAP50 vs. Compression Ratio / Sparsity) ---
        plt.figure(figsize=(8, 5))
        for pt in df_curves["Pruning Type"].unique():
            df_pt = df_curves[df_curves["Pruning Type"] == pt]
            plt.plot(df_pt["Sparsity"] * 100, df_pt["mAP50"], marker="o", label=f"{pt} Pruning", linewidth=2)
        
        if not df_quant.empty:
            plt.scatter(df_quant["Sparsity"] * 100, df_quant["mAP50"], color="red", marker="X", s=100, label="INT8 Quantized", zorder=5)
            
        plt.title(f"{model_name} Accuracy vs Sparsity Trade-off", fontsize=14, fontweight="bold", pad=10)
        plt.xlabel("Sparsity (%)", fontsize=12)
        plt.ylabel("mAP50 Accuracy", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_accuracy_vs_compression.png"), dpi=200)
        plt.close()

        # --- Chart 2: FPS vs Compression (FPS vs. Compression Ratio) ---
        plt.figure(figsize=(8, 5))
        for pt in df_curves["Pruning Type"].unique():
            df_pt = df_curves[df_curves["Pruning Type"] == pt]
            plt.plot(df_pt["Compression Ratio"], df_pt["FPS"], marker="s", label=f"{pt} Pruning", linewidth=2)
            
        if not df_quant.empty:
            plt.scatter(df_quant["Compression Ratio"], df_quant["FPS"], color="red", marker="X", s=100, label="INT8 Quantized", zorder=5)
            
        plt.title(f"{model_name} Processing Speed (FPS) vs Compression", fontsize=14, fontweight="bold", pad=10)
        plt.xlabel("Compression Ratio (x)", fontsize=12)
        plt.ylabel("Throughput (FPS)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_fps_vs_compression.png"), dpi=200)
        plt.close()

        # --- Chart 3: Latency vs Compression ---
        plt.figure(figsize=(8, 5))
        for pt in df_curves["Pruning Type"].unique():
            df_pt = df_curves[df_curves["Pruning Type"] == pt]
            plt.plot(df_pt["Compression Ratio"], df_pt["Latency (ms)"], marker="^", label=f"{pt} Pruning", linewidth=2)
            
        if not df_quant.empty:
            plt.scatter(df_quant["Compression Ratio"], df_quant["Latency (ms)"], color="red", marker="X", s=100, label="INT8 Quantized", zorder=5)
            
        plt.title(f"{model_name} Inference Latency vs Compression", fontsize=14, fontweight="bold", pad=10)
        plt.xlabel("Compression Ratio (x)", fontsize=12)
        plt.ylabel("Latency (ms)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_latency_vs_compression.png"), dpi=200)
        plt.close()

        # --- Chart 4: FLOPs vs Compression ---
        plt.figure(figsize=(8, 5))
        for pt in df_curves["Pruning Type"].unique():
            df_pt = df_curves[df_curves["Pruning Type"] == pt]
            plt.plot(df_pt["Sparsity"] * 100, df_pt["FLOPs"] / 1e6, marker="d", label=f"{pt} Pruning", linewidth=2)
            
        plt.title(f"{model_name} Computational Complexity (MFLOPs) vs Sparsity", fontsize=14, fontweight="bold", pad=10)
        plt.xlabel("Sparsity (%)", fontsize=12)
        plt.ylabel("Complexity (MFLOPs)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_flops_vs_compression.png"), dpi=200)
        plt.close()

        # --- Chart 5: Parameters vs Compression ---
        plt.figure(figsize=(8, 5))
        
        # Sort and group configs by bar names
        configs_to_show = df_model.sort_values(by="Sparsity")
        config_names = configs_to_show["Config"].tolist()
        param_counts = (configs_to_show["Params"] / 1e6).tolist()
        
        x = np.arange(len(config_names))
        plt.bar(x, param_counts, color="#2196F3", edgecolor="black", width=0.6)
        plt.xticks(x, config_names, rotation=45, ha="right")
        plt.title(f"{model_name} Parameter Count Reduction Comparison", fontsize=14, fontweight="bold", pad=10)
        plt.ylabel("Parameters (Millions)", fontsize=12)
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_params_vs_compression.png"), dpi=200)
        plt.close()
        
        # For legacy Streamlit compatibility, also save copies under reports/ without model name prefix
        # accuracy_vs_compression.png, fps_latency_vs_compression.png, resource_reduction.png
        # Let's save accuracy_vs_compression
        plt.figure(figsize=(8, 5))
        for pt in df_curves["Pruning Type"].unique():
            df_pt = df_curves[df_curves["Pruning Type"] == pt]
            plt.plot(df_pt["Sparsity"] * 100, df_pt["mAP50"], marker="o", label=f"{pt} Pruning", linewidth=2)
        if not df_quant.empty:
            plt.scatter(df_quant["Sparsity"] * 100, df_quant["mAP50"], color="red", marker="X", s=100, label="INT8 Quantized", zorder=5)
        plt.title(f"Accuracy vs Compression ({model_name})")
        plt.xlabel("Sparsity (%)")
        plt.ylabel("mAP50")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_accuracy_vs_compression.png"), dpi=200)
        plt.close()

    print("Successfully generated all Matplotlib visualizations.")
