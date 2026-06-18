import os
import time
import torch
import pandas as pd
import numpy as np
from models.models import get_model
from pruning.pruner import prune_by_magnitude, get_effective_metrics

def main():
    print("--- Starting ONNX CPU Benchmarking for YOLOv5s Magnitude Pruning ---")
    
    import sys
    # Try installing onnx and onnxruntime if not available
    try:
        import onnx
        import onnxruntime as ort
        print("onnx and onnxruntime are already installed.")
    except ImportError:
        print("Installing onnx and onnxruntime...")
        os.system(f"{sys.executable} -m pip install onnx onnxruntime")
        import onnx
        import onnxruntime as ort

    device = torch.device("cpu")
    model_name = "yolov5"
    img_size = 640
    dummy_input = torch.zeros(1, 3, img_size, img_size)
    
    # Read existing mAP results from benchmark_results.csv if available
    mAP_map = {}
    if os.path.exists("benchmark_results.csv"):
        try:
            df = pd.read_csv("benchmark_results.csv")
            df_yolo = df[df["Model"] == "YOLOV5"]
            for _, row in df_yolo.iterrows():
                config = row["Config"]
                mAP_map[config] = row["mAP50-95"]
        except Exception as e:
            print(f"Warning reading benchmark_results.csv: {e}")

    configs = [
        ("Baseline FP32", 0.0),
        ("Magnitude Pruned (30%)", 0.3),
        ("Magnitude Pruned (50%)", 0.5),
        ("Magnitude Pruned (70%)", 0.7),
    ]

    results = []

    for config_name, sparsity in configs:
        print(f"\nProcessing {config_name}...")
        
        # Load baseline model
        model = get_model(model_name, model_size="n").to(device)
        model.eval()
        
        # Apply pruning if needed
        if sparsity > 0.0:
            model = prune_by_magnitude(model, sparsity)
            
        # Get parameters and FLOPs
        eff = get_effective_metrics(model, (1, 3, img_size, img_size))
        params_m = eff["active_params"] / 1e6
        flops_g = eff["flops"] / 1e9
        
        # Path for temporary ONNX model
        onnx_path = f"weights/yolov5_{config_name.replace(' ', '_').replace('(', '').replace(')', '')}.onnx"
        os.makedirs("weights", exist_ok=True)
        
        # Export core model to ONNX (using model.model to export raw neural network structure, avoiding postprocessing loops)
        print("Exporting to ONNX...")
        try:
            torch.onnx.export(
                model.model,
                dummy_input,
                onnx_path,
                opset_version=11,
                input_names=["images"],
                output_names=["output"],
                dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            
            # Benchmark latency on ONNX CPU
            print("Benchmarking ONNX CPU latency...")
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            
            # Warmup
            numpy_input = dummy_input.numpy()
            for _ in range(10):
                session.run(None, {input_name: numpy_input})
                
            # Measure
            runs = 50
            start_time = time.perf_counter()
            for _ in range(runs):
                session.run(None, {input_name: numpy_input})
            latency_ms = ((time.perf_counter() - start_time) / runs) * 1000.0
            print(f"ONNX CPU Latency: {latency_ms:.2f} ms")
        except Exception as e:
            print(f"ONNX export/benchmark failed for {config_name}: {e}. Falling back to PyTorch CPU benchmark.")
            # Fallback to PyTorch CPU latency
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model.model(dummy_input)
            # Measure
            runs = 50
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(runs):
                    model.model(dummy_input)
            latency_ms = ((time.perf_counter() - start_time) / runs) * 1000.0
            print(f"PyTorch CPU Latency: {latency_ms:.2f} ms")

        # Clean up temporary ONNX file
        if os.path.exists(onnx_path):
            try:
                os.remove(onnx_path)
            except:
                pass
                
        # Retrieve mAP
        map_val = mAP_map.get(config_name, 0.0)
        
        results.append({
            "Model": f"YOLOv5s {config_name}",
            "size (pixels)": f"{img_size}x{img_size}",
            "mAPbox 50-95": f"{map_val:.4f}",
            "mAPmask": "0.0 (N/A)",
            "Speed ONNX CPU (ms)": f"{latency_ms:.2f} ms",
            "params (M)": f"{params_m:.3f}M",
            "FLOPs (G)": f"{flops_g:.3f}G"
        })

    # Generate Markdown Table
    df_res = pd.DataFrame(results)
    print("\n=== FINAL BENCHMARK TABLE ===")
    print(df_res.to_markdown(index=False))
    print("=============================")
    
    # Save to file
    df_res.to_csv("benchmark/yolov5_onnx_benchmark.csv", index=False)
    
if __name__ == "__main__":
    main()
