#!/usr/bin/env python3
import subprocess
import os
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch runner for magnitude YOLOv5s models.")
    parser.add_argument("--script", type=str, default="yolo_car_counter_5.py", 
                        choices=["yolo_car_counter_4.py", "yolo_car_counter_5.py"],
                        help="The counter script to run (default: yolo_car_counter_5.py)")
    parser.add_argument("--max-frames", type=int, default=300, 
                        help="Maximum frames to process per video (default: 300, set to -1 for full video)")
    args = parser.parse_args()

    # Define environment and python path
    python_exec = "/home/huy/miniconda3/envs/env_cv/bin/python"
    script_path = args.script
    
    # Check if python executable exists
    if not os.path.exists(python_exec):
        print(f"Error: Python executable not found at {python_exec}")
        sys.exit(1)
        
    # Check if the counter script exists
    if not os.path.exists(script_path):
        print(f"Error: Counter script not found at {script_path}")
        sys.exit(1)

    # Determine filename suffix based on script
    suffix = "4_" if args.script == "yolo_car_counter_4.py" else ""

    # Define the 4 runs
    runs = [
        {
            "name": "YOLOv5s Baseline",
            "checkpoint": "checkpoints/yolov5s/baseline/best.pt",
            "output": f"DATA/OUTPUTS/car_counter_{suffix}yolov5s_baseline.mp4"
        },
        {
            "name": "YOLOv5s Magnitude Pruned 30%",
            "checkpoint": "checkpoints/yolov5s/pruned/magnitude_0.3.pt",
            "output": f"DATA/OUTPUTS/car_counter_{suffix}yolov5s_magnitude_0.3.mp4"
        },
        {
            "name": "YOLOv5s Magnitude Pruned 50%",
            "checkpoint": "checkpoints/yolov5s/pruned/magnitude_0.5.pt",
            "output": f"DATA/OUTPUTS/car_counter_{suffix}yolov5s_magnitude_0.5.mp4"
        },
        {
            "name": "YOLOv5s Magnitude Pruned 70%",
            "checkpoint": "checkpoints/yolov5s/pruned/magnitude_0.7.pt",
            "output": f"DATA/OUTPUTS/car_counter_{suffix}yolov5s_magnitude_0.7.mp4"
        }
    ]

    print("==================================================")
    print(f"STARTING BATCH PROCESSING FOR 4 YOLOv5s MODELS USING {args.script.upper()}")
    print(f"Limit per video: {args.max_frames} frames" if args.max_frames > 0 else "Limit per video: Full video")
    print("==================================================")
    
    os.makedirs("DATA/OUTPUTS", exist_ok=True)
    
    total_start_time = time.time()
    
    for i, run in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] Running {run['name']}...")
        print(f"  Checkpoint: {run['checkpoint']}")
        print(f"  Saving to:  {run['output']}")
        
        # Build command
        cmd = [
            python_exec, script_path,
            "--model", "yolov5s",
            "--checkpoint", run["checkpoint"],
            "--output", run["output"],
            "--dataset", "coco",
            "--max-frames", str(args.max_frames),
            "--headless"
        ]
        
        start_time = time.time()
        try:
            # Run the counter script and stream the output to the console
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Print output in real time
            for line in process.stdout:
                print(f"    {line.strip()}")
                
            process.wait()
            
            if process.returncode == 0:
                elapsed = time.time() - start_time
                print(f"✓ Successfully finished {run['name']} in {elapsed:.2f}s")
            else:
                print(f"✗ Failed {run['name']} with exit code {process.returncode}")
                
        except Exception as e:
            print(f"✗ Exception occurred while running {run['name']}: {e}")
            
    total_elapsed = time.time() - total_start_time
    print("\n==================================================")
    print(f"BATCH PROCESSING COMPLETED IN {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
    print("Generated videos are available in DATA/OUTPUTS/:")
    for run in runs:
        print(f"  - {run['output']}")
    print("==================================================")

if __name__ == "__main__":
    main()
