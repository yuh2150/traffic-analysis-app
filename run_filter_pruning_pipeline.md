# YOLOv5s & DETR Filter Pruning Pipeline Execution Guide (COCO Dataset)

This guide contains the exact commands required to run baseline setup, structured filter pruning, recovery fine-tuning, and benchmarking for both YOLOv5s and DETR models using the codebase's modular CLI scripts.

---

## 1. Quick Baseline Setup (Using Pretrained Hub Weights)
To save time, run these commands to download the official pretrained weights (on COCO 80 classes) and save them as the project baseline checkpoints.

### YOLOv5s Baseline
```bash
python -c "import os, torch; from models.factory import ModelFactory; from utils.artifact_manager import ArtifactManager; mgr = ArtifactManager(); path = mgr.get_baseline_checkpoint('yolov5s', 'best'); os.makedirs(os.path.dirname(path), exist_ok=True); model = ModelFactory.load('yolov5s', num_classes=80); torch.save(model.state_dict(), path)"
```

### DETR Baseline
```bash
python -c "import os, torch; from models.factory import ModelFactory; from utils.artifact_manager import ArtifactManager; mgr = ArtifactManager(); path = mgr.get_baseline_checkpoint('detr', 'best'); os.makedirs(os.path.dirname(path), exist_ok=True); model = ModelFactory.load('detr', num_classes=80); torch.save(model.state_dict(), path)"
```

---

## 2. Model Pruning (Stage B)
Prune the baseline models using **Structured Filter Pruning** (which physically removes channels/filters using Torch-Pruning) at different sparsity ratios ($30\%$, $50\%$, $70\%$).

### Prune YOLOv5s
```bash
# 30% Sparsity (Filter Pruning)
python scripts/prune.py --model yolov5s --prune-type filter --sparsity 0.3 --dataset coco --force

# 50% Sparsity (Filter Pruning)
python scripts/prune.py --model yolov5s --prune-type filter --sparsity 0.5 --dataset coco --force

# 70% Sparsity (Filter Pruning)
python scripts/prune.py --model yolov5s --prune-type filter --sparsity 0.7 --dataset coco --force
```

### Prune DETR
```bash
# 30% Sparsity (Filter Pruning)
python scripts/prune.py --model detr --prune-type filter --sparsity 0.3 --dataset coco --force

# 50% Sparsity (Filter Pruning)
python scripts/prune.py --model detr --prune-type filter --sparsity 0.5 --dataset coco --force

# 70% Sparsity (Filter Pruning)
python scripts/prune.py --model detr --prune-type filter --sparsity 0.7 --dataset coco --force
```

---

## 3. Recovery Fine-Tuning (Stage C)
Fine-tune the pruned weights on COCO to restore model mAP accuracy (running for 10 epochs).

### Recover YOLOv5s
```bash
# 30% Sparsity (Filter Pruning)
python scripts/recover.py --epochs 10 --model yolov5s --prune-type filter --sparsity 0.3 --dataset coco --force

# 50% Sparsity (Filter Pruning)
python scripts/recover.py --epochs 10 --model yolov5s --prune-type filter --sparsity 0.5 --dataset coco --force

# 70% Sparsity (Filter Pruning)
python scripts/recover.py --epochs 10 --model yolov5s --prune-type filter --sparsity 0.7 --dataset coco --force
```

### Recover DETR
```bash
# 30% Sparsity (Filter Pruning)
python scripts/recover.py --epochs 10 --model detr --prune-type filter --sparsity 0.3 --dataset coco --force

# 50% Sparsity (Filter Pruning)
python scripts/recover.py --epochs 10 --model detr --prune-type filter --sparsity 0.5 --dataset coco --force

# 70% Sparsity (Filter Pruning)
python scripts/recover.py --epochs 10 --model detr --prune-type filter --sparsity 0.7 --dataset coco --force
```

---

## 4. Benchmark & Evaluate Results (Stage D)
Profile and evaluate the metrics (Precision, Recall, mAP50, Latency, FPS, Params, FLOPs) of the models.

### A. Evaluating the Raw Pruned Models (Before Recovery)
Evaluate models immediately after pruning to observe the initial performance drop and verify the latency/FPS improvements.

#### YOLOv5s (Pruned)
```bash
# Baseline original
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/baseline/best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/pruned/filter_0.3.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/pruned/filter_0.5.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/pruned/filter_0.7.pt --dataset coco --max-samples 500
```

#### DETR (Pruned)
```bash
# Baseline original
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/baseline/best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/pruned/filter_0.3.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/pruned/filter_0.5.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/pruned/filter_0.7.pt --dataset coco --max-samples 500
```

### B. Evaluating the Recovered Models (After Recovery Training)
Evaluate the models after recovery training to check the restored mAP accuracy.

#### YOLOv5s (Recovered)
```bash
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/recovered/filter_0.3_best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/recovered/filter_0.5_best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/recovered/filter_0.7_best.pt --dataset coco --max-samples 500
```

#### DETR (Recovered)
```bash
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/recovered/filter_0.3_best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/recovered/filter_0.5_best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/recovered/filter_0.7_best.pt --dataset coco --max-samples 500
```

---

## 5. Automated Multi-Model Matrix Experiment (Stage E)
To run the entire pipeline (Prune $\to$ Recover $\to$ Benchmark $\to$ Graph Generation) for both models and all filter-pruning sparsities in a single command, use:

```bash
python scripts/experiment.py --model all --prune-types filter --sparsities 0.3 0.5 0.7 --dataset coco --epochs-recover 10 --max-samples 500
```
