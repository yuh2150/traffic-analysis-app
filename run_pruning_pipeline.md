# YOLOv5s & DETR Pruning Pipeline execution Guide (COCO Dataset)

This guide contains the exact commands required to run baseline setup, pruning, recovery fine-tuning, and benchmarking for both YOLOv5s and DETR models.

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
Prune the baseline models using **Unstructured Magnitude Pruning** at different sparsity ratios ($30\%$, $50\%$, $70\%$).

### Prune YOLOv5s
```bash
# 30% Sparsity
python scripts/prune.py --model yolov5s --prune-type magnitude --sparsity 0.3 --dataset coco --force

# 50% Sparsity
python scripts/prune.py --model yolov5s --prune-type magnitude --sparsity 0.5 --dataset coco --force

# 70% Sparsity
python scripts/prune.py --model yolov5s --prune-type magnitude --sparsity 0.7 --dataset coco --force
```

### Prune DETR
```bash
# 30% Sparsity
python scripts/prune.py --model detr --prune-type magnitude --sparsity 0.3 --dataset coco --force

# 50% Sparsity
python scripts/prune.py --model detr --prune-type magnitude --sparsity 0.5 --dataset coco --force

# 70% Sparsity
python scripts/prune.py --model detr --prune-type magnitude --sparsity 0.7 --dataset coco --force
```

---

## 3. Recovery Fine-Tuning (Stage C)
Fine-tune the pruned weights on COCO to restore model mAP accuracy (running for 5 epochs).

### Recover YOLOv5s
```bash
# 30% Sparsity
python scripts/recover.py --epochs 10 --model yolov5s --prune-type magnitude --sparsity 0.3 --dataset coco --force

# 50% Sparsity
python scripts/recover.py --epochs 10 --model yolov5s --prune-type magnitude --sparsity 0.5 --dataset coco --force

# 70% Sparsity
python scripts/recover.py --epochs 10 --model yolov5s --prune-type magnitude --sparsity 0.7 --dataset coco --force
```

### Recover DETR
```bash
# 30% Sparsity
python scripts/recover.py --epochs 10 --model detr --prune-type magnitude --sparsity 0.3 --dataset coco --force

# 50% Sparsity
python scripts/recover.py --epochs 10 --model detr --prune-type magnitude --sparsity 0.5 --dataset coco --force

# 70% Sparsity
python scripts/recover.py --epochs 10 --model detr --prune-type magnitude --sparsity 0.7 --dataset coco --force
```

---

## 4. Benchmark & Evaluate Results (Stage D)
Profile and evaluate the metrics (Precision, Recall, mAP50, Latency, FPS) of the recovered models.

### Benchmark YOLOv5s
```bash
# Baseline original
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/baseline/best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/pruned/magnitude_0.3.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/pruned/magnitude_0.5.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model yolov5s --checkpoint checkpoints/yolov5s/pruned/magnitude_0.7.pt --dataset coco --max-samples 500
```

### Benchmark DETR
```bash
# Baseline original
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/baseline/best.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/pruned/magnitude_0.3.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/pruned/magnitude_0.5.pt --dataset coco --max-samples 500
python scripts/benchmark.py --model detr --checkpoint checkpoints/detr/pruned/magnitude_0.7.pt --dataset coco --max-samples 500
```

---

## 5. Automated Multi-Model Matrix Experiment (Stage E)
To run the entire pipeline (Prune $\to$ Recover $\to$ Benchmark $\to$ Graph Generation) for both models and all sparsities in a single command, use:

```bash
python scripts/experiment.py --model all --prune-types magnitude --sparsities 0.3 0.5 0.7 --dataset coco --epochs-recover 10 --max-samples 500

```

