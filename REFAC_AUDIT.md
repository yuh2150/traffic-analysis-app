# Phase 1 Project Refactoring Audit

This document identifies duplicate code, legacy logic, and technical debt across the traffic-analysis-app codebase.

| File | Issue | Severity | Suggested Action |
| --- | --- | --- | --- |
| `models/models.py` | Contains legacy, buggy YOLOv5 and torchvision DETR wrappers. The YOLO wrapper forces evaluation mode and disables gradient tracking during training, completely breaking fine-tuning. | **Critical** | Delete. Replaced by `models/yolov5_wrapper.py` and `models/detr_wrapper.py`. |
| `pruning/pruner.py` | Contains legacy structured, layer, and channel pruning. Treating DAG models as simple sequential chains causes indexing crashes on Concat and SPPF layers. Structured pruning randomly re-initializes neck layer weights. | **Critical** | Delete. Replaced by modular, dimension-safe masked pruners under `pruning/`. |
| `pruning/layer/layer_pruner.py` | Legacy layer-wise bottleneck pruner. Replaced by updated C3 bottleneck pruning inside the parent package. | **High** | Delete the file and the `pruning/layer/` subdirectory. |
| `training/train.py` | Legacy training script containing duplicate training loops. Relies on buggy models wrapper which locks gradient calculation. | **Critical** | Delete the file and the `training/` subdirectory. |
| `validate_model.py` | Legacy validation script. Contains FLOPs unpacking bugs and 5D tensor profiling crashes when run standalone. | **Critical** | Delete. Replaced by root-level `validate.py`. |
| `benchmark/benchmark.py` | Legacy benchmarking script. Relies on buggy model wrappers and outdated pruners. | **Critical** | Delete the file and the `benchmark/` subdirectory. |
| `scripts/generate_weights.py` | Outdated script generating model weights. Relies on buggy `pruning/pruner.py` and crashes when running channel pruning. Standardizes DETR weights as `.pt` instead of `.pth` expected by the Streamlit app. | **High** | Delete. Weight generation logic is consolidated inside `benchmark.py`. |
| `demo/app.py` | Imports from missing `quantization` module, causing startup crashes. Loads models via obsolete `models.models` path. | **High** | Reorganize under `app/app.py`. Stub missing quantization imports and update model/tracker references. |
| `scripts/onnx_benchmark.py` | Imports model loading and magnitude pruning from obsolete legacy paths. Contains duplicated parameter/FLOPs analysis logic. | **Medium** | Reorganize under `scripts/`. Update imports and reuse model properties. |
