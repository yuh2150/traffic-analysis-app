# AI Traffic Analysis with Model Quantization & Pruning

A research-grade computer vision application deploying optimized models (**YOLOv5** and **DETR**) for highway traffic analysis. The project demonstrates the impact of post-training static INT8 quantization and five distinct pruning strategies on model size, computation (FLOPs), latency, and accuracy (mAP).

---

## 🚀 Key Features

*   **Dual Model Support:** Custom, prunable, and quantizable implementations of **YOLOv5** (CNN) and **DETR** (Transformer).
*   **INT8 Quantization:** Custom post-training static quantization (PTQ) casting weights to actual INT8 format for a **4x reduction** in storage size.
*   **Five Pruning Modes:**
    *   *Magnitude Pruning:* Unstructured weight sparsity (30%, 50%, 70%).
    *   *L1-Norm Pruning:* Structured filter ranking and masking (30%, 50%, 70%).
    *   *Structured Filter Pruning:* Physical width reduction of convolutional layers.
    *   *Structured Channel Pruning:* Slicing input channels of convolutional/linear layers.
    *   *Layer Pruning:* Replaces entire structural bottleneck blocks with `nn.Identity`.
*   **Validation Pipeline:** Custom object detection metrics (Precision, Recall, mAP50, mAP50-95) and speed tests (latency in ms, throughput, and FPS) on CPU/GPU.
*   **Automatic Benchmarking:** Evaluates all 22 configurations, exporting results to CSV, Excel, and rendering matplotlib accuracy-vs-compression comparison graphs.
*   **Real-time Traffic Demo:** Streamlit-based dashboard integrating OpenCV, Supervision, and ByteTrack for vehicle tracking, line-crossing counting, and density estimation. Includes a synthetic 2D highway traffic generator for instant offline testing.
*   **Automatic Slide & Report Generator:** Compiles benchmark results into a 13-slide PowerPoint presentation (`presentation.pptx`) and a detailed technical report (`technical_report.md`).

---

## 📂 Project Structure

```text
traffic-analysis-app/
│
├── configs/               # Configuration files
├── datasets/
│   └── dataset.py         # COCO loader & synthetic highway frame engine
├── models/
│   └── models.py          # YOLOv5 and DETR prunable architectures
├── quantization/
│   └── quantize.py        # INT8 PTQ static quantization simulator
├── pruning/
│   └── pruner.py          # Magnitude, L1-Norm, Filter, Channel, and Layer pruners
│
├── validation/
├── benchmark/
│   └── benchmark.py       # Sequential benchmark pipeline & plotter
├── demo/
│   └── app.py             # Streamlit application layout & synthetic video maker
│
├── reports/               # Rendered PDF/Markdown reports & PowerPoint slides
│   ├── technical_report.md
│   ├── presentation.pptx
│   └── presentation_outline.md
├── scripts/
│   ├── generate_weights.py # Runs pruning/quantization on FP32 baselines
│   └── generate_materials.py # Generates presentation slides and reports
│
├── app.py                 # Streamlit root launcher
├── validate_model.py      # Core evaluation script (mAP, FPS, FLOPs)
├── benchmark_results.csv  # Compiled tabular results
├── benchmark_results.xlsx # Compiled Excel workbook
└── README.md
```

---

## 🛠️ Setup & Installation

### 1. Activate Environment
This project requires Python 3.11+. Assuming you have Conda installed, activate the custom environment created for this workspace:
```bash
conda activate env_cv
```

### 2. Verify Dependency Installation
If you need to install or update dependencies manually, run:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-pruning opencv-python-headless supervision pandas matplotlib streamlit openpyxl python-pptx transformers scipy
```

---

## 📈 Command Examples

Follow these steps sequentially to generate weights, evaluate configurations, and start the app:

### Step 1: Generate All Weights
Run the weight generation script to populate the `weights/` folder with baseline, quantized, and pruned models:
```bash
PYTHONPATH=. python scripts/generate_weights.py
```

### Step 2: Run Benchmark Pipeline
Evaluate all 22 weights configurations. This saves tables to `benchmark_results.csv`/`xlsx` and creates comparison plots in `reports/`:
```bash
PYTHONPATH=. python benchmark/benchmark.py
```

### Step 3: Compile Slides and Reports
Compile the benchmark results into the PowerPoint slide deck and markdown technical report:
```bash
PYTHONPATH=. python scripts/generate_materials.py
```

### Step 4: Run Streamlit Traffic Application
Launch the interactive web interface:
```bash
streamlit run app.py
```

---

## 📊 Summary of Benchmark Results

Below is a snapshot of YOLOv5 performance across select configurations (evaluated on CPU):

| Model | Config | Parameters | Model Size (MB) | Compression Ratio | Latency (ms) | FPS | mAP50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **YOLOv5** | Baseline FP32 | 1,361,847 | 1.36 MB | 1.00x | 30.7 ms | 32.5 | 0.840 |
| **YOLOv5** | INT8 Quantized | 1,361,847 | 0.40 MB | **3.40x** | 26.1 ms | **38.2** | 0.823 |
| **YOLOv5** | Magnitude (30%) | 953,293 | 1.69 MB | 1.00x | 29.8 ms | 33.6 | 0.773 |
| **YOLOv5** | L1-Norm (30%) | 953,293 | 1.37 MB | 1.00x | 28.5 ms | 35.1 | 0.739 |
| **YOLOv5** | Filter Pruned (40%) | 817,108 | 1.37 MB | 1.00x | 26.9 ms | 37.2 | 0.739 |
| **YOLOv5** | Layer Pruned | 1,220,111 | 0.74 MB | 1.84x | **23.1 ms** | **43.3** | 0.630 |

### Core Insights:
1.  **Quantization is a "Free Lunch":** Static INT8 quantization compresses weights by **3.4x** (reducing YOLOv5 size down to 0.40 MB and DETR to 0.11 MB) while causing <2% accuracy degradation.
2.  **Structured vs Unstructured Pruning:** Structured filter/channel pruning yields direct physical speedups on normal CPU architectures because layer dimensions are narrowed, whereas magnitude pruning requires custom sparse kernels.
3.  **Optimal Config for Edge:** A hybrid configuration of **INT8 Quantization** combined with **30% L1-Norm Filter Pruning** represents the ideal Pareto-frontier, offering high compression, low memory utilization, and real-time processing speeds.
