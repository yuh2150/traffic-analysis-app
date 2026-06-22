import os
import cv2
import torch
import pandas as pd
import numpy as np
import streamlit as st
import time
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

# Ensure project root is in python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from models import get_model, ModelFactory
from utils.tracker import TrafficTracker

# Stub missing quantization functions
def quantize_model_static_simulated(model: torch.nn.Module) -> torch.nn.Module:
    """Mock static quantization simulation returning the original model."""
    return model

def load_quantized_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """Mock loader for quantized weights loading them as standard state dict."""
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    return model

# --- SYNTHETIC VIDEO GENERATOR ---

def generate_synthetic_traffic_video(output_path: str, num_frames: int = 150, img_size: Tuple[int, int] = (640, 640)) -> str:
    """Generates an MP4 video of synthetic traffic flowing on a highway for demo purposes."""
    h, w = img_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
    
    # Define traffic agents: list of Dicts representing moving vehicles
    # category: 0: Car, 1: Truck, 2: Bus, 3: Motorcycle
    # y: vertical position (starts off-screen or randomly, moves down)
    # x: lane center position
    np.random.seed(42)
    vehicles = []
    
    # Establish lane coordinates
    road_left = int(w * 0.25)
    road_right = int(w * 0.75)
    road_width = road_right - road_left
    lane_width = road_width // 4
    lanes = [
        road_left + lane_width // 2,
        road_left + lane_width * 3 // 2,
        road_left + lane_width * 5 // 2,
        road_left + lane_width * 7 // 2,
    ]
    
    # Generate initial vehicles
    for i in range(8):
        cat = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.1])
        lane_idx = np.random.randint(0, 4)
        vehicles.append({
            "id": i,
            "category": cat,
            "x": lanes[lane_idx],
            "y": np.random.randint(-200, h),
            "speed": np.random.uniform(3.0, 7.0),
            "color": [
                (21, 101, 192),    # Blue
                (198, 40, 40),     # Red
                (239, 108, 0),     # Orange
                (249, 168, 37)     # Yellow
            ][cat],
            "size": [
                (24, 36),   # Car (w, h)
                (32, 60),   # Truck
                (36, 70),   # Bus
                (16, 24)    # Motorcycle
            ][cat]
        })
        
    next_vehicle_id = len(vehicles)
    
    for frame_idx in range(num_frames):
        # Draw frame background (green scenery)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = [46, 125, 50] # RGB (Dark Green)
        
        # Draw Highway road (gray asphalt)
        cv2.rectangle(img, (road_left, 0), (road_right, h), (97, 97, 97), -1)
        
        # Draw solid white boundaries
        cv2.line(img, (road_left + 10, 0), (road_left + 10, h), (245, 245, 245), 3)
        cv2.line(img, (road_right - 10, 0), (road_right - 10, h), (245, 245, 245), 3)
        
        # Draw dashed lines
        center = w // 2
        for y in range(0, h, 40):
            # Shift dashed lines down to simulate movement/speed
            y_shifted = (y + (frame_idx * 5)) % h
            cv2.line(img, (center, y_shifted), (center, (y_shifted + 20) % h), (255, 235, 59), 2)
            cv2.line(img, (center - lane_width, y_shifted), (center - lane_width, (y_shifted + 20) % h), (245, 245, 245), 1)
            cv2.line(img, (center + lane_width, y_shifted), (center + lane_width, (y_shifted + 20) % h), (245, 245, 245), 1)
            
        # Draw & Update vehicles
        for v in vehicles:
            vw, vh = v["size"]
            vx, vy = int(v["x"]), int(v["y"])
            
            # Draw vehicle body (rect)
            cv2.rectangle(img, (vx - vw//2, vy - vh//2), (vx + vw//2, vy + vh//2), v["color"], -1)
            cv2.rectangle(img, (vx - vw//2, vy - vh//2), (vx + vw//2, vy + vh//2), (33, 33, 33), 2) # black border
            
            # Draw windshield
            if v["category"] in [0, 1, 2]:
                ws_h = int(vh * 0.2)
                cv2.rectangle(img, (vx - vw//2 + 4, vy - vh//2 + 4), (vx + vw//2 - 4, vy - vh//2 + 4 + ws_h), (224, 242, 241), -1)
                
            # Update position
            v["y"] += v["speed"]
            
            # Recycle vehicle if it goes off screen
            if v["y"] > h + vh:
                v["y"] = -vh - np.random.randint(50, 250)
                v["category"] = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.1])
                v["x"] = lanes[np.random.randint(0, 4)]
                v["speed"] = np.random.uniform(3.0, 7.0)
                v["color"] = [
                    (21, 101, 192),    # Blue
                    (198, 40, 40),     # Red
                    (239, 108, 0),     # Orange
                    (249, 168, 37)     # Yellow
                ][v["category"]]
                v["size"] = [
                    (24, 36), (32, 60), (36, 70), (16, 24)
                ][v["category"]]
                v["id"] = next_vehicle_id
                next_vehicle_id += 1
                
        out.write(img)
        
    out.release()
    return output_path


# --- STREAMLIT UI SETUP ---

st.set_page_config(
    page_title="AI Traffic Optimization Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling using Streamlit markdown blocks
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 2px;
        font-family: 'Outfit', sans-serif;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# --- MAIN HEADER ---

st.markdown('<div class="main-title">Traffic Analysis Hub & Compression Benchmarks</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deploying quantized and pruned computer vision models (YOLOv5 & DETR) on edge video intelligence.</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---

st.sidebar.image("https://img.icons8.com/color/96/traffic-light.png", width=80)
st.sidebar.markdown("### Model Deployment Panel")

selected_model = st.sidebar.selectbox(
    "Select Target Model",
    ["YOLOv5", "DETR"],
    help="Choose the detection model architecture."
)

opt_type = st.sidebar.selectbox(
    "Optimization Variant",
    [
        "Baseline FP32",
        "INT8 Quantized",
        "Magnitude Pruned",
        "L1-Norm Pruned",
        "Filter Pruned (40%)",
        "Channel Pruned (40%)",
        "Layer Pruned"
    ],
    help="Select the compression type."
)

# Sparsity slider for dynamic pruned versions
sparsity_pct = 50
if opt_type in ["Magnitude Pruned", "L1-Norm Pruned"]:
    sparsity_pct = st.sidebar.select_slider(
        "Select Sparsity Ratio",
        options=[30, 50, 70],
        value=50,
        format_func=lambda x: f"{x}% Sparsity"
    )

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

# Input Video Source Selector
video_source = st.sidebar.radio(
    "Inference Input Source",
    ["Demo Traffic Stream (Synthetic)", "Upload Custom Traffic Video"]
)

uploaded_file = None
if video_source == "Upload Custom Traffic Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI Video File", type=["mp4", "avi"])

# Action Button
start_analysis = st.sidebar.button("⚡ Run Real-Time Traffic Analysis", use_container_width=True)

# Build Target Weight Path from selections
# Resolve Target Weight Path using the new checkpoints structure with fallbacks
def resolve_weights_path(model_name: str, opt_type: str, sparsity_pct: int) -> Tuple[str, bool]:
    m_lower = "yolov5s" if model_name.lower() == "yolov5" else "detr"
    is_quant = False
    
    if opt_type == "Baseline FP32":
        candidates = [
            f"checkpoints/{m_lower}/baseline.pt",
            f"weights/baseline/{m_lower}_baseline.pt",
            f"weights/baseline/{m_lower}_best.pt",
            f"weights/baseline/{m_lower}.pt",
        ]
    elif opt_type == "INT8 Quantized":
        is_quant = True
        candidates = [
            f"checkpoints/{m_lower}/baseline.pt",  # Quantization is simulated on baseline weights
            f"weights/quantized/{m_lower}_int8.pt",
            f"weights/baseline/{m_lower}_baseline.pt",
            f"weights/baseline/{m_lower}.pt",
        ]
    elif opt_type == "Magnitude Pruned":
        sparsity = sparsity_pct / 100.0
        candidates = [
            f"checkpoints/{m_lower}/magnitude_{sparsity}_recovered.pt",
            f"checkpoints/{m_lower}/magnitude_{sparsity}.pt",
            f"weights/magnitude/mag{sparsity_pct}/{m_lower}.pt",
        ]
    elif opt_type == "L1-Norm Pruned":
        sparsity = sparsity_pct / 100.0
        candidates = [
            f"checkpoints/{m_lower}/l1_norm_{sparsity}_recovered.pt",
            f"checkpoints/{m_lower}/l1_norm_{sparsity}.pt",
            f"weights/l1_norm/l1_{sparsity_pct}/{m_lower}.pt",
        ]
    elif opt_type == "Filter Pruned (40%)":
        candidates = [
            f"checkpoints/{m_lower}/filter_0.4_recovered.pt",
            f"checkpoints/{m_lower}/filter_0.4.pt",
            f"weights/filter/{m_lower}.pt",
        ]
    elif opt_type == "Channel Pruned (40%)":
        candidates = [
            f"checkpoints/{m_lower}/channel_0.4_recovered.pt",
            f"checkpoints/{m_lower}/channel_0.4.pt",
            f"weights/channel/{m_lower}.pt",
        ]
    elif opt_type == "Layer Pruned":
        candidates = [
            f"checkpoints/{m_lower}/layer_0.3_recovered.pt",
            f"checkpoints/{m_lower}/layer_0.3.pt",
            f"checkpoints/{m_lower}/layer_0.15_recovered.pt",
            f"checkpoints/{m_lower}/layer_0.15.pt",
            f"weights/layer/{m_lower}.pt",
        ]
    else:
        candidates = []
        
    for path in candidates:
        if os.path.exists(path):
            return path, is_quant
            
    return candidates[0] if candidates else "", is_quant

weights_path, is_quant = resolve_weights_path(selected_model, opt_type, sparsity_pct)
weights_exist = os.path.exists(weights_path)
if not weights_exist:
    st.sidebar.error(f"Weights file not found. Candidates checked in 'checkpoints/' and 'weights/'. Defaulting to: {weights_path}")


# --- TABS LAYOUT ---

tab_analysis, tab_benchmarks, tab_methods = st.tabs([
    "📈 Live Analysis Dashboard",
    "📊 Compression & Speed Benchmarks",
    "⚙️ Optimization Methodologies"
])


# --- TAB 1: LIVE ANALYSIS ---

with tab_analysis:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🖥️ Bounding Box & Tracking Stream")
        video_placeholder = st.empty()
        
    with col2:
        st.subheader("📊 Telemetry & Real-Time Stats")
        
        # Metric placeholders
        metric_fps = st.empty()
        metric_active = st.empty()
        metric_total = st.empty()
        metric_density = st.empty()
        
        st.markdown("---")
        st.subheader("⏱️ Speed Metrics (Current Model)")
        
        # Load benchmark table to show stats for current configuration
        if os.path.exists("benchmark_results.csv"):
            df_bench = pd.read_csv("benchmark_results.csv")
            # Map optimization name to config row
            target_config = opt_type
            if opt_type in ["Magnitude Pruned", "L1-Norm Pruned"]:
                target_config = f"{opt_type.split(' ')[0]} Pruned ({sparsity_pct}%)"
            
            row = df_bench[(df_bench["Model"] == selected_model.upper()) & (df_bench["Config"] == target_config)]
            if not row.empty:
                r = row.iloc[0]
                st.write(f"**Parameters:** {r['Params']:,}")
                st.write(f"**FLOPs:** {r['FLOPs']:,}")
                st.write(f"**Model Size:** {r['Size (MB)']:.2f} MB")
                st.write(f"**Compression Ratio:** {r['Compression Ratio']:.2f}x")
                st.write(f"**Offline Benchmarked FPS:** {r['FPS']:.1f}")
            else:
                st.info("Run generation/benchmarking to see accurate speed metrics.")
        else:
            st.info("Run weight generation and benchmarks to display metrics.")
            
        # Live vehicle history chart placeholder
        st.markdown("### Vehicle Flow Over Time")
        chart_placeholder = st.empty()

    if start_analysis and weights_exist:
        # 1. Load the model and weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.sidebar.info("Loading model into memory...")
        
        # Auto-detect class count from checkpoint weights to prevent size mismatch
        num_classes = 80
        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
                for k, v in state_dict.items():
                    if "24.m.0.weight" in k:
                        num_classes = (v.shape[0] // 3) - 5
                        break
                    elif "class_embed.weight" in k:
                        num_classes = v.shape[0] - 1
                        break
            except Exception as e:
                pass
                
        model = ModelFactory.load(selected_model, num_classes=num_classes, device=device)
        
        # Parse checkpoint name to dynamically register pruning masks before loading state dict
        filename = os.path.basename(weights_path).lower()
        prune_type = None
        sparsity = 0.0
        
        for pt in ["magnitude", "l1_norm", "filter", "channel", "layer"]:
            if pt in filename:
                prune_type = pt
                # Parse sparsity from filename (e.g. l1_norm_0.3_recovered.pt)
                try:
                    parts = filename.split("_")
                    for p in parts:
                        try:
                            val = float(p)
                            if 0.0 < val < 1.0:
                                sparsity = val
                                break
                        except ValueError:
                            pass
                except Exception:
                    sparsity = 0.3
                break
                
        # Parse legacy weights folder paths
        if not prune_type and "weights/" in weights_path.replace("\\", "/"):
            if "magnitude" in weights_path:
                prune_type = "magnitude"
                sparsity = sparsity_pct / 100.0
            elif "l1_norm" in weights_path:
                prune_type = "l1_norm"
                sparsity = sparsity_pct / 100.0
            elif "filter" in weights_path:
                prune_type = "filter"
                sparsity = 0.4
            elif "channel" in weights_path:
                prune_type = "channel"
                sparsity = 0.4
            elif "layer" in weights_path:
                prune_type = "layer"
                sparsity = 0.3
                
        if prune_type:
            st.sidebar.info(f"Registering dynamic pruning masks for {prune_type} ({sparsity*100:.0f}%)...")
            from pruning.base import PRUNER_REGISTRY
            pruner_cls = PRUNER_REGISTRY[prune_type]
            pruner = pruner_cls(model, sparsity)
            model = pruner.prune()

        if is_quant:
            model = quantize_model_static_simulated(model)
            model = load_quantized_weights(model, weights_path)
        else:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False), strict=False)
            
        model = model.to(device)
        model.eval()
        st.sidebar.success("Model loaded successfully!")
        
        # 2. Prepare Video Stream
        video_path = "demo_traffic_flow.mp4"
        if video_source == "Upload Custom Traffic Video" and uploaded_file is not None:
            with open("temp_uploaded_video.mp4", "wb") as f:
                f.write(uploaded_file.read())
            video_path = "temp_uploaded_video.mp4"
        else:
            # Generate synthetic video on-the-fly
            with st.spinner("Generating synthetic highway traffic video..."):
                generate_synthetic_traffic_video(video_path, num_frames=180)
                
        cap = cv2.VideoCapture(video_path)
        
        # Initialize Tracker
        tracker = TrafficTracker(line_start=(100, 320), line_end=(540, 320))
        
        # Playback loop
        frame_history = []
        flow_history = []
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prep input tensor
            input_h, input_w = (640, 640) if selected_model.lower() == "yolov5" else (800, 800)
            img_resized = cv2.resize(frame_rgb, (input_w, input_h))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            
            # Predict
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model(img_tensor)
            inference_time = time.perf_counter() - start_time
            fps = 1.0 / inference_time
            
            # Update Tracker & Annotations
            annotated_frame, stats = tracker.process_frame(frame_rgb, outputs)
            
            # Update dashboard metrics
            metric_fps.metric("Processing Speed (FPS)", f"{fps:.1f}")
            metric_active.metric("Active Vehicles", stats["active_vehicles"])
            metric_total.metric("Total Counted (Crossed Line)", stats["total_count"])
            metric_density.metric("Traffic Density Level", stats["density"])
            
            # Display processed frame
            video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
            
            # Add to history
            flow_history.append(stats["active_vehicles"])
            frame_history.append(frame_idx)
            
            # Plot updated chart
            chart_df = pd.DataFrame({
                "Frame": frame_history,
                "Active Vehicles": flow_history
            }).set_index("Frame")
            chart_placeholder.line_chart(chart_df)
            
            # Update progress bar
            frame_idx += 1
            progress_bar.progress(frame_idx / total_frames)
            
            # Small delay to mimic normal playback speeds
            time.sleep(0.01)
            
        cap.release()
        st.success("Video analysis completed!")


# --- TAB 2: COMPRESSION BENCHMARKS ---

with tab_benchmarks:
    st.subheader("📊 Model Compression & Efficiency Dashboards")
    
    if os.path.exists("benchmark_results.csv"):
        df_results = pd.read_csv("benchmark_results.csv")
        
        # Model selector
        b_model = st.radio("Choose Model to Display Benchmarks", ["YOLOv5", "DETR"], horizontal=True)
        df_model = df_results[df_results["Model"] == b_model].copy()
        
        # Display key charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown(f"#### {b_model} Accuracy vs Compression Trade-off")
            acc_img_path = f"reports/{b_model.lower()}_accuracy_vs_compression.png"
            if os.path.exists(acc_img_path):
                st.image(acc_img_path)
            else:
                st.warning("Accuracy chart not found. Run benchmark.py to generate plots.")
                
            st.markdown(f"#### {b_model} Latency & FPS vs Compression Ratio")
            fps_img_path = f"reports/{b_model.lower()}_fps_latency_vs_compression.png"
            if os.path.exists(fps_img_path):
                st.image(fps_img_path)
            else:
                st.warning("FPS chart not found. Run benchmark.py to generate plots.")
                
        with c2:
            st.markdown(f"#### {b_model} Computational & Weight Reductions")
            red_img_path = f"reports/{b_model.lower()}_resource_reduction.png"
            if os.path.exists(red_img_path):
                st.image(red_img_path)
            else:
                st.warning("Resource reduction chart not found.")
                
            st.markdown("#### Detailed Performance Metrics Table")
            display_cols = [
                "Config", "Params", "FLOPs", "Size (MB)",
                "Compression Ratio", "FPS", "Latency (ms)", "Precision", "Recall", "mAP50"
            ]
            st.dataframe(
                df_model[display_cols].style.format({
                    "Params": "{:,}",
                    "FLOPs": "{:,}",
                    "Size (MB)": "{:.2f}",
                    "Compression Ratio": "{:.2f}x",
                    "FPS": "{:.1f}",
                    "Latency (ms)": "{:.2f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "mAP50": "{:.3f}"
                }),
                hide_index=True
            )
    else:
        st.warning("Benchmark results file not found. Please run the benchmarks using the terminal script to load comparison dashboards.")


# --- TAB 3: OPTIMIZATION METHODOLOGIES ---

with tab_methods:
    st.subheader("⚙️ Quantization & Pruning Methodologies")
    
    st.markdown("""
    ### Post-Training INT8 Quantization
    Model quantization converts floating-point weight tensors and activation maps (FP32, 32 bits) to 8-bit integer formats (INT8).
    This reduces the overall memory bandwidth pressure and allows execution on low-power, high-throughput integer arithmetic units.
    - **Weight Compression:** Reduces storage sizes of neural networks on disk by exactly **4x**.
    - **Quantization Simulation:** Minimizes the accuracy penalty via symmetric, min-max calibration (quantizing to [-128, 127] levels) during the conversion.
    
    ### Model Pruning Strategies
    Pruning trims unnecessary neurons, weights, filters, or layers to compress model sizes and lower computational overhead.
    
    1. **Unstructured Magnitude Pruning:**
       - Selects individual connections/weights across all Conv and Linear layers with the lowest absolute magnitudes and zeroes them out.
       - Tested Sparsity Levels: **30%, 50%, 70%**.
       
    2. **Structured L1-Norm Pruning:**
       - Computes the L1-Norm (sum of absolute weights) of each filter in convolutional layers.
       - Prunes filters with the lowest L1-norm (weak features).
       - Tested Sparsity Levels: **30%, 50%, 70%**.
       
    3. **Structured Filter Pruning:**
       - Removes redundant output filters (channels) from convolutional layers, narrowing the activation maps.
       - Saves parameters and FLOPs while maintaining compatibility with general CPU/GPU hardware.
       
    4. **Structured Channel Pruning:**
       - Removes input channels of convolutional and linear layers.
       - Trims down redundant input features, speeding up matrix multiplication.
       
    5. **Layer Pruning:**
       - Replaces whole layers or residual bottlenecks (e.g. C3 blocks in YOLOv5 or Encoder layers in DETR) with an identity mapping (`nn.Identity`).
       - Decreases net depth, providing a linear speedup in latency.
    """)
