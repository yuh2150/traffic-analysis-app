import os
import cv2
import torch
import pandas as pd
import numpy as np
import streamlit as st
import time
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

# Local imports
from models.models import get_model
from quantization.quantize import quantize_model_static_simulated, load_quantized_weights
from tracking.tracker import TrafficTracker

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
""", unsafe_html=True)


# --- MAIN HEADER ---

st.markdown('<div class="main-title">Traffic Analysis Hub & Compression Benchmarks</div>', unsafe_html=True)
st.markdown('<div class="subtitle">Deploying quantized and pruned computer vision models (YOLOv5 & DETR) on edge video intelligence.</div>', unsafe_html=True)

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
weights_path = ""
is_quant = False

m_lower = selected_model.lower()
ext = ".pt" if m_lower == "yolov5" else ".pth"

if opt_type == "Baseline FP32":
    weights_path = f"weights/baseline/{m_lower}{ext}"
elif opt_type == "INT8 Quantized":
    weights_path = f"weights/quantized/{m_lower}_int8{ext}"
    is_quant = True
elif opt_type == "Magnitude Pruned":
    weights_path = f"weights/magnitude/mag{sparsity_pct}/{m_lower}{ext}"
elif opt_type == "L1-Norm Pruned":
    weights_path = f"weights/l1_norm/l1_{sparsity_pct}/{m_lower}{ext}"
elif opt_type == "Filter Pruned (40%)":
    weights_path = f"weights/filter/{m_lower}{ext}"
elif opt_type == "Channel Pruned (40%)":
    weights_path = f"weights/channel/{m_lower}{ext}"
elif opt_type == "Layer Pruned":
    weights_path = f"weights/layer/{m_lower}{ext}"

# Check weights availability
weights_exist = os.path.exists(weights_path)
if not weights_exist:
    st.sidebar.error(f"Weights file not found at: {weights_path}. Please run weight generation first!")


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
        
        model = get_model(selected_model)
        if is_quant:
            model = quantize_model_static_simulated(model)
            model = load_quantized_weights(model, weights_path)
        else:
            model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
            
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
