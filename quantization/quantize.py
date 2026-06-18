import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Quantization")

# --- SIMULATED INT8 PTQ ---

class QuantizedConv2dSim(nn.Module):
    """Simulated INT8 Quantized Convolutional Layer."""
    def __init__(self, ref_conv: nn.Conv2d):
        super().__init__()
        self.in_channels = ref_conv.in_channels
        self.out_channels = ref_conv.out_channels
        self.kernel_size = ref_conv.kernel_size
        self.stride = ref_conv.stride
        self.padding = ref_conv.padding
        self.dilation = ref_conv.dilation
        self.groups = ref_conv.groups
        
        # Save floating point weights & biases
        self.weight = nn.Parameter(ref_conv.weight.data.clone())
        self.bias = nn.Parameter(ref_conv.bias.data.clone()) if ref_conv.bias is not None else None
        
        # Quantization parameters
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zero_point", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("act_scale", torch.tensor(1.0))
        self.register_buffer("act_zero_point", torch.tensor(0, dtype=torch.int32))
        
        self.calibrate_weights()

    def calibrate_weights(self):
        """Calibrate weight quantization parameters (Min-Max quantization)."""
        w_min = self.weight.data.min().item()
        w_max = self.weight.data.max().item()
        
        # Map [w_min, w_max] to [-128, 127]
        max_val = max(abs(w_min), abs(w_max), 1e-8)
        self.w_scale.copy_(torch.tensor(max_val / 127.0))
        self.w_zero_point.copy_(torch.tensor(0, dtype=torch.int32)) # Symmetric quantization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize weights to INT8 and dequantize to simulate precision loss
        w_q = torch.clamp(torch.round(self.weight / self.w_scale) + self.w_zero_point, -128, 127)
        w_dq = (w_q - self.w_zero_point) * self.w_scale
        
        # 2. Quantize activations (inputs)
        # For simplicity, we use running min-max for inputs
        x_min = x.detach().min().item()
        x_max = x.detach().max().item()
        max_val = max(abs(x_min), abs(x_max), 1e-8)
        act_scale = max_val / 127.0
        
        x_q = torch.clamp(torch.round(x / act_scale), -128, 127)
        x_dq = x_q * act_scale
        
        # 3. Perform Conv2d with quantized values
        out = F.conv2d(x_dq, w_dq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class QuantizedLinearSim(nn.Module):
    """Simulated INT8 Quantized Linear Layer."""
    def __init__(self, ref_linear: nn.Linear):
        super().__init__()
        self.in_features = ref_linear.in_features
        self.out_features = ref_linear.out_features
        
        self.weight = nn.Parameter(ref_linear.weight.data.clone())
        self.bias = nn.Parameter(ref_linear.bias.data.clone()) if ref_linear.bias is not None else None
        
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zero_point", torch.tensor(0, dtype=torch.int32))
        
        self.calibrate_weights()

    def calibrate_weights(self):
        w_min = self.weight.data.min().item()
        w_max = self.weight.data.max().item()
        max_val = max(abs(w_min), abs(w_max), 1e-8)
        self.w_scale.copy_(torch.tensor(max_val / 127.0))
        self.w_zero_point.copy_(torch.tensor(0, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weight
        w_q = torch.clamp(torch.round(self.weight / self.w_scale) + self.w_zero_point, -128, 127)
        w_dq = (w_q - self.w_zero_point) * self.w_scale
        
        # Quantize activation
        x_min = x.detach().min().item()
        x_max = x.detach().max().item()
        max_val = max(abs(x_min), abs(x_max), 1e-8)
        act_scale = max_val / 127.0
        
        x_q = torch.clamp(torch.round(x / act_scale), -128, 127)
        x_dq = x_q * act_scale
        
        out = F.linear(x_dq, w_dq, self.bias)
        return out


def quantize_model_static_simulated(model: nn.Module) -> nn.Module:
    """Recursively replaces Conv2d and Linear layers with simulated INT8 quantized versions."""
    # Deep copy the model to keep original intact
    import copy
    model_quant = copy.deepcopy(model)
    
    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, name, QuantizedConv2dSim(child))
            elif isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinearSim(child))
            else:
                replace_layers(child)
                
    replace_layers(model_quant)
    logger.info("Successfully converted model to simulated static INT8 quantized model.")
    return model_quant


def save_quantized_weights(model: nn.Module, filepath: str):
    """
    Saves weights of the model as actual INT8 tensors to simulate compression.
    Standard state_dict is saved, but Conv2d/Linear weights are cast to int8
    along with scale and zero_point so the file is 4x smaller.
    """
    state_dict = model.state_dict()
    quantized_state_dict = {}
    
    for k, v in state_dict.items():
        if "weight" in k and len(v.shape) >= 2:
            # Check if this is a weight we can quantize to int8
            w_min = v.min().item()
            w_max = v.max().item()
            max_val = max(abs(w_min), abs(w_max), 1e-8)
            scale = max_val / 127.0
            
            # Cast to int8
            w_q = torch.clamp(torch.round(v / scale), -128, 127).to(torch.int8)
            quantized_state_dict[k] = w_q
            quantized_state_dict[f"{k}_scale"] = torch.tensor(scale, dtype=torch.float32)
        else:
            # Biases and other buffers kept as float32
            quantized_state_dict[k] = v
            
    torch.save(quantized_state_dict, filepath)
    logger.info(f"Saved compressed INT8 weights to {filepath} (Simulated 4x weight size reduction).")


def load_quantized_weights(model: nn.Module, filepath: str) -> nn.Module:
    """
    Loads quantized weights from disk, dequantizes them, and loads into FP32 model.
    """
    quantized_state_dict = torch.load(filepath)
    fp32_state_dict = {}
    
    for k in list(quantized_state_dict.keys()):
        if k.endswith("_scale"):
            continue
            
        v = quantized_state_dict[k]
        if v.dtype == torch.int8:
            scale = quantized_state_dict[f"{k}_scale"]
            # Dequantize to float32
            fp32_state_dict[k] = v.to(torch.float32) * scale
        else:
            fp32_state_dict[k] = v
            
    model.load_state_dict(fp32_state_dict, strict=False)
    logger.info(f"Successfully loaded and dequantized weights from {filepath}.")
    return model


if __name__ == "__main__":
    from models.models import YOLOv5Detector, DETRDetector
    
    # Test quantization on YOLOv5
    yolo = YOLOv5Detector()
    yolo_q = quantize_model_static_simulated(yolo)
    
    # Save quantized weights
    os.makedirs("weights/quantized", exist_ok=True)
    save_quantized_weights(yolo, "weights/quantized/yolov5_int8.pt")
    
    # Load and verify size
    yolo_loaded = YOLOv5Detector()
    load_quantized_weights(yolo_loaded, "weights/quantized/yolov5_int8.pt")
    
    fp32_size = os.path.getsize("weights/quantized/yolov5_int8.pt") / (1024 * 1024)
    print(f"Quantized weight file size: {fp32_size:.2f} MB")
