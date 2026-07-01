#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import ModelFactory, extract_model_state_dict
from utils.pipeline_utils import load_checkpoint_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ExportONNX")


class YOLOv5ExportWrapper(nn.Module):
    """Wrapper to make YOLOv5s ONNX export clean and consistent in evaluation mode."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.forward_module = base_model.model if hasattr(base_model, 'model') else base_model
        self.forward_module.eval()

    def forward(self, x):
        # Forward pass through the inner YOLOv5 model
        out = self.forward_module(x)
        if isinstance(out, (tuple, list)):
            decoded = out[0]
        else:
            decoded = out
            
        # decoded shape: [B, num_boxes, 5 + num_classes]
        cx = decoded[..., 0]
        cy = decoded[..., 1]
        w = decoded[..., 2]
        h = decoded[..., 3]
        
        # Convert center-xywh to x1y1x2y2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)
        
        # Normalize to [0, 1] relative coordinates using input shape.
        # Note: In this codebase's YOLOv5Wrapper, decoded predictions are in absolute 
        # pixel coordinates (e.g. 640x640 range), so they must be normalized here.
        img_h, img_w = float(x.shape[2]), float(x.shape[3])
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=x.device)
        boxes_norm = boxes_xyxy / scale
        
        # Calculate scores (objectness * class probabilities)
        obj_conf = decoded[..., 4:5]
        cls_conf = decoded[..., 5:]
        scores = obj_conf * cls_conf
        
        # Get maximum confidence score and class index
        max_scores, class_ids = scores.max(-1)
        
        return boxes_norm, max_scores, class_ids


class DETRExportWrapper(nn.Module):
    """Wrapper to make DETR ONNX export clean and consistent in evaluation mode."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.forward_module = base_model.model if hasattr(base_model, 'model') else base_model
        self.forward_module.eval()

    def forward(self, x):
        # Forward pass through the inner DETR model
        out = self.forward_module(x)
        logits = out["pred_logits"]
        boxes = out["pred_boxes"]
        
        # Calculate class probabilities (excluding the background class at index -1)
        probs = F.softmax(logits, dim=-1)[:, :, :-1]
        scores, class_ids = probs.max(-1)
        
        # Convert relative cxcywh coordinates to relative x1y1x2y2
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)
        
        # Map class IDs if using 91-class pretrained head for COCO
        inner_model = self.forward_module
        num_classes = getattr(inner_model, "num_classes", getattr(self.base_model, "num_classes", 0))
        if num_classes == 91:
            coco_cat_to_idx = {
                1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
                11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18,
                21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
                33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36,
                42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45,
                52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
                61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
                74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
                84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
            }
            mapping_arr = [-1] * 92
            for cat_id, idx in coco_cat_to_idx.items():
                mapping_arr[cat_id] = idx
            mapping_tensor = torch.tensor(mapping_arr, dtype=torch.long, device=x.device)
            mapped_class_ids = mapping_tensor[class_ids]
        else:
            mapped_class_ids = class_ids
            
        return boxes_xyxy, scores, mapped_class_ids


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoints to ONNX format")
    parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "detr"], help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to input PyTorch checkpoint (.pt file)")
    parser.add_argument("--output", type=str, default="", help="Path to save output ONNX model")
    parser.add_argument("--export-style", type=str, default="decoded", choices=["decoded", "raw"], 
                        help="Export style: 'decoded' (consistent boxes, scores, classes) or 'raw' (original outputs)")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "detrac"], help="Dataset format/classes")
    parser.add_argument("--num-classes", type=int, default=80, help="Fallback class count if not in checkpoint")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch sizes and dimensions")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model using onnxsim")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()

    # Determine default output path if not provided
    if not args.output:
        base_dir = os.path.dirname(args.checkpoint)
        filename = os.path.basename(args.checkpoint)
        if filename.endswith(".pt"):
            filename = filename[:-3]
        filename = f"{filename}_{args.export_style}.onnx"
        args.output = os.path.join(base_dir, filename)

    logger.info(f"Loading model architecture: {args.model}")
    
    # Load checkpoint details
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)
    
    # Detect if checkpoint has weight masks
    has_masks = any(k.endswith(".weight_mask") for k in state_dict.keys())
    
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    prune_type = config.get("prune_type", config.get("pruning_method", "None"))
    sparsity = config.get("sparsity", 0.0)
    
    prune_type_str = str(prune_type)
    sparsity_val = float(sparsity)
    
    # Auto-detect magnitude pruning if masks are present in checkpoint but config is missing/None
    if has_masks and (prune_type_str.lower() == "none" or sparsity_val == 0.0):
        prune_type_str = "magnitude"
        total_elements = 0
        zero_elements = 0
        for k, v in state_dict.items():
            if k.endswith(".weight_mask"):
                total_elements += v.numel()
                zero_elements += (v == 0.0).sum().item()
        if total_elements > 0:
            sparsity_val = zero_elements / total_elements
        else:
            sparsity_val = 0.3
    
    # Auto-detect class count from checkpoint weights
    num_classes = args.num_classes
    for k, v in state_dict.items():
        if "24.m.0.weight" in k:
            num_classes = (v.shape[0] // 3) - 5
            break
        elif "class_embed.weight" in k:
            num_classes = v.shape[0] - 1
            break
    logger.info(f"Detected configurations -> num_classes: {num_classes}, prune_type: {prune_type_str}, sparsity: {sparsity_val}, has_masks: {has_masks}")

    # Instantiate model structure
    model = ModelFactory.load(args.model, num_classes=num_classes)
    
    # Register pruning masks if required
    from pruning.base import PRUNER_REGISTRY
    from utils.pipeline_utils import bake_pruned_weights
    
    should_prune = False
    if prune_type_str.lower() != "none" and sparsity_val > 0.0:
        if prune_type_str.lower() == "magnitude":
            should_prune = has_masks
        else:
            should_prune = True
            
    if should_prune:
        logger.info(f"Registering pruning structure for type '{prune_type_str}'...")
        prune_type_lower = prune_type_str.lower()
        if prune_type_lower in PRUNER_REGISTRY:
            pruner_cls = PRUNER_REGISTRY[prune_type_lower]
            pruner = pruner_cls(model, sparsity_val)
            model = pruner.prune()
            
    # Load checkpoint weights
    load_checkpoint_weights(model, args.checkpoint, torch.device("cpu"))
    
    # Bake weights (remove hooks and finalize weights to boost inference speed)
    if has_masks:
        logger.info("Baking pruning masks into weights...")
        bake_pruned_weights(model)
        
    model.eval()
    model.cpu()

    img_size = model.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size)
    logger.info(f"Created dummy input tensor with shape: {dummy_input.shape}")

    # Set up wrapper based on export style
    if args.export_style == "decoded":
        logger.info("Wrapping model in decoded export wrapper for consistent ONNX output format...")
        if args.model == "yolov5s":
            export_model = YOLOv5ExportWrapper(model)
            output_names = ["boxes", "scores", "class_ids"]
        else:
            export_model = DETRExportWrapper(model)
            output_names = ["boxes", "scores", "class_ids"]
    else:
        logger.info("Using raw model forward pass for ONNX export...")
        export_model = model
        if args.model == "yolov5s":
            output_names = ["outputs"]
        else:
            output_names = ["pred_logits", "pred_boxes"]

    # Dynamic axes setup
    dynamic_axes = {
        "images": {0: "batch_size"}
    }
    if args.dynamic:
        # Also allow dynamic height/width for fully flexible deployments
        dynamic_axes["images"][2] = "height"
        dynamic_axes["images"][3] = "width"

    for out_name in output_names:
        if args.dynamic and args.model == "yolov5s":
            dynamic_axes[out_name] = {0: "batch_size", 1: "num_boxes"}
        else:
            dynamic_axes[out_name] = {0: "batch_size"}

    export_model.eval()
    
    # Run a test forward pass to verify shapes before export
    logger.info("Verifying model output shapes with a dummy forward pass...")
    with torch.no_grad():
        test_out = export_model(dummy_input)
        if isinstance(test_out, tuple):
            for i, name in enumerate(output_names):
                logger.info(f" - Output '{name}' shape: {list(test_out[i].shape)}")
        elif isinstance(test_out, dict):
            for name, tensor in test_out.items():
                logger.info(f" - Output '{name}' shape: {list(tensor.shape)}")
        else:
            logger.info(f" - Output shape: {list(test_out.shape)}")

    logger.info(f"Exporting model to ONNX path: {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes if args.dynamic or args.export_style == "decoded" else None
    )
    logger.info("ONNX export completed successfully!")

    # Embed external weights data back into the main .onnx file if split by exporter
    import onnx
    data_path = args.output + ".data"
    alt_data_path = os.path.splitext(args.output)[0] + ".data"
    target_data_path = None
    if os.path.exists(data_path):
        target_data_path = data_path
    elif os.path.exists(alt_data_path):
        target_data_path = alt_data_path
        
    if target_data_path:
        logger.info(f"Embedding external weight data ({target_data_path}) back into single ONNX file...")
        onnx_model = onnx.load(args.output)
        onnx.save(onnx_model, args.output)
        os.remove(target_data_path)
        logger.info(f"Removed temporary external data file: {target_data_path}")

    # Simplify ONNX using onnxsim if requested
    if args.simplify:
        try:
            import onnxsim
            logger.info("Simplifying ONNX model graph...")
            import onnx
            onnx_model = onnx.load(args.output)
            model_simp, check = onnxsim.simplify(onnx_model)
            assert check, "Simplified ONNX model validation failed"
            onnx.save(model_simp, args.output)
            logger.info("ONNX model simplified successfully!")
        except ImportError:
            logger.warning("onnx-simplifier is not installed. Skipping simplification.")
        except Exception as e:
            logger.error(f"Failed to simplify ONNX model: {e}")

    # Verify ONNX model against PyTorch output
    if args.verify:
        try:
            import onnxruntime as ort
            logger.info("Verifying ONNX model outputs using ONNX Runtime...")
            
            # Run PyTorch
            with torch.no_grad():
                torch_out = export_model(dummy_input)
            
            # Run ONNX Runtime
            ort_sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
            ort_inputs = {"images": dummy_input.numpy()}
            ort_outs = ort_sess.run(None, ort_inputs)
            
            # Compare output count
            if len(ort_outs) != (1 if isinstance(torch_out, torch.Tensor) else len(torch_out)):
                logger.error("ONNX verification failed: Output count mismatch!")
                sys.exit(1)

            # Compare outputs numerically
            logger.info("Comparing PyTorch and ONNX Runtime outputs:")
            if isinstance(torch_out, torch.Tensor):
                diff = np.abs(torch_out.numpy() - ort_outs[0])
                max_diff = diff.max()
                logger.info(f" - Output 'outputs' max absolute difference: {max_diff:.6e}")
            elif isinstance(torch_out, tuple):
                for i, name in enumerate(output_names):
                    pytorch_tensor = torch_out[i].cpu().numpy()
                    onnx_tensor = ort_outs[i]
                    diff = np.abs(pytorch_tensor - onnx_tensor)
                    max_diff = diff.max()
                    logger.info(f" - Output '{name}' max absolute difference: {max_diff:.6e}")
            elif isinstance(torch_out, dict):
                # Raw DETR outputs dict
                for i, name in enumerate(output_names):
                    pytorch_tensor = torch_out[name].cpu().numpy()
                    onnx_tensor = ort_outs[i]
                    diff = np.abs(pytorch_tensor - onnx_tensor)
                    max_diff = diff.max()
                    logger.info(f" - Output '{name}' max absolute difference: {max_diff:.6e}")
            
            logger.info("ONNX verification complete. Model matches PyTorch outputs closely.")
        except ImportError:
            logger.warning("onnxruntime is not installed. Skipping verification.")
        except Exception as e:
            logger.error(f"Verification encountered an error: {e}")


if __name__ == "__main__":
    main()
