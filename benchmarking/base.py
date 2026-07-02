import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from evaluation.validator import benchmark_latency_fps

logger = logging.getLogger("BaseBenchmark")


class BaseBenchmark:
    """Abstract base benchmark for profiling deep learning models."""

    def __init__(self, model_name: str, device: torch.device, img_size: Tuple[int, int, int] = (3, 640, 640)):
        self.model_name = model_name
        self.device = device
        self.img_size = img_size

    def profile_model(self, model: nn.Module) -> Dict[str, Any]:
        """Calculates structural metrics of the model: parameters, FLOPs, size, and global weight sparsity."""
        model = model.to(self.device)
        model.eval()
        
        # Calculate parameters count (total parameters)
        params = sum(p.numel() for p in model.parameters())
        
        # Calculate model size in MB (exclude pruning_mask buffers — they are temporary artifacts)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for name, b in model.named_buffers() if "pruning_mask" not in name)
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Calculate FLOPs using validator/thop/fvcore if available
        flops = None
        try:
            from thop import profile
            try:
                x = torch.zeros(1, *self.img_size, device=self.device)
                macs, _ = profile(model, inputs=(x,), verbose=False)
                flops = int(macs * 2)
            except Exception as e:
                logger.error(f"Error executing thop profiling: {e}")
        except ImportError:
            logger.warning("thop library is not installed.")

        if flops is None:
            try:
                # pyrefly: ignore [missing-import]
                from fvcore.nn import FlopCountAnalysis
                try:
                    x = torch.zeros(1, *self.img_size, device=self.device)
                    flops = FlopCountAnalysis(model, x).total()
                except Exception as e:
                    logger.error(f"Error executing fvcore profiling: {e}")
            except ImportError:
                logger.warning("fvcore library is not installed.")

        if flops is None:
            logger.warning("Using fallback estimation for FLOPs.")
            active_params = sum(p.numel() for p in model.parameters() if (p != 0).any())
            flops = int(active_params * 2 * 10)
                
        # Calculate actual global sparsity
        total_weights = 0
        zero_weights = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.data
                total_weights += w.numel()
                zero_weights += (w == 0.0).sum().item()
        sparsity = zero_weights / total_weights if total_weights > 0 else 0.0

        return {
            "params": params,
            "flops": flops,
            "size_mb": size_mb,
            "sparsity": sparsity,
        }

    def profile_speed(self, model: nn.Module, num_runs: int = 50) -> Dict[str, float]:
        """Profiles the execution latency (ms) and throughput (FPS) on the device."""
        model = model.to(self.device)
        res = benchmark_latency_fps(model, self.device, num_runs=num_runs, img_size=self.img_size)
        return {
            "latency_ms": res[0],
            "fps": res[1],
            "latency_std_ms": res[2],
            "latency_median_ms": res[3],
        }
