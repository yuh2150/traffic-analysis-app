import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Sets standard and PyTorch seeds to ensure reproducibility across experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic CudNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Deterministic seed {seed} set globally.")
