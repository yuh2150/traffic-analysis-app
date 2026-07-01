#!/usr/bin/env python3
import os
import sys
import time
import torch
import gzip
import zipfile
import bz2
import tempfile
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import extract_model_state_dict

def analyze_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    state_dict = extract_model_state_dict(checkpoint)
    
    # Calculate sparsity of Conv/Linear layers
    total_elements = 0
    zero_elements = 0
    prunable_keys = []
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2: # Linear and Conv weights
            total_elements += v.numel()
            zeros = (v == 0.0).sum().item()
            zero_elements += zeros
            prunable_keys.append(k)
            
    sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
    print(f"Total elements in prunable layers: {total_elements:,}")
    print(f"Zero elements in prunable layers:  {zero_elements:,}")
    print(f"Actual sparsity:                   {sparsity*100:.2f}%")
    
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Original file size:                 {original_size:.2f} MB\n")
    
    results = {}
    
    # 1. Base Dense Saving (reference)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        dense_path = tmp.name
    try:
        t0 = time.time()
        torch.save(state_dict, dense_path)
        t_save = time.time() - t0
        
        t0 = time.time()
        _ = torch.load(dense_path, map_location="cpu")
        t_load = time.time() - t0
        
        size = os.path.getsize(dense_path) / (1024 * 1024)
        results["Dense"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(dense_path):
            os.remove(dense_path)
            
    # 2. Compressed Dense using gzip
    with tempfile.NamedTemporaryFile(suffix=".pt.gz", delete=False) as tmp:
        gzip_path = tmp.name
    try:
        t0 = time.time()
        with gzip.open(gzip_path, "wb") as f:
            torch.save(state_dict, f)
        t_save = time.time() - t0
        
        t0 = time.time()
        with gzip.open(gzip_path, "rb") as f:
            _ = torch.load(f, map_location="cpu")
        t_load = time.time() - t0
        
        size = os.path.getsize(gzip_path) / (1024 * 1024)
        results["Dense + Gzip"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(gzip_path):
            os.remove(gzip_path)

    # 3. Compressed Dense using zipfile (DEFLATE)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
    try:
        t0 = time.time()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            # Save state dict into a temp file, write to zip, remove temp
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as inner_tmp:
                inner_path = inner_tmp.name
            try:
                torch.save(state_dict, inner_path)
                z.write(inner_path, arcname="state_dict.pt")
            finally:
                if os.path.exists(inner_path):
                    os.remove(inner_path)
        t_save = time.time() - t0
        
        t0 = time.time()
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open("state_dict.pt") as f:
                _ = torch.load(f, map_location="cpu")
        t_load = time.time() - t0
        
        size = os.path.getsize(zip_path) / (1024 * 1024)
        results["Dense + Zip (Deflate)"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # 4. PyTorch Sparse COO Representation
    sparse_coo_sd = {}
    for k, v in state_dict.items():
        if k in prunable_keys:
            # Only convert to sparse if it has zeros
            if (v == 0.0).any():
                sparse_coo_sd[k] = v.to_sparse_coo()
            else:
                sparse_coo_sd[k] = v
        else:
            sparse_coo_sd[k] = v
            
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        sparse_coo_path = tmp.name
    try:
        t0 = time.time()
        torch.save(sparse_coo_sd, sparse_coo_path)
        t_save = time.time() - t0
        
        t0 = time.time()
        loaded = torch.load(sparse_coo_path, map_location="cpu")
        # Inflate back to dense
        inflated = {}
        for k, v in loaded.items():
            if isinstance(v, torch.Tensor) and v.is_sparse:
                inflated[k] = v.to_dense()
            else:
                inflated[k] = v
        t_load = time.time() - t0
        
        size = os.path.getsize(sparse_coo_path) / (1024 * 1024)
        results["PyTorch Sparse COO"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(sparse_coo_path):
            os.remove(sparse_coo_path)

    # 5. PyTorch Sparse COO + Gzip
    with tempfile.NamedTemporaryFile(suffix=".pt.gz", delete=False) as tmp:
        sparse_coo_gz_path = tmp.name
    try:
        t0 = time.time()
        with gzip.open(sparse_coo_gz_path, "wb") as f:
            torch.save(sparse_coo_sd, f)
        t_save = time.time() - t0
        
        t0 = time.time()
        with gzip.open(sparse_coo_gz_path, "rb") as f:
            loaded = torch.load(f, map_location="cpu")
        # Inflate back to dense
        inflated = {}
        for k, v in loaded.items():
            if isinstance(v, torch.Tensor) and v.is_sparse:
                inflated[k] = v.to_dense()
            else:
                inflated[k] = v
        t_load = time.time() - t0
        
        size = os.path.getsize(sparse_coo_gz_path) / (1024 * 1024)
        results["Sparse COO + Gzip"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(sparse_coo_gz_path):
            os.remove(sparse_coo_gz_path)

    # 6. Custom Mask + 1D Values Representation
    # We store: mask (bool tensor) and non-zero values (1D tensor)
    custom_sparse_sd = {}
    for k, v in state_dict.items():
        if k in prunable_keys:
            if (v == 0.0).any():
                mask = v != 0.0
                values = v[mask]
                shape = v.shape
                custom_sparse_sd[k] = {"mask": mask, "values": values, "shape": shape}
            else:
                custom_sparse_sd[k] = v
        else:
            custom_sparse_sd[k] = v

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        custom_sparse_path = tmp.name
    try:
        t0 = time.time()
        torch.save(custom_sparse_sd, custom_sparse_path)
        t_save = time.time() - t0
        
        t0 = time.time()
        loaded = torch.load(custom_sparse_path, map_location="cpu")
        # Inflate back to dense
        inflated = {}
        for k, v in loaded.items():
            if isinstance(v, dict) and "mask" in v and "values" in v and "shape" in v:
                dense_v = torch.zeros(v["shape"], dtype=v["values"].dtype, device=v["values"].device)
                dense_v[v["mask"]] = v["values"]
                inflated[k] = dense_v
            else:
                inflated[k] = v
        t_load = time.time() - t0
        
        size = os.path.getsize(custom_sparse_path) / (1024 * 1024)
        results["Custom Mask+1D (Uncompressed)"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(custom_sparse_path):
            os.remove(custom_sparse_path)

    # 7. Custom Mask + 1D Values + Gzip
    with tempfile.NamedTemporaryFile(suffix=".pt.gz", delete=False) as tmp:
        custom_sparse_gz_path = tmp.name
    try:
        t0 = time.time()
        with gzip.open(custom_sparse_gz_path, "wb") as f:
            torch.save(custom_sparse_sd, f)
        t_save = time.time() - t0
        
        t0 = time.time()
        with gzip.open(custom_sparse_gz_path, "rb") as f:
            loaded = torch.load(f, map_location="cpu")
        # Inflate back to dense
        inflated = {}
        for k, v in loaded.items():
            if isinstance(v, dict) and "mask" in v and "values" in v and "shape" in v:
                dense_v = torch.zeros(v["shape"], dtype=v["values"].dtype, device=v["values"].device)
                dense_v[v["mask"]] = v["values"]
                inflated[k] = dense_v
            else:
                inflated[k] = v
        t_load = time.time() - t0
        
        size = os.path.getsize(custom_sparse_gz_path) / (1024 * 1024)
        results["Custom Mask+1D + Gzip"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(custom_sparse_gz_path):
            os.remove(custom_sparse_gz_path)

    # 8. Custom Bit-Packed Mask + 1D Values
    # We pack the boolean mask into a numpy uint8 array (8 bits per byte)
    bitpacked_sd = {}
    for k, v in state_dict.items():
        if k in prunable_keys:
            if (v == 0.0).any():
                mask = (v != 0.0).cpu().numpy()
                packed_mask = np.packbits(mask.flatten())
                values = v[v != 0.0]
                bitpacked_sd[k] = {
                    "packed_mask": packed_mask,
                    "values": values,
                    "shape": v.shape,
                    "numel": v.numel()
                }
            else:
                bitpacked_sd[k] = v
        else:
            bitpacked_sd[k] = v

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        bitpacked_path = tmp.name
    try:
        t0 = time.time()
        torch.save(bitpacked_sd, bitpacked_path)
        t_save = time.time() - t0
        
        t0 = time.time()
        loaded = torch.load(bitpacked_path, map_location="cpu", weights_only=False)
        # Inflate back to dense
        inflated = {}
        for k, v in loaded.items():
            if isinstance(v, dict) and "packed_mask" in v and "values" in v and "shape" in v:
                unpacked = np.unpackbits(v["packed_mask"])[:v["numel"]]
                mask = torch.from_numpy(unpacked.astype(bool)).view(v["shape"])
                dense_v = torch.zeros(v["shape"], dtype=v["values"].dtype, device=v["values"].device)
                dense_v[mask] = v["values"]
                inflated[k] = dense_v
            else:
                inflated[k] = v
        t_load = time.time() - t0
        
        size = os.path.getsize(bitpacked_path) / (1024 * 1024)
        results["Bit-Packed Mask+1D (Uncompressed)"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(bitpacked_path):
            os.remove(bitpacked_path)

    # 9. Custom Bit-Packed Mask + 1D Values + Zip (Deflate)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        bitpacked_zip_path = tmp.name
    try:
        t0 = time.time()
        with zipfile.ZipFile(bitpacked_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as inner_tmp:
                inner_path = inner_tmp.name
            try:
                torch.save(bitpacked_sd, inner_path)
                z.write(inner_path, arcname="state_dict.pt")
            finally:
                if os.path.exists(inner_path):
                    os.remove(inner_path)
        t_save = time.time() - t0
        
        t0 = time.time()
        with zipfile.ZipFile(bitpacked_zip_path, "r") as z:
            with z.open("state_dict.pt") as f:
                loaded = torch.load(f, map_location="cpu", weights_only=False)
        # Inflate back to dense
        inflated = {}
        for k, v in loaded.items():
            if isinstance(v, dict) and "packed_mask" in v and "values" in v and "shape" in v:
                unpacked = np.unpackbits(v["packed_mask"])[:v["numel"]]
                mask = torch.from_numpy(unpacked.astype(bool)).view(v["shape"])
                dense_v = torch.zeros(v["shape"], dtype=v["values"].dtype, device=v["values"].device)
                dense_v[mask] = v["values"]
                inflated[k] = dense_v
            else:
                inflated[k] = v
        t_load = time.time() - t0
        
        size = os.path.getsize(bitpacked_zip_path) / (1024 * 1024)
        results["Bit-Packed Mask+1D + Zip (Deflate)"] = {"size": size, "save_time": t_save, "load_time": t_load}
    finally:
        if os.path.exists(bitpacked_zip_path):
            os.remove(bitpacked_zip_path)

    # Print results in a neat markdown table
    print("| Method | Size (MB) | Compression Ratio | Save Time (s) | Load Time (s) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for name, data in results.items():
        comp_ratio = results["Dense"]["size"] / data["size"]
        print(f"| {name} | {data['size']:.2f} MB | {comp_ratio:.2f}x | {data['save_time']:.4f}s | {data['load_time']:.4f}s |")
    
    return results

if __name__ == "__main__":
    # Choose 0.7 magnitude pruned model
    chk = "checkpoints/yolov5s/pruned/magnitude_0.7.pt"
    if not os.path.exists(chk):
        # Fallback to check if magnitude_0.3.pt exists
        chk = "checkpoints/yolov5s/pruned/magnitude_0.3.pt"
    
    if os.path.exists(chk):
        analyze_checkpoint(chk)
    else:
        print(f"Checkpoint {chk} not found. Please run the pruning script first.")
