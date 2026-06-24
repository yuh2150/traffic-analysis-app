import json
import os
import sys

notebooks_dir = "d:/Project/traffic-analysis-app/notebooks"
prefixes = ["channel_pruning", "filter_pruning", "layer_pruning", "magnitude_pruning", "l1_norm_pruning"]

errors = 0
for prefix in prefixes:
    filename = f"{prefix}.ipynb"
    path = os.path.join(notebooks_dir, filename)
    print(f"Verifying {path}...")
    
    if not os.path.exists(path):
        print(f"  Error: File does not exist!")
        errors += 1
        continue
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        # Check basic notebook structure
        if "cells" not in nb:
            print("  Error: Missing 'cells' list!")
            errors += 1
            continue
            
        if "metadata" not in nb:
            print("  Error: Missing 'metadata'!")
            errors += 1
            continue
            
        if nb.get("nbformat") != 4:
            print(f"  Warning: Unexpected nbformat {nb.get('nbformat')} (expected 4)")
            
        cells_count = len(nb["cells"])
        print(f"  Success: Valid Jupyter Notebook with {cells_count} cells.")
        
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON format! Details: {e}")
        errors += 1

if errors > 0:
    print(f"\nVerification FAILED with {errors} errors.")
    sys.exit(1)
else:
    print("\nAll notebooks verified successfully!")
    sys.exit(0)
