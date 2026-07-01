import torch
from models.factory import ModelFactory

print("=== Checking Raw Checkpoint Keys ===")
checkpoint_path = "checkpoints/yolov5s/pruned/magnitude_0.4.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Get the raw inflated state dict
from utils.sparsity import state_dict_to_dense
raw_sd = state_dict_to_dense(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)

model = ModelFactory.load("yolov5s", num_classes=80)
model_keys = list(model.state_dict().keys())
raw_keys = list(raw_sd.keys())

print(f"Model keys (first 3): {model_keys[:3]}")
print(f"Raw checkpoint keys (first 3): {raw_keys[:3]}")
matched_raw = set(model_keys).intersection(raw_keys)
print(f"Number of matched keys with raw checkpoint: {len(matched_raw)} / {len(model_keys)}")
