import os
import json
import uuid
import torch


def save_profiler_results(layer_means, model_name="unknown", out_dir="results/profiling"):
    os.makedirs(out_dir, exist_ok=True)

    safe_name = model_name.replace("/", "_").lower()
    filename = f"profiling_{safe_name}.json"

    path = os.path.join(out_dir, filename)

    # convert tensors to plain floats before saving
    data = {
        layer: {f"feature_{i}": val.item() for i, val in enumerate(layer_mean_tensor)}
        for layer, layer_mean_tensor in layer_means.items()
    }

    # write JSON file
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SAVE] Profiling results for '{model_name}' saved to: {path}\n")
    return path


def save_forward_results(deviations, model_name="unknown", out_dir="results/forward_analysis"):
    os.makedirs(out_dir, exist_ok=True)

    uid = str(uuid.uuid4())[:4]
    safe_name = model_name.replace("/", "_").lower()
    filename = f"{safe_name}_{uid}.json"
    path = os.path.join(out_dir, filename)

    # convert tensors to plain floats before saving
    data = {
        layer: {f"neuron_{i}": val.item() for i, val in enumerate(delta_tensor)}
        for layer, delta_tensor in deviations.items()
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SAVE] Forward analyzer results for '{model_name}' saved to: {path}\n")
    return path


def load_profiler_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profiling file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # convert lists back to torch tensors
    profiling_means = {
        layer: torch.tensor([v for v in layer_vals.values()])
        for layer, layer_vals in data.items()
    }

    print(f"[LOAD] Loaded profiling means for {len(profiling_means)} layers from: {path}")
    return profiling_means