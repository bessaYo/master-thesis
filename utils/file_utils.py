import os
import json
import uuid


def save_result(
    data_dict,
    model_name="unknown",
    out_dir="results",
    prefix="result",
    key_prefix="item",
):
    os.makedirs(out_dir, exist_ok=True)

    safe_name = model_name.replace("/", "_").lower()
    uid = str(uuid.uuid4())[:4]
    filename = f"{prefix}_{safe_name}_{uid}.json"

    path = os.path.join(out_dir, filename)

    # Convert tensor-like structures to floats
    json_ready = {
        layer: {f"{key_prefix}_{i}": float(val) for i, val in enumerate(tensor)}
        for layer, tensor in data_dict.items()
    }

    with open(path, "w") as f:
        json.dump(json_ready, f, indent=2)

    print(f"[SAVE] '{prefix}' results for '{model_name}' saved to: {path}")
    return path
