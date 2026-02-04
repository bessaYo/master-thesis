"""
Generic slicing script for ResNet models on CIFAR-10

Usage:
------
python scripts/slicing/run_slicing.py --model resnet18
python scripts/slicing/run_slicing.py --model resnet34 --workers 4
"""

import torch
from torchvision import datasets, transforms
import json
import time
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

from models import get_model
from core.slicer import Slicer


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CONFIGS = [
    {"name": "baseline",     "theta": 0.3, "channel_mode": False, "block_mode": False},
    {"name": "channel_0.50", "theta": 0.3, "channel_mode": True,  "block_mode": False, "channel_alpha": 0.50},
    {"name": "channel_0.20", "theta": 0.3, "channel_mode": True,  "block_mode": False, "channel_alpha": 0.20},
    {"name": "channel_0.10", "theta": 0.3, "channel_mode": True,  "block_mode": False, "channel_alpha": 0.10},
]

NUM_SAMPLES = 5


def run_slice(args):
    """Worker function for parallel execution."""
    sample_idx, image, label, config, model_name, profile_path, debug = args
    
    # Load model and profile in each worker
    model = get_model(model_name, pretrained=True)
    model.eval()
    profile = torch.load(profile_path, map_location="cpu", weights_only=False)

    slicer = Slicer(
        model=model,
        input_sample=image.unsqueeze(0),
        precomputed_profile=profile,
        debug=debug,
    )

    slicer.profile()
    slicer.forward()

    backward_kwargs = {
        "target_index": label,
        "theta": config["theta"],
        "channel_mode": config["channel_mode"],
        "block_mode": config["block_mode"],
    }

    if "channel_alpha" in config:
        backward_kwargs["channel_alpha"] = config["channel_alpha"]

    slicer.backward(**backward_kwargs)
    br = slicer.backward_result

    return {
        "sample_idx": sample_idx,
        "class_id": label,
        "class_name": CIFAR10_CLASSES[label],
        "model": model_name,
        "config_name": config["name"],
        "backward_time_sec": round(br["backward_time_sec"], 4),
        "total_neurons": br["total_neurons"],
        "slice_neurons": br["slice_neurons"],
        "total_synapses": br["total_synapses"],
        "slice_synapses": br["slice_synapses"],
        "config": br["config"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    n_workers = min(args.workers, cpu_count())

    print("\n" + "=" * 70)
    print(f"MODEL:              {model_name}")
    print(f"SAMPLES:            {NUM_SAMPLES}")
    print(f"CONFIGS:            {len(CONFIGS)}")
    print(f"TOTAL SLICES:       {NUM_SAMPLES * len(CONFIGS)}")
    print(f"WORKERS:            {n_workers}")
    print("=" * 70 + "\n")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    # Collect samples
    samples = []
    for idx, (img, label) in enumerate(dataset):
        samples.append((idx, img, label))
        if len(samples) >= NUM_SAMPLES:
            break

    # Build task list
    profile_path = f"profiles/cifar10_{model_name}.pt"
    tasks = []
    for sample_idx, img, label in samples:
        for config in CONFIGS:
            tasks.append((sample_idx, img, label, config, model_name, profile_path, args.debug))

    # Run parallel
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(run_slice, tasks), total=len(tasks), desc="Slicing"))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(tasks):.2f}s/slice)")

    # Save
    out_dir = Path("evaluation/results/slices")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_slices.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()