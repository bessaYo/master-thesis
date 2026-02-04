"""
Counterfactual Slicing for ResNet

Computes a few slices with full data for counterfactual evaluation.
1 sample × 3 configs = 3 slices per model

Usage:
    python scripts/slicing/slice_counterfactual.py --model resnet18
    python scripts/slicing/slice_counterfactual.py --model resnet34
"""

import argparse
import torch
from torchvision import datasets, transforms
import time
from pathlib import Path

from models import get_model
from core.slicer import Slicer


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CONFIGS_FULL = [
    {"theta": 0.3, "channel_mode": False, "block_mode": False, "name": "baseline"},
    {"theta": 0.3, "channel_mode": True, "block_mode": False, "channel_alpha": 0.10, "name": "channel"},
    {"theta": 0.3, "channel_mode": True, "block_mode": True, "channel_alpha": 0.10, "block_beta": 0.8, "name": "channel_block"},
]

# For ResNet34: skip baseline (too slow)
CONFIGS_FAST = [
    {"theta": 0.3, "channel_mode": True, "block_mode": False, "channel_alpha": 0.10, "name": "channel"},
    {"theta": 0.3, "channel_mode": True, "block_mode": True, "channel_alpha": 0.10, "block_beta": 0.8, "name": "channel_block"},
]


def run_slice(model, profile, image, label, config):
    """Run slicing and return full results."""
    slicer = Slicer(
        model=model,
        input_sample=image.unsqueeze(0),
        precomputed_profile=profile
    )

    slicer.profile()
    slicer.forward()

    backward_kwargs = {
        "target_index": label,
        "theta": config["theta"],
        "channel_mode": config["channel_mode"],
        "block_mode": config["block_mode"],
        "debug": False,
    }
    if "channel_alpha" in config:
        backward_kwargs["channel_alpha"] = config["channel_alpha"]
    if "block_beta" in config:
        backward_kwargs["block_beta"] = config["block_beta"]

    slicer.backward(**backward_kwargs)

    return slicer.backward_result


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Slicing")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet34"])
    parser.add_argument("--class_id", type=int, default=3, help="Class to slice (default: 3=cat)")
    args = parser.parse_args()

    model_name = args.model
    target_class = args.class_id

    print("=" * 60)
    print(f"COUNTERFACTUAL SLICING - {model_name.upper()}")
    print("=" * 60)

    # Load model
    print(f"\nLoading {model_name}...")
    model = get_model(model_name, pretrained=True)
    model.eval()

    # Load profile
    profile = torch.load(f"profiles/cifar10_{model_name}.pt", map_location="cpu", weights_only=False)

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    # Find first sample of target class
    sample_idx = None
    sample_image = None
    for idx, (img, label) in enumerate(dataset):
        if label == target_class:
            sample_idx = idx
            sample_image = img
            break

    print(f"Sample: idx={sample_idx}, class={target_class} ({CIFAR10_CLASSES[target_class]})")
    
    # Choose configs based on model
    configs = CONFIGS_FAST if model_name == "resnet34" else CONFIGS_FULL
    print(f"Configs: {len(configs)} ({', '.join(c['name'] for c in configs)})")

    # Create output dir
    output_dir = Path(f"results/counterfactual/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run slicing for each config
    for config in configs:
        print(f"\n{'-' * 40}")
        print(f"Config: {config['name']}")
        print(f"{'-' * 40}")

        t0 = time.time()
        result = run_slice(model, profile, sample_image, target_class, config)
        elapsed = time.time() - t0

        # Save full slice data
        output_file = output_dir / f"{config['name']}.pt"
        torch.save({
            "model": model_name,
            "sample_idx": sample_idx,
            "class_id": target_class,
            "class_name": CIFAR10_CLASSES[target_class],
            "config": config,
            "neuron_contributions": result["neuron_contributions"],
            "synapse_contributions": result["synapse_contributions"],
            "backward_time_sec": result["backward_time_sec"],
            "total_synapses": result["total_synapses"],
            "slice_synapses": result["slice_synapses"],
            "total_neurons": result["total_neurons"],
            "slice_neurons": result["slice_neurons"],
            "kept_blocks": result["kept_blocks"],
            "skipped_blocks": result["skipped_blocks"],
        }, output_file)

        reduction = (1 - result["slice_synapses"] / result["total_synapses"]) * 100
        print(f"Time: {elapsed:.2f}s")
        print(f"Synapses: {result['slice_synapses']:,} / {result['total_synapses']:,} ({reduction:.1f}% reduction)")
        print(f"Saved: {output_file}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()