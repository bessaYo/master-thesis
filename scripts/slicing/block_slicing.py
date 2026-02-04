"""
Generic slicing script for ResNet models on CIFAR-10
Channel + Block slicing (combined)

Design decisions:
- Fixed channel_alpha = 0.10 (best trade-off from prior experiments)
- Vary block_beta to analyze structural pruning
- Few samples (fast, sufficient for comparison)
- Parallel execution with multiprocessing

Usage:
------
python scripts/slicing/run_slicing_block_channel.py --model resnet18
python scripts/slicing/run_slicing_block_channel.py --model resnet34 --workers 4
python scripts/slicing/run_slicing_block_channel.py --model resnet50 --workers 6
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


# ---------------------------------------------------------------------
# CIFAR-10 classes
# ---------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ---------------------------------------------------------------------
# Slice configurations (Channel + Block)
# ---------------------------------------------------------------------

CONFIGS = [
    # {
    #     "name": "channel_0.10_block_0.90",
    #     "theta": 0.3,
    #     "channel_mode": True,
    #     "block_mode": True,
    #     "channel_alpha": 0.10,
    #     "block_beta": 0.90,
    # },
    #  {
    #     "name": "channel_0.10_block_0.80",
    #     "theta": 0.3,
    #     "channel_mode": True,
    #     "block_mode": True,
    #     "channel_alpha": 0.10,
    #     "block_beta": 0.80,
    # },
    {
        "name": "channel_0.10_block_0.70",
        "theta": 0.3,
        "channel_mode": True,
        "block_mode": True,
        "channel_alpha": 0.10,
        "block_beta": 0.70,
    },
]

# Small number of samples (fast, sufficient)
NUM_SAMPLES = 5


# ---------------------------------------------------------------------
# Worker function (parallel execution)
# ---------------------------------------------------------------------

def run_slice(args):
    sample_idx, image, label, config, model_name, profile_path, debug = args

    # Load model & profile inside worker (safe for multiprocessing)
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
    if "block_beta" in config:
        backward_kwargs["block_beta"] = config["block_beta"]

    slicer.backward(**backward_kwargs)
    br = slicer.backward_result

    return {
        "sample_idx": sample_idx,
        "class_id": label,
        "class_name": CIFAR10_CLASSES[label],
        "model": model_name,
        "config_name": config["name"],

        "backward_time_sec": round(br["backward_time_sec"], 4),
        "total_blocks": br["total_blocks"],
        "skipped_blocks": br["skipped_blocks"],

        "total_neurons": br["total_neurons"],
        "slice_neurons": br["slice_neurons"],

        "total_synapses": br["total_synapses"],
        "slice_synapses": br["slice_synapses"],

        "config": br["config"],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    n_workers = min(args.workers, cpu_count())

    print("\n" + "=" * 72)
    print(f"MODEL:              {model_name}")
    print(f"SAMPLES:            {NUM_SAMPLES}")
    print(f"CONFIGS:            {len(CONFIGS)}")
    print(f"TOTAL SLICES:       {NUM_SAMPLES * len(CONFIGS)}")
    print(f"WORKERS:            {n_workers}")
    print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # Take first NUM_SAMPLES samples (deterministic)
    samples = []
    for idx, (img, label) in enumerate(dataset):
        samples.append((idx, img, label))
        if len(samples) >= NUM_SAMPLES:
            break

    # ------------------------------------------------------------------
    # Build task list
    # ------------------------------------------------------------------

    profile_path = f"profiles/cifar10_{model_name}.pt"
    tasks = []
    for sample_idx, img, label in samples:
        for config in CONFIGS:
            tasks.append((
                sample_idx,
                img,
                label,
                config,
                model_name,
                profile_path,
                args.debug,
            ))

    # ------------------------------------------------------------------
    # Run slicing (parallel)
    # ------------------------------------------------------------------

    t0 = time.time()
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(run_slice, tasks),
                total=len(tasks),
                desc="Slicing",
            )
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(tasks):.2f}s / slice)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    out_dir = Path("evaluation/results/slices")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{model_name}_channel_block_slices.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()