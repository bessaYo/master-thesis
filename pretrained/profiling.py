import argparse
import os
import torch
from tqdm import tqdm

from models import get_model
from core.tracing.profiler import Profiler
from utils.data import load_cifar10, load_imagenet_val


MODELS = {
    "resnet_cifar": ("cifar10", "pretrained/profiles/cifar10_resnet_cifar.pt"),
    "resnet18": ("imagenet", "pretrained/profiles/imagenet_resnet18.pt"),
    "resnet34": ("imagenet", "pretrained/profiles/imagenet_resnet34.pt"),
    "resnet50": ("imagenet", "pretrained/profiles/imagenet_resnet50.pt"),
    "resnet101": ("imagenet", "pretrained/profiles/imagenet_resnet101.pt"),
}


def main():
    parser = argparse.ArgumentParser(description="Precompute activation profiles")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to profile",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, out_path = MODELS[args.model]

    print(f"Loading {args.model} ({dataset_name})...")
    model = get_model(args.model, pretrained=True).to(device).eval()

    if dataset_name == "cifar10":
        dataset = load_cifar10(train=True)
    elif dataset_name == "imagenet":
        dataset = load_imagenet_val()

    # Load all samples into a single tensor
    print(f"Loading {len(dataset)} samples...")
    samples = torch.stack([x for x, _ in tqdm(dataset, desc="Preparing samples")]).to(device)

    print(f"Profiling {args.model} on {dataset_name} ({len(samples)} samples)...")
    profiler = Profiler(model)
    profile = profiler.execute(samples)

    # Save profile
    save_dict = {
        "neuron_means": profile["neuron_means"],
        "channel_means": profile["channel_means"],
        "layer_means": profile["layer_means"],
    }
    if "block_means" in profile:
        save_dict["block_means"] = profile["block_means"]
    if "blocks" in profile:
        save_dict["blocks"] = profile["blocks"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(save_dict, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
