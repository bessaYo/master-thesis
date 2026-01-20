# scripts/profile/profile_resnet18_cifar.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from core.tracing.profiler import Profiler


# Directory to store profiling results
PROFILE_DIR = "profiles"
os.makedirs(PROFILE_DIR, exist_ok=True)

# CIFAR-10 normalization (standard)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

# Load CIFAR-10 training set for profiling
dataset = datasets.CIFAR10(
    root="data",
    train=True,
    transform=transform,
    download=True,
)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Load pretrained ResNet-18
model = get_model("resnet18", pretrained=True)
model.eval()

print("Collecting CIFAR-10 samples for ResNet-18 profiling...")
all_samples = torch.cat(
    [batch for batch, _ in tqdm(loader)],
    dim=0,
)

print(f"Total samples: {all_samples.shape[0]}")
print("Running profiler...")

# Run profiling
profiler = Profiler(model)
profile = profiler.execute(all_samples)

# Print layer means
print("\n" + "=" * 60)
print("LAYER MEANS")
print("=" * 60)
print(f"{'Layer':<30} | {'Mean':>12} | {'Abs Mean':>12}")
print("-" * 60)
for layer_name, mean in profile["layer_means"].items():
    mean_val = mean.item() if isinstance(mean, torch.Tensor) else mean
    abs_mean = abs(mean_val)
    print(f"{layer_name:<30} | {mean_val:>12.4f} | {abs_mean:>12.4f}")

# Print block means
print("\n" + "=" * 60)
print("BLOCK MEANS")
print("=" * 60)
print(f"{'Block':<30} | {'Mean':>12} | {'Layers'}")
print("-" * 60)
for block_name, mean in profile["block_means"].items():
    mean_val = mean.item() if isinstance(mean, torch.Tensor) else mean
    layers = profile["blocks"][block_name]
    print(f"{block_name:<30} | {mean_val:>12.4f} | {len(layers)} layers")

# Print detailed block information
print("\n" + "=" * 60)
print("BLOCK DETAILS")
print("=" * 60)
for block_name, layers in profile["blocks"].items():
    print(f"\n{block_name}:")
    for layer in layers:
        mean = profile["layer_means"].get(layer)
        if mean is not None:
            mean_val = mean.item() if isinstance(mean, torch.Tensor) else mean
            print(f"  {layer:<28} | {mean_val:>12.4f}")

# Save profile data
torch.save(
    {
        "neuron_means": profile["neuron_means"],
        "channel_means": profile["channel_means"],
        "layer_means": profile["layer_means"],
        "block_means": profile["block_means"],
        "blocks": profile["blocks"],
        "meta": {
            "dataset": "CIFAR-10",
            "model": "ResNet-18",
            "num_samples": all_samples.shape[0],
        },
    },
    os.path.join(PROFILE_DIR, "cifar10_resnet18.pt"),
)

print(f"\nProfile saved to {PROFILE_DIR}/cifar10_resnet18.pt")