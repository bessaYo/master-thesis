# scripts/slicing/slice_resnet18.py

import torch
from torchvision import datasets, transforms

from models import get_model
from core.slicer import Slicer


# Load pretrained ResNet-18 model
model = get_model("resnet18", pretrained=True)
model.eval()

# Load precomputed profile
profile = torch.load("profiles/cifar10_resnet18.pt")

# CIFAR-10 preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    transform=transform,
    download=True,
)

# Select one test sample
image, label = test_data[0]
image = image.unsqueeze(0)  # [1, 3, 32, 32]

print(f"Sample label: {label}")
print("=" * 60)

# === Baseline ===
print("\n=== Baseline ===")
slicer1 = Slicer(
    model=model,
    input_sample=image,
    precomputed_profile=profile,
)
slicer1.profile()
slicer1.forward()
slicer1.backward(
    target_index=label,
    theta=0.3,
    channel_mode=False,
    block_mode=False,
    debug=True,
)
baseline_time = slicer1.backward_result["backward_time_sec"]

# === With Channel Filter ===
print("\n=== With Channel Filter (top 20%) ===")
slicer2 = Slicer(
    model=model,
    input_sample=image,
    precomputed_profile=profile,
)
slicer2.profile()
slicer2.forward()
slicer2.backward(
    target_index=label,
    theta=0.3,
    channel_mode=True,
    block_mode=False,
    filter_percent=0.2,
    debug=True,
)
channel_time = slicer2.backward_result["backward_time_sec"]

# === With Channel Filter + Block Skip ===
print("\n=== With Channel Filter + Block Skip ===")
slicer3 = Slicer(
    model=model,
    input_sample=image,
    precomputed_profile=profile,
)
slicer3.profile()
forward_result = slicer3.forward()
print(f"[DEBUG] blocks: {forward_result['blocks']}")
print(f"[DEBUG] block_deltas: {forward_result['block_deltas']}")
slicer3.backward(
    target_index=label,
    theta=0.3,
    channel_mode=True,
    block_mode=True,
    filter_percent=0.2,
    block_threshold=0.5,
    debug=True,
)
combined_time = slicer3.backward_result["backward_time_sec"]

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"{'Method':<30} | {'Time':>10} | {'Speedup':>10}")
print("-" * 60)
print(f"{'Baseline':<30} | {baseline_time:>10.3f}s | {1.0:>10.2f}x")
print(f"{'Channel Filter':<30} | {channel_time:>10.3f}s | {baseline_time/channel_time:>10.2f}x")
print(f"{'Channel + Block Skip':<30} | {combined_time:>10.3f}s | {baseline_time/combined_time:>10.2f}x")