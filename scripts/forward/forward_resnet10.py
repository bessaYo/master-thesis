# scripts/test_forward_deltas.py

import torch
from models import get_model
from core.tracing.forward import ForwardAnalyzer
from torchvision import datasets, transforms

# Load profile
PROFILE_PATH = "profiles/cifar10_resnet10.pt"
profile = torch.load(PROFILE_PATH)

print(f"Loaded profile from {PROFILE_PATH}")
print(f"Dataset: {profile['meta']['dataset']}")
print(f"Model: {profile['meta']['model']}")
print(f"Samples: {profile['meta']['num_samples']}")

# Load model
model = get_model("resnet10", pretrained=True)
model.eval()

# Load a sample from CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

dataset = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
sample, label = dataset[0]

print(f"\nSample label: {label}")

# Run forward analysis
forward_analyzer = ForwardAnalyzer(model, profile)
result = forward_analyzer.execute(sample)

# === Layer Deltas ===
print("\n" + "=" * 70)
print("LAYER DELTAS")
print("=" * 70)
print(f"{'Layer':<30} | {'Delta':>12}")
print("-" * 70)
for layer_name, delta in result["layer_deltas"].items():
    delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
    print(f"{layer_name:<30} | {delta_val:>12.4f}")

# === Channel Deltas (summary) ===
print("\n" + "=" * 70)
print("CHANNEL DELTAS (per Conv layer)")
print("=" * 70)
print(f"{'Layer':<30} | {'Channels':>10} | {'Mean':>10} | {'Max':>10} | {'Min':>10}")
print("-" * 70)
for layer_name, deltas in result["channel_deltas"].items():
    deltas_squeezed = deltas.squeeze()
    n_channels = deltas_squeezed.numel()
    mean_val = deltas_squeezed.mean().item()
    max_val = deltas_squeezed.max().item()
    min_val = deltas_squeezed.min().item()
    print(f"{layer_name:<30} | {n_channels:>10} | {mean_val:>10.4f} | {max_val:>10.4f} | {min_val:>10.4f}")

# === Block Deltas ===
print("\n" + "=" * 70)
print("BLOCK DELTAS")
print("=" * 70)
print(f"{'Block':<30} | {'Delta':>12} | {'Layers'}")
print("-" * 70)
for block_name, delta in result["block_deltas"].items():
    delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
    layers = result["blocks"][block_name]
    print(f"{block_name:<30} | {delta_val:>12.4f} | {len(layers)} layers")

# === Block Details ===
print("\n" + "=" * 70)
print("BLOCK DETAILS")
print("=" * 70)
for block_name, layers in result["blocks"].items():
    block_delta = result["block_deltas"][block_name]
    block_delta_val = block_delta.item() if isinstance(block_delta, torch.Tensor) else block_delta
    print(f"\n{block_name} (delta: {block_delta_val:.4f}):")
    for layer in layers:
        delta = result["layer_deltas"].get(layer)
        if delta is not None:
            delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
            print(f"  {layer:<28} | {delta_val:>12.4f}")

# === Skip Candidates ===
print("\n" + "=" * 70)
print("BLOCK SKIP CANDIDATES (delta < 0.3)")
print("=" * 70)
threshold = 0.3
for block_name, delta in result["block_deltas"].items():
    delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
    if delta_val < threshold:
        print(f"  ✓ {block_name:<28} | delta: {delta_val:.4f} → SKIP CANDIDATE")
    else:
        print(f"    {block_name:<28} | delta: {delta_val:.4f}")