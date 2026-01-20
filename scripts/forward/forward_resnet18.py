# scripts/forward/forward_resnet18.py

import torch
from models import get_model
from core.tracing.forward import ForwardAnalyzer
from torchvision import datasets, transforms

# Load profile
PROFILE_PATH = "profiles/cifar10_resnet18.pt"
profile = torch.load(PROFILE_PATH)

print(f"Loaded profile from {PROFILE_PATH}")
print(f"Dataset: {profile['meta']['dataset']}")
print(f"Model: {profile['meta']['model']}")
print(f"Samples: {profile['meta']['num_samples']}")

# Load model
model = get_model("resnet18", pretrained=True)
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
print(f"{'Block':<30} | {'Delta':>12} | {'Layers':>10} | {'Has Identity Shortcut'}")
print("-" * 70)
for block_name, delta in result["block_deltas"].items():
    delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
    layers = result["blocks"][block_name]
    
    # Check if block has identity shortcut (no shortcut.0 conv layer)
    has_shortcut_conv = any("shortcut.0" in l for l in layers)
    shortcut_type = "Conv" if has_shortcut_conv else "Identity"
    
    print(f"{block_name:<30} | {delta_val:>12.4f} | {len(layers):>10} | {shortcut_type}")

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
            # Highlight shortcut vs main path
            marker = " [shortcut]" if "shortcut" in layer else ""
            print(f"  {layer:<28} | {delta_val:>12.4f}{marker}")

# === Main Path vs Shortcut Analysis ===
print("\n" + "=" * 70)
print("MAIN PATH vs SHORTCUT ANALYSIS")
print("=" * 70)
print(f"{'Block':<20} | {'Main Path':>12} | {'Shortcut':>12} | {'Ratio':>10} | {'Skip?'}")
print("-" * 70)

for block_name, layers in result["blocks"].items():
    main_path_deltas = []
    shortcut_deltas = []
    
    for layer in layers:
        delta = result["layer_deltas"].get(layer)
        if delta is not None:
            delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
            if "shortcut" in layer:
                shortcut_deltas.append(delta_val)
            elif "conv" in layer:
                main_path_deltas.append(delta_val)
    
    main_avg = sum(main_path_deltas) / len(main_path_deltas) if main_path_deltas else 0
    short_avg = sum(shortcut_deltas) / len(shortcut_deltas) if shortcut_deltas else 0
    
    # For identity shortcuts, use the input delta (relu from previous block)
    if not shortcut_deltas:
        short_avg = main_avg * 0.5  # Placeholder - identity shortcut
    
    ratio = main_avg / short_avg if short_avg > 0 else float('inf')
    skip_candidate = "✓ YES" if ratio < 1.5 else "NO"
    
    print(f"{block_name:<20} | {main_avg:>12.4f} | {short_avg:>12.4f} | {ratio:>10.2f} | {skip_candidate}")

# === Skip Candidates Summary ===
print("\n" + "=" * 70)
print("BLOCK SKIP CANDIDATES")
print("=" * 70)
threshold = 0.5
print(f"Threshold: block_delta < {threshold}")
print("-" * 70)

skip_candidates = []
for block_name, delta in result["block_deltas"].items():
    delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
    layers = result["blocks"][block_name]
    has_shortcut_conv = any("shortcut.0" in l for l in layers)
    
    if delta_val < threshold:
        skip_candidates.append(block_name)
        print(f"  ✓ {block_name:<28} | delta: {delta_val:.4f} → SKIP CANDIDATE")
    else:
        print(f"    {block_name:<28} | delta: {delta_val:.4f}")

if skip_candidates:
    print(f"\n→ {len(skip_candidates)} block(s) can be skipped: {skip_candidates}")
else:
    print(f"\n→ No blocks below threshold {threshold}")