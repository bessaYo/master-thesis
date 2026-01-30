# scripts/eval/test_interpretability.py

import torch
from torchvision import datasets, transforms

from models import get_model
from core.slicer import Slicer


model = get_model("resnet18", pretrained=True)
model.eval()
profile = torch.load("profiles/cifar10_resnet18.pt")

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

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def get_first_sample_by_class(dataset, class_idx):
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == class_idx:
            return i
    return None


def compute_slice(model, profile, image, label):
    slicer = Slicer(
        model=model,
        input_sample=image,
        precomputed_profile=profile,
    )
    slicer.profile()
    slicer.forward()
    slicer.backward(
        target_index=label,
        theta=0.3,
        channel_mode=True,
        block_mode=True,
        filter_percent=0.05,
        block_threshold=0.5,
        debug=False,
    )
    return slicer.backward_result["neuron_contributions"]


def get_active_channels_per_layer(contributions):
    """Return dict: layer -> set of active channel indices."""
    result = {}
    for layer, contrib in contributions.items():
        if layer == "input":
            continue
        if contrib.dim() == 4:  # Conv: [1, C, H, W]
            active = set()
            for c in range(contrib.shape[1]):
                if (contrib[0, c] != 0).any():
                    active.add(c)
            result[layer] = active
    return result


def jaccard(set_a, set_b):
    if len(set_a | set_b) == 0:
        return 0
    return len(set_a & set_b) / len(set_a | set_b)


# === MAIN ===

ANALYZE_CLASSES = [0, 3, 5, 6, 8]  # airplane, cat, dog, frog, ship

print("=" * 70)
print("CHANNEL-LEVEL INTERPRETABILITY ANALYSIS")
print("=" * 70)

# Compute slices
active_channels = {}

for class_idx in ANALYZE_CLASSES:
    print(f">>> Computing slice for: {CLASSES[class_idx]}")
    
    idx = get_first_sample_by_class(test_data, class_idx)
    image, label = test_data[idx]
    
    contributions = compute_slice(model, profile, image.unsqueeze(0), label)
    active_channels[class_idx] = get_active_channels_per_layer(contributions)

# === ANALYZE SPECIFIC LAYERS ===
# Frühe Layer = shared features, Späte Layer = class-specific

analyze_layers = [
    "conv1",           # Sehr früh
    "layer1.2.conv2",  # Früh
    "layer2.3.conv2",  # Mitte
    "layer3.5.conv2",  # Spät
    "layer4.2.conv2",  # Sehr spät
]

for layer in analyze_layers:
    # Check if layer exists for all classes
    if not all(layer in active_channels[c] for c in ANALYZE_CLASSES):
        continue
    
    print(f"\n{'='*70}")
    print(f"LAYER: {layer}")
    print(f"{'='*70}")
    
    # Print channel counts
    print("\nActive channels per class:")
    for class_idx in ANALYZE_CLASSES:
        n_active = len(active_channels[class_idx][layer])
        print(f"  {CLASSES[class_idx]:>10}: {n_active} channels")
    
    # Jaccard matrix for this layer
    print(f"\nJaccard similarity:")
    
    header = f"{'':>10} |"
    for c in ANALYZE_CLASSES:
        header += f" {CLASSES[c]:>8} |"
    print(header)
    print("-" * len(header))
    
    for class_a in ANALYZE_CLASSES:
        row = f"{CLASSES[class_a]:>10} |"
        for class_b in ANALYZE_CLASSES:
            sim = jaccard(
                active_channels[class_a][layer],
                active_channels[class_b][layer]
            )
            row += f" {sim*100:>7.1f}% |"
        print(row)

# === UNIQUE CHANNELS ===
print(f"\n{'='*70}")
print("UNIQUE CHANNELS (only active for one class)")
print(f"{'='*70}")

for layer in ["layer3.5.conv2", "layer4.2.conv2"]:
    if not all(layer in active_channels[c] for c in ANALYZE_CLASSES):
        continue
    
    print(f"\n{layer}:")
    
    for class_idx in ANALYZE_CLASSES:
        my_channels = active_channels[class_idx][layer]
        other_channels = set()
        for other_idx in ANALYZE_CLASSES:
            if other_idx != class_idx:
                other_channels |= active_channels[other_idx][layer]
        
        unique = my_channels - other_channels
        print(f"  {CLASSES[class_idx]:>10}: {len(unique)} unique channels")

# === SHARED VS UNIQUE ===
print(f"\n{'='*70}")
print("SHARED VS CLASS-SPECIFIC CHANNELS")
print(f"{'='*70}")

for layer in ["layer2.3.conv2", "layer4.2.conv2"]:
    if not all(layer in active_channels[c] for c in ANALYZE_CLASSES):
        continue
    
    all_channels = set()
    shared_channels = None
    
    for class_idx in ANALYZE_CLASSES:
        channels = active_channels[class_idx][layer]
        all_channels |= channels
        if shared_channels is None:
            shared_channels = channels.copy()
        else:
            shared_channels &= channels
    
    print(f"\n{layer}:")
    print(f"  Total unique channels used: {len(all_channels)}")
    print(f"  Shared by ALL classes:      {len(shared_channels)}")
    print(f"  Class-specific:             {len(all_channels) - len(shared_channels)}")