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

# === Baseline (no channel filtering) ===
print("\n=== Baseline (without Channel Filter) ===")
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
    filter_channels=False,
    debug=True,
)

baseline_time = slicer.backward_result["backward_time_sec"]

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
    filter_channels=True,
    filter_percent=0.2,
    debug=True,
)

filtered_time = slicer2.backward_result["backward_time_sec"]

# === Comparison ===
print(f"\n=== Comparison ===")
print(f"Baseline:  {baseline_time:.3f}s")
print(f"Filtered:  {filtered_time:.3f}s")
print(f"Speedup:   {baseline_time / filtered_time:.2f}x")