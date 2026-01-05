# scripts/slicing/slice_lenet.py

import torch
from torchvision import datasets, transforms
from models import get_model
from core.slicer import Slicer


# Load pretrained LeNet model
model = get_model("lenet", pretrained=True)
model.eval()

# Load precomputed profile
profile = torch.load("profiles/lenet_mnist.pt")

# Load an MNIST sample
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST("data", train=False, transform=transform)

# Take first sample
image, label = test_data[0]
image = image.unsqueeze(0)  # [1, 1, 28, 28]

# Test without channel filter
print("=== Without Channel Filter ===")
slicer = Slicer(
    model,
    input_sample=image,
    precomputed_profile=profile,
)
slicer.profile()
slicer.forward()
slicer.backward(target_index=label, theta=0.3, filter_channels=False)
time_baseline = slicer.backward_result["backward_time_sec"]

# Test with channel filter
print("\n=== With Channel Filter (top 20%) ===")
slicer2 = Slicer(
    model,
    input_sample=image,
    precomputed_profile=profile,
)
slicer2.profile()
slicer2.forward()
slicer2.backward(target_index=label, theta=0.3, filter_channels=True, filter_percent=0.2)
time_filtered = slicer2.backward_result["backward_time_sec"]

# === Comparison ===
print(f"\n=== Comparison ===")
print(f"Baseline:  {time_baseline:.3f}s")
print(f"Filtered:  {time_filtered:.3f}s")
print(f"Speedup:   {time_baseline / time_filtered:.2f}x")