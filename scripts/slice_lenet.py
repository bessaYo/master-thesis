# scripts/test_slicer_lenet.py

import torch
from torchvision import datasets, transforms
from models import get_model
from core.slicer import Slicer


# Load pretrained LeNet model
model = get_model("lenet", pretrained=True)
model.eval()

# Load an MNIST sample
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST("data", train=False, transform=transform)

# Take first sample
image, label = test_data[0]
image = image.unsqueeze(0)  # [1, 1, 28, 28]

# Execute Slicer
slicer = Slicer(
    model,
    profiling_samples=torch.zeros_like(image),
    input_sample=image
)

slicer.execute(target_index=label, theta=0.3)

