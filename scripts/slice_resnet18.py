# scripts/test_slicer_resnet18.py

import torch
from torchvision import datasets, transforms

from models import get_model
from core.slicer import Slicer


# Load pretrained ResNet-18 model
model = get_model("resnet18", pretrained=True)
model.eval()


# Cifar-10 processing
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]
)

test_data = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)

# Sample image from Cifar-10 test set
image, label = test_data[0]
image = image.unsqueeze(0)  # Shape: [1, 3, 32, 32]


# Execute Slicer
slicer = Slicer(
    model=model,
    profiling_samples=torch.zeros_like(image),  # Zero baseline
    input_sample=image,
)

slicer.execute(target_index=label, theta=0.3)
