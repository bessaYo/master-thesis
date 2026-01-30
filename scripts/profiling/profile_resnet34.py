# scripts/profiling/profile_resnet34.py

import torch
from torchvision import datasets, transforms
from models import get_model
from core.tracing.profiler import Profiler

model = get_model("resnet34", pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    transform=transform,
    download=True,
)

# Use subset for profiling (z.B. 5000 samples)
samples = torch.stack([train_data[i][0] for i in range(5000)])

profiler = Profiler(model)
profile = profiler.execute(samples)

torch.save(profile, "profiles/cifar10_resnet34.pt")
print("Profile saved!")