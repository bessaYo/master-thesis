# scripts/resnet10/profile_resnet10_cifar.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_model
from core.analysis.profiler import Profiler
from tqdm import tqdm
import os

PROFILE_DIR = "profiles"
os.makedirs(PROFILE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

model = get_model("resnet10", pretrained=True)
model.eval()

print("Collecting CIFAR-10 samples...")
all_samples = torch.cat([batch for batch, _ in tqdm(loader)], dim=0)

print(f"Total samples: {all_samples.shape[0]}")
print("Running profiler...")

profiler = Profiler(model)
profile = profiler.execute(all_samples)

torch.save({
    "neuron_means": profile["neuron_means"],
    "channel_means": profile["channel_means"],
    "layer_means": profile["layer_means"],
    "meta": {
        "dataset": "CIFAR-10",
        "model": "ResNet-10",
        "num_samples": all_samples.shape[0],
    }
}, os.path.join(PROFILE_DIR, "cifar10_resnet10.pt"))

print(f"Profile saved to {PROFILE_DIR}/cifar10_resnet10.pt")