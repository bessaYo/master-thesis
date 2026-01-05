import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from core.analysis.profiler import Profiler


# Directory to store profiling results
PROFILE_DIR = "profiles"
os.makedirs(PROFILE_DIR, exist_ok=True)

# CIFAR-10 normalization (standard)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

# Load CIFAR-10 training set for profiling
dataset = datasets.CIFAR10(
    root="data",
    train=True,
    transform=transform,
    download=True,
)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Load pretrained ResNet-18
model = get_model("resnet18", pretrained=True)
model.eval()

print("Collecting CIFAR-10 samples for ResNet-18 profiling...")
all_samples = torch.cat(
    [batch for batch, _ in tqdm(loader)],
    dim=0,
)

print(f"Total samples: {all_samples.shape[0]}")
print("Running profiler...")

# Run profiling
profiler = Profiler(model)
profile = profiler.execute(all_samples)

# Store profiling statistics
torch.save(
    {
        "neuron_means": profile["neuron_means"],
        "channel_means": profile["channel_means"],
        "layer_means": profile["layer_means"],
        "meta": {
            "dataset": "CIFAR-10",
            "model": "ResNet-18",
            "num_samples": all_samples.shape[0],
        },
    },
    os.path.join(PROFILE_DIR, "cifar10_resnet18.pt"),
)

print(f"Profile saved to {PROFILE_DIR}/cifar10_resnet18.pt")