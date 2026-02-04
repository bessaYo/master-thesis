import argparse
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from core.tracing.profiler import Profiler


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    choices=["resnet18", "resnet34", "resnet50"],
    help="ResNet variant to profile"
)
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

MODEL_NAME = args.model
BATCH_SIZE = args.batch_size

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cpu")
print("Device:", device)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,   # 🔥 bewusst KEIN multiprocessing
)

print("=" * 70)
print(f"PROFILING {MODEL_NAME.upper()} ON CIFAR-10")
print("=" * 70)
print(f"Dataset size: {len(dataset)}")
# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
model = get_model(MODEL_NAME, pretrained=True)
model.eval()

# ------------------------------------------------------------
# Collect samples
# ------------------------------------------------------------
all_samples = []

for batch, _ in tqdm(loader, desc="Collecting samples"):
    all_samples.append(batch)

all_samples = torch.cat(all_samples, dim=0)
print(f"Collected samples: {all_samples.shape[0]}")

# ------------------------------------------------------------
# Run profiler
# ------------------------------------------------------------
print("\nRunning profiler...")
profiler = Profiler(model)
profile = profiler.execute(all_samples)

# ------------------------------------------------------------
# Save profile
# ------------------------------------------------------------
os.makedirs("profiles", exist_ok=True)

out_path = f"profiles/cifar10_{MODEL_NAME}.pt"
torch.save(
    {
        "neuron_means": profile["neuron_means"],
        "channel_means": profile["channel_means"],
        "layer_means": profile["layer_means"],
        "block_means": profile["block_means"],
        "blocks": profile["blocks"],
        "meta": {
            "dataset": "CIFAR-10",
            "model": MODEL_NAME,
            "num_samples": all_samples.shape[0],
        },
    },
    out_path,
)

print("\nProfile saved to:", out_path)
print("Done.")