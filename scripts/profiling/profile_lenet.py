import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_model
from core.tracing.profiler import Profiler
from tqdm import tqdm
import os

DATA_ROOT = "data"
PROFILE_DIR = "profiles"
BATCH_SIZE = 128
NUM_WORKERS = 0
DEVICE = "cpu"

os.makedirs(PROFILE_DIR, exist_ok=True)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,     
    transform=transform,
    download=True
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# Load pretrained LeNet model
model = get_model("lenet", pretrained=True)
model.eval()
model.to(DEVICE)

print("Collecting MNIST samples for profiling...")

all_batches = []

with torch.no_grad():
    for batch, _ in tqdm(loader):
        all_batches.append(batch.to(DEVICE))

# [N, 1, 28, 28]
all_samples = torch.cat(all_batches, dim=0)

print(f"Total samples collected: {all_samples.shape[0]}")


print("Running profiler...")

profiler = Profiler(model)
profile = profiler.execute(all_samples)

profile_path = os.path.join(PROFILE_DIR, "lenet_mnist.pt")

torch.save(
    {
        "neuron_means": profile["neuron_means"],
        "channel_means": profile["channel_means"],
        "layer_means": profile["layer_means"],
        "meta": {
            "dataset": "MNIST",
            "split": "train",
            "model": "LeNet",
            "num_samples": all_samples.shape[0],
            "normalization": "mean=0.1307 std=0.3081",
        }
    },
    profile_path
)

print(f"\nProfile successfully saved to:\n{profile_path}")

print("\n=== Layer means (sanity check) ===")
for layer, value in profile["layer_means"].items():
    print(f"{layer:30s} : {float(value):.6f}")

print("\nProfiling finished successfully.")