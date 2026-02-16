# pretrained/profile.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import get_model
from core.tracing.profiler import Profiler

# --- Config ---
MODEL_NAME = "resnet-cifar"
BATCH_SIZE = 128

# Maps model name to dataset and output path
MODELS = {
    "resnet-cifar": ("cifar10", "pretrained/profiles/cifar10_resnet-cifar.pt"),
    "resnet18": ("imagenet", "pretrained/profiles/imagenet_resnet18.pt"),
    "resnet34": ("imagenet", "pretrained/profiles/imagenet_resnet34.pt"),
    "resnet50": ("imagenet", "pretrained/profiles/imagenet_resnet50.pt"),
    "resnet101": ("imagenet", "pretrained/profiles/imagenet_resnet101.pt"),
}


def build_loader(dataset_name, batch_size):
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    elif dataset_name == "imagenet":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        dataset = datasets.ImageFolder(root="data/imagenet/val", transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, out_path = MODELS[MODEL_NAME]

    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    loader = build_loader(dataset_name, BATCH_SIZE)
    print(f"Profiling {MODEL_NAME} on {dataset_name}")

    profiler = Profiler(model)
    profile = profiler.execute_loader(loader)

    # Save profile
    save_dict = {
        "neuron_means": profile["neuron_means"],
        "channel_means": profile["channel_means"],
        "layer_means": profile["layer_means"],
    }
    if "block_means" in profile:
        save_dict["block_means"] = profile["block_means"]
    if "blocks" in profile:
        save_dict["blocks"] = profile["blocks"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(save_dict, out_path)
    print(f"Saved to {out_path}")
