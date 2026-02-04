"""Shared utilities for evaluation scripts.

Provides data loading, slicing, aggregation, pruning, and evaluation
functions used across benchmark and evaluation scripts.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm
import copy
from multiprocessing import Pool
from functools import partial

from models import get_model
from core.slicer import Slicer

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# =============================================================================
# Data
# =============================================================================

def load_cifar10(train: bool = False):
    return datasets.CIFAR10(root='data', train=train, transform=CIFAR10_TRANSFORM, download=True)


def get_samples_for_classes(dataset, classes: List[int], per_class: int) -> List[tuple]:
    """Get samples for specified classes."""
    samples = defaultdict(list)
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label in classes and len(samples[label]) < per_class:
            samples[label].append((img, label, idx))
        if all(len(samples[c]) >= per_class for c in classes):
            break
    return [s for c in classes for s in samples[c]]


# =============================================================================
# Slicing
# =============================================================================

def compute_single_slice(sample_data, model_name, profile_path, theta,
                         channel_mode, channel_alpha, block_mode, block_beta):
    """Compute slice for a single image (multiprocessing worker)."""
    image, label, idx = sample_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(model_name, pretrained=True).to(device).eval()
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    slicer = Slicer(model=model, input_sample=image.unsqueeze(0).to(device),
                    precomputed_profile=profile, debug=False)
    slicer.profile()
    slicer.forward()
    slicer.backward(target_index=label, theta=theta,
                    channel_mode=channel_mode, channel_alpha=channel_alpha,
                    block_mode=block_mode, block_beta=block_beta)

    return {k: v.cpu() for k, v in slicer.backward_result["neuron_contributions"].items()}


def compute_single_slice_timed(sample_data, model_name, profile_path, theta,
                               channel_mode, channel_alpha, block_mode, block_beta):
    """Like compute_single_slice but returns backward time instead of contributions."""
    image, label, _ = sample_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(model_name, pretrained=True).to(device).eval()
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    slicer = Slicer(model=model, input_sample=image.unsqueeze(0).to(device),
                    precomputed_profile=profile)
    slicer.profile()
    slicer.forward()
    slicer.backward(target_index=label, theta=theta,
                    channel_mode=channel_mode, channel_alpha=channel_alpha,
                    block_mode=block_mode, block_beta=block_beta)

    return slicer.backward_result["backward_time_sec"]


def compute_slices(samples, model_name, profile_path, theta=0.3,
                   channel_mode=False, channel_alpha=0.8,
                   block_mode=False, block_beta=0.7,
                   num_workers=4, desc="Slicing"):
    """Compute slices in parallel."""
    worker_fn = partial(compute_single_slice, model_name=model_name,
                        profile_path=profile_path, theta=theta,
                        channel_mode=channel_mode, channel_alpha=channel_alpha,
                        block_mode=block_mode, block_beta=block_beta)

    with Pool(num_workers) as pool:
        slices = list(tqdm(pool.imap(worker_fn, samples), total=len(samples),
                          desc=f"  {desc}", leave=False))
    return slices


def aggregate_slices(slices: List[Dict]) -> Dict[str, torch.Tensor]:
    """Aggregate slices via union: sum of absolute contributions."""
    aggregated = {}
    for key in slices[0]:
        stacked = torch.stack([s[key].float() for s in slices])
        aggregated[key] = stacked.abs().sum(dim=0)
    return aggregated


def compute_slice_size(aggregated: Dict[str, torch.Tensor]) -> float:
    """Fraction of active channels in aggregated slice."""
    total_channels = 0
    active_channels = 0
    for key, tensor in aggregated.items():
        if tensor.dim() == 4:
            channel_contrib = tensor.abs().sum(dim=(2, 3)).squeeze(0)
            total_channels += channel_contrib.numel()
            active_channels += (channel_contrib > 0).sum().item()
    return active_channels / total_channels if total_channels > 0 else 0


# =============================================================================
# Pruning & Evaluation
# =============================================================================

def prune_model(model, contributions: Dict[str, torch.Tensor], device):
    """Prune conv channels with zero aggregated contribution.
    Never touches the final linear layer."""
    pruned = copy.deepcopy(model).to(device).eval()
    total_channels = 0
    active_channels = 0

    with torch.no_grad():
        for name, module in pruned.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if name not in contributions:
                continue

            contrib = contributions[name].to(device).squeeze(0)
            if contrib.dim() == 3:
                channel_contrib = contrib.abs().sum(dim=(1, 2))
            elif contrib.dim() == 1:
                channel_contrib = contrib.abs()
            else:
                continue

            mask = (channel_contrib > 0).float()
            C_out = module.weight.shape[0]
            module.weight.data *= mask.view(C_out, 1, 1, 1)

            total_channels += C_out
            active_channels += int(mask.sum().item())

    return pruned, active_channels / total_channels if total_channels > 0 else 0


def evaluate_per_class(model, dataset, device, num_classes=10):
    """Single-pass evaluation returning per-class and overall accuracy."""
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct = [0] * num_classes
    total = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for cls in range(num_classes):
                mask = (y == cls)
                total[cls] += mask.sum().item()
                correct[cls] += (preds[mask] == cls).sum().item()

    per_class = {c: correct[c] / total[c] if total[c] > 0 else 0 for c in range(num_classes)}
    overall = sum(correct) / sum(total) if sum(total) > 0 else 0
    return per_class, overall
