# utils/evaluation.py

"""Slicing and evaluation utilities for evaluation scripts"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from models import get_model
from core.slicer import Slicer
from utils.data import IMAGENETTE_TO_IMAGENET, stratified_subset


def compute_single_slice(
    sample_data,
    model_name,
    profile_path,
    theta,
    channel_mode,
    channel_alpha,
    block_mode,
    block_beta,
):
    """Compute slice for a single image (multiprocessing worker)."""
    image, label, idx = sample_data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(model_name, pretrained=True).to(device).eval()
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    slicer = Slicer(
        model=model,
        input_sample=image.unsqueeze(0).to(device),
        precomputed_profile=profile,
        debug=False,
    )
    slicer.profile()
    slicer.forward()
    slicer.backward(
        target_index=label,
        theta=theta,
        channel_mode=channel_mode,
        channel_alpha=channel_alpha,
        block_mode=block_mode,
        block_beta=block_beta,
    )

    return {
        "contributions": {
            k: v.cpu()
            for k, v in slicer.backward_result["neuron_contributions"].items()
        },
        "total_blocks": slicer.backward_result["total_blocks"],
        "skipped_blocks": slicer.backward_result["skipped_blocks"],
    }


def compute_slices(
    samples,
    model_name,
    profile_path,
    theta=0.3,
    channel_mode=False,
    channel_alpha=0.8,
    block_mode=False,
    block_beta=0.7,
    num_workers=4,
    desc="Slicing",
):
    """Compute slices in parallel."""
    worker_fn = partial(
        compute_single_slice,
        model_name=model_name,
        profile_path=profile_path,
        theta=theta,
        channel_mode=channel_mode,
        channel_alpha=channel_alpha,
        block_mode=block_mode,
        block_beta=block_beta,
    )

    with Pool(num_workers) as pool:
        slices = list(
            tqdm(
                pool.imap(worker_fn, samples),
                total=len(samples),
                desc=f"  {desc}",
                leave=False,
            )
        )
    return slices


def aggregate_slices(slices: List[Dict]) -> Dict[str, torch.Tensor]:
    """Aggregate slices via union: sum of absolute contributions.

    Handles slices with different key sets (e.g. from block filtering)
    by treating missing keys as zero contributions.
    """
    contributions = [s["contributions"] for s in slices]
    all_keys = set()
    for c in contributions:
        all_keys.update(c.keys())

    aggregated = {}
    for key in all_keys:
        tensors = [s[key].float() for s in contributions if key in s]
        stacked = torch.stack(tensors)
        aggregated[key] = stacked.abs().sum(dim=0)
    return aggregated


def compute_slice_size(aggregated: Dict[str, torch.Tensor], model=None) -> float:
    """Fraction of active channels in aggregated slice.

    If model is provided, counts all conv layers (including those not in
    aggregated, e.g. skipped blocks) as having zero active channels.
    """
    total_channels = 0
    active_channels = 0

    if model is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                C_out = module.weight.shape[0]
                total_channels += C_out
                if name in aggregated:
                    tensor = aggregated[name]
                    if tensor.dim() == 4:
                        ch = tensor.abs().sum(dim=(2, 3)).squeeze(0)
                        active_channels += (ch > 0).sum().item()
    else:
        for key, tensor in aggregated.items():
            if tensor.dim() == 4:
                channel_contrib = tensor.abs().sum(dim=(2, 3)).squeeze(0)
                total_channels += channel_contrib.numel()
                active_channels += (channel_contrib > 0).sum().item()

    return active_channels / total_channels if total_channels > 0 else 0


def evaluate_per_class(
    model, dataset, device, num_classes=10, eval_samples: Optional[int] = None
):
    """Single-pass evaluation returning per-class and overall accuracy.

    Args:
        eval_samples: If set, evaluate on at most this many samples per class
                      instead of the full dataset.
    """
    if eval_samples is not None:
        dataset = stratified_subset(dataset, num_classes, eval_samples)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct = [0] * num_classes
    total = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for cls in range(num_classes):
                mask = y == cls
                total[cls] += mask.sum().item()
                correct[cls] += (preds[mask] == cls).sum().item()

    per_class = {
        c: correct[c] / total[c] if total[c] > 0 else 0 for c in range(num_classes)
    }
    overall = sum(correct) / sum(total) if sum(total) > 0 else 0
    return per_class, overall


def evaluate_per_class_imagenette(
    model, dataset, device, eval_samples: Optional[int] = None
):
    """Evaluate a 1000-class ImageNet model on ImageNette (10 classes).

    Maps ImageNet predictions to local ImageNette indices before comparison.
    """
    num_classes = 10
    if eval_samples is not None:
        dataset = stratified_subset(dataset, num_classes, eval_samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    imagenet_indices = IMAGENETTE_TO_IMAGENET
    correct = [0] * num_classes
    total = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu()
            # Only look at the 10 ImageNette logits
            imagenette_logits = logits[:, imagenet_indices]
            preds = imagenette_logits.argmax(1)
            for cls in range(num_classes):
                mask = y == cls
                total[cls] += mask.sum().item()
                correct[cls] += (preds[mask] == cls).sum().item()

    per_class = {
        c: correct[c] / total[c] if total[c] > 0 else 0 for c in range(num_classes)
    }
    overall = sum(correct) / sum(total) if sum(total) > 0 else 0
    return per_class, overall
