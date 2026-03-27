import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from models import get_model
from core.slicer import Slicer


def compute_single_slice(
    sample_data,
    model_name,
    profile_path,
    theta,
    channel_alpha,
    block_beta,
):
    """Compute slice for a single image."""
    image, label, idx = sample_data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(model_name, pretrained=True).to(device).eval()
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    slicer = Slicer(
        model=model,
        input_sample=image.unsqueeze(0).to(device),
        precomputed_profile=profile,
    )
    slicer.profile()
    slicer.forward()
    slicer.backward(
        target_index=label,
        theta=theta,
        channel_alpha=channel_alpha,
        block_beta=block_beta,
    )

    return {
        "contributions": {k: v.cpu() for k, v in slicer.backward_result["neuron_contributions"].items()},
        "synapse_contributions": {k: v.cpu() for k, v in slicer.backward_result["synapse_contributions"].items()},
        "total_blocks": slicer.backward_result["total_blocks"],
        "skipped_blocks": slicer.backward_result["skipped_blocks"],
    }


def compute_slices(
    samples,
    model_name,
    profile_path,
    theta=0.2,
    channel_alpha=None,
    block_beta=None,
    num_workers=4,
    desc="Slicing",
):
    """Compute slices for multiple samples in parallel."""
    worker_fn = partial(
        compute_single_slice,
        model_name=model_name,
        profile_path=profile_path,
        theta=theta,
        channel_alpha=channel_alpha,
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


def aggregate_slices(slices):
    """Aggregate multiple slices via union (sum of abs contributions)."""
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


def aggregate_synapse_contribs(slices):
    """Aggregate synapse contributions via union (sum of abs contributions)."""
    all_keys = set()
    for s in slices:
        all_keys.update(s["synapse_contributions"].keys())

    aggregated = {}
    for key in all_keys:
        tensors = [s["synapse_contributions"][key].float() for s in slices if key in s["synapse_contributions"]]
        stacked = torch.stack(tensors)
        aggregated[key] = stacked.abs().sum(dim=0)
    return aggregated


def compute_slice_size(aggregated, model=None):
    """Fraction of active channels in the aggregated slice."""
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

    if total_channels == 0:
        return 0
    return active_channels / total_channels


def evaluate_per_class(model, dataset, device, num_classes=10, eval_samples=None):
    """Evaluate per-class and overall accuracy"""
    if eval_samples is not None:
        counts = defaultdict(int)
        indices = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            if counts[label] < eval_samples:
                indices.append(i)
                counts[label] += 1
            if len(indices) >= eval_samples * num_classes:
                break
        dataset = Subset(dataset, indices)
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

    per_class = {}
    for c in range(num_classes):
        if total[c] > 0:
            per_class[c] = correct[c] / total[c]
        else:
            per_class[c] = 0

    overall = sum(correct) / sum(total) if sum(total) > 0 else 0
    return per_class, overall
