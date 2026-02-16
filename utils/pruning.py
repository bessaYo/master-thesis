# utils/pruning.py

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict


def prune_model(model, contributions: Dict[str, torch.Tensor], device):
    """Zero conv channels with no aggregated contribution"""
    pruned = copy.deepcopy(model).to(device).eval()
    total_channels = 0
    active_channels = 0

    with torch.no_grad():
        for name, module in pruned.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue

            C_out = module.weight.shape[0]
            total_channels += C_out

            if name not in contributions:
                # Skipped block: zero all channels
                module.weight.data.zero_()
                continue

            contrib = contributions[name].to(device).squeeze(0)
            if contrib.dim() == 3:
                channel_contrib = contrib.abs().sum(dim=(1, 2))
            elif contrib.dim() == 1:
                channel_contrib = contrib.abs()
            else:
                continue

            mask = (channel_contrib > 0).float()
            module.weight.data *= mask.view(C_out, 1, 1, 1)
            active_channels += int(mask.sum().item())

    return pruned, active_channels / total_channels if total_channels > 0 else 0


def get_weight_contributions(aggregated, model):
    """Per-weight contribution scores: channel_energy(out_ch) * |weight|"""
    all_contribs, weight_map = [], []
    offset = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        W = module.weight
        n = W.numel()

        if name in aggregated:
            ch_energy = aggregated[name].abs().sum(dim=(0, 2, 3))
            contrib = ch_energy.view(-1, 1, 1, 1).expand_as(W) * W.abs()
            all_contribs.append(contrib.reshape(-1))
        else:
            all_contribs.append(torch.zeros(n))

        weight_map.append((name, offset, offset + n))
        offset += n

    return torch.cat(all_contribs), weight_map, offset


def prune_by_ratio(model, contributions, weight_map, ratio, device):
    """Zero out the lowest-contribution weights per layer"""
    pruned = copy.deepcopy(model).to(device).eval()

    with torch.no_grad():
        for name, start, end in weight_map:
            module = dict(pruned.named_modules())[name]
            layer_c = contributions[start:end]
            n_prune = int(layer_c.numel() * ratio)
            if n_prune == 0:
                continue

            threshold = layer_c.sort()[0][n_prune - 1].item()
            mask = layer_c > threshold

            # Handle ties at threshold
            if (~mask).sum().item() < n_prune:
                ties = (layer_c == threshold).nonzero(as_tuple=True)[0]
                mask[ties[: n_prune - (~mask).sum().item()]] = False

            module.weight.data *= mask.reshape(module.weight.shape).float().to(device)

    return pruned


def prune_random(model, weight_map, ratio, device, seed=0):
    """Zero out random weights per layer"""
    pruned = copy.deepcopy(model).to(device).eval()
    rng = np.random.RandomState(seed)

    with torch.no_grad():
        for name, start, end in weight_map:
            module = dict(pruned.named_modules())[name]
            n = end - start
            mask = np.ones(n, dtype=np.float32)
            mask[rng.choice(n, size=int(n * ratio), replace=False)] = 0.0
            module.weight.data *= torch.from_numpy(mask).reshape(module.weight.shape).to(device)

    return pruned
