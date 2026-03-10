import torch
import torch.nn as nn
import copy


def prune_model(model, neuron_contributions, device, synapse_contributions=None):
    """Prune model at block, channel and weight level based on slice contributions"""
    pruned = copy.deepcopy(model).to(device).eval()
    total_weights = 0
    active_weights = 0
    total_channels = 0
    active_channels = 0

    with torch.no_grad():
        for name, module in pruned.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue

            C_out = module.weight.shape[0]
            total_channels += C_out
            total_weights += module.weight.numel()

            # Block pruning
            if name not in neuron_contributions:
                module.weight.data.zero_()
                continue

            # Channel pruning
            contrib = neuron_contributions[name].to(device).squeeze(0)
            if contrib.dim() == 3:
                channel_contrib = contrib.abs().sum(dim=(1, 2))
            elif contrib.dim() == 1:
                channel_contrib = contrib.abs()
            else:
                continue

            channel_mask = (channel_contrib > 0).float()
            module.weight.data *= channel_mask.view(C_out, 1, 1, 1)
            active_channels += int(channel_mask.sum().item())

            # Weight pruning
            if synapse_contributions is not None and name in synapse_contributions:
                syn = synapse_contributions[name].to(device)
                module.weight.data *= (syn.abs() > 0).float()

            active_weights += int((module.weight.data != 0).sum().item())

    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    active_params = 0
    for p in pruned.parameters():
        active_params += (p != 0).sum().item()

    stats = {
        "total_channels": total_channels,
        "active_channels": active_channels,
        "channel_ratio": active_channels / total_channels if total_channels > 0 else 0,
        "total_weights": total_weights,
        "active_weights": active_weights,
        "weight_ratio": active_weights / total_weights if total_weights > 0 else 0,
        "total_params": total_params,
        "active_params": int(active_params),
        "model_size": int(active_params) / total_params if total_params > 0 else 0,
    }
    return pruned, stats
