import json
import os
import torch.nn as nn


def _layer_synapses(mod, n_active):
    """Synapse count for a single layer given active output neurons"""
    if isinstance(mod, nn.Conv2d):
        return mod.in_channels * n_active
    elif isinstance(mod, nn.Linear):
        return mod.in_features * n_active
    return 0


def _total_synapses(mod, n_total):
    """Total synapse count for a single layer"""
    if isinstance(mod, nn.Conv2d):
        return mod.in_channels * n_total
    elif isinstance(mod, nn.Linear):
        return mod.in_features * mod.out_features
    return 0


def _model_totals(model, neuron_deltas):
    """Total neuron/synapse/channel counts using forward deltas (stable with block skipping)"""
    modules = dict(model.named_modules())
    total_n, total_s, total_ch = 0, 0, 0
    for name in neuron_deltas:
        if name not in modules:
            continue
        mod = modules[name]
        if isinstance(mod, nn.Conv2d):
            n = neuron_deltas[name].numel()
            total_n += n
            total_s += mod.in_channels * n
            total_ch += mod.out_channels
        elif isinstance(mod, nn.Linear):
            total_n += mod.out_features
            total_s += mod.in_features * mod.out_features
    return total_n, total_s, total_ch


def print_header(model_name, model, dataset_name, class_names, target_idx,
                 image_label, dataset_idx, theta, channel_alpha, block_beta):
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 100)
    print("SLICING REPORT")
    print("=" * 100)
    print(f"  Model:        {model_name} ({total_params:,} params)")
    print(f"  Dataset:      {dataset_name.upper()}")
    if class_names:
        print(f"  Target class: {class_names[target_idx]} (index {target_idx})")
        print(f"  Input image:  test set #{dataset_idx} (true label: {class_names[image_label]})")
    else:
        print(f"  Target class: {target_idx}")
        print(f"  Input image:  test set #{dataset_idx} (true label: {image_label})")
    print(f"  Theta:        {theta}")
    print(f"  Channel mode: {'ON (alpha=' + str(channel_alpha) + ')' if channel_alpha is not None else 'OFF'}")
    print(f"  Block mode:   {'ON (beta=' + str(block_beta) + ')' if block_beta is not None else 'OFF'}")


def print_neuron_table(neuron_contributions, model, neuron_deltas):
    print()
    print("-" * 100)
    print("LAYER-BY-LAYER CONTRIBUTIONS")
    print("-" * 100)

    modules = dict(model.named_modules())

    # All compute layers from forward deltas (includes layers in skipped blocks)
    compute_layers = [n for n in neuron_deltas
                      if n in modules and isinstance(modules[n], (nn.Conv2d, nn.Linear))]

    header = f"  {'Layer':<24} {'|Contrib|':>10}  {'Neurons':>16}  {'Synapses':>18}  {'Channels':>8}  {'Ratio':>6}"
    print(header)
    print("  " + "-" * 98)

    sum_n, sum_active_n = 0, 0
    sum_s, sum_active_s = 0, 0
    sum_contrib = 0.0

    for name in compute_layers:
        mod = modules[name]
        n_total = neuron_deltas[name].numel()

        # Active neurons and contribution value from backward pass
        if name in neuron_contributions:
            tensor = neuron_contributions[name]
            n_active = (tensor != 0).sum().item()
            contrib_val = tensor.abs().sum().item()
        else:
            # Layer was in a skipped block
            n_active, contrib_val = 0, 0.0

        s_total = _total_synapses(mod, n_total)
        s_active = _layer_synapses(mod, n_active)

        sum_n += n_total
        sum_active_n += n_active
        sum_s += s_total
        sum_active_s += s_active
        sum_contrib += contrib_val

        # Channel info only for conv layers
        ch_str = ""
        if isinstance(mod, nn.Conv2d) and name in neuron_contributions:
            ch_contrib = neuron_contributions[name][0].abs().sum(dim=(1, 2))
            ch_str = f"{(ch_contrib > 0).sum().item()}/{ch_contrib.numel()}"

        neuron_str = f"{n_active:,}/{n_total:,}"
        syn_str = f"{s_active:,}/{s_total:,}"
        ratio = f"{100.0 * n_active / n_total:.1f}%" if n_total > 0 else "0.0%"
        print(f"  {name:<24} {contrib_val:>10.1f}  {neuron_str:>16}  {syn_str:>18}  {ch_str:>8}  {ratio:>6}")

    print("  " + "-" * 98)
    ratio = f"{100.0 * sum_active_n / sum_n:.1f}%" if sum_n > 0 else "0.0%"
    total_neuron_str = f"{sum_active_n:,}/{sum_n:,}"
    total_syn_str = f"{sum_active_s:,}/{sum_s:,}"
    print(f"  {'TOTAL':<24} {sum_contrib:>10.1f}  {total_neuron_str:>16}  {total_syn_str:>18}  {'':>8}  {ratio:>6}")


def print_block_analysis(forward_result, backward_result, neuron_contributions):
    blocks = forward_result.get("blocks", {})
    if not blocks:
        return

    block_deltas = forward_result.get("block_deltas", {})

    print()
    print("-" * 100)
    print("BLOCK ANALYSIS")
    print("-" * 100)
    print(f"  {'Block':<24} {'Delta':>8}  {'Status':<8}")
    print("  " + "-" * 98)

    for block_name in sorted(blocks):
        delta = float(block_deltas.get(block_name, 0.0))

        # Check if any main-path conv has nonzero contributions
        active = any(
            (neuron_contributions[ln] != 0).any()
            for ln in blocks[block_name]
            if "conv" in ln and "shortcut" not in ln and ln in neuron_contributions
        )
        print(f"  {block_name:<24} {delta:>8.4f}  {'KEPT' if active else 'SKIPPED':<8}")

    print("  " + "-" * 98)
    print(f"  Total: {backward_result['total_blocks']} blocks, "
          f"{backward_result['skipped_blocks']} skipped")


def print_slice_summary(backward_result, model, neuron_contributions, t_backward, neuron_deltas):
    modules = dict(model.named_modules())
    total_n, total_s, total_ch = _model_totals(model, neuron_deltas)

    # Count active neurons, synapses, channels from backward contributions
    slice_n, active_s, active_ch = 0, 0, 0
    for name, tensor in neuron_contributions.items():
        if name not in modules:
            continue
        mod = modules[name]
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            n_active = (tensor != 0).sum().item()
            slice_n += n_active
            active_s += _layer_synapses(mod, n_active)
        if isinstance(mod, nn.Conv2d) and tensor.dim() == 4:
            active_ch += (tensor[0].abs().sum(dim=(1, 2)) > 0).sum().item()

    print()
    print("-" * 100)
    print("SLICE SUMMARY")
    print("-" * 100)
    print(f"  Time:            {t_backward:.2f}s")
    n_pct = f"{100.0 * slice_n / total_n:.1f}%" if total_n > 0 else "0.0%"
    s_pct = f"{100.0 * active_s / total_s:.1f}%" if total_s > 0 else "0.0%"
    print(f"  Neurons:         {slice_n:,}/{total_n:,} ({n_pct})")
    print(f"  Synapses:        {active_s:,}/{total_s:,} ({s_pct})")
    if total_ch > 0:
        ch_pct = f"{100.0 * active_ch / total_ch:.1f}%"
        print(f"  Channels:        {active_ch:,}/{total_ch:,} ({ch_pct})")
    print(f"  Blocks total:    {backward_result['total_blocks']}")
    print(f"  Blocks skipped:  {backward_result['skipped_blocks']}")
    print()
    print("=" * 100)


def save_slice_json(args, dataset_name, class_names, backward_result, model, neuron_deltas):
    nc = backward_result["neuron_contributions"]
    modules = dict(model.named_modules())
    total_n, total_s, _ = _model_totals(model, neuron_deltas)

    # Count active slice neurons and synapses
    slice_n, active_s = 0, 0
    for name, tensor in nc.items():
        if name not in modules:
            continue
        mod = modules[name]
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            n_active = (tensor != 0).sum().item()
            slice_n += n_active
            active_s += _layer_synapses(mod, n_active)

    # Build filename from slicing config
    fname = f"{args.model}_t{args.target}_img{args.image_index}"
    if args.channel_alpha is not None:
        fname += f"_ch{args.channel_alpha}"
    if args.block_beta is not None:
        fname += f"_bl{args.block_beta}"

    out_dir = "evaluation/slices"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname}.json")

    data = {
        "config": {
            "model": args.model,
            "dataset": dataset_name,
            "target_class": args.target,
            "target_class_name": class_names[args.target] if class_names else str(args.target),
            "image_index": args.image_index,
            "theta": args.theta,
            "channel_alpha": args.channel_alpha,
            "block_beta": args.block_beta,
        },
        "results": {
            "backward_time": round(backward_result["backward_time"], 4),
            "total_neurons": total_n,
            "slice_neurons": slice_n,
            "neurons_pct": round(100.0 * slice_n / total_n, 1) if total_n > 0 else 0.0,
            "total_synapses": total_s,
            "slice_synapses": active_s,
            "synapses_pct": round(100.0 * active_s / total_s, 1) if total_s > 0 else 0.0,
            "total_blocks": backward_result["total_blocks"],
            "skipped_blocks": backward_result["skipped_blocks"],
        },
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {out_path}")
