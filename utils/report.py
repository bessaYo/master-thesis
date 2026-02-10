"""Report printing for single-image slicing"""

import torch.nn as nn

W = 100  # report width


def compute_layer_synapses(model, neuron_contributions):
    """Compute per-layer synapse counts (total and active) at pixel level"""
    modules = {name: mod for name, mod in model.named_modules()}
    layer_synapses = {}

    for name, tensor in neuron_contributions.items():
        if name not in modules:
            continue
        mod = modules[name]

        if isinstance(mod, nn.Conv2d):
            out_neurons = tensor[0].numel()
            total = mod.in_channels * out_neurons
            active_out = (tensor[0] != 0).sum().item()
            active = mod.in_channels * active_out
            layer_synapses[name] = (total, min(active, total))

        elif isinstance(mod, nn.Linear):
            total = mod.in_features * mod.out_features
            active_out = (tensor.view(-1) != 0).sum().item()
            active = mod.in_features * active_out
            layer_synapses[name] = (total, min(active, total))

    return layer_synapses


def compute_layer_channels(model, neuron_contributions):
    """Compute per-layer active/total channel counts for Conv2d layers"""
    modules = {name: mod for name, mod in model.named_modules()}
    layer_channels = {}

    for name, tensor in neuron_contributions.items():
        if tensor.dim() != 4:
            continue
        if name not in modules or not isinstance(modules[name], nn.Conv2d):
            continue
        ch_contrib = tensor[0].abs().sum(dim=(1, 2))
        n_total = ch_contrib.numel()
        n_active = (ch_contrib > 0).sum().item()
        layer_channels[name] = (n_total, n_active)

    return layer_channels


def print_header(
    model_name,
    model,
    dataset_name,
    class_names,
    target_idx,
    image_label,
    dataset_idx,
    theta,
    channel_mode,
    channel_alpha,
    block_mode,
    block_beta,
):
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * W)
    print("SLICING REPORT")
    print("=" * W)
    print(f"  Model:        {model_name} ({total_params:,} params)")
    print(f"  Dataset:      {dataset_name.upper()}")
    if class_names:
        print(f"  Target class: {class_names[target_idx]} (index {target_idx})")
        print(
            f"  Input image:  test set #{dataset_idx} (true label: {class_names[image_label]})"
        )
    else:
        print(f"  Target class: {target_idx}")
        print(f"  Input image:  test set #{dataset_idx} (true label: {image_label})")
    print(f"  Theta:        {theta}")
    ch_str = f"ON (alpha={channel_alpha})" if channel_mode else "OFF"
    bl_str = f"ON (beta={block_beta})" if block_mode else "OFF"
    print(f"  Channel mode: {ch_str}")
    print(f"  Block mode:   {bl_str}")


def print_neuron_table(neuron_contributions, model):
    print()
    print("-" * W)
    print("LAYER-BY-LAYER CONTRIBUTIONS")
    print("-" * W)

    modules = {name: mod for name, mod in model.named_modules()}
    compute_layers = {}
    for name, tensor in neuron_contributions.items():
        if name in modules and isinstance(modules[name], (nn.Conv2d, nn.Linear)):
            compute_layers[name] = tensor

    layer_syn = compute_layer_synapses(model, neuron_contributions)
    layer_ch = compute_layer_channels(model, neuron_contributions)

    header = f"  {'Layer':<24} {'|Contrib|':>10}  {'Neurons':>16}  {'Synapses':>18}  {'Channels':>8}  {'Ratio':>6}"
    print(header)
    print("  " + "-" * (W - 2))

    total_n = 0
    active_n = 0
    total_s = 0
    active_s = 0
    total_contrib = 0.0

    for name, tensor in compute_layers.items():
        n_total = tensor.numel()
        n_active = (tensor != 0).sum().item()
        total_n += n_total
        active_n += n_active

        contrib_val = tensor.abs().sum().item()
        total_contrib += contrib_val

        neuron_str = f"{n_active:,}/{n_total:,}"

        s_total, s_active = layer_syn.get(name, (0, 0))
        total_s += s_total
        active_s += s_active
        syn_str = f"{s_active:,}/{s_total:,}"

        if name in layer_ch:
            ch_total, ch_active = layer_ch[name]
            ch_str = f"{ch_active}/{ch_total}"
        else:
            ch_str = ""

        ratio = f"{100.0 * n_active / n_total:.1f}%" if n_total > 0 else "0.0%"
        print(
            f"  {name:<24} {contrib_val:>10.1f}  {neuron_str:>16}  {syn_str:>18}  {ch_str:>8}  {ratio:>6}"
        )

    print("  " + "-" * (W - 2))
    total_neuron_str = f"{active_n:,}/{total_n:,}"
    total_syn_str = f"{active_s:,}/{total_s:,}"
    ratio = f"{100.0 * active_n / total_n:.1f}%" if total_n > 0 else "0.0%"
    print(
        f"  {'TOTAL':<24} {total_contrib:>10.1f}  {total_neuron_str:>16}  {total_syn_str:>18}  {'':>8}  {ratio:>6}"
    )


def print_block_analysis(forward_result, backward_result, neuron_contributions):
    blocks = forward_result.get("blocks", {})
    if not blocks:
        return

    block_deltas = forward_result.get("block_deltas", {})
    skipped = backward_result.get("skipped_blocks", 0)
    total = backward_result.get("total_blocks", 0)

    print()
    print("-" * W)
    print("BLOCK ANALYSIS")
    print("-" * W)

    header = f"  {'Block':<24} {'Delta':>8}  {'Status':<8}"
    print(header)
    print("  " + "-" * (W - 2))

    sorted_blocks = sorted(blocks.keys())
    for block_name in sorted_blocks:
        layer_list = blocks[block_name]
        delta = block_deltas.get(block_name, 0.0)
        delta_val = delta if isinstance(delta, float) else float(delta)

        main_conv_active = False
        for layer_name in layer_list:
            is_conv = "conv" in layer_name and "shortcut" not in layer_name
            if is_conv and layer_name in neuron_contributions:
                if (neuron_contributions[layer_name] != 0).any():
                    main_conv_active = True
                    break

        status = "KEPT" if main_conv_active else "SKIPPED"
        print(f"  {block_name:<24} {delta_val:>8.4f}  {status:<8}")

    print("  " + "-" * (W - 2))
    print(f"  Total: {total} blocks, {skipped} skipped")


def print_slice_summary(backward_result, model, neuron_contributions, t_backward):
    modules = {name: mod for name, mod in model.named_modules()}
    total_n = 0
    slice_n = 0
    for name, tensor in neuron_contributions.items():
        if name in modules and isinstance(modules[name], (nn.Conv2d, nn.Linear)):
            total_n += tensor.numel()
            slice_n += (tensor != 0).sum().item()

    layer_syn = compute_layer_synapses(model, neuron_contributions)
    total_s = sum(t for t, _ in layer_syn.values())
    active_s = sum(a for _, a in layer_syn.values())

    layer_ch = compute_layer_channels(model, neuron_contributions)
    total_ch = sum(t for t, _ in layer_ch.values())
    active_ch = sum(a for _, a in layer_ch.values())

    print()
    print("-" * W)
    print("SLICE SUMMARY")
    print("-" * W)
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
    print("=" * W)
