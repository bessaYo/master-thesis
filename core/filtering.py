import torch


class ThetaSlicer:
    """Slice neurons/synapses by cumulative magnitude threshold (theta)"""

    def __init__(self, theta=0.0):
        self.theta = theta

    def filter_linear(self, magnitude, layer_activation):
        """Theta filter for linear layers"""
        output_neurons, _ = magnitude.shape
        keep = torch.ones_like(magnitude, dtype=torch.bool)

        # Loop over output neurons and apply energy thresholding
        for neuron in range(output_neurons):
            activation = layer_activation[neuron].abs()

            if activation == 0:
                keep[neuron] = False
                continue

            energies = magnitude[neuron].abs()
            mask = energy_threshold(energies, 1 - self.theta)

            if mask is None:
                keep[neuron] = False
            else:
                keep[neuron] = mask

        return keep

    def filter_conv(self, magnitude, layer_activation):
        """Theta filter for conv2d: filter per output channel across all spatial positions"""
        # Sum absolute magnitude across spatial positions per kernel element
        channel_mag = magnitude.abs().sum(dim=2)  # [out_ch, kernel_size]
        out_channels = channel_mag.shape[0]

        keep = torch.zeros_like(channel_mag, dtype=torch.bool)

        # Loop over output channels and for each channel apply energy thresholding
        for ch in range(out_channels):
            activation = layer_activation[ch].abs().sum()

            if activation == 0:
                continue

            mask = energy_threshold(channel_mag[ch], 1 - self.theta)
            if mask is not None:
                keep[ch] = mask

        magnitude = magnitude * keep.unsqueeze(2).float()
        return magnitude


class ChannelSlicer:
    """Keep only channels where cumulative energy >= alpha of total energy"""

    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self._channel_masks = {}

    def compute_active_channels(self, channel_deltas):
        """Based on channel deltas, compute a boolean mask to indicate which channels to keep for each node"""
        self._channel_masks = {}

        for node_key, deltas in channel_deltas.items():
            deltas = deltas.squeeze()
            energies = deltas.abs()
            mask = energy_threshold(energies, self.alpha)

            if mask is not None:
                self._channel_masks[node_key] = mask

    def get_channel_mask(self, node_key):
        """Return channel mask for given node"""
        return self._channel_masks.get(node_key)


class BlockSlicer:
    """Keep only ResNet blocks where cumulative energy >= alpha of total energy"""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.skip_blocks = set()

    def identify_skip_blocks(self, block_deltas, blocks):
        self.skip_blocks = set()

        block_energy = {}
        for block_name, delta in block_deltas.items():
            if isinstance(delta, torch.Tensor):
                val = delta.item()
            else:
                val = delta
            block_energy[block_name] = abs(val)

        total_energy = sum(block_energy.values())
        if total_energy == 0:
            print("[BlockSlicer] Warning: total block energy is zero.")
            return set()

        sorted_blocks = sorted(block_energy.items(), key=lambda x: x[1], reverse=True)

        kept_blocks = set()
        cum_energy = 0.0
        for block_name, energy in sorted_blocks:
            kept_blocks.add(block_name)
            cum_energy += energy
            if cum_energy / total_energy >= self.alpha:
                break

        protected_blocks = set()
        for block_name in block_energy.keys():
            if block_name in kept_blocks:
                continue

            layers = blocks.get(block_name, [])
            has_conv_shortcut = False
            for layer in layers:
                if "shortcut.0" in layer:
                    has_conv_shortcut = True
                    break

            if not has_conv_shortcut:
                self.skip_blocks.add(block_name)
            else:
                protected_blocks.add(block_name)

        actually_kept = len(kept_blocks) + len(protected_blocks)
        actually_skipped = len(self.skip_blocks)

        print(f"[BlockSlicer] Energy threshold: {actually_kept}/{len(block_energy)} blocks (a={self.alpha:.2f}, energy={cum_energy / total_energy:.2%})")
        if protected_blocks:
            print(f"[BlockSlicer] Protected (conv shortcut): {protected_blocks}")
        print(f"[BlockSlicer] Skip: {actually_skipped}, Keep: {actually_kept}")

        return self.skip_blocks

    def get_skip_blocks(self):
        return self.skip_blocks


def energy_threshold(energies, hyperparameter):
    """Sort by energy and keep top elements based on hyperparameter -> return boolean tensor of kept elements"""
    total = energies.sum().item()
    if total == 0:
        return None

    sorted_idx = energies.argsort(descending=True)  # Sort in descending order
    cumsum = energies[sorted_idx].cumsum(0)  # Cumulative sum of sorted energies
    threshold = hyperparameter * total
    cutoff = (cumsum >= threshold).nonzero()[0].item()  # Find first element where threshold is met

    # Create boolean mask of kept elements
    energy_mask = torch.zeros(len(energies), dtype=torch.bool, device=energies.device)
    energy_mask[sorted_idx[: cutoff + 1]] = True
    return energy_mask
