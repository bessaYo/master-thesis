# core/blocks/block_filter.py

import torch

class ChannelBlockFilter():
    """Energy-based channel filtering.

    Keeps the minimal set of channels whose cumulative delta energy
    explains at least ``alpha`` fraction of the total channel energy.
    This is analogous to the block-level energy filter but applied
    per conv layer, and avoids the cascade problem of a fixed top-k%.
    """

    def __init__(self, alpha=0.8):
        self.alpha = alpha

    def get_active_channels(self, channel_deltas, contrib_channels=None):
        """Return channels that explain alpha fraction of energy."""
        deltas = channel_deltas.squeeze()  # [C]

        if contrib_channels is not None and len(contrib_channels) > 0:
            contrib_list = list(contrib_channels)
            contrib_indices = torch.tensor(contrib_list, dtype=torch.long)
            energies = deltas[contrib_indices].abs()
        else:
            contrib_indices = torch.arange(len(deltas))
            energies = deltas.abs()

        total_energy = energies.sum().item()
        if total_energy < 1e-9:
            return set(contrib_indices.tolist())

        # Sort by descending energy, keep until alpha is reached
        sorted_idx = energies.argsort(descending=True)
        cum_energy = 0.0
        active = set()

        for idx in sorted_idx:
            active.add(int(contrib_indices[idx].item()))
            cum_energy += energies[idx].item()
            if cum_energy / total_energy >= self.alpha:
                break

        return active


class ResBlockFilter:
    """
    Energy-based filtering of ResNet blocks.
    Keeps the minimal set of blocks whose accumulated contribution
    explains at least alpha fraction of total block energy.
    """

    def __init__(self, alpha=0.9):
        """
        alpha ∈ (0, 1]: fraction of total block contribution to preserve
        """
        self.alpha = alpha
        self.skip_blocks = set()

    def identify_skip_blocks(self, block_deltas, blocks):
        """
        block_deltas: Dict[str, Tensor or float]
        blocks: Dict[str, List[layer_names]]
        """
        self.skip_blocks = set()

        # 1. Compute energy per block
        block_energy = {}
        for block_name, delta in block_deltas.items():
            val = delta.item() if isinstance(delta, torch.Tensor) else delta
            block_energy[block_name] = abs(val)

        total_energy = sum(block_energy.values())
        if total_energy == 0:
            print("[BlockFilter] Warning: total block energy is zero.")
            return set()

        # 2. Sort blocks by descending energy
        sorted_blocks = sorted(
            block_energy.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 3. Keep blocks until alpha energy is reached
        kept_blocks = set()
        cum_energy = 0.0

        for block_name, energy in sorted_blocks:
            kept_blocks.add(block_name)
            cum_energy += energy
            if cum_energy / total_energy >= self.alpha:
                break

        # 4. Skip remaining blocks (if safe)
        protected_blocks = set()  # NEW
        for block_name in block_energy.keys():
            if block_name in kept_blocks:
                continue

            layers = blocks.get(block_name, [])
            has_conv_shortcut = any("shortcut.0" in layer for layer in layers)

            # Only skip identity blocks
            if not has_conv_shortcut:
                self.skip_blocks.add(block_name)
            else:
                protected_blocks.add(block_name) 

        actually_kept = len(kept_blocks) + len(protected_blocks)
        actually_skipped = len(self.skip_blocks)

        print(
            f"[BlockFilter] Energy threshold: {actually_kept}/{len(block_energy)} blocks "
            f"(α={self.alpha:.2f}, energy={cum_energy/total_energy:.2%})"
        )
        if protected_blocks:
            print(f"[BlockFilter] Protected (conv shortcut): {protected_blocks}")
        print(f"[BlockFilter] Skip Blocks: {actually_skipped}, Keep Blocks: {actually_kept}")

        return self.skip_blocks

    def should_skip_layer(self, layer_name):
        for block_name in self.skip_blocks:
            if layer_name.startswith(block_name + "."):
                if "shortcut" not in layer_name:
                    return True
        return False

    def get_skip_blocks(self):
        return self.skip_blocks