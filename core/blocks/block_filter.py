# core/blocks/block_filter.py

from abc import ABC, abstractmethod
import torch


class BlockFilter(ABC):
    """Abstract base class for block filtering strategies."""

    @abstractmethod
    def get_active_blocks(self, deltas, contrib_mask=None):
        """Retrieve the set of active blocks after filtering."""
        pass


class ChannelBlockFilter(BlockFilter):
    """Filters channels by keeping only top-k% by delta magnitude."""

    def __init__(self, percent=0.2):
        self.percent = percent

    def get_active_blocks(self, channel_deltas, contrib_channels=None):
        """Return top-k% channels by delta magnitude."""
        deltas = channel_deltas.squeeze()  # [C]

        if contrib_channels is not None and len(contrib_channels) > 0:
            # Filter only within channels that have contributions
            contrib_list = list(contrib_channels)
            contrib_indices = torch.tensor(contrib_list, dtype=torch.long)
            contrib_deltas = deltas[contrib_indices]

            k = max(1, int(len(contrib_list) * self.percent))
            top_k_idx = torch.topk(contrib_deltas.abs(), k).indices

            return set(contrib_indices[top_k_idx].tolist())
        else:
            # Fallback: top-k% of all channels
            k = max(1, int(len(deltas) * self.percent))
            _, top_indices = torch.topk(deltas.abs(), k)
            return set(top_indices.tolist())


class ResBlockFilter:
    """Filters ResNet blocks by skipping those with low delta values."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.skip_blocks = set()

    def identify_skip_blocks(self, block_deltas, blocks):
        """Identify blocks to skip based on delta threshold."""
        print(f"[Block Filter] Using threshold: {self.threshold}")
        self.skip_blocks = set()

        for block_name, delta in block_deltas.items():
            delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta

            # Check if block has conv shortcut (not identity)
            layers = blocks.get(block_name, [])
            has_conv_shortcut = any("shortcut.0" in layer for layer in layers)

            # Only skip blocks with identity shortcut and low delta
            if delta_val < self.threshold and not has_conv_shortcut:
                self.skip_blocks.add(block_name)

        return self.skip_blocks

    def should_skip_layer(self, layer_name):
        """Check if a layer should be skipped (main path of skipped block)."""
        for block_name in self.skip_blocks:
            if layer_name.startswith(block_name + "."):
                # Skip main path layers, but NOT shortcut layers
                if "shortcut" not in layer_name:
                    return True
        return False

    def get_skip_blocks(self):
        """Return set of blocks to skip."""
        return self.skip_blocks