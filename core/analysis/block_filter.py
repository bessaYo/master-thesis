# core/analysis/block_filter.py

"""
Block-based filtering for hierarchical slicing.

Block types:
1. Channel blocks: Filter entire channels based on contribution thresholds.
2. Layer blocks: Filter entire layers based on contribution thresholds.
"""

from typing import Set, Optional
from abc import ABC, abstractmethod
import torch


class BlockFilter(ABC):

    @abstractmethod
    def get_active_blocks(self, deltas: torch.Tensor) -> Set[int]:
        """Retrieve the set of active blocks after filtering."""
        pass


# Channel Block Filter that selects top-k% channels by delta magnitude
class ChannelBlockFilter(BlockFilter):
    def __init__(self, percent: float = 0.2):
        self.percent = percent

    def get_active_blocks(self, channel_deltas: torch.Tensor) -> Set[int]:
        """
        Args:
            channel_deltas: Tensor of shape [1, C] - already computed in ForwardAnalyzer
        """
        channel_magnitudes = channel_deltas.squeeze()  # [C]
        k = max(1, int(len(channel_magnitudes) * self.percent))
        _, top_indices = torch.topk(channel_magnitudes, k)

        # Return as a set of channel indices to process
        return set(top_indices.tolist())
