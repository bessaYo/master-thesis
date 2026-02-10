# tests/test_block_slicer.py
"""Tests for channel and block slicing"""

import torch
from core.block.block_slicer import ChannelSlicer, BlockSlicer


def test_channel_slicer_keeps_top_channels():
    """Alpha=0.8 should keep the channels that explain 80% of energy"""
    slicer = ChannelSlicer(alpha=0.8)
    # Energies: [10, 1, 1, 1, 1] -> total=14, 80%=11.2
    # Cumsum sorted: 10, 11, 12 -> cutoff at index 2, so 3 channels
    deltas = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0])
    mask = slicer.get_active_mask(deltas)

    assert mask[0] == True
    assert mask.sum().item() == 3


def test_channel_slicer_alpha1_keeps_all():
    """Alpha=1.0 should keep all channels with nonzero energy"""
    slicer = ChannelSlicer(alpha=1.0)
    deltas = torch.tensor([5.0, 3.0, 2.0])
    mask = slicer.get_active_mask(deltas)

    assert mask.all()


def test_block_slicer_skips_low_energy():
    """Blocks below the energy threshold should be skipped"""
    slicer = BlockSlicer(alpha=0.9)
    block_deltas = {"layer1.0": 10.0, "layer1.1": 1.0, "layer2.0": 0.5}
    blocks = {"layer1.0": ["layer1.0.conv1"], "layer1.1": ["layer1.1.conv1"], "layer2.0": ["layer2.0.conv1"]}

    skipped = slicer.identify_skip_blocks(block_deltas, blocks)

    assert "layer1.0" not in skipped
    assert len(skipped) > 0
