# tests/test_forward.py
"""Tests for the forward analysis phase"""

import torch
import pytest
from core.slicer import Slicer
from models.simple import PaperNN


def test_forward_keys_match_profile():
    """Forward and profile should have the same layer keys"""
    model = PaperNN()
    slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
    slicer.profile(profiling_samples=torch.zeros(1, 2))
    result = slicer.forward()

    profile_keys = set(slicer.profiler_result["neuron_means"].keys())
    forward_keys = set(result["neuron_deltas"].keys())
    assert profile_keys == forward_keys


def test_deltas_are_nonzero():
    """Deltas should be nonzero when input differs from profiling mean"""
    model = PaperNN()
    slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
    slicer.profile(profiling_samples=torch.zeros(1, 2))
    result = slicer.forward()

    assert result["layer_deltas"]["fc1"].item() > 0


def test_hooks_removed():
    """No hooks should remain on the model after forward"""
    model = PaperNN()
    slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
    slicer.profile(profiling_samples=torch.zeros(1, 2))
    slicer.forward()

    for module in slicer.model.modules():
        assert len(module._forward_hooks) == 0


def test_requires_profile():
    """forward() without profile should raise RuntimeError"""
    slicer = Slicer(PaperNN(), input_sample=torch.tensor([[1.0, 2.0]]))
    with pytest.raises(RuntimeError):
        slicer.forward()
