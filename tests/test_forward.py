# tests/test_forward.py
"""Tests for the forward analysis phase (profiling + forward deltas)."""

import torch
import pytest
from core.slicer import Slicer
from models.simple import SimpleNN, SimpleCNN


class TestSimpleNNForward:

    @pytest.fixture
    def model(self):
        return SimpleNN()

    @pytest.fixture
    def input_sample(self):
        return torch.tensor([[1.0, 2.0]])

    def test_forward_pass_values(self, model, input_sample):
        """Verify known output: [1,2] -> fc1 -> [5,2,2] -> fc2 -> [9,-3] -> fc3 -> [15,0]"""
        output = model(input_sample)
        expected = torch.tensor([[15.0, 0.0]])
        assert torch.allclose(output, expected)

    def test_forward_deltas_with_zero_profiling(self, model, input_sample):
        """With zero profiling, deltas should equal activations."""
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 2))
        result = slicer.forward()

        expected_deltas = {
            "fc1": torch.tensor([[5.0, 2.0, 2.0]]),
            "fc2": torch.tensor([[9.0, -3.0]]),
            "fc3": torch.tensor([[15.0, 0.0]]),
        }
        for layer, exp in expected_deltas.items():
            assert torch.allclose(
                result["neuron_deltas"][layer], exp
            ), f"{layer} delta mismatch"

    def test_input_delta_stored(self, model, input_sample):
        """Input delta should be stored in forward result."""
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 2))
        result = slicer.forward()

        assert "input" in result["neuron_deltas"]
        assert torch.allclose(result["neuron_deltas"]["input"], input_sample)

    def test_layer_deltas_stored(self, model, input_sample):
        """Layer-level deltas should exist for all layers."""
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 2))
        result = slicer.forward()

        for layer in ("fc1", "fc2", "fc3"):
            assert layer in result["layer_deltas"]


class TestSimpleCNNForward:

    @pytest.fixture
    def model(self):
        return SimpleCNN()

    @pytest.fixture
    def input_sample(self):
        return torch.tensor([[[[1., 2., 0.], [0., 1., 1.], [2., 0., 1.]]]])

    def test_forward_produces_output(self, model, input_sample):
        output = model(input_sample)
        assert output.shape[0] == 1

    def test_forward_deltas_have_spatial_dims(self, model, input_sample):
        """Conv layer deltas should have 4D shape (B, C, H, W)."""
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros_like(input_sample))
        result = slicer.forward()

        conv_delta = result["neuron_deltas"]["conv"]
        assert conv_delta.dim() == 4

    def test_channel_deltas_for_conv(self, model, input_sample):
        """Channel deltas should exist for conv layers."""
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros_like(input_sample))
        result = slicer.forward()

        assert "conv" in result["channel_deltas"]

    def test_activations_stored(self, model, input_sample):
        """Raw activations should be stored for each layer."""
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros_like(input_sample))
        result = slicer.forward()

        assert "conv" in result["activations"]
        assert "relu" in result["activations"]
        assert "fc" in result["activations"]
