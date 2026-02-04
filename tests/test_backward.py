# tests/test_backward.py
"""Tests for the backward analysis phase (contribution propagation)."""

import torch
import pytest
import torch.nn as nn
from core.slicer import Slicer
from models.simple import SimpleNN, SimpleCNN


class TestSimpleNNBackward:

    @pytest.fixture
    def model(self):
        return SimpleNN()

    @pytest.fixture
    def slicer(self, model):
        input_sample = torch.tensor([[1.0, 2.0]])
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 2))
        slicer.forward()
        return slicer

    def test_output_initialization(self, slicer):
        """Target neuron should get contribution, others should be zero."""
        slicer.backward(target_index=0, theta=0.0)
        contrib = slicer.backward_result["neuron_contributions"]

        assert contrib["fc3"][0, 0] != 0.0
        assert contrib["fc3"][0, 1] == 0.0

    def test_backward_produces_slice(self, slicer):
        result = slicer.backward(target_index=0, theta=0.3)
        assert result["slice_neurons"] > 0
        assert result["slice_neurons"] <= result["total_neurons"]

    def test_backward_produces_synapses(self, slicer):
        result = slicer.backward(target_index=0, theta=0.3)
        assert result["slice_synapses"] > 0

    def test_contributions_are_integers(self, slicer):
        """Contributions are accumulated signs, so they should be integers."""
        slicer.backward(target_index=0, theta=0.3)
        contrib = slicer.backward_result["neuron_contributions"]

        for name, tensor in contrib.items():
            # sign() accumulates across multiple parents, so values can be > 1 or < -1
            assert torch.all(tensor == tensor.round()), (
                f"Layer {name} has non-integer contribution values"
            )


class TestSimpleCNNBackward:

    @pytest.fixture
    def model(self):
        """SimpleCNN with AvgPool (original architecture)."""
        return SimpleCNN()

    @pytest.fixture
    def slicer(self, model):
        input_sample = torch.tensor([[[[1., 2., 0.], [0., 1., 1.], [2., 0., 1.]]]])
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 1, 3, 3))
        slicer.forward()
        return slicer

    def test_backward_produces_contributions(self, slicer):
        result = slicer.backward(target_index=0, theta=0.3)
        assert result["slice_neurons"] > 0
        assert result["slice_synapses"] > 0

    def test_contributions_for_all_layers(self, slicer):
        slicer.backward(target_index=0, theta=0.3)
        contrib = slicer.backward_result["neuron_contributions"]

        assert any("conv" in k for k in contrib)
        assert any("fc" in k for k in contrib)

    def test_synapse_contributions_exist(self, slicer):
        slicer.backward(target_index=0, theta=0.3)
        synapses = slicer.backward_result["synapse_contributions"]
        assert len(synapses) > 0

    def test_conv_contributions_have_spatial_dims(self, slicer):
        """Conv layer contributions should be 4D tensors."""
        slicer.backward(target_index=0, theta=0.3)
        contrib = slicer.backward_result["neuron_contributions"]

        for name, tensor in contrib.items():
            if "conv" in name:
                assert tensor.dim() == 4, f"{name} should be 4D, got {tensor.dim()}D"
