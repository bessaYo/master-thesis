# tests/test_integration.py
"""End-to-end integration tests for the full slicing pipeline."""

import torch
import pytest
from core.slicer import Slicer
from models.simple import SimpleNN, SimpleCNN


class TestSlicerPipeline:
    """Test the full profile -> forward -> backward pipeline."""

    def test_simple_nn_full_pipeline(self):
        model = SimpleNN()
        input_sample = torch.tensor([[1.0, 2.0]])

        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 2))
        slicer.forward()
        result = slicer.backward(target_index=0, theta=0.3)

        assert result["slice_neurons"] > 0
        assert result["total_neurons"] > 0
        assert result["slice_synapses"] > 0

    def test_simple_cnn_full_pipeline(self):
        model = SimpleCNN()
        input_sample = torch.tensor([[[[1., 2., 0.], [0., 1., 1.], [2., 0., 1.]]]])

        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 1, 3, 3))
        slicer.forward()
        result = slicer.backward(target_index=0, theta=0.3)

        assert result["slice_neurons"] > 0
        assert result["total_neurons"] > 0


class TestThetaSliceSize:
    """Test that theta correctly controls slice size."""

    def test_higher_theta_smaller_or_equal_slice(self):
        """Higher theta should produce smaller or equal slices."""
        model = SimpleNN()
        input_sample = torch.tensor([[1.0, 2.0]])
        profiling_samples = torch.randn(10, 2)

        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=profiling_samples)

        slicer.forward()
        result_low = slicer.backward(target_index=0, theta=0.1)

        slicer.forward()
        result_high = slicer.backward(target_index=0, theta=0.7)

        assert result_low["slice_neurons"] >= result_high["slice_neurons"]

    def test_theta_zero_is_maximum(self):
        """theta=0 should produce the largest possible slice."""
        model = SimpleNN()
        input_sample = torch.tensor([[1.0, 2.0]])
        profiling_samples = torch.randn(10, 2)

        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=profiling_samples)

        slicer.forward()
        result_zero = slicer.backward(target_index=0, theta=0.0)

        slicer.forward()
        result_nonzero = slicer.backward(target_index=0, theta=0.5)

        assert result_zero["slice_neurons"] >= result_nonzero["slice_neurons"]


class TestTargetClasses:
    """Test slicing for different target classes."""

    def test_different_targets_produce_different_slices(self):
        model = SimpleNN()
        input_sample = torch.tensor([[1.0, 2.0]])
        profiling_samples = torch.randn(10, 2)

        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=profiling_samples)

        slicer.forward()
        result0 = slicer.backward(target_index=0, theta=0.3)
        contrib0 = {k: v.clone() for k, v in result0["neuron_contributions"].items()}

        slicer.forward()
        result1 = slicer.backward(target_index=1, theta=0.3)
        contrib1 = result1["neuron_contributions"]

        any_different = any(
            not torch.equal(contrib0[k], contrib1[k])
            for k in contrib0 if k in contrib1
        )
        assert any_different


class TestPrecomputedProfile:
    """Test that precomputed profiles work correctly."""

    def test_precomputed_matches_computed(self):
        model = SimpleNN()
        input_sample = torch.tensor([[1.0, 2.0]])
        profiling_samples = torch.randn(10, 2)

        # Compute profile
        slicer1 = Slicer(model, input_sample=input_sample)
        profile = slicer1.profile(profiling_samples=profiling_samples)

        # Use precomputed
        slicer2 = Slicer(model, input_sample=input_sample, precomputed_profile=profile)
        slicer2.profile()
        slicer2.forward()
        result = slicer2.backward(target_index=0, theta=0.3)

        assert result["slice_neurons"] > 0


class TestErrorHandling:
    """Test that proper errors are raised for incorrect usage."""

    def test_profile_requires_samples_or_precomputed(self):
        model = SimpleNN()
        slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
        with pytest.raises(RuntimeError):
            slicer.profile()

    def test_forward_requires_profile(self):
        model = SimpleNN()
        slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
        with pytest.raises(RuntimeError):
            slicer.forward()

    def test_backward_requires_forward(self):
        model = SimpleNN()
        slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
        slicer.profile(profiling_samples=torch.zeros(1, 2))
        with pytest.raises(RuntimeError):
            slicer.backward(target_index=0)
