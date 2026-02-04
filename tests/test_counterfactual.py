# tests/test_counterfactual.py
"""Tests for the counterfactual evaluation."""

import torch
import pytest
from core.slicer import Slicer
from evaluation.counterfactual import CounterfactualEvaluator
from models.simple import SimpleNN


class TestCounterfactualEvaluator:

    @pytest.fixture
    def model(self):
        return SimpleNN()

    @pytest.fixture
    def input_sample(self):
        return torch.tensor([[1.0, 2.0]])

    @pytest.fixture
    def contributions(self, model, input_sample):
        """Compute a full slice (theta=0) for testing."""
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.zeros(1, 2))
        slicer.forward()
        slicer.backward(target_index=0, theta=0.0)
        return slicer.backward_result["neuron_contributions"]

    def test_returns_all_keys(self, model, input_sample, contributions):
        evaluator = CounterfactualEvaluator(model)
        result = evaluator.evaluate(input_sample, contributions, target_class=0)

        assert "original" in result
        assert "keep" in result
        assert "remove" in result

    def test_result_contains_metrics(self, model, input_sample, contributions):
        evaluator = CounterfactualEvaluator(model)
        result = evaluator.evaluate(input_sample, contributions, target_class=0)

        for key in ("original", "keep", "remove"):
            assert "pred" in result[key]
            assert "target_prob" in result[key]
            assert "target_logit" in result[key]
            assert "entropy" in result[key]
            assert "margin" in result[key]

    def test_keep_slice_produces_valid_output(self, model, input_sample, contributions):
        """Keeping slice neurons should produce valid probabilities."""
        evaluator = CounterfactualEvaluator(model)
        result = evaluator.evaluate(input_sample, contributions, target_class=0)

        # Probabilities should be valid (sum to ~1, all >= 0)
        assert result["keep"]["target_prob"] >= 0.0
        assert result["keep"]["target_prob"] <= 1.0

    def test_remove_slice_reduces_target_prob(self, model, input_sample):
        """Removing slice neurons should reduce target class probability."""
        # Use non-trivial profiling for meaningful slice
        slicer = Slicer(model, input_sample=input_sample)
        slicer.profile(profiling_samples=torch.randn(50, 2))
        slicer.forward()
        slicer.backward(target_index=0, theta=0.0)
        contributions = slicer.backward_result["neuron_contributions"]

        evaluator = CounterfactualEvaluator(model)
        result = evaluator.evaluate(input_sample, contributions, target_class=0)

        assert result["remove"]["target_prob"] <= result["original"]["target_prob"]

    def test_soft_mask_option(self, model, input_sample, contributions):
        """Soft mask should not fully zero out non-slice neurons."""
        evaluator = CounterfactualEvaluator(model, soft_mask=True, min_mask_value=0.1)
        result = evaluator.evaluate(input_sample, contributions, target_class=0)

        # Should still produce valid results
        assert "keep" in result
        assert result["keep"]["target_prob"] >= 0.0

    def test_hooks_cleaned_up(self, model, input_sample, contributions):
        """After evaluate(), no hooks should remain on the model."""
        evaluator = CounterfactualEvaluator(model)
        evaluator.evaluate(input_sample, contributions, target_class=0)

        assert len(evaluator.hooks) == 0
