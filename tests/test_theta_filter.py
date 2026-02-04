# tests/test_theta_filter.py
"""Tests for the theta filter mechanism.

The theta filter controls slice size by removing low-magnitude contributions.
Paper formula: |Σ(i=1..j) wi·Δxi / y| < θ
where y is the activation value (NOT the delta).
"""

import torch
import pytest
import torch.nn as nn
from core.tracing.operations import BackwardOperations


class TestThetaFilterBasic:
    """Basic theta filter behavior."""

    @pytest.fixture
    def ops(self):
        return BackwardOperations(theta=0.3)

    def test_filters_small_contributions(self, ops):
        """Candidates whose cumulative w*dx stays below theta should be removed."""
        cands = [
            {"i": 0, "w_dx": 0.1, "local": torch.tensor(0.1)},
            {"i": 1, "w_dx": 0.5, "local": torch.tensor(0.5)},
            {"i": 2, "w_dx": 2.0, "local": torch.tensor(2.0)},
        ]
        # y=10: cumulative (0.1+0.5+2.0)/10=0.26 < 0.3 -> all filtered
        result = ops._theta_filter(cands, 10.0)
        assert len(result) == 0

    def test_keeps_large_contributions(self, ops):
        """Contributions that push cumulative above theta should be kept."""
        cands = [
            {"i": 0, "w_dx": 0.1, "local": torch.tensor(0.1)},
            {"i": 1, "w_dx": 5.0, "local": torch.tensor(5.0)},
        ]
        # y=10: 0.1/10=0.01<0.3 (remove), (0.1+5.0)/10=0.51>=0.3 (keep i=1)
        result = ops._theta_filter(cands, 10.0)
        assert len(result) == 1
        assert result[0]["i"] == 1

    def test_theta_zero_keeps_all(self):
        """With theta=0, nothing should be filtered."""
        ops = BackwardOperations(theta=0.0)
        cands = [
            {"i": 0, "w_dx": 0.001, "local": torch.tensor(0.001)},
            {"i": 1, "w_dx": 0.002, "local": torch.tensor(0.002)},
        ]
        result = ops._theta_filter(cands, 10.0)
        assert len(result) == 2

    def test_empty_candidates(self, ops):
        """Empty candidate list should return empty."""
        result = ops._theta_filter([], 10.0)
        assert len(result) == 0

    def test_near_zero_output_returns_all(self, ops):
        """When output value is ~0, all candidates should be returned (no filtering)."""
        cands = [
            {"i": 0, "w_dx": 0.1, "local": torch.tensor(0.1)},
        ]
        result = ops._theta_filter(cands, 1e-12)
        assert len(result) == 1

    def test_sorted_by_magnitude(self, ops):
        """Candidates should be sorted by |w_dx| before filtering."""
        cands = [
            {"i": 0, "w_dx": 5.0, "local": torch.tensor(5.0)},
            {"i": 1, "w_dx": 0.1, "local": torch.tensor(0.1)},
            {"i": 2, "w_dx": 1.0, "local": torch.tensor(1.0)},
        ]
        # y=10: sorted by |w_dx|: [0.1, 1.0, 5.0]
        # 0.1/10=0.01<0.3 (remove i=1)
        # (0.1+1.0)/10=0.11<0.3 (remove i=2)
        # (0.1+1.0+5.0)/10=0.61>=0.3 (keep i=0)
        result = ops._theta_filter(cands, 10.0)
        assert len(result) == 1
        assert result[0]["i"] == 0


class TestThetaFilterDenominator:
    """Test that the theta filter uses the correct denominator.

    The paper specifies y (activation) as denominator, NOT Δy (delta).
    This distinction matters when a neuron's activation is far from its delta.
    """

    def test_activation_vs_delta_as_denominator(self):
        """Using activation y vs delta Δy should produce different filtering."""
        ops = BackwardOperations(theta=0.3)

        cands = [
            {"i": 0, "w_dx": 0.5, "local": torch.tensor(0.5)},
            {"i": 1, "w_dx": 1.0, "local": torch.tensor(1.0)},
            {"i": 2, "w_dx": 3.0, "local": torch.tensor(3.0)},
        ]

        # With activation y=10 (correct): cumulative 4.5/10=0.45
        # Only i=2 survives (cumulative at i=1: 1.5/10=0.15 < 0.3)
        result_activation = ops._theta_filter(cands, 10.0)

        # With delta=0.5 (wrong): 0.5/0.5=1.0 >= 0.3 -> keep ALL
        result_delta = ops._theta_filter(cands, 0.5)

        assert len(result_activation) < len(result_delta)
        assert len(result_activation) == 1
        assert len(result_delta) == 3

    def test_linear_uses_activation_not_delta(self):
        """linear() should pass activation y to theta filter, not delta Δy."""
        layer = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            layer.weight[:] = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])

        ops = BackwardOperations(theta=0.3)

        CONTRIB_n = torch.tensor([[1.0, 0.0]])
        delta_n = torch.tensor([[0.1, 0.0]])       # Small delta (neuron near mean)
        delta_i = torch.tensor([[1.0, 1.0, 1.0]])
        activation_n = torch.tensor([[5.0, 0.0]])   # Large activation

        # With activation (y=5.0): more filtering, smaller slice
        _, contrib_act = ops.linear(layer, CONTRIB_n, delta_n, delta_i, activation_n)

        # Without activation (falls back to delta y=0.1): less filtering, larger slice
        _, contrib_delta = ops.linear(layer, CONTRIB_n, delta_n, delta_i, None)

        active_act = (contrib_act != 0).sum().item()
        active_delta = (contrib_delta != 0).sum().item()

        assert active_act <= active_delta

    def test_conv2d_uses_activation_not_delta(self):
        """conv2d() should pass activation y to theta filter, not delta Δy."""
        layer = nn.Conv2d(1, 1, kernel_size=2, bias=False)
        with torch.no_grad():
            layer.weight[:] = torch.tensor([[[[1.0, 0.5], [0.3, 0.1]]]])

        ops = BackwardOperations(theta=0.3)

        CONTRIB_n = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
        delta_n = torch.tensor([[[[0.01, 0.01], [0.01, 0.01]]]])   # Small delta
        delta_i = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])
        activation_n = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])  # Large activation

        # With activation: denominator is large -> more filtering
        _, contrib_act = ops.conv2d(layer, CONTRIB_n, delta_n, delta_i, None, activation_n)

        # Without activation: denominator is small delta -> less filtering
        _, contrib_delta = ops.conv2d(layer, CONTRIB_n, delta_n, delta_i, None, None)

        active_act = (contrib_act != 0).sum().item()
        active_delta = (contrib_delta != 0).sum().item()

        assert active_act <= active_delta
