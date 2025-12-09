# tests/test_nnslicer.py

import torch
import pytest
import torch.nn as nn
from core.slicer import Slicer


# Neural Network from paper
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)
        self.fc3 = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.fc1.weight[:] = torch.tensor([[3.0, 1.0], [-2.0, 2.0], [0.0, 1.0]])
            self.fc2.weight[:] = torch.tensor([[1.0, 0.0, 2.0], [-1.0, 3.0, -2.0]])
            self.fc3.weight[:] = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Convolutional Neural Network with given weights
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.fc = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.conv.weight[:] = torch.tensor(
                [[[[1.0, -1.0], [0.0, 2.0]]], [[[0.0, 1.0], [1.0, -1.0]]]]
            )
            self.fc.weight[:] = torch.tensor(
                [
                    [4.0, -2.0],
                    [1.0, 3.0],
                ]
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# NN Tests
class TestSimpleNN:

    @pytest.fixture
    def model(self):
        return SimpleNN()

    @pytest.fixture
    def input_sample(self):
        return torch.tensor([[1.0, 2.0]])

    @pytest.fixture
    def zero_profiling(self, input_sample):
        """Zero input for profiling → average activations = 0 → deltas = activations."""
        return torch.zeros_like(input_sample)

    def test_forward_pass(self, model, input_sample):
        """Verify model output."""
        output = model(input_sample)
        # [1,2] → [5,2,2] → [9,-3] → [15,0]
        expected = torch.tensor([[15.0, 0.0]])
        assert torch.allclose(output, expected)


    def test_forward_deltas_with_zero_profiling(self, model, input_sample, zero_profiling):
        """With zero profiling, deltas should equal activations."""
        slicer = Slicer(
            model, profiling_samples=zero_profiling, input_sample=input_sample
        )
        slicer.profile()
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

    def test_backward_output_initialization(self, model, input_sample, zero_profiling):
        """Test that output layer gets correct initialization."""
        slicer = Slicer(
            model, profiling_samples=zero_profiling, input_sample=input_sample
        )
        result = slicer.execute(target_index=0, theta=0.0)
        assert result["slice"] is not None

        contrib = result["slice"]["neuron_contributions"]

        # Output layer: target=0 should have 1, target=1 should have 0
        assert contrib["fc3"][0, 0] == 1.0
        assert contrib["fc3"][0, 1] == 0.0


# CNN Tests
class TestSimpleCNN:

    @pytest.fixture
    def model(self):
        return SimpleCNN()

    @pytest.fixture
    def input_sample(self):
        return torch.tensor([[[[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0]]]])

    @pytest.fixture
    def zero_profiling(self, input_sample):
        """Zero input for profiling → average activations = 0 → deltas = activations."""
        return torch.zeros_like(input_sample)

    def test_forward_pass(self, model, input_sample):
        """Verify model output."""
        output = model(input_sample)
        expected = torch.tensor([[10.0, 13.0]])
        assert torch.allclose(output, expected)

    def test_backward_input_contributions(self, model, input_sample, zero_profiling):
        """Test full backward pass to input layer (hand-calculated)."""
        slicer = Slicer(
            model, profiling_samples=zero_profiling, input_sample=input_sample
        )
        result = slicer.execute(target_index=0, theta=0.3)
        assert result["slice"] is not None

        contrib = result["slice"]["neuron_contributions"]

        expected_input = torch.tensor(
            [[[[0.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 0.0]]]]
        )

        assert torch.equal(
            contrib["input"], expected_input
        ), f"Expected:\n{expected_input}\nGot:\n{contrib['input']}"

    def test_backward_layer_contributions(self, model, input_sample, zero_profiling):
        """Test intermediate layer contributions."""
        slicer = Slicer(
            model, profiling_samples=zero_profiling, input_sample=input_sample
        )
        result = slicer.execute(target_index=0, theta=0.3)
        assert result["slice"] is not None

        contrib = result["slice"]["neuron_contributions"]

        # Pool: Channel 0 positive, Channel 1 negative
        expected_pool = torch.tensor([[[[1.0]], [[-1.0]]]])
        assert torch.equal(contrib["pool"], expected_pool)

        # ReLU: passthrough from pool via maxpool indices
        expected_relu = torch.tensor(
            [[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [-1.0, 0.0]]]]
        )
        assert torch.equal(contrib["relu"], expected_relu)

    def test_synapse_counts(self, model, input_sample, zero_profiling):
        """Verify correct number of synapses per layer."""
        slicer = Slicer(
            model, profiling_samples=zero_profiling, input_sample=input_sample
        )
        result = slicer.execute(target_index=0, theta=0.3)
        assert result["slice"] is not None

        synapses = result["slice"]["synapse_contributions"]

        assert len(synapses["fc"]) == 2
        assert len(synapses["pool"]) == 2
        assert len(synapses["conv"]) == 4

    def test_synapse_values(self, model, input_sample, zero_profiling):
        """Verify specific synapse contributions."""
        slicer = Slicer(
            model, profiling_samples=zero_profiling, input_sample=input_sample
        )
        result = slicer.execute(target_index=0, theta=0.3)
        assert result["slice"] is not None

        fc_synapses = result["slice"]["synapse_contributions"]["fc"]

        # FC: input 0 (pool ch0) → +1, input 1 (pool ch1) → -1
        fc_dict = {s["i"]: s["sign"] for s in fc_synapses}
        assert fc_dict[0] == 1.0
        assert fc_dict[1] == -1.0