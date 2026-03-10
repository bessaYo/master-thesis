import torch
import pytest
from core.slicer import Slicer
from models.simple import PaperNN


def test_contributions_theta0_target0():
    """Neuron contributions for theta=0, target=0"""
    model = PaperNN()
    slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
    slicer.profile(profiling_samples=torch.zeros(1, 2))
    slicer.forward()
    slicer.backward(target_index=0, theta=0.0)
    contrib = slicer.backward_result["neuron_contributions"]

    assert torch.equal(contrib["fc3"], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(contrib["fc2"], torch.tensor([[1.0, -1.0]]))
    assert torch.equal(contrib["fc1"], torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.equal(contrib["input"], torch.tensor([[-1.0, 1.0]]))


def test_contributions_theta03_target0():
    """Neuron contributions for theta=0.3, target=0"""
    model = PaperNN()
    slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
    slicer.profile(profiling_samples=torch.zeros(1, 2))
    slicer.forward()
    slicer.backward(target_index=0, theta=0.3)
    contrib = slicer.backward_result["neuron_contributions"]

    assert torch.equal(contrib["fc3"], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(contrib["fc2"], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(contrib["fc1"], torch.tensor([[1.0, 0.0, 1.0]]))
    assert torch.equal(contrib["input"], torch.tensor([[1.0, 2.0]]))


def test_slice_counts():
    """Slice neuron and total neuron counts should match manual calculation"""
    model = PaperNN()
    slicer = Slicer(model, input_sample=torch.tensor([[1.0, 2.0]]))
    slicer.profile(profiling_samples=torch.zeros(1, 2))
    slicer.forward()
    result = slicer.backward(target_index=0, theta=0.0)

    # PaperNN: input(2) + fc1(3) + fc2(2) + fc3(2) = 9 total
    assert result["total_neurons"] == 9
    # 6 neurons have nonzero contribution for theta=0, target=0
    assert result["slice_neurons"] == 6


def test_requires_forward():
    """backward() without forward should raise RuntimeError"""
    slicer = Slicer(PaperNN(), input_sample=torch.tensor([[1.0, 2.0]]))
    slicer.profile(profiling_samples=torch.zeros(1, 2))
    with pytest.raises(RuntimeError):
        slicer.backward(target_index=0)
