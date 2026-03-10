import torch
import pytest
from core.slicer import Slicer
from models.simple import PaperNN


def test_neuron_means_match_manual():
    """Mean of input samples should match hand-computed value"""
    model = PaperNN()
    slicer = Slicer(model)
    samples = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0]])
    result = slicer.profile(profiling_samples=samples)

    expected = samples.mean(dim=0)
    assert torch.allclose(result["neuron_means"]["input"], expected)


def test_all_layers_profiled():
    """Profiler should capture means for input and all three layers"""
    model = PaperNN()
    slicer = Slicer(model)
    result = slicer.profile(profiling_samples=torch.zeros(1, 2))

    assert "input" in result["neuron_means"]
    assert "fc1" in result["neuron_means"]
    assert "fc2" in result["neuron_means"]
    assert "fc3" in result["neuron_means"]


def test_layer_means_are_scalars():
    """Layer means should be scalar values"""
    model = PaperNN()
    slicer = Slicer(model)
    result = slicer.profile(profiling_samples=torch.zeros(1, 2))

    for name in ("fc1", "fc2", "fc3"):
        assert result["layer_means"][name].dim() == 0


def test_requires_samples():
    """profile() without samples or precomputed should raise RuntimeError"""
    slicer = Slicer(PaperNN(), input_sample=torch.tensor([[1.0, 2.0]]))
    with pytest.raises(RuntimeError):
        slicer.profile()
