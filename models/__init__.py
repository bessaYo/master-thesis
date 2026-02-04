# models/__init__.py

import torch

from .lenet import LeNet
from .simple import SimpleNN, SimpleCNN
from .resnet import ResNet18, ResNet34, ResNet50

MODELS = {
    "lenet": LeNet,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "simple_nn": SimpleNN,
    "simple_cnn": SimpleCNN,
}


def get_model(name, pretrained=False):
    """Get a model by name, optionally loading pretrained weights."""

    if name == "resnet18":
        model = ResNet18()
        path = "checkpoints/resnet18_cifar10.pt"
    
    elif name == "resnet34":
        model = ResNet34()
        path = "checkpoints/resnet34_cifar10.pt"

    elif name == "resnet50":
        model = ResNet50()
        path = "checkpoints/resnet50_cifar10.pt"
    elif name == "lenet":
        model = LeNet()
        path = "checkpoints/lenet_mnist.pt"

    else:
        raise ValueError(f"Unknown model: {name}")

    if pretrained:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model
