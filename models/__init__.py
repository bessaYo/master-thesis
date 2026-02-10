# models/__init__.py

import torch

from .lenet import LeNet
from .simple import PaperNN, SimpleCNN
from .resnet import ResNet_CIFAR
from .imagenet import IMAGENET_MODELS, load_resnet


def get_model(name, pretrained=False):
    """Get a model by name, optionally loading pretrained weights."""

    if name in IMAGENET_MODELS:
        return load_resnet(name, pretrained=pretrained)

    if name == "resnet-cifar":
        model = ResNet_CIFAR()
        path = "pretrained/checkpoints/resnet-cifar_cifar10.pt"
    elif name == "lenet":
        model = LeNet()
        path = "pretrained/checkpoints/lenet_mnist.pt"
    else:
        raise ValueError(f"Unknown model: {name}")

    if pretrained:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model
