# models/__init__.py

import torch

from .resnet import ResNet_CIFAR
from .imagenet import IMAGENET_MODELS, load_resnet


def get_model(name, pretrained=False):
    """Get a model by name. Optionally with pretrained weights"""

    if name in IMAGENET_MODELS:
        return load_resnet(name, pretrained=pretrained)

    if name == "resnet_cifar":
        model = ResNet_CIFAR()
        path = "pretrained/checkpoints/resnet_cifar10.pt"
    else:
        raise ValueError(f"Unknown model: {name}")

    if pretrained:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model
