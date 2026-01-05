# models/__init__.py

import torch

from .lenet import LeNet
from .simple import SimpleNN, SimpleCNN
from .resnet import ResNet10, ResNet18

MODELS = {
    "lenet": LeNet,
    "resnet10": ResNet10,
    "resnet18": ResNet18,
    "simple_nn": SimpleNN,
    "simple_cnn": SimpleCNN,
}


def get_model(name, pretrained=False):
    if name == "resnet18":
        model = ResNet18()
        path = "checkpoints/resnet18_cifar10.pth"

    elif name == "resnet10":
        model = ResNet10()
        path = "checkpoints/resnet10_cifar10.pth"

    elif name == "lenet":
        model = LeNet()
        path = "checkpoints/lenet_mnist.pth"

    else:
        raise ValueError(f"Unknown model: {name}")

    if pretrained:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model
