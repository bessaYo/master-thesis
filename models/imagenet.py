# Torchvision reuses the same ReLU module for multiple activations in each block.
# For slicing we need separate ReLU instances to trace each activation.

import torch.nn as nn
import torchvision.models as tv_models

IMAGENET_MODELS = {"resnet18", "resnet34", "resnet50", "resnet101"}


class BasicBlock(nn.Module):
    """BasicBlock with separate ReLU instances for ResNet-18 and ResNet-34"""

    def __init__(self, orig_block):
        super().__init__()
        self.conv1 = orig_block.conv1
        self.bn1 = orig_block.bn1
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = orig_block.conv2
        self.bn2 = orig_block.bn2
        self.downsample = orig_block.downsample
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck with separate ReLU instances for ResNet-50 and ResNet-101"""

    def __init__(self, orig_block):
        super().__init__()
        self.conv1 = orig_block.conv1
        self.bn1 = orig_block.bn1
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = orig_block.conv2
        self.bn2 = orig_block.bn2
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = orig_block.conv3
        self.bn3 = orig_block.bn3
        self.downsample = orig_block.downsample
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu3(out)
        return out


def _replace_blocks(model, block_type):
    """Replace all blocks with slicer compatible wrappers"""
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            layer[i] = block_type(layer[i])
    return model


def load_resnet(model_name, pretrained=False):
    """Load ResNets from torchvision and replace blocks"""

    if model_name == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
        return _replace_blocks(model, BasicBlock)
    elif model_name == "resnet34":
        model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT if pretrained else None)
        return _replace_blocks(model, BasicBlock)
    elif model_name == "resnet50":
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
        return _replace_blocks(model, Bottleneck)
    elif model_name == "resnet101":
        model = tv_models.resnet101(weights=tv_models.ResNet101_Weights.DEFAULT if pretrained else None)
        return _replace_blocks(model, Bottleneck)
    else:
        raise ValueError(f"Unknown model: {model_name}")
