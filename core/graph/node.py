# core/graph/node.py
import torch
from typing import Optional, Tuple, Union


class Node:
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.type = module.__class__.__name__.lower() if module else "input"

        self.parents: list["Node"] = []
        self.children: list["Node"] = []

        # Weights and Bias
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

        # Convolution-specific metadata
        self.kernel_size: Optional[Tuple[int, ...]] = None
        self.stride: Optional[Tuple[int, ...]] = None
        self.padding: Optional[Union[int, Tuple[int, ...], str]] = None
        self.in_channels: Optional[int] = None
        self.out_channels: Optional[int] = None

        # Linear-specific metadata
        self.in_features: Optional[int] = None
        self.out_features: Optional[int] = None

        # BatchNorm-specific metadata
        self.running_mean: Optional[torch.Tensor] = None
        self.running_var: Optional[torch.Tensor] = None

    def __repr__(self):
        return f"Node(name={self.name}, type={self.type})"