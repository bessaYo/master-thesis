# core/graph/graph.py
import torch.nn as nn
from core.graph.node import Node


class Graph:
    def __init__(self, model):
        self.model = model
        self.nodes = {}
        self.edges = []

    # Build the graph from the model and forward results
    def build(self):
        self.nodes.clear()
        self.edges.clear()

        # Add input layer as first node
        previous = "input"
        self.nodes[previous] = Node(previous, None)

        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue

            node = self._create_node(name, module)
            self.nodes[name] = node
            self._extract_metadata(node)

            self._add_edge(previous, name)
            previous = name

    # Create a new node
    def _create_node(self, name, module):
        return Node(name, module)

    # Extract weights, biases, and other metadata
    def _extract_metadata(self, node):
        module = node.module

        if isinstance(module, nn.Conv2d):
            node.weight = module.weight.detach()
            node.bias = module.bias.detach() if module.bias is not None else None
            node.kernel_size = module.kernel_size
            node.stride = module.stride
            node.padding = module.padding
            node.in_channels = module.in_channels
            node.out_channels = module.out_channels

        elif isinstance(module, nn.Linear):
            node.weight = module.weight.detach()
            node.bias = module.bias.detach() if module.bias is not None else None
            node.in_features = module.in_features
            node.out_features = module.out_features

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            node.kernel_size = module.kernel_size
            node.stride = module.stride
            node.padding = module.padding

        elif isinstance(module, nn.BatchNorm2d):
            node.running_mean = module.running_mean.detach() if module.running_mean is not None else None
            node.running_var = module.running_var.detach() if module.running_var is not None else None
            node.weight = module.weight.detach() if module.weight is not None else None
            node.bias = module.bias.detach() if module.bias is not None else None

    # Add edge between parent and child nodes
    def _add_edge(self, parent_name, child_name):
        if parent_name is None:
            return

        parent = self.nodes[parent_name]
        child = self.nodes[child_name]

        self.edges.append((parent_name, child_name))
        parent.children.append(child)
        child.parents.append(parent)

    # Print a summary of the graph
    def summary(self):
        print("\n================== GRAPH SUMMARY ==================")

        print("\n-- Nodes --")
        for name, node in self.nodes.items():
            print(f"{name}: type={node.type}")

        print("\n-- Edges --")
        for parent, child in self.edges:
            print(f"{parent} -> {child}")

        print("\n-- Layer Metadata --")
        for name, node in self.nodes.items():
            if isinstance(node.module, nn.Conv2d):
                print(f"{name}: Conv2D weights={tuple(node.weight.shape)}")
                print(f"      kernel={node.kernel_size}, stride={node.stride}, pad={node.padding}")

            elif isinstance(node.module, nn.Linear):
                print(f"{name}: Linear weights={tuple(node.weight.shape)}")
                print(f"      in_features={node.in_features}, out_features={node.out_features}")

            elif isinstance(node.module, (nn.MaxPool2d, nn.AvgPool2d)):
                print(f"{name}: Pool kernel={node.kernel_size}, stride={node.stride}, pad={node.padding}")

        print("\n===================================================\n")

    def __repr__(self):
        return f"Graph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"
