# core/graph/graph.py

import torch
import torch.nn as nn
import torch.fx as fx
import operator


class Graph:
    """Wrapper around torch.fx graph for easier traversal and analysis."""

    def __init__(self, model):
        self.model = model
        self.traced = fx.symbolic_trace(model)
        self.graph = self.traced.graph
        self.modules = dict(self.traced.named_modules())

        self._passthrough_types = {
            "method_view",
            "method_flatten",
            "method_size",
            "method_contiguous",
            "method_reshape",
            "<built-in function getitem>",
        }

    def get_nodes(self):
        """Return all nodes in the graph."""
        return list(self.graph.nodes)

    def get_module(self, node):
        """Get the nn.Module for a call_module node."""
        if node.op == "call_module":
            return self.modules.get(node.target)
        return None

    def key(self, node):
        """Get unique string key for node (used for dict lookups)."""
        if node.op == "call_module":
            return str(node.target)
        if node.op == "placeholder":
            return "input"
        return node.name

    def get_type(self, node):
        """Get semantic type of node (e.g. 'conv2d', 'relu', 'add')."""
        if node.op == "placeholder":
            return "input"

        if node.op == "output":
            return "output"

        if node.op == "call_module":
            module = self.get_module(node)
            if module is None:
                return "unknown"
            return module.__class__.__name__.lower()

        if node.op == "call_function":
            if node.target in (operator.add, torch.add):
                return "add"
            if node.target in (operator.mul, torch.mul):
                return "mul"
            if node.target == torch.flatten:
                return "flatten"
            if node.target in (torch.relu, torch.nn.functional.relu):
                return "relu"
            return str(node.target)

        if node.op == "call_method":
            return f"method_{node.target}"

        if node.op == "get_attr":
            return "attr"

        return "unknown"

    def get_parent_nodes(self, node):
        """Get direct parent nodes (inputs to this node)."""
        parents = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                parents.append(arg)
            elif isinstance(arg, (list, tuple)):
                parents.extend(x for x in arg if isinstance(x, fx.Node))
        return parents

    def is_passthrough(self, node):
        """Check if node is a passthrough (reshape, view, etc.)."""
        return self.get_type(node) in self._passthrough_types

    def skip_passthrough(self, node):
        """Follow parent chain until non-passthrough node is found."""
        while self.is_passthrough(node):
            parents = self.get_parent_nodes(node)
            if not parents:
                break
            node = parents[0]
        return node

    def get_compute_parents(self, node):
        """Get parent nodes, skipping passthrough operations."""
        parents = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                parents.append(self.skip_passthrough(arg))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, fx.Node):
                        parents.append(self.skip_passthrough(item))
        return parents

    def last_compute_node(self):
        """Get the last computational node (typically the output layer)."""
        for node in reversed(list(self.graph.nodes)):
            if node.op not in ("output", "get_attr"):
                return node
        raise RuntimeError("No compute node found")