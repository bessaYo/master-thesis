# core/graph/fx_graph.py

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Optional, List, cast
import operator


class Graph:
    def __init__(self, model: nn.Module):
        self.model = model
        self.traced = fx.symbolic_trace(model)
        self.graph = self.traced.graph
        self.modules = dict(self.traced.named_modules())

        # Pass-through node types (reshape, indexing, etc.)
        self._passthrough_types = {
            "method_view",
            "method_flatten",
            "method_size",
            "method_contiguous",
            "method_reshape",
            "<built-in function getitem>",
        }

    def get_nodes(self) -> List[fx.Node]:
        return list(self.graph.nodes)

    def get_module(self, node: fx.Node) -> Optional[nn.Module]:
        if node.op == "call_module":
            assert isinstance(node.target, str)
            return self.modules.get(node.target)
        return None

    def key(self, node: fx.Node) -> str:
        if node.op == "call_module":
            assert isinstance(node.target, str)
            return cast(str, node.target)

        if node.op == "placeholder":
            return "input"

        return node.name

    def get_type(self, node: fx.Node) -> str:
        if node.op == "placeholder":
            return "input"

        elif node.op == "output":
            return "output"

        elif node.op == "call_module":
            module = self.get_module(node)
            if module is None:
                return "unknown"
            return module.__class__.__name__.lower()

        elif node.op == "call_function":
            if node.target in (operator.add, torch.add):
                return "add"
            elif node.target in (operator.mul, torch.mul):
                return "mul"
            elif node.target == torch.flatten:
                return "flatten"
            elif node.target in (torch.relu, torch.nn.functional.relu):
                return "relu"
            return str(node.target)

        elif node.op == "call_method":
            return f"method_{node.target}"

        elif node.op == "get_attr":
            return "attr"

        return "unknown"

    def get_parent_nodes(self, node: fx.Node) -> List[fx.Node]:
        parents = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                parents.append(arg)
            elif isinstance(arg, (list, tuple)):
                parents.extend(x for x in arg if isinstance(x, fx.Node))
        return parents

    def is_passthrough(self, node: fx.Node) -> bool:
        return self.get_type(node) in self._passthrough_types

    def skip_passthrough(self, node: fx.Node) -> fx.Node:
        while self.is_passthrough(node):
            parents = self.get_parent_nodes(node)
            if not parents:
                break
            node = parents[0]
        return node

    def get_compute_parents(self, node: fx.Node) -> List[fx.Node]:
        parents = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                parents.append(self.skip_passthrough(arg))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, fx.Node):
                        parents.append(self.skip_passthrough(item))
        return parents

    def last_compute_node(self) -> fx.Node:
        for node in reversed(list(self.graph.nodes)):
            if node.op not in ("output", "get_attr"):
                return node
        raise RuntimeError("No compute node found")
