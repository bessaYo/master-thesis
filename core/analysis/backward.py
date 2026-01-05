import time
from typing import Dict, List, Any, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from core.graph import Graph
from core.analysis.operations import BackwardOperations
from core.analysis.block_filter import ChannelBlockFilter
from utils.logging import SlicerLogger


class BackwardAnalyzer:
    """
    Performs backward contribution tracing.
    Propagates neuron-level contributions (CONTRIB) backward from
    a given target neuron, following the computational graph in reverse order.
    Local contribution rules are delegated to BackwardOperations.
    """

    def __init__(
        self,
        graph: Graph,
        forward_result: Dict[str, Any],
        target_index: int,
        theta: float = 0.3,
        filter_channels: bool = False,
        filter_percent: float = 0.2,
        debug: bool = False,
    ):

        self.graph = graph
        self.target_index = target_index

        # Optional channel-level block filter (applied before local conv operations)
        self.channel_filter = (
            ChannelBlockFilter(percent=filter_percent) if filter_channels else None
        )

        # Forward pass information
        self.activations = forward_result["activations"]
        self.neuron_deltas = forward_result["neuron_deltas"]
        self.layer_deltas = forward_result["layer_deltas"]
        self.channel_deltas = forward_result.get("channel_deltas", {})
        self.pool_indices = forward_result.get("pool_indices", {})

        # Handler for operator-local backward contribution rules
        self.ops = BackwardOperations(theta=theta)

        # Accumulated neuron and synapse contributions
        self.neuron_contributions: Dict[str, torch.Tensor] = {}
        self.synapse_contributions: Dict[str, List[Dict[str, Any]]] = {}

        # Timing information
        self.backward_time_sec: float = 0.0

        # Optional logger for debugging and visualization
        self.logger = SlicerLogger(enabled=debug)

    # Main backward tracing routine
    def trace(self):
        # Log basic model and slicing information
        self.logger.model_summary(
            self.graph, self.neuron_deltas, self.target_index, self.ops.theta
        )
        self.logger.layer_table(self.graph, self.neuron_deltas, self._key)

        start = time.perf_counter()

        # Initialize contribution tensors for all relevant nodes
        self._init_contributions()

        # Reverse topological traversal of the graph
        for node in reversed(list(self.graph.get_nodes())):
            if node.op in ("output", "get_attr"):
                continue

            node_key = self._key(node)
            node_type = self.graph.get_type(node)

            # Skip nodes without registered contributions
            if node_key not in self.neuron_contributions:
                continue

            CONTRIB_n = self.neuron_contributions[node_key]

            # Skip nodes whose contribution tensor is entirely zero
            if (CONTRIB_n != 0).sum().item() == 0:
                continue

            # delta_n corresponds to Δy: relative output activation of this node
            delta_n = self._get_delta(node)

            # Determine parent nodes that receive propagated contributions
            parents = self.graph.get_compute_parents(node)
            expanded_parents: List[fx.Node] = []

            # Special handling for additive fan-in (residual connections)
            for p in parents:
                if self.graph.get_type(p) == "add":
                    for ap in self.graph.get_parent_nodes(p):
                        expanded_parents.append(self.graph.skip_passthrough(ap))

                    # Propagate contribution to the add node itself
                    add_key = self._key(p)
                    if add_key in self.neuron_contributions:
                        self.neuron_contributions[add_key] += CONTRIB_n
                else:
                    expanded_parents.append(p)

            # Propagate contributions to all expanded parents
            for parent in expanded_parents:
                parent_key = self._key(parent)

                if parent_key not in self.neuron_deltas:
                    continue

                # delta_i corresponds to Δx: relative input activation of the operator
                delta_i = self._get_delta(parent)

                # Dispatch operator-specific backward contribution rule
                syn, contrib_parent = self._backward_dispatch(
                    node, parent, CONTRIB_n, delta_n, delta_i
                )

                # Accumulate synapse-level contributions
                if syn:
                    self.synapse_contributions.setdefault(node_key, []).extend(syn)

                # Accumulate neuron-level contributions
                self.neuron_contributions.setdefault(
                    parent_key, torch.zeros_like(delta_i)
                )
                self.neuron_contributions[parent_key] += contrib_parent

        self.backward_time_sec = time.perf_counter() - start

        # Final logging of results
        self.logger.results(
            self.graph,
            self.neuron_contributions,
            self.synapse_contributions,
            self.backward_time_sec,
            self._key,
            self.neuron_deltas,
        )

    # Operator-specific backward contribution dispatch
    # delta_n : Δy (output delta of the current node)
    # delta_i : Δx (input delta of the parent node)
    def _backward_dispatch(
        self,
        node: fx.Node,
        parent: fx.Node,
        CONTRIB_n: torch.Tensor,
        delta_n: torch.Tensor,
        delta_i: torch.Tensor,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:

        node_type = self.graph.get_type(node)
        node_key = self._key(node)

        if node_type == "linear":
            module = self.graph.get_module(node)
            return self.ops.linear(module, CONTRIB_n, delta_n, delta_i)

        if node_type == "conv2d":
            module = self.graph.get_module(node)
            active_channels = None

            # Optional channel-level block filtering
            if self.channel_filter and node_key in self.channel_deltas:
                active_channels = self.channel_filter.get_active_blocks(
                    self.channel_deltas[node_key]
                )

            return self.ops.conv2d(module, CONTRIB_n, delta_n, delta_i, active_channels)

        if node_type == "batchnorm2d":
            return self.ops.batchnorm2d(CONTRIB_n, delta_n, delta_i)

        if node_type == "relu":
            return self.ops.relu(
                self._get_activation(node), CONTRIB_n, delta_n, delta_i
            )

        if node_type == "maxpool2d":
            return self.ops.maxpool2d(
                self.pool_indices[node_key], CONTRIB_n, delta_n, delta_i
            )

        if node_type == "avgpool2d":
            module = self.graph.get_module(node)
            return self.ops.avgpool2d(CONTRIB_n, delta_n, delta_i, module)

        if node_type == "adaptiveavgpool2d":
            return self.ops.avgpool2d(CONTRIB_n, delta_n, delta_i, None)

        if node_type == "add":
            return self.ops.add(CONTRIB_n, delta_n, delta_i)

        if node_type in ("flatten", "method_view", "method_flatten"):
            return self.ops.flatten(CONTRIB_n, delta_i)

        # Fallback: identity propagation
        return [], CONTRIB_n.clone()

    # Returns a stable string identifier for a graph node. Used to index activations, deltas, and contributions.
    def _key(self, node: fx.Node) -> str:
        if node.op == "call_module":
            return str(node.target)
        if node.op == "placeholder":
            return "input"
        return node.name

    # Retrieves the relative activation delta for a node.
    # For call_function / call_method nodes, assumes single-input semantics and propagates the delta of the first parent.
    def _get_delta(self, node: fx.Node) -> torch.Tensor:
        if node.op == "placeholder":
            return self.neuron_deltas["input"]

        if node.op == "call_module":
            return self.neuron_deltas[str(node.target)]

        if node.op in ("call_function", "call_method"):
            parents = self.graph.get_parent_nodes(node)
            if parents:
                return self._get_delta(parents[0])

        raise KeyError(f"No delta for node {node.name}")

    # Retrieves the activation tensor for a node.
    def _get_activation(self, node: fx.Node) -> torch.Tensor:
        if node.op == "call_module":
            return self.activations[str(node.target)]
        if node.op == "placeholder":
            return self.neuron_deltas["input"]
        raise RuntimeError(f"No activation for node {node.name}")

    # Identifies the last compute node in the graph (before output)
    def _get_last_compute_node(self) -> fx.Node:
        for node in reversed(list(self.graph.get_nodes())):
            if node.op not in ("output", "get_attr"):
                return node
        raise RuntimeError("No compute node found")

    # Initializes neuron contribution tensors and sets slicing criterion
    def _init_contributions(self):
        for node in self.graph.get_nodes():
            if node.op in ("call_module", "placeholder"):
                key = self._key(node)
                if key in self.neuron_deltas:
                    self.neuron_contributions[key] = torch.zeros_like(
                        self.neuron_deltas[key]
                    )

            elif node.op == "call_function" and self.graph.get_type(node) == "add":
                parent = self.graph.get_parent_nodes(node)[0]
                parent_key = self._key(parent)
                self.neuron_contributions[node.name] = torch.zeros_like(
                    self.neuron_deltas[parent_key]
                )

        # Initialize slicing criterion at the output layer
        last_node = self._get_last_compute_node()
        last_key = self._key(last_node)
        contrib = self.neuron_contributions[last_key]

        if contrib.dim() == 2:
            contrib[0, self.target_index] = 1.0
        elif contrib.dim() == 4:
            contrib[0, self.target_index, :, :] = 1.0
