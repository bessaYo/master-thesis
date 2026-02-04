# core/tracing/backward.py

import time
import torch

from core.graph import Graph
from core.tracing.operations import BackwardOperations
from core.blocks.block_filter import ChannelBlockFilter, ResBlockFilter
from core.blocks.block_structure import BlockStructureAnalyzer
from utils.logging import SlicerLogger


class BackwardAnalyzer:
    """Propagates contributions backward through the graph to compute slices."""

    def __init__(
        self,
        graph,
        forward_result,
        target_index,
        theta=0.3,
        channel_mode=False,
        block_mode=False,
        channel_alpha=0.8,
        block_beta=0.9,
        debug=False,
    ):
        self.graph = graph
        self.target_index = target_index
        self.debug = debug

        # Filters
        self.channel_filter = channel_mode and ChannelBlockFilter(alpha=channel_alpha)
        self.block_filter = block_mode and ResBlockFilter(alpha=block_beta)

        # Backward operations handler
        self.ops = BackwardOperations(theta=theta)

        # Forward pass data
        self.activations = forward_result["activations"]
        self.neuron_deltas = forward_result["neuron_deltas"]
        self.layer_deltas = forward_result["layer_deltas"]
        self.channel_deltas = forward_result.get("channel_deltas", {})
        self.block_deltas = forward_result.get("block_deltas", {})
        self.blocks = forward_result.get("blocks", {})
        self.pool_indices = forward_result.get("pool_indices", {})

        # Contributions
        self.neuron_contributions = {}
        self.synapse_contributions = {}

        # Block structure
        self.block_analyzer = None
        self.skip_main_path_nodes = set()

        # Timing
        self.backward_time_sec = 0.0
        self.logger = SlicerLogger(enabled=debug)

        # Analyze blocks if present
        if self.blocks:
            self.block_analyzer = BlockStructureAnalyzer(graph, self.blocks, debug)
            self.block_analyzer.analyze()

        # Identify skip nodes if block filtering enabled
        if self.block_filter and self.block_deltas and self.blocks:
            self.block_filter.identify_skip_blocks(self.block_deltas, self.blocks)
            skip_blocks = self.block_filter.get_skip_blocks()
            self.skip_main_path_nodes = self.block_analyzer.get_skip_nodes(skip_blocks)

    def trace(self):
        """Main entry point. Propagates contributions from target neuron backward."""
        self._log_start()
        skip_blocks = self._get_skip_blocks()

        start = time.perf_counter()
        self._init_contributions()

        for node in reversed(list(self.graph.get_nodes())):
            if self._should_skip(node):
                node_type = self.graph.get_type(node)
                print(f"[DEBUG] Skipping Node: {node.name}. Type: {node_type}") if self.debug else None
                continue
            self._propagate_contributions(node, skip_blocks)

        self.backward_time_sec = time.perf_counter() - start
        self._log_results()

    def _should_skip(self, node):
        if node.op in ("output", "get_attr"):
            return True

        node_key = self.graph.key(node)

        if node_key not in self.neuron_contributions:
            return True

        if node_key in self.skip_main_path_nodes:
            return True

        contrib_sum = (self.neuron_contributions[node_key] != 0).sum().item()
        if contrib_sum == 0:
            return True

        return False

    def _propagate_contributions(self, node, skip_blocks):
        """Propagate contributions from node to its parents."""
        node_key = self.graph.key(node)
        CONTRIB_n = self.neuron_contributions[node_key]
        delta_n = self._get_delta(node)

        for parent in self._get_parents(node, skip_blocks, CONTRIB_n):
            parent_key = self.graph.key(parent)

            if parent_key not in self.neuron_deltas:
                continue

            delta_i = self._get_delta(parent)
            syn, contrib = self._backward_dispatch(node, parent, CONTRIB_n, delta_n, delta_i)

            if syn:
                self.synapse_contributions.setdefault(node_key, []).extend(syn)

            self.neuron_contributions.setdefault(parent_key, torch.zeros_like(delta_i))
            self.neuron_contributions[parent_key] += contrib

    def _get_parents(self, node, skip_blocks, CONTRIB_n):
        """Get parents to propagate to. Skips flatten and expands add nodes."""
        parents = []

        for p in self.graph.get_compute_parents(node):
            p_type = self.graph.get_type(p)
            
            # Skip through flatten - get its parent instead
            if p_type in ("flatten", "method_view", "method_flatten"):
                parents.extend(self.graph.get_compute_parents(p))
            elif p_type == "add":
                parents.extend(self._expand_add_node(p, skip_blocks, CONTRIB_n))
            else:
                parents.append(p)

        return parents

    def _expand_add_node(self, add_node, skip_blocks, CONTRIB_n):
        """Expand add node to its parents. Blocks main path if block is skipped."""
        add_key = self.graph.key(add_node)
        expanded = []

        skipped_block = self.block_analyzer.get_block_for_add(add_node.name) if self.block_analyzer else None
        should_skip_main = skipped_block in skip_blocks if skipped_block else False
        main_path_end = self.block_analyzer.get_main_path_end(skipped_block) if skipped_block else None

        for ap in self.graph.get_parent_nodes(add_node):
            ap_actual = self.graph.skip_passthrough(ap)
            ap_key = self.graph.key(ap_actual)

            if should_skip_main and ap_key == main_path_end:
                continue
            expanded.append(ap_actual)

        if add_key in self.neuron_contributions:
            self.neuron_contributions[add_key] += CONTRIB_n

        return expanded

    def _backward_dispatch(self, node, parent, CONTRIB_n, delta_n, delta_i):
        """Dispatch to operator-specific backward contribution rule."""
        node_type = self.graph.get_type(node)
        node_key = self.graph.key(node)

        print(f"[DEBUG] Backward - Processing Node: {node.name}, Type: {node_type}") if self.debug else None

        if node_type == "linear":
            module = self.graph.get_module(node)
            activation_n = self.activations.get(node_key)
            return self.ops.linear(module, CONTRIB_n, delta_n, delta_i, activation_n)

        if node_type == "conv2d":
            module = self.graph.get_module(node)
            active_channels = None

            if self.channel_filter and node_key in self.channel_deltas:
                contrib_channels = set(
                    c for c in range(CONTRIB_n.shape[1]) if (CONTRIB_n[0, c] != 0).any()
                )
                active_channels = self.channel_filter.get_active_channels(
                    self.channel_deltas[node_key], contrib_channels
                )

                # Zero out non-active channels and update CONTRIB_n for conv2d
                if active_channels is not None:
                    mask = torch.zeros(CONTRIB_n.shape[1], dtype=torch.bool, device=CONTRIB_n.device)
                    for ch in active_channels:
                        mask[ch] = True
                    CONTRIB_n = CONTRIB_n * mask.view(1, -1, 1, 1)
                    self.neuron_contributions[node_key] = CONTRIB_n

            activation_n = self.activations.get(node_key)
            return self.ops.conv2d(module, CONTRIB_n, delta_n, delta_i, active_channels, activation_n)

        if node_type == "batchnorm2d":
            return self.ops.batchnorm2d(CONTRIB_n, delta_n, delta_i)

        if node_type == "relu":
            activation = self._get_activation(node)
            return self.ops.relu(activation, CONTRIB_n, delta_n, delta_i)

        if node_type == "maxpool2d":
            return self.ops.maxpool2d(self.pool_indices[node_key], CONTRIB_n, delta_n, delta_i)

        if node_type == "avgpool2d":
            module = self.graph.get_module(node)
            return self.ops.avgpool2d(CONTRIB_n, delta_n, delta_i, module)

        if node_type == "adaptiveavgpool2d":
            return self.ops.avgpool2d(CONTRIB_n, delta_n, delta_i, None)

        if node_type == "add":
            return self.ops.add(CONTRIB_n, delta_n, delta_i)

        if node_type in ("flatten", "method_view", "method_flatten", "function_flatten", "builtin_flatten"):
            return self.ops.flatten(CONTRIB_n, delta_n, delta_i)

        return [], CONTRIB_n.clone()

    def _get_delta(self, node):
        """Get delta tensor for node."""
        if node.op in ("call_function", "call_method"):
            parent = self.graph.get_parent_nodes(node)[0]
            return self._get_delta(parent)
        
        key = self.graph.key(node)
        if key not in self.neuron_deltas:
            raise RuntimeError(f"No delta for node {node.name}")
        return self.neuron_deltas[key]

    def _get_activation(self, node):
        """Get activation tensor for node."""
        key = self.graph.key(node)
        if key not in self.activations:
            raise RuntimeError(f"No activation for node {node.name}")
        return self.activations[key]

    def _init_contributions(self):
        """Initialize all contributions to zero, set target neuron to 1.0."""
        for node in self.graph.get_nodes():
            if node.op in ("call_module", "placeholder"):
                key = self.graph.key(node)
                if key in self.neuron_deltas:
                    self.neuron_contributions[key] = torch.zeros_like(self.neuron_deltas[key])

            elif node.op == "call_function" and self.graph.get_type(node) == "add":
                parent = self.graph.get_parent_nodes(node)[0]
                parent_key = self.graph.key(parent)
                self.neuron_contributions[node.name] = torch.zeros_like(self.neuron_deltas[parent_key])

        last_node = self.graph.last_compute_node()
        last_key = self.graph.key(last_node)

        # Direkt ins Dict schreiben statt über Variable
        if self.neuron_contributions[last_key].dim() == 2:
            self.neuron_contributions[last_key][0, self.target_index] = 1.0
        elif self.neuron_contributions[last_key].dim() == 4:
            self.neuron_contributions[last_key][0, self.target_index, :, :] = 1.0

    def _get_skip_blocks(self):
        """Get set of blocks to skip."""
        if self.block_filter:
            return self.block_filter.get_skip_blocks()
        return set()

    def _log_start(self):
        """Log model summary and layer table."""
        self.logger.model_summary(
            self.graph, self.neuron_deltas, self.target_index, self.ops.theta
        )
        self.logger.layer_table(self.graph, self.neuron_deltas, self.graph.key)

    def _log_results(self):
        """Log final results."""
        self.logger.results(
            self.graph,
            self.neuron_contributions,
            self.synapse_contributions,
            self.backward_time_sec,
            self.graph.key,
            self.neuron_deltas,
        )