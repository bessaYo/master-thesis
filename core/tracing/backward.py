import time
import torch
from core.tracing.operations import BackwardOperations
from core.filtering import ChannelSlicer, BlockSlicer
from core.block import BlockStructureAnalyzer


class BackwardAnalyzer:
    """Propagates contributions backward through the graph to compute slices"""

    def __init__(self, graph, forward_result, target_index, theta=0.3, channel_alpha=None, block_beta=None):
        self.graph = graph
        self.target_index = target_index

        # Initialize instances for slicing and backward operations
        self.channel_slicer = None
        if channel_alpha is not None:
            self.channel_slicer = ChannelSlicer(alpha=channel_alpha)
        self.block_slicer = None
        if block_beta is not None:
            self.block_slicer = BlockSlicer(alpha=block_beta)
        self.ops = BackwardOperations(theta=theta)
        self.block_analyzer = None

        # Get activations and deltas from forward pass
        self.activations = forward_result["activations"]
        self.neuron_deltas = forward_result["neuron_deltas"]
        self.layer_deltas = forward_result["layer_deltas"]
        self.channel_deltas = forward_result.get("channel_deltas", {})
        self.pool_inputs = forward_result.get("pool_inputs", {})
        self.block_deltas = forward_result.get("block_deltas", {})
        self.blocks = forward_result.get("blocks", {})

        self.skip_main_path_nodes = set()
        self.backward_time = 0.0

        # Store contribution tensors for each node
        self.neuron_contributions = {}
        self.synapse_contributions = {}  

    def trace(self):
        """Main entry point. Propagates backwards from node to node and calculates contributions"""
        start = time.perf_counter()

        # Identify blocks to skip based on block energy
        self._init_block()
        skip_blocks = self.block_slicer.get_skip_blocks() if self.block_slicer else set()

        # Compute channel masks -> channels we want to keep, zero out rest
        if self.channel_slicer and self.channel_deltas:
            self.channel_slicer.compute_active_channels(self.channel_deltas)

        with torch.no_grad():
            self._init_contributions()

            # Go through graph in reverse order to propagate contributions
            for node in reversed(list(self.graph.get_nodes())):
                if self._skip_node(node):
                    continue
                self._process_node(node, skip_blocks)

        self.backward_time = time.perf_counter() - start

    # Get parent nodes for a given node and add contribution
    def _process_node(self, node, skip_blocks):
        node_key = self.graph.get_key(node)
        CONTRIB_n = self.neuron_contributions[node_key]
        delta_n = self._get_delta(node)

        # Calculate contribution for each parent
        for parent in self._get_parents(node, skip_blocks, CONTRIB_n):
            parent_key = self.graph.get_key(parent)
            delta_i = self._get_delta(parent)
            contrib = self._compute_contribution(node, CONTRIB_n, delta_n, delta_i)

            # Initiliaze zero contribution tensor if first time we reach it
            if parent_key not in self.neuron_contributions:
                self.neuron_contributions[parent_key] = torch.zeros_like(delta_i)

        # Accumulate contributions -> can have multiple children
            self.neuron_contributions[parent_key] += contrib


    # Apply backward operation based on node type
    def _compute_contribution(self, node, CONTRIB_n, delta_n, delta_i):
        node_key = self.graph.get_key(node)
        node_type = self.graph.get_type(node)

        if node_type == "linear":
            module = self.graph.get_module(node)
            activation_n = self.activations.get(node_key)
            result = self.ops.linear(module, CONTRIB_n, delta_n, delta_i, activation_n)
            # Store synapse weight contributions
            if hasattr(self.ops, '_last_synapse_contrib'):
                self.synapse_contributions[node_key] = self.ops._last_synapse_contrib
            return result

        if node_type == "conv2d":
            # For channel slicing -> zero out inactive channels before conv2d
            if self.channel_slicer:
                mask = self.channel_slicer.get_channel_mask(node_key)
                if mask is not None:
                    self.neuron_contributions[node_key][:, ~mask] = 0
            module = self.graph.get_module(node)
            activation_n = self.activations.get(node_key)
            result = self.ops.conv2d(module, CONTRIB_n, delta_n, delta_i, activation_n)
            # Store synapse weight contributions
            if hasattr(self.ops, '_last_synapse_contrib'):
                self.synapse_contributions[node_key] = self.ops._last_synapse_contrib
            return result

        if node_type == "batchnorm2d":
            return self.ops.batchnorm2d(CONTRIB_n, delta_n, delta_i)

        if node_type == "relu":
            activation = self._get_activation(node)
            return self.ops.relu(activation, CONTRIB_n, delta_n, delta_i)

        if node_type == "maxpool2d":
            module = self.graph.get_module(node)
            pool_input = self.pool_inputs.get(node_key)
            return self.ops.maxpool(module, CONTRIB_n, delta_n, delta_i, pool_input)

        if node_type in ("avgpool2d", "adaptiveavgpool2d"):
            return self.ops.avgpool(CONTRIB_n, delta_n, delta_i)

        if node_type == "add":
            return self.ops.add(CONTRIB_n, delta_n, delta_i)

        if node_type in ("flatten", "method_view", "method_flatten", "function_flatten", "builtin_flatten"):
            return self.ops.flatten(CONTRIB_n, delta_n, delta_i)

        return CONTRIB_n.clone()

    # Skip output node and nodes with zero contribution
    def _skip_node(self, node):
        if node.op in ("output", "get_attr"):  # output = return value node, get_attr = weight/constant loading
            return True

        node_key = self.graph.get_key(node)
        if node_key not in self.neuron_contributions:
            return True
        # Skip if node is in main path of a skipped block
        if node_key in self.skip_main_path_nodes:
            return True
        # Return if zero contribution
        if not self.neuron_contributions[node_key].any():
            return True

        return False

    # Get parent nodes for a given node
    def _get_parents(self, node, skip_blocks, CONTRIB_n):
        node_type = self.graph.get_type(node)

        # Add nodes need skipblock filtering on their own parents
        if node_type == "add":
            return self._get_add_parents(node, skip_blocks)

        parents = []
        for parent in self.graph.get_compute_parents(node):
            parent_type = self.graph.get_type(parent)

            # Skip flatten/view and get their parent nodes instead
            if parent_type in ("flatten", "method_view", "method_flatten"):
                parents.extend(self.graph.get_compute_parents(parent))
            elif parent_type == "add":
                parents.extend(self._get_add_parents(parent, skip_blocks))
            else:
                parents.append(parent)

        return parents

    def _get_add_parents(self, add_node, skip_blocks):
        """Get parents of an add node, skipping main path end if block is skipped"""
        block = None
        if self.block_analyzer:
            block = self.block_analyzer.get_block_for_add(add_node.name)
        block_skip = block and block in skip_blocks
        main_path_end = None
        if block:
            main_path_end = self.block_analyzer.get_main_path_end(block)

        # We keep shortcut parent only for skipped blocks
        # contributions flows through shortcut
        parents = []
        for parent in self.graph.get_parent_nodes(add_node):
            parent = self.graph.skip_passthrough(parent)
            parent_key = self.graph.get_key(parent)
            if block_skip and parent_key == main_path_end:
                continue
            parents.append(parent)
        return parents

    # Get delta for a given node
    def _get_delta(self, node):
        if node.op in ("call_function", "call_method"):  # e.g. flatten or view
            parent = self.graph.get_parent_nodes(node)[0]
            return self._get_delta(parent)

        key = self.graph.get_key(node)
        if key not in self.neuron_deltas:
            raise RuntimeError(f"No delta for node {node.name}")
        return self.neuron_deltas[key]

    # Get activation for a given node
    def _get_activation(self, node):
        key = self.graph.get_key(node)
        if key not in self.activations:
            raise RuntimeError(f"No activation for node {node.name}")
        return self.activations[key]

    # Initialize contributions for all nodes to zero, except target neuron to 1.0
    def _init_contributions(self):
        for node in self.graph.get_nodes():
            if node.op in (
                "call_module",
                "placeholder",
            ):  # placeholder = input node, call_module = any layer
                key = self.graph.get_key(node)
                # Skip for nodes we know will be skipped
                if key in self.skip_main_path_nodes:
                    continue
                if key in self.neuron_deltas:
                    self.neuron_contributions[key] = torch.zeros_like(self.neuron_deltas[key])

            # Add nodes are "call functions" and have no own delta -> initialize contribution shape based on their single parent
            elif node.op == "call_function" and self.graph.get_type(node) == "add":
                parent = self.graph.get_parent_nodes(node)[0]
                parent_key = self.graph.get_key(parent)
                self.neuron_contributions[node.name] = torch.zeros_like(self.neuron_deltas[parent_key])

        last_node = self.graph.last_compute_node()
        last_key = self.graph.get_key(last_node)

        # Target neuron starts with 1.0 contribution, rest are zero
        # dim=2 for linear (batch, classes)
        # dim=4 for conv (batch, ch, h, w)
        if self.neuron_contributions[last_key].dim() == 2:
            self.neuron_contributions[last_key][0, self.target_index] = 1.0
        elif self.neuron_contributions[last_key].dim() == 4:
            self.neuron_contributions[last_key][0, self.target_index, :, :] = 1.0

    # Initialize block analyzer and identify blocks to skip
    def _init_block(self):
        if self.blocks:
            self.block_analyzer = BlockStructureAnalyzer(self.graph, self.blocks)
            self.block_analyzer.analyze()

        if self.block_slicer and self.block_deltas and self.blocks:
            self.block_slicer.identify_skip_blocks(self.block_deltas, self.blocks)
            skip_blocks = self.block_slicer.get_skip_blocks()
            self.skip_main_path_nodes = self.block_analyzer.get_skip_nodes(skip_blocks)
