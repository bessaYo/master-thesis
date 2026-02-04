# core/blocks/block_structure.py


class BlockStructureAnalyzer:
    """Analyzes ResNet block structure from graph."""

    SHORTCUT_PATTERNS = ["shortcut", "downsample", "skip", "projection"]
    MAIN_PATH_END_PATTERNS = ["bn2", "bn3", "norm2", "norm3"]

    def __init__(self, graph, blocks, debug=False):
        self.graph = graph
        self.blocks = blocks
        self.debug = debug

        self.block_info = {}
        self.add_to_block = {}

    def analyze(self):
        """Analyze all blocks, return block_info and add_to_block mappings."""
        add_nodes_info = self._find_add_nodes()

        for block_name, block_layers in self.blocks.items():
            info = self._analyze_single_block(block_name, block_layers, add_nodes_info)
            if info:
                self.block_info[block_name] = info
                if info["add_node"]:
                    self.add_to_block[info["add_node"]] = block_name

        if self.debug:
            self._print_debug_info()

        return self.block_info, self.add_to_block

    def get_skip_nodes(self, skip_blocks):
        """Given skip_blocks, return set of main path nodes to skip."""
        skip_nodes = set()
        for block_name in skip_blocks:
            if block_name in self.block_info:
                info = self.block_info[block_name]
                skip_nodes.update(info["main_path_nodes"])
                if info["post_add_node"]:
                    skip_nodes.discard(info["post_add_node"])

        return skip_nodes

    def get_block_for_add(self, add_node_name):
        """Get block name for a given add node."""
        return self.add_to_block.get(add_node_name)

    def get_main_path_end(self, block_name):
        """Get the main path end node for a block."""
        if block_name in self.block_info:
            return self.block_info[block_name]["main_path_end"]
        return None

    def _find_add_nodes(self):
        """Find all add nodes and their parent/child connections."""
        add_nodes = {}

        for node in self.graph.get_nodes():
            if self.graph.get_type(node) == "add":
                parents = []
                for p in self.graph.get_parent_nodes(node):
                    actual = self.graph.skip_passthrough(p)
                    parents.append(self.graph.key(actual))

                children = []
                for other_node in self.graph.get_nodes():
                    if node in self.graph.get_parent_nodes(other_node):
                        children.append(self.graph.key(other_node))

                add_nodes[node.name] = {
                    "parents": parents,
                    "children": children,
                    "node": node,
                }

        return add_nodes

    def _analyze_single_block(self, block_name, block_layers, add_nodes_info):
        """Analyze a single residual block's structure."""
        matching_add = None
        main_path_end = None
        shortcut_input = None

        for add_name, add_info in add_nodes_info.items():
            parents = add_info["parents"]

            block_parent = None
            external_parent = None

            for parent in parents:
                if self._belongs_to_block(parent, block_name):
                    # Check if it's a shortcut layer
                    if self._is_shortcut_layer(parent):
                        shortcut_parent = parent
                    else:
                        block_parent = parent
                else:
                    # External parent (identity shortcut case)
                    shortcut_parent = parent

            if block_parent and shortcut_parent:
                if self._is_main_path_end(block_parent, block_layers):
                    matching_add = add_name
                    main_path_end = block_parent
                    shortcut_input = shortcut_parent
                    break

        if not matching_add:
            return None

        # Find post-add node (usually relu)
        post_add_node = None
        add_info = add_nodes_info[matching_add]
        if add_info["children"]:
            post_add_node = add_info["children"][0]

        # Identify main path nodes
        main_path_nodes = set()
        for layer in block_layers:
            if not self._is_shortcut_layer(layer) and layer != post_add_node:
                main_path_nodes.add(layer)

        return {
            "add_node": matching_add,
            "main_path_end": main_path_end,
            "shortcut_input": shortcut_input,
            "post_add_node": post_add_node,
            "main_path_nodes": main_path_nodes,
        }

    def _belongs_to_block(self, layer_name, block_name):
        """Check if layer belongs to block."""
        return layer_name.startswith(block_name + ".")

    def _is_main_path_end(self, layer_name, block_layers):
        """Check if layer is end of main path (bn2/bn3)."""
        for pattern in self.MAIN_PATH_END_PATTERNS:
            if pattern in layer_name:
                return True

        if "bn" in layer_name.lower() or "norm" in layer_name.lower():
            try:
                idx = block_layers.index(layer_name)
                if idx >= len(block_layers) * 0.5:
                    return True
            except ValueError:
                pass
        return False

    def _is_shortcut_layer(self, layer_name):
        """Check if layer is part of shortcut connection."""
        return any(p in layer_name.lower() for p in self.SHORTCUT_PATTERNS)

    def _print_debug_info(self):
        """Print analyzed block structure for debugging."""
        print("[DEBUG] Analyzed block structure:")
        for block_name, info in self.block_info.items():
            print(f"  {block_name}:")
            print(f"    add_node: {info['add_node']}")
            print(f"    main_path_end: {info['main_path_end']}")
            print(f"    shortcut_input: {info['shortcut_input']}")
            print(f"    post_add_node: {info['post_add_node']}")