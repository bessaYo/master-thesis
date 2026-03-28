SHORTCUT_PATTERNS = ["shortcut", "downsample"]
MAIN_PATH_END_PATTERNS = ["bn2", "bn3"]


class BlockStructureAnalyzer:
    """Analyzes ResNet block structure from the fx graph to separate main path from shortcut nodes"""

    def __init__(self, graph, blocks):
        self.graph = graph
        self.blocks = blocks

        self.block_info = {}
        self.add_to_block = {}  # add_node_name -> block_name

    def analyze(self):
        """Finds the add node, main path and shortcut for each block"""
        # First build a child map so we can find what comes after each add node
        child_map = {}
        for node in self.graph.get_nodes():
            for p in self.graph.get_parent_nodes(node):
                child_map.setdefault(p, []).append(node)

        # Collect all add nodes with their parents and children
        add_nodes = {}
        for node in self.graph.get_nodes():
            if self.graph.get_type(node) != "add":
                continue
            parents = [self.graph.get_key(self.graph.skip_passthrough(p))
                       for p in self.graph.get_parent_nodes(node)]
            children = [self.graph.get_key(c) for c in child_map.get(node, [])]
            add_nodes[node.name] = {"parents": parents, "children": children}

        # Match each block to its add node
        for block_name, block_layers in self.blocks.items():
            info = self._match_block(block_name, block_layers, add_nodes)
            if info:
                self.block_info[block_name] = info
                self.add_to_block[info["add_node"]] = block_name

        return self.block_info, self.add_to_block

    def get_skip_nodes(self, skip_blocks):
        """Returns main path nodes that should be skipped during backward"""
        skip_nodes = set()
        for block_name in skip_blocks:
            if block_name not in self.block_info:
                continue
            info = self.block_info[block_name]
            skip_nodes.update(info["main_path_nodes"])
            # Keep relu node after add node
            if info["post_add_node"]:
                skip_nodes.discard(info["post_add_node"])
        return skip_nodes

    def get_block_for_add(self, add_node_name):
        return self.add_to_block.get(add_node_name)

    def get_main_path_end(self, block_name):
        if block_name in self.block_info:
            return self.block_info[block_name]["main_path_end"]
        return None

    def _match_block(self, block_name, block_layers, add_nodes):
        """Matches a block to its add node and extracts the structure"""
        for add_name, add_info in add_nodes.items():
            parents = add_info["parents"]

            block_parent = None
            shortcut_parent = None
            for parent in parents:
                if parent.startswith(block_name + "."):
                    if _is_shortcut(parent):
                        shortcut_parent = parent
                    else:
                        block_parent = parent
                else:
                    # External parent = identity shortcut (no conv downsample)
                    shortcut_parent = parent

            if not (block_parent and shortcut_parent):
                continue
            if not _is_main_path_end(block_parent):
                continue

            # Found the matching add node for this block
            post_add = add_info["children"][0] if add_info["children"] else None

            # Main path = everything except shortcut layers and post-add relu
            main_path_nodes = set()
            for layer in block_layers:
                if not _is_shortcut(layer) and layer != post_add:
                    main_path_nodes.add(layer)

            return {
                "add_node": add_name,
                "main_path_end": block_parent,
                "post_add_node": post_add,
                "main_path_nodes": main_path_nodes,
            }

        return None


def _is_shortcut(layer_name):
    return any(p in layer_name.lower() for p in SHORTCUT_PATTERNS)


def _is_main_path_end(layer_name):
    """Checks if this is the last layer on the main path (bn2 or bn3)"""
    return any(p in layer_name for p in MAIN_PATH_END_PATTERNS)
