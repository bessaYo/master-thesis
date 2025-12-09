import matplotlib.pyplot as plt
import networkx as nx

def visualize_slice(graph, slice_result, input_tensor=None):
    deltas = graph["deltas"]
    weights = graph["weights"]
    order = graph["execution_order"]
    neuron_slice = slice_result["neuron_slice"]
    synapse_slice = slice_result["synapse_slice"]
    
    G = nx.DiGraph()

    # -----------------------------------------------------
    # 0) extract *real* layers (ignore "input")
    # -----------------------------------------------------
    real_layers = [l for l in order if l != "input"]
    if not real_layers:
        raise ValueError("No non-input layers found in execution_order.")

    first_layer = real_layers[0]

    # -----------------------------------------------------
    # 1) Create INPUT nodes
    # -----------------------------------------------------
    in_features = weights[first_layer].shape[1]
    input_values = (
        input_tensor.flatten().tolist()
        if input_tensor is not None
        else None
    )
    
    for i in range(in_features):
        G.add_node(f"input.neuron_{i}", layer="input", idx=i)
    
    # -----------------------------------------------------
    # 2) Create nodes for all real layers
    # -----------------------------------------------------
    for layer in real_layers:
        for i in range(len(deltas[layer])):
            G.add_node(f"{layer}.neuron_{i}", layer=layer, idx=i)
    
    # -----------------------------------------------------
    # 3) Create edges
    # -----------------------------------------------------
    def add_edges(parent, child, W):
        out_f, in_f = W.shape
        for o in range(out_f):
            for i in range(in_f):
                src = f"{parent}.neuron_{i}"
                dst = f"{child}.neuron_{o}"
                syn_id = f"{src} -> {dst}"
                G.add_edge(
                    src, dst,
                    weight=W[o, i].item(),
                    selected=(synapse_slice.get(syn_id, 0) != 0)
                )

    # input → first layer
    add_edges("input", first_layer, weights[first_layer])

    # layer → layer edges (skip the fake "input" edges)
    for parent, child in graph["edges"]:
        if parent == "input":
            continue
        if child not in weights:
            continue
        add_edges(parent, child, weights[child])
    
    # -----------------------------------------------------
    # 4) Layout: layered, neuron_0 at top
    # -----------------------------------------------------
    layers = ["input"] + real_layers

    layer_nodes = {
        l: sorted(
            [n for n in G.nodes() if G.nodes[n]["layer"] == l],
            key=lambda n: G.nodes[n]["idx"]
        )
        for l in layers
    }

    pos = {}
    for lx, layer in enumerate(layers):
        nodes = layer_nodes[layer]
        y_start = (len(nodes) - 1) * 1.3 / 2
        for ny, node in enumerate(nodes):
            pos[node] = (lx * 2.5, y_start - ny * 1.3)

    # -----------------------------------------------------
    # 5) Draw graph
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # edges
    for u, v, d in G.edges(data=True):
        color, width = ("#ff4c4c", 2.5) if d["selected"] else ("#999999", 1.0)
        ax.annotate(
            "",
            xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->",
                color=color, lw=width,
                shrinkA=10, shrinkB=10
            )
        )

    # nodes
    node_colors = [
        "#ff4c4c" if neuron_slice.get(n, 0) != 0 else "#dddddd"
        for n in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        linewidths=1.5,
        ax=ax
    )

    # labels
    labels = {}
    for node, d in G.nodes(data=True):
        idx = d["idx"]
        layer = d["layer"]

        if layer == "input":
            labels[node] = f"{input_values[idx]:.1f}" if input_values else f"x{idx}"
        else:
            labels[node] = f"{deltas[layer][idx].item():.1f}"

    nx.draw_networkx_labels(
        G, pos, labels,
        font_size=10, font_weight="bold",
        ax=ax,
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white", ec="black", alpha=0.8
        )
    )

    ax.set_title("Dynamic Slice", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.show()