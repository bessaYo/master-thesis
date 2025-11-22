# core/graph_builder.py

class GraphBuilder:
    @staticmethod
    def build_graph(deltas, execution_order, edges, weights, input_tensor, profiling_means):
        graph = {}

        # ----------------------------------------------------
        # 0) Shallow copy basic structures
        # ----------------------------------------------------
        # (Never return references to mutable user dicts)
        graph["deltas"] = dict(deltas)                     # layer -> Δy tensor
        graph["execution_order"] = list(execution_order)   # preserve order
        graph["edges"] = list(edges)                       # list of (parent, child)
        graph["weights"] = dict(weights)                   # layer -> weight matrix


        # ----------------------------------------------------
        # 1) Ensure input is part of execution order
        # ----------------------------------------------------
        if len(graph["execution_order"]) == 0 or graph["execution_order"][0] != "input":
            graph["execution_order"].insert(0, "input")


        # ----------------------------------------------------
        # 2) Compute Δ(input) correctly
        # ----------------------------------------------------
        # forward analyzer passes raw input batch (1, n)
        flat_input = input_tensor.flatten()

        if profiling_means is not None and "input" in profiling_means:
            # profiling_means["input"] has same shape (n,)
            mean_input = profiling_means["input"].flatten().to(flat_input.device)
            delta_input = flat_input - mean_input
        else:
            # fallback: treat Δx as raw input
            delta_input = flat_input.clone()

        graph["deltas"]["input"] = delta_input


        # ----------------------------------------------------
        # 3) Ensure Input → First-Layer edge exists
        # ----------------------------------------------------
        # first actual model layer = execution_order[1]
        if len(graph["execution_order"]) > 1:
            first_layer = graph["execution_order"][1]
            input_edge = ("input", first_layer)

            # Add only if missing
            if input_edge not in graph["edges"]:
                graph["edges"].insert(0, input_edge)


        # ----------------------------------------------------
        # 4) Everything ready
        # ----------------------------------------------------
        return graph