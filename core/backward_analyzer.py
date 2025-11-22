import math
from core.neuron_operations import weighted_sum_operation


class BackwardAnalyzer:
    def __init__(self, graph, theta: float = 0.2, debug: bool = True):
        """
        BackwardAnalyzer:
        - graph: output of ForwardAnalyzer / GraphBuilder
        - theta: threshold for filtering small local contributions
        - debug: if True, prints detailed backtracking logs
        """
        self.deltas = graph["deltas"]              # Δy per layer (incl. "input")
        self.weights = graph["weights"]            # weight matrices per layer
        self.edges = graph["edges"]                # list of (parent, child) layer names
        self.order = graph["execution_order"]      # execution order incl. "input"
        self.theta = theta
        self.debug = debug

        # global CONTRIB tables for neurons and synapses (slice result)
        self.neuron_CONTRIB = {}
        self.synapse_CONTRIB = {}

    # =====================================================================
    # Main entrypoint: compute dynamic slice
    # =====================================================================
    def compute_slice(self, target_neurons, input_tensor=None):
        """
        Compute a backward slice for a given list of target neurons.
        target_neurons: list like ["fc3.neuron_0"]
        input_tensor:   optional, only used for debug printing
        """

        if self.debug:
            print("\n========== COMPUTE SLICE ==========")
            print(f"Target neurons: {target_neurons}")
            if input_tensor is not None:
                print(f"Input tensor (for reference): {input_tensor}")

        # determine input size from Δ(input)
        if "input" not in self.deltas:
            raise KeyError(
                "Graph does not contain 'input' deltas. "
                "Make sure GraphBuilder.add_input_deltas was used."
            )
        input_size = len(self.deltas["input"])

        # initialize neuron and synapse CONTRIB tables (including input layer)
        self.initialize_contributions(input_size)

        # CONTRIB marks which neurons are active for backtracking (0 or 1)
        CONTRIB = {nid: 0.0 for nid in self.neuron_CONTRIB.keys()}

        # mark target neurons with CONTRIB = 1
        for t in target_neurons:
            if t not in CONTRIB:
                raise KeyError(f"Target neuron {t} not found in CONTRIB table.")
            CONTRIB[t] = 1.0
            self.neuron_CONTRIB[t] = 1
            if self.debug:
                print(f"→ Mark target {t} = 1")

        # =================================================================
        # Backward pass over layers (from last to first)
        # =================================================================
        for layer in reversed(self.order):

            # we stop at the input layer: no parents before input
            if layer == "input":
                continue

            if self.debug:
                print(f"\n--- Backtracking layer: {layer} ---")

            num_neurons = len(self.deltas[layer])

            for n_idx in range(num_neurons):
                n_id = f"{layer}.neuron_{n_idx}"
                CONTRIB_n = CONTRIB.get(n_id, 0.0)

                # if this neuron is not part of the current slice frontier → skip
                if CONTRIB_n == 0.0:
                    continue

                delta_n = self.deltas[layer][n_idx].item()
                if math.isclose(delta_n, 0.0, abs_tol=1e-8):
                    # if Δy is ~0, influence ratios become ill-defined → skip
                    continue

                if self.debug:
                    print(f"\nNeuron {n_id}: CONTRIB={CONTRIB_n:.4f}, Δy={delta_n:.4f}")

                # find parent layers
                parents = [p for (p, c) in self.edges if c == layer]

                if len(parents) == 0:
                    # direct successor of input
                    parent_layer = "input"
                else:
                    parent_layer = parents[0]

                if parent_layer not in self.deltas:
                    raise KeyError(
                        f"Parent layer '{parent_layer}' has no deltas in graph['deltas']."
                    )

                if layer not in self.weights:
                    # need to handle later
                    if self.debug:
                        print(f"  (skip layer {layer}: no weights present)")
                    continue

                W = self.weights[layer]               # shape [out_features, in_features]
                in_features = W.shape[1]
                delta_parent = self.deltas[parent_layer]

                local_contribs = []

                # compute local contributions for all parent neurons
                for i_idx in range(in_features):
                    w_i = W[n_idx, i_idx].item()
                    delta_i = delta_parent[i_idx].item()

                    # local contribution (only used for magnitude + sign)
                    contrib_i = weighted_sum_operation(CONTRIB_n, delta_n, w_i, delta_i)
                    i_id = f"{parent_layer}.neuron_{i_idx}"

                    local_contribs.append((i_id, contrib_i, w_i, delta_i))

                    if self.debug:
                        print(
                            f"  local {i_id}: w={w_i:.2f}, Δx={delta_i:.2f}, "
                            f"local_contrib={contrib_i:.4f}"
                        )

                # apply filtering based on theta
                survivors = self.filter_contributions(local_contribs, delta_n)

                # propagate only survivors
                for i_id, contrib_i, w_i, delta_i in survivors:
                    syn_id = f"{i_id} -> {layer}.neuron_{n_idx}"

                    if self.debug:
                        print(
                            f"    ✓ propagate survivor {i_id}: "
                            f"contrib={contrib_i:.4f} → mark for backtracking"
                        )

                    self.accumulate_CONTRIB(i_id, syn_id, contrib_i)
                    
                    # mark for backtracking
                    CONTRIB[i_id] = 1.0

        if self.debug:
            print("\n======== DONE SLICE ========\n")

        return {
            "neuron_slice": self.neuron_CONTRIB,
            "synapse_slice": self.synapse_CONTRIB,
        }

    # =====================================================================
    # Filter local contributions by smallest-first until theta is reached
    # =====================================================================
    def filter_contributions(self, local_contribs, delta_n):
        theta = self.theta

        if not local_contribs:
            return []

        # sort the list of contributions in ascending order by |contrib|
        local_contribs = sorted(local_contribs, key=lambda x: abs(x[1]))

        removable_ids = set()
        influence_sum = 0.0
        y = delta_n if abs(delta_n) > 1e-8 else 1e-8

        for i_id, contrib_i, w_i, delta_i in local_contribs:
            # influence: |w_i * Δx_i / Δy_n|
            influence = abs(w_i * delta_i) / abs(y)

            # if we can still remove this one without exceeding theta -> drop it
            if influence_sum + influence <= theta:
                removable_ids.add(i_id)
                influence_sum += influence
            else:
                break

        # keep only survivors (not removed)
        survivors = [entry for entry in local_contribs if entry[0] not in removable_ids]
        return survivors

    # =====================================================================
    # Initialize CONTRIB tables to zero for all neurons and synapses
    # =====================================================================
    def initialize_contributions(self, input_size: int):
        self.neuron_CONTRIB = {}
        self.synapse_CONTRIB = {}

        # Set all hidden/output neurons to zero
        for layer_name, delta_vec in self.deltas.items():
            if layer_name == "input":
                continue
            for i in range(len(delta_vec)):
                nid = f"{layer_name}.neuron_{i}"
                self.neuron_CONTRIB[nid] = 0

        # Set input neurons to zero
        for i in range(input_size):
            nid = f"input.neuron_{i}"
            self.neuron_CONTRIB[nid] = 0

        # Set all synapses (parent -> child) to zero
        for parent, child in self.edges:
            if child not in self.weights:
                continue
            W = self.weights[child]
            out_f, in_f = W.shape
            for o in range(out_f):
                for i in range(in_f):
                    sid = f"{parent}.neuron_{i} -> {child}.neuron_{o}"
                    self.synapse_CONTRIB[sid] = 0

        # If you want to be extra safe, you can ensure Input->first still exists
        # but edges should already contain ("input", first_layer) from GraphBuilder.

    # =====================================================================
    # Accumulate the CONTRIB value by normalizing to -1, 0, or 1
    # =====================================================================
    def accumulate_CONTRIB(self, neuron_i: str, synapse_i: str, contrib_i: float):
        if contrib_i > 0:
            s = 1
        elif contrib_i < 0:
            s = -1
        else:
            s = 0

        # Initialize if not present (defensive)
        if neuron_i not in self.neuron_CONTRIB:
            self.neuron_CONTRIB[neuron_i] = 0
        if synapse_i not in self.synapse_CONTRIB:
            self.synapse_CONTRIB[synapse_i] = 0

        self.neuron_CONTRIB[neuron_i] += s
        self.synapse_CONTRIB[synapse_i] += s