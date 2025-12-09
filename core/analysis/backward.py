# core/analysis/backward.py

import torch


class BackwardAnalyzer:
    def __init__(
        self, graph, forward_result, target_index, theta=0.3, channel_mode=False
    ):
        self.graph = graph
        self.channel_mode = channel_mode
        self.theta = theta

        self.activations = forward_result["activations"]
        self.neuron_deltas = forward_result["neuron_deltas"]
        self.layer_deltas = forward_result["layer_deltas"]
        self.pool_indices = forward_result.get("pool_indices", {})

        self.output_layer = list(self.graph.nodes.keys())[-1]
        self.target_index = target_index

        # results
        self.neuron_contributions = {}
        self.synapse_contributions = {}

    # Main tracing function (entry point)
    def trace(self):
        self._init_contributions()

        layer_order = list(self.graph.nodes.keys())[::-1]

        for layer_name in layer_order:
            node = self.graph.nodes[layer_name]
            CONTRIB_n = self.neuron_contributions[layer_name]
            delta_n = self.neuron_deltas[layer_name]

            # No parents → input layer reached
            if not node.parents:
                continue

            for parent in node.parents:
                p_name = parent.name
                delta_i = self.neuron_deltas[p_name]

                syn, contrib_parent = self._backward_layer(
                    node, CONTRIB_n, delta_n, delta_i
                )

                # store synapses sparse
                if syn:
                    self.synapse_contributions.setdefault(layer_name, []).extend(syn)

                # accumulate neuron contributions
                self.neuron_contributions[p_name] += contrib_parent

    # Initialize contributions to zero, set target neurons to 1
    def _init_contributions(self):
        for layer_name, delta in self.neuron_deltas.items():
            contrib = torch.zeros_like(delta)

            # OUTPUT LAYER gets 1 at target neuron
            if layer_name == self.output_layer:
                # Case: linear output [B, out]
                if contrib.dim() == 2:
                    contrib[0, self.target_index] = 1.0

                # Case: conv feature output [B, C, H, W]
                elif contrib.dim() == 4:
                    contrib[0, self.target_index, :, :] = 1.0

            self.neuron_contributions[layer_name] = contrib

    # Theta filtering for linear and conv layers
    def _theta_filter(self, candidates, output_value):
        """
        Sort by |w_i * Δx_i| ascending, remove smallest
        while |Σ removed| / |y| < θ
        """
        if not candidates:
            return []

        # Sort by magnitude (ascending)
        sorted_cands = sorted(candidates, key=lambda c: abs(c["w_dx"]))

        # Avoid division by zero
        if abs(output_value) < 1e-9:
            return sorted_cands

        # Find cutoff: remove smallest while influence < θ
        removed_sum = 0.0
        cutoff_idx = 0

        for i, c in enumerate(sorted_cands):
            w_dx = c["w_dx"]
            # Check: can we still remove this?
            if abs(removed_sum + w_dx) / abs(output_value) < self.theta:
                removed_sum += w_dx
                cutoff_idx = i + 1
            else:
                break

        # Keep everything from cutoff_idx onwards
        return sorted_cands[cutoff_idx:]

    # Dispatch to layer-specific backward methods
    def _backward_layer(self, node, CONTRIB_n, delta_n, delta_i):
        if node.type == "linear":
            return self._backward_linear(node, CONTRIB_n, delta_n, delta_i)

        elif node.type == "conv2d":
            return self._backward_conv2d(node, CONTRIB_n, delta_n, delta_i)

        elif node.type == "avgpool2d":
            return self._backward_avgpool2d(node, CONTRIB_n, delta_n, delta_i)

        elif node.type == "maxpool2d":
            return self._backward_maxpool2d(node, CONTRIB_n, delta_n, delta_i)

        elif node.type == "relu":
            return self._backward_relu(node, CONTRIB_n, delta_n, delta_i)

        elif node.type == "batchnorm2d":
            return self._backward_batchnorm2d(node, CONTRIB_n, delta_n, delta_i)

        else:
            raise NotImplementedError(node.type)

    # Linear backward operation
    def _backward_linear(self, node, CONTRIB_n, delta_n, delta_i):
        W = node.weight  # [out, in]

        # Output Site
        out_contrib = CONTRIB_n.squeeze()  # [out]
        out_delta = delta_n.squeeze()  # [out]

        # Input Site
        orig_shape = delta_i.shape  # Remember for reshape at the end

        # If delta_i has more than 2 dims, flatten it
        if delta_i.dim() > 2:
            in_delta = delta_i.view(-1)  # [in] flat
        else:
            in_delta = delta_i.squeeze()  # [in]

        parent_contrib_flat = torch.zeros_like(in_delta)
        syn_list = []

        for j in range(W.shape[0]):
            if out_contrib[j] == 0:
                continue

            dy = out_delta[j]

            candidates = []

            for i in range(W.shape[1]):
                w_val = W[j, i]
                dx = in_delta[i]

                local = out_contrib[j] * dy * w_val * dx
                w_dx = w_val * dx

                candidates.append(
                    {
                        "i": i,
                        "local": local,
                        "w_dx": float(w_dx) if torch.is_tensor(w_dx) else w_dx,
                    }
                )

            # Apply theta filtering
            filtered = self._theta_filter(candidates, float(dy))

            # Store contributions for neurons and synapses
            for cand in filtered:
                s = torch.sign(cand["local"])
                idx = cand["i"]

                parent_contrib_flat[idx] += s
                syn_list.append(
                    {
                        "i": int(idx),
                        "j": int(j),
                        "sign": float(s),
                    }
                )

        # Back to original shape for parent contributions
        parent_contrib = parent_contrib_flat.view(orig_shape)

        return syn_list, parent_contrib

    # Conv2d backward operation
    def _backward_conv2d(self, node, CONTRIB_n, delta_n, delta_i):
        W = node.weight  # [C_out, C_in, kH, kW]
        stride_h, stride_w = node.stride
        pad_h, pad_w = node.padding

        _, C_out, H_out, W_out = CONTRIB_n.shape
        _, C_in, H_in, W_in = delta_i.shape
        kH, kW = W.shape[2:]

        input_contrib = torch.zeros_like(delta_i)
        syn_list = []

        for co in range(C_out):
            for ho in range(H_out):
                for wo in range(W_out):

                    c_out = CONTRIB_n[0, co, ho, wo]
                    if c_out == 0:
                        continue

                    dy = delta_n[0, co, ho, wo]

                    h_start = ho * stride_h - pad_h
                    w_start = wo * stride_w - pad_w

                    # Collect all contributions for this output neuron
                    candidates = []

                    for ci in range(C_in):
                        for kh in range(kH):
                            for kw in range(kW):

                                h_in = h_start + kh
                                w_in = w_start + kw

                                if h_in < 0 or h_in >= H_in or w_in < 0 or w_in >= W_in:
                                    continue

                                dx = delta_i[0, ci, h_in, w_in]
                                w_val = W[co, ci, kh, kw]

                                local = c_out * dy * w_val * dx
                                w_dx = w_val * dx

                                candidates.append(
                                    {
                                        "ci": ci,
                                        "h_in": h_in,
                                        "w_in": w_in,
                                        "local": local,
                                        "w_dx": (
                                            float(w_dx)
                                            if torch.is_tensor(w_dx)
                                            else w_dx
                                        ),
                                    }
                                )

                    # Apply theta filtering
                    filtered = self._theta_filter(candidates, float(dy))

                    # Store contributions
                    for cand in filtered:
                        s = torch.sign(cand["local"])

                        ci = cand["ci"]
                        h_in = cand["h_in"]
                        w_in = cand["w_in"]

                        # flat indices
                        in_idx = ci * H_in * W_in + h_in * W_in + w_in
                        out_idx = co * H_out * W_out + ho * W_out + wo

                        syn_list.append(
                            {"i": int(in_idx), "j": int(out_idx), "sign": float(s)}
                        )
                        input_contrib[0, ci, h_in, w_in] += s

        return syn_list, input_contrib

    # AvgPool2d backward operation
    def _backward_avgpool2d(self, node, CONTRIB_n, delta_n, delta_i):
        kH, kW = node.kernel_size
        sH, sW = node.stride
        pH, pW = node.padding

        B, C, H_out, W_out = CONTRIB_n.shape
        _, _, H_in, W_in = delta_i.shape

        syn_list = []
        input_contrib = torch.zeros_like(delta_i)

        scale = 1.0 / (kH * kW)

        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):

                    c_out = CONTRIB_n[0, c, ho, wo]
                    if c_out == 0:
                        continue

                    dy = delta_n[0, c, ho, wo]

                    h_start = ho * sH - pH
                    w_start = wo * sW - pW

                    # Collect all contributions for this output neuron
                    candidates = []

                    for kh in range(kH):
                        for kw in range(kW):

                            h_in = h_start + kh
                            w_in = w_start + kw

                            if h_in < 0 or h_in >= H_in or w_in < 0 or w_in >= W_in:
                                continue

                            dx = delta_i[0, c, h_in, w_in]
                            local = c_out * dy * dx * scale
                            w_dx = dx * scale  # "weight" is uniform 1/k²

                            candidates.append(
                                {
                                    "h_in": h_in,
                                    "w_in": w_in,
                                    "local": local,
                                    "w_dx": (
                                        float(w_dx) if torch.is_tensor(w_dx) else w_dx
                                    ),
                                }
                            )

                    # 2. Theta filtering
                    filtered = self._theta_filter(candidates, float(dy))

                    # 3. Store contributions
                    for cand in filtered:
                        s = torch.sign(cand["local"])

                        h_in = cand["h_in"]
                        w_in = cand["w_in"]

                        in_idx = c * H_in * W_in + h_in * W_in + w_in
                        out_idx = c * H_out * W_out + ho * W_out + wo

                        syn_list.append(
                            {"i": int(in_idx), "j": int(out_idx), "sign": float(s)}
                        )
                        input_contrib[0, c, h_in, w_in] += s

        return syn_list, input_contrib

    # MaxPool2d backward operation
    def _backward_maxpool2d(self, node, CONTRIB_n, delta_n, delta_i):
        indices = self.pool_indices[node.name]  # GLOBAL flat indices

        B, C, H_out, W_out = CONTRIB_n.shape
        _, _, H_in, W_in = delta_i.shape

        input_contrib = torch.zeros_like(delta_i)
        syn_list = []

        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):

                    c_out = CONTRIB_n[0, c, ho, wo]
                    if c_out == 0:
                        continue

                    dy = delta_n[0, c, ho, wo]

                    # Global flat index into input tensor
                    flat = int(indices[0, c, ho, wo].item())

                    # Convert to (h_in, w_in)
                    h_in = flat // W_in
                    w_in = flat % W_in

                    dx = delta_i[0, c, h_in, w_in]

                    # Local contribution (paper formula)
                    local = c_out * dy * dx

                    # No theta filter for maxpool (only 1 input contributes)
                    s = torch.sign(local)

                    # Synapse flat indices
                    in_idx = c * H_in * W_in + flat
                    out_idx = c * H_out * W_out + ho * W_out + wo

                    syn_list.append(
                        {"i": in_idx, "j": out_idx, "sign": float(s.item())}
                    )

                    input_contrib[0, c, h_in, w_in] += s

        return syn_list, input_contrib

    # Relu backward operation
    def _backward_relu(self, node, CONTRIB_n, delta_n, delta_i):
        activation = self.activations[node.name]
        mask = (activation > 0).float()

        # Local contribution: CONTRIB_n × Δy × w × Δx
        local = CONTRIB_n * delta_n * delta_i * mask
        input_contrib = torch.sign(local)
        syn_list = []

        nonzero = (input_contrib != 0).nonzero(as_tuple=False)

        for idx in nonzero:
            if input_contrib.dim() == 4:
                _, c, h, w = idx.tolist()
                C, H, W = input_contrib.shape[1:]
                flat_idx = c * H * W + h * W + w
            else:
                flat_idx = idx[-1].item()

            syn_list.append(
                {
                    "i": int(flat_idx),
                    "j": int(flat_idx),
                    "sign": float(input_contrib[tuple(idx)].item()),
                }
            )

        return syn_list, input_contrib

    # BatchNorm2d backward operation
    def _backward_batchnorm2d(self, node, CONTRIB_n, delta_n, delta_i):
        local = CONTRIB_n * delta_n * delta_i
        return [], torch.sign(local)
