# core/tracing/operations.py

import torch


class BackwardOperations:
    """
    Layer-specific contribution operations for backward analysis.
    
    Each operation computes:
    1. Contributions of neurons in the previous layer
    2. Contributions of synapses (connections)
    """

    def __init__(self, theta=0.0):
        self.theta = theta

    def linear(self, module, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for fully connected layer."""
        W = module.weight.detach()
        out_c = CONTRIB_n.squeeze()
        out_d = delta_n.squeeze()

        flat_in = delta_i.reshape(-1)
        parent = torch.zeros_like(flat_in)
        syn = []

        for j in range(W.shape[0]):
            if out_c[j] == 0:
                continue

            dy = out_d[j]
            cands = []

            for i in range(W.shape[1]):
                w, dx = W[j, i], flat_in[i]
                cands.append({
                    "i": i,
                    "local": out_c[j] * dy * w * dx,
                    "w_dx": float(w * dx),
                })

            for c in self._theta_filter(cands, float(dy)):
                s = torch.sign(c["local"])
                parent[c["i"]] += s
                syn.append({"i": c["i"], "j": j, "sign": float(s)})

        return syn, parent.reshape(delta_i.shape)

    def conv2d(self, module, CONTRIB_n, delta_n, delta_i, active_channels=None):
        """Backward contribution for convolutional layer."""
        W = module.weight.detach()

        sH, sW = (
            module.stride
            if isinstance(module.stride, tuple)
            else (module.stride, module.stride)
        )
        pH, pW = (
            module.padding
            if isinstance(module.padding, tuple)
            else (module.padding, module.padding)
        )

        groups = module.groups

        _, C_out, H_out, W_out = CONTRIB_n.shape
        _, _, H_in, W_in = delta_i.shape
        kH, kW = W.shape[2:]

        Cin_g = W.shape[1]
        Cout_g = C_out // groups

        input_contrib = torch.zeros_like(delta_i)
        syn = []

        for co in range(C_out):
            # Skip non-active channels (channel filtering)
            if active_channels is not None and co not in active_channels:
                continue

            g = co // Cout_g
            ci_start = g * Cin_g

            for ho in range(H_out):
                for wo in range(W_out):
                    if CONTRIB_n[0, co, ho, wo] == 0:
                        continue

                    dy = delta_n[0, co, ho, wo]
                    hs = ho * sH - pH
                    ws = wo * sW - pW

                    cands = []

                    for ci in range(Cin_g):
                        for kh in range(kH):
                            for kw in range(kW):
                                hi, wi = hs + kh, ws + kw
                                if 0 <= hi < H_in and 0 <= wi < W_in:
                                    dx = delta_i[0, ci_start + ci, hi, wi]
                                    w = W[co, ci, kh, kw]
                                    cands.append({
                                        "ci": ci_start + ci,
                                        "h": hi,
                                        "w": wi,
                                        "local": CONTRIB_n[0, co, ho, wo] * dy * w * dx,
                                        "w_dx": float(w * dx),
                                    })

                    for c in self._theta_filter(cands, float(dy)):
                        s = torch.sign(c["local"])
                        input_contrib[0, c["ci"], c["h"], c["w"]] += s
                        syn.append({
                            "i": int(c["ci"] * H_in * W_in + c["h"] * W_in + c["w"]),
                            "j": int(co * H_out * W_out + ho * W_out + wo),
                            "sign": float(s),
                        })

        return syn, input_contrib

    def maxpool2d(self, indices, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for max pooling layer."""
        _, C, H_out, W_out = CONTRIB_n.shape
        _, _, H_in, W_in = delta_i.shape

        out = torch.zeros_like(delta_i)
        syn = []

        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):
                    if CONTRIB_n[0, c, ho, wo] == 0:
                        continue

                    flat = int(indices[0, c, ho, wo].item())
                    hi = flat // W_in
                    wi = flat % W_in

                    local = (
                        CONTRIB_n[0, c, ho, wo]
                        * delta_n[0, c, ho, wo]
                        * delta_i[0, c, hi, wi]
                    )
                    s = torch.sign(local)
                    out[0, c, hi, wi] += s

                    syn.append({
                        "i": int(c * H_in * W_in + hi * W_in + wi),
                        "j": int(c * H_out * W_out + ho * W_out + wo),
                        "sign": float(s.item()),
                    })

        return syn, out

    def avgpool2d(self, CONTRIB_n, delta_n, delta_i, module=None):
        """Backward contribution for average pooling layer."""
        _, C, H_out, W_out = CONTRIB_n.shape
        _, _, H_in, W_in = delta_i.shape

        # Extract params for fixed pooling
        if module is not None:
            kH, kW = (
                module.kernel_size
                if isinstance(module.kernel_size, tuple)
                else (module.kernel_size, module.kernel_size)
            )
            sH, sW = (
                module.stride
                if isinstance(module.stride, tuple)
                else (module.stride, module.stride)
            )
            pH, pW = (
                module.padding
                if isinstance(module.padding, tuple)
                else (module.padding, module.padding)
            )

        out = torch.zeros_like(delta_i)
        syn = []

        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):
                    if CONTRIB_n[0, c, ho, wo] == 0:
                        continue

                    # Compute pooling region
                    if module is None:  # Adaptive
                        hs = ho * H_in // H_out
                        he = (ho + 1) * H_in // H_out
                        ws = wo * W_in // W_out
                        we = (wo + 1) * W_in // W_out
                    else:  # Fixed kernel
                        hs = ho * sH - pH
                        he = hs + kH
                        ws = wo * sW - pW
                        we = ws + kW

                    # Clamp to valid input range
                    hs_clamped = max(0, hs)
                    he_clamped = min(H_in, he)
                    ws_clamped = max(0, ws)
                    we_clamped = min(W_in, we)

                    scale = 1.0 / ((he - hs) * (we - ws))

                    for hi in range(hs_clamped, he_clamped):
                        for wi in range(ws_clamped, we_clamped):
                            local = (
                                CONTRIB_n[0, c, ho, wo]
                                * delta_n[0, c, ho, wo]
                                * delta_i[0, c, hi, wi]
                                * scale
                            )
                            s = torch.sign(local)
                            out[0, c, hi, wi] += s

                            if s != 0:
                                syn.append({
                                    "i": int(c * H_in * W_in + hi * W_in + wi),
                                    "j": int(c * H_out * W_out + ho * W_out + wo),
                                    "sign": float(s.item()),
                                })

        return syn, out

    def relu(self, activation, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for ReLU activation."""
        mask = (activation > 0).float()
        return self._passthrough(CONTRIB_n, delta_n, delta_i * mask)

    def batchnorm2d(self, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for batch normalization."""
        return self._passthrough(CONTRIB_n, delta_n, delta_i)

    def add(self, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for residual add operation."""
        return self._passthrough(CONTRIB_n, delta_n, delta_i)

    def flatten(self, CONTRIB_n, delta_i):
        """Backward contribution for flatten/reshape operation."""
        return [], CONTRIB_n.reshape(delta_i.shape)

    def _passthrough(self, CONTRIB_n, delta_n, delta_i, track_synapses=True):
        """Generic passthrough for layers that don't modify shape."""
        local = CONTRIB_n * delta_n * delta_i
        contrib = torch.sign(local)

        if not track_synapses:
            return [], contrib

        flat = contrib.flatten()
        syn = [
            {"i": i, "j": i, "sign": float(flat[i].item())}
            for i in range(flat.numel())
            if flat[i] != 0
        ]
        return syn, contrib

    def _theta_filter(self, candidates, output_value):
        """Filter synapse candidates based on theta threshold."""
        if not candidates or abs(output_value) < 1e-9:
            return candidates

        candidates = sorted(candidates, key=lambda c: abs(c["w_dx"]))
        removed = 0.0
        cut_idx = 0

        for i, c in enumerate(candidates):
            if abs(removed + c["w_dx"]) / abs(output_value) < self.theta:
                removed += c["w_dx"]
                cut_idx = i + 1
            else:
                break

        return candidates[cut_idx:]