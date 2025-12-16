# core/analysis/backward.py

import time
from typing import Dict, List, Any, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from core.graph import Graph
from utils.logging import SlicerLogger


class BackwardAnalyzer:
    def __init__(
        self,
        graph: Graph,
        forward_result: Dict[str, Any],
        target_index: int,
        theta: float = 0.3,
        channel_mode: bool = False,
        debug: bool = False,
    ):
        self.graph = graph
        self.theta = theta
        self.channel_mode = channel_mode
        self.target_index = target_index

        self.activations = forward_result["activations"]
        self.neuron_deltas = forward_result["neuron_deltas"]
        self.layer_deltas = forward_result["layer_deltas"]
        self.pool_indices = forward_result.get("pool_indices", {})

        # Results
        self.neuron_contributions: Dict[str, torch.Tensor] = {}
        self.synapse_contributions: Dict[str, List[Dict[str, Any]]] = {}

        # Timing
        self.backward_time_sec: float = 0.0

        # Logger
        self.logger = SlicerLogger(enabled=debug)

    def _key(self, node: fx.Node) -> str:
        """Stable string key for activations / deltas / contributions."""
        if node.op == "call_module":
            return str(node.target)
        if node.op == "placeholder":
            return "input"
        return node.name

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

    def _get_activation(self, node: fx.Node) -> torch.Tensor:
        if node.op == "call_module":
            return self.activations[str(node.target)]
        if node.op == "placeholder":
            return self.neuron_deltas["input"]
        raise RuntimeError(f"No activation for node {node.name}")

    def _get_last_compute_node(self) -> fx.Node:
        for node in reversed(list(self.graph.get_nodes())):
            if node.op not in ("output", "get_attr"):
                return node
        raise RuntimeError("No compute node found")


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

        last_node = self._get_last_compute_node()
        last_key = self._key(last_node)
        contrib = self.neuron_contributions[last_key]

        if contrib.dim() == 2:
            contrib[0, self.target_index] = 1.0
        elif contrib.dim() == 4:
            contrib[0, self.target_index, :, :] = 1.0

    def trace(self):
        """Main backward tracing function."""

        self.logger.model_summary(
            self.graph, self.neuron_deltas, self.target_index, self.theta
        )
        self.logger.layer_table(self.graph, self.neuron_deltas, self._key)

        # Start timer
        start = time.perf_counter()

        self._init_contributions()
        
        # Collect trace steps for later logging
        trace_steps: List[Tuple[str, str, int, int, List[str]]] = []

        for node in reversed(list(self.graph.get_nodes())):
            if node.op in ("output", "get_attr"):
                continue

            node_key = self._key(node)
            node_type = self.graph.get_type(node)

            if node_key not in self.neuron_contributions:
                continue

            CONTRIB_n = self.neuron_contributions[node_key]
            contrib_in = (CONTRIB_n != 0).sum().item()

            if contrib_in == 0:
                continue

            delta_n = self._get_delta(node)

            parents = self.graph.get_compute_parents(node)
            expanded_parents: List[fx.Node] = []

            for p in parents:
                if self.graph.get_type(p) == "add":
                    for ap in self.graph.get_parent_nodes(p):
                        expanded_parents.append(self.graph.skip_passthrough(ap))
                    add_key = self._key(p)
                    if add_key in self.neuron_contributions:
                        self.neuron_contributions[add_key] += CONTRIB_n
                else:
                    expanded_parents.append(p)

            contrib_out_total = 0
            parent_names: List[str] = []

            for parent in expanded_parents:
                parent_key = self._key(parent)
                if parent_key not in self.neuron_deltas:
                    continue

                parent_names.append(parent_key)
                delta_i = self._get_delta(parent)

                syn, contrib_parent = self._backward_dispatch(
                    node, parent, CONTRIB_n, delta_n, delta_i
                )

                if syn:
                    self.synapse_contributions.setdefault(node_key, []).extend(syn)

                self.neuron_contributions.setdefault(
                    parent_key, torch.zeros_like(delta_i)
                )
                self.neuron_contributions[parent_key] += contrib_parent
                contrib_out_total += (contrib_parent != 0).sum().item()

            trace_steps.append((node_key, node_type, int(contrib_in), contrib_out_total, parent_names))

        # Stop Timer
        self.backward_time_sec = time.perf_counter() - start
            
        self.logger.results(
            self.graph,
            self.neuron_contributions,
            self.synapse_contributions,
            self.backward_time_sec,
            self._key,
            self.neuron_deltas,
        )

    def _backward_dispatch(
        self,
        node: fx.Node,
        parent: fx.Node,
        CONTRIB_n: torch.Tensor,
        delta_n: torch.Tensor,
        delta_i: torch.Tensor,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:

        node_type = self.graph.get_type(node)

        if node_type == "linear":
            module = self.graph.get_module(node)
            assert isinstance(module, nn.Linear)
            return self._backward_linear(module, CONTRIB_n, delta_n, delta_i)

        if node_type == "conv2d":
            module = self.graph.get_module(node)
            assert isinstance(module, nn.Conv2d)
            return self._backward_conv2d(module, CONTRIB_n, delta_n, delta_i)

        if node_type == "batchnorm2d":
            return self._backward_batchnorm2d(CONTRIB_n, delta_n, delta_i)

        if node_type == "relu":
            return self._backward_relu(
                self._get_activation(node), CONTRIB_n, delta_n, delta_i
            )

        if node_type == "maxpool2d":
            return self._backward_maxpool2d(
                self.pool_indices[self._key(node)], CONTRIB_n, delta_n, delta_i
            )

        if node_type == "avgpool2d":
            module = self.graph.get_module(node)
            assert isinstance(module, nn.AvgPool2d)
            return self._backward_avgpool2d(module, CONTRIB_n, delta_n, delta_i)

        if node_type == "adaptiveavgpool2d":
            return self._backward_adaptiveavgpool2d(CONTRIB_n, delta_n, delta_i)

        if node_type == "add":
            return self._backward_add(CONTRIB_n, delta_n, delta_i)

        if node_type in ("flatten", "method_view", "method_flatten"):
            return self._backward_flatten(CONTRIB_n, delta_i)

        return [], CONTRIB_n.clone()

    def _theta_filter(
        self,
        candidates: List[Dict[str, Any]],
        output_value: float,
    ) -> List[Dict[str, Any]]:
        """
        Filter synapse candidates by theta threshold.

        Sort by |w_i * Δx_i| ascending, remove smallest
        while |Σ removed| / |y| < θ
        """
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

    def _backward_linear(
        self,
        module: nn.Linear,
        CONTRIB_n: torch.Tensor,
        delta_n: torch.Tensor,
        delta_i: torch.Tensor,
    ):
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
                cands.append(
                    {
                        "i": i,
                        "local": out_c[j] * dy * w * dx,
                        "w_dx": float(w * dx),
                    }
                )

            for c in self._theta_filter(cands, float(dy)):
                s = torch.sign(c["local"])
                parent[c["i"]] += s
                syn.append({"i": c["i"], "j": j, "sign": float(s)})

        return syn, parent.reshape(delta_i.shape)

    def _backward_conv2d(
        self,
        module: nn.Conv2d,
        CONTRIB_n: torch.Tensor,
        delta_n: torch.Tensor,
        delta_i: torch.Tensor,
    ):
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
                                    cands.append(
                                        {
                                            "ci": ci_start + ci,
                                            "h": hi,
                                            "w": wi,
                                            "local": CONTRIB_n[0, co, ho, wo]
                                            * dy
                                            * w
                                            * dx,
                                            "w_dx": float(w * dx),
                                        }
                                    )

                    for c in self._theta_filter(cands, float(dy)):
                        s = torch.sign(c["local"])
                        input_contrib[0, c["ci"], c["h"], c["w"]] += s
                        syn.append(
                            {
                                "i": int(
                                    c["ci"] * H_in * W_in + c["h"] * W_in + c["w"]
                                ),
                                "j": int(co * H_out * W_out + ho * W_out + wo),
                                "sign": float(s),
                            }
                        )

        return syn, input_contrib

    def _backward_relu(
        self,
        activation: torch.Tensor,
        CONTRIB_n: torch.Tensor,
        delta_n: torch.Tensor,
        delta_i: torch.Tensor,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """ReLU backward pass - 1:1 synapse per neuron."""
        mask = (activation > 0).float()
        local = CONTRIB_n * delta_n * delta_i * mask
        contrib = torch.sign(local)

        # 1:1 Synapsen für aktive Neuronen
        flat = contrib.flatten()
        syn = [
            {"i": i, "j": i, "sign": float(flat[i].item())}
            for i in range(flat.numel())
            if flat[i] != 0
        ]

        return syn, contrib

    def _backward_maxpool2d(self, indices, CONTRIB_n, delta_n, delta_i):
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

                    # FEHLTE!
                    syn.append(
                        {
                            "i": int(c * H_in * W_in + hi * W_in + wi),
                            "j": int(c * H_out * W_out + ho * W_out + wo),
                            "sign": float(s.item()),
                        }
                    )

        return syn, out

    def _backward_avgpool2d(
        self,
        module: nn.AvgPool2d,
        CONTRIB_n: torch.Tensor,
        delta_n: torch.Tensor,
        delta_i: torch.Tensor,
    ):
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

        _, C, H_out, W_out = CONTRIB_n.shape
        _, _, H_in, W_in = delta_i.shape

        scale = 1.0 / (kH * kW)
        out = torch.zeros_like(delta_i)
        syn = []

        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):
                    if CONTRIB_n[0, c, ho, wo] == 0:
                        continue

                    hs = ho * sH - pH
                    ws = wo * sW - pW

                    for kh in range(kH):
                        for kw in range(kW):
                            hi = hs + kh
                            wi = ws + kw
                            if 0 <= hi < H_in and 0 <= wi < W_in:
                                local = (
                                    CONTRIB_n[0, c, ho, wo]
                                    * delta_n[0, c, ho, wo]
                                    * delta_i[0, c, hi, wi]
                                    * scale
                                )
                                s = torch.sign(local)
                                out[0, c, hi, wi] += s

                                # FEHLTE!
                                if s != 0:
                                    syn.append(
                                        {
                                            "i": int(c * H_in * W_in + hi * W_in + wi),
                                            "j": int(
                                                c * H_out * W_out + ho * W_out + wo
                                            ),
                                            "sign": float(s.item()),
                                        }
                                    )

        return syn, out

    def _backward_adaptiveavgpool2d(self, CONTRIB_n, delta_n, delta_i):
        _, C, H_out, W_out = CONTRIB_n.shape
        _, _, H_in, W_in = delta_i.shape

        out = torch.zeros_like(delta_i)
        syn = []

        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):
                    if CONTRIB_n[0, c, ho, wo] == 0:
                        continue

                    hs = ho * H_in // H_out
                    he = (ho + 1) * H_in // H_out
                    ws = wo * W_in // W_out
                    we = (wo + 1) * W_in // W_out
                    scale = 1.0 / ((he - hs) * (we - ws))

                    for hi in range(hs, he):
                        for wi in range(ws, we):
                            local = (
                                CONTRIB_n[0, c, ho, wo]
                                * delta_n[0, c, ho, wo]
                                * delta_i[0, c, hi, wi]
                                * scale
                            )
                            s = torch.sign(local)
                            out[0, c, hi, wi] += s

                            if s != 0:
                                syn.append(
                                    {
                                        "i": int(c * H_in * W_in + hi * W_in + wi),
                                        "j": int(c * H_out * W_out + ho * W_out + wo),
                                        "sign": float(s.item()),
                                    }
                                )

        return syn, out

    def _backward_batchnorm2d(self, CONTRIB_n, delta_n, delta_i):
        local = CONTRIB_n * delta_n * delta_i
        contrib = torch.sign(local)

        # 1:1 Synapsen
        flat = contrib.flatten()
        syn = [
            {"i": i, "j": i, "sign": float(flat[i].item())}
            for i in range(flat.numel())
            if flat[i] != 0
        ]

        return syn, contrib

    def _backward_add(self, CONTRIB_n, delta_n, delta_i):
        local = CONTRIB_n * delta_n * delta_i
        contrib = torch.sign(local)

        # 1:1 Synapsen
        flat = contrib.flatten()
        syn = [
            {"i": i, "j": i, "sign": float(flat[i].item())}
            for i in range(flat.numel())
            if flat[i] != 0
        ]

        return syn, contrib

    def _backward_flatten(self, CONTRIB_n, delta_i):
        return [], CONTRIB_n.reshape(delta_i.shape)
