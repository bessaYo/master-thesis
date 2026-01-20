# utils/logging.py

import torch
from typing import Dict, List, Any, Callable
from core.graph import Graph


class SlicerLogger:
    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose 

    def log(self, *args):
        if self.enabled:
            print(*args)

    def header(self, title: str):
        self.log("\n" + "=" * 80)
        self.log(title)
        self.log("=" * 80)

    def separator(self):
        self.log("-" * 80)

    def model_summary(
        self,
        graph: "Graph",
        neuron_deltas: Dict[str, torch.Tensor],
        target_index: int,
        theta: float,
    ):
        nodes = list(graph.get_nodes())
        compute_nodes = [
            n for n in nodes
            if n.op in ("call_module", "call_function")
            and graph.get_type(n) not in ("output", "attr")
        ]

        total_params = 0
        total_neurons = sum(d.numel() for d in neuron_deltas.values())

        for node in nodes:
            if node.op == "call_module":
                module = graph.get_module(node)
                if module:
                    for p in module.parameters():
                        total_params += p.numel()

        self.header("MODEL SUMMARY")
        self.log(f"  Model:   {graph.model.__class__.__name__}")
        self.log(f"  Layers:  {len(compute_nodes)} | Params: {total_params:,} | Neurons: {total_neurons:,}")
        self.log(f"  Target:  index {target_index} | Theta: {theta}")

    def layer_table(
        self,
        graph: "Graph",
        neuron_deltas: Dict[str, torch.Tensor],
        key_fn: Callable,
    ):
        
        if not self.verbose:
            return
            
        self.header("LAYER DETAILS")
        self.log(f"{'Layer':<30} | {'Type':<15} | {'Neurons':>10}")
        self.separator()

        for node in graph.get_nodes():
            if node.op not in ("call_module", "placeholder"):
                continue

            key = key_fn(node)
            if key not in neuron_deltas:
                continue

            delta = neuron_deltas[key]
            node_type = graph.get_type(node)
            neurons = delta.numel()

            self.log(f"{key:<30} | {node_type:<15} | {neurons:>10,}")

    def results(
        self,
        graph: "Graph",
        neuron_contributions: Dict[str, torch.Tensor],
        synapse_contributions: Dict[str, List[Dict[str, Any]]],
        backward_time: float,
        key_fn: Callable,
        neuron_deltas: Dict[str, torch.Tensor],
    ):
        total_neurons = 0
        slice_neurons = 0
        slice_synapses = 0

        layer_results = []

        for node in graph.get_nodes():
            if node.op not in ("call_module", "placeholder", "call_function"):
                continue

            key = key_fn(node)
            if key not in neuron_contributions:
                continue

            contrib = neuron_contributions[key]
            n_slice = (contrib != 0).sum().item()
            n_total = contrib.numel()
            s_slice = len(synapse_contributions.get(key, []))

            total_neurons += n_total
            slice_neurons += n_slice
            slice_synapses += s_slice

            layer_results.append({
                "key": key,
                "n_slice": n_slice,
                "n_total": n_total,
                "s_slice": s_slice,
            })

        # Show layer-wise results
        if self.verbose:
            self.header("SLICE RESULTS")
            self.log(f"{'Layer':<30} | {'Neurons in Slice':>22} | {'Synapses':>10}")
            self.separator()

            for r in layer_results:
                pct = 100 * r["n_slice"] / r["n_total"] if r["n_total"] > 0 else 0
                neuron_str = f"{r['n_slice']:>6} / {r['n_total']:<6} ({pct:>5.1f}%)"
                self.log(f"{r['key']:<30} | {neuron_str} | {r['s_slice']:>10}")

            self.separator()

        # Summary
        pct_n = 100 * slice_neurons / total_neurons if total_neurons > 0 else 0
        
        self.header("SLICE SUMMARY")
        self.log(f"  Neurons:  {slice_neurons:,} / {total_neurons:,} ({pct_n:.1f}%)")
        self.log(f"  Synapses: {slice_synapses:,}")
        self.log(f"  Time:     {backward_time:.4f}s")