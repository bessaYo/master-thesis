import torch
import torch.nn as nn
from core.graph_builder import GraphBuilder


class ForwardAnalyzer:
    def __init__(self, model, profiling_means):
        self.model = model
        self.profiling_means = profiling_means   # includes "input" now
        self.hooks = []

        self.deltas = {}             # Δy values per layer
        self.execution_order = []    # order of executed layers
        self.tensor_origin = {}      # map tensor id -> layer name
        self.edges = []              # connections between modules
        self.weights = {}            # weight matrices
        self.modules = {}            # store modules

        self._last_seen = set()      # detect first execution of a module
        self._edge_set = set()       # avoid duplicate edges

        self._input_tensor = None    # store raw input for Δx


    # ----------------------------------------------------------------------
    # Create forward hook for a module
    # ----------------------------------------------------------------------
    def _hook_fn(self, name):
        def hook(module, input, output):

            # track execution order (only first call)
            if name not in self._last_seen:
                self.execution_order.append(name)
                self._last_seen.add(name)

            # register parent → child edges via tensor origin tracking
            if isinstance(input, (tuple, list)) and len(input) > 0:
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        tid = id(inp)
                        if tid in self.tensor_origin:
                            parent = self.tensor_origin[tid]
                            edge = (parent, name)

                            if edge not in self._edge_set:
                                self.edges.append(edge)
                                self._edge_set.add(edge)

            # tag the output as originating from this module
            if isinstance(output, torch.Tensor):
                self.tensor_origin[id(output)] = name

            # compute Δy = y - profiling_mean for this layer
            if isinstance(output, torch.Tensor) and name in self.profiling_means:
                output_mean = output.mean(dim=0).flatten()
                ref_mean = self.profiling_means[name].to(output.device)
                self.deltas[name] = output_mean - ref_mean

            # store weight matrix (if present)
            if hasattr(module, "weight"):
                self.weights[name] = module.weight.detach().clone()

            # store module reference
            self.modules[name] = module

        return hook


    # ----------------------------------------------------------------------
    # Register hooks on all modules
    # ----------------------------------------------------------------------
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name == "":
                continue  # skip top-level container
            self.hooks.append(module.register_forward_hook(self._hook_fn(name)))


    # ----------------------------------------------------------------------
    # Remove hooks from all modules
    # ----------------------------------------------------------------------
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


    # ----------------------------------------------------------------------
    # Run one forward pass and collect:
    # - deltas
    # - edges
    # - execution order
    # - weights
    # ----------------------------------------------------------------------
    def analyze(self, x):
        """
        Runs the model once with x, gathers deltas, edges, weights, execution order,
        and constructs a unified graph via GraphBuilder.
        """

        # store raw input (for Δx = x - mean_input)
        self._input_tensor = x.clone().detach()

        self.register_hooks()

        with torch.no_grad():
            _ = self.model(x)

        self.remove_hooks()

        # Build graph
        graph = GraphBuilder.build_graph(
            deltas=self.deltas,
            execution_order=self.execution_order,
            edges=self.edges,
            weights=self.weights,
            input_tensor=self._input_tensor,
            profiling_means=self.profiling_means
        )

        return graph