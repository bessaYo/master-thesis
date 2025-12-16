# core/analysis/forward.py

import torch
import torch.nn as nn


class ForwardAnalyzer:
    def __init__(self, model: nn.Module, profiler_result):
        self.model = model
        self.hooks = []

        # Set means from profiler result
        self.neuron_means = profiler_result["neuron_means"]
        self.layer_means = profiler_result["layer_means"]

        # Store deltas (sample activations - profiler means)
        self.activations = {}
        self.pool_indices = {}
        self.neuron_deltas = {}
        self.layer_deltas = {}

    # Function that registers hook function to each layer
    def _register_hooks(self):
        for name, layer in self.model.named_modules():

            # Skip layer with children (like sequentials or blocks) for now
            if len(list(layer.children())) > 0:
                continue

            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)

    # Hook function to capture activations
    def _hook_fn(self, layer_name):
        def hook(module, input, output):

            # Handle MaxPool2d with return_indices=True
            if isinstance(module, nn.MaxPool2d) and module.return_indices:
                pooled, indices = output
                self.activations[layer_name] = pooled.detach()
                self.pool_indices[layer_name] = indices.detach()
            else:
                self.activations[layer_name] = output.detach()

        return hook

    # Remove all registered hooks
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    # Compute input delta
    def _compute_input_delta(self, sample):
        mean = self.neuron_means["input"].to(sample.device)
        self.input_delta = sample - mean

        # Store neuron and layer deltas for input
        self.neuron_deltas["input"] = self.input_delta
        self.layer_deltas["input"] = self.input_delta.mean()

    # Compute neuron deltas for a given layer
    def _compute_neuron_deltas(self, layer_name):
        activation = self.activations[layer_name]
        mean = self.neuron_means[layer_name].to(activation.device)

        delta = activation - mean
        self.neuron_deltas[layer_name] = delta

    # Compute layer delta for a given layer
    def _compute_layer_delta(self, layer_name):
        self.layer_deltas[layer_name] = self.neuron_deltas[layer_name].mean()

    # Execute forward analysis for a given sample
    def execute(self, sample):
        self._reset_stats()
        self._register_hooks()
        self.model.eval()

        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)

        with torch.no_grad():
            _ = self.model(sample)

        # Compute input delta
        self._compute_input_delta(sample)

        # Compute deltas for each layer
        for layer_name, _ in self.activations.items():
            self._compute_neuron_deltas(layer_name)
            self._compute_layer_delta(layer_name)

        self._remove_hooks()

        return {
            "activations": self.activations,
            "neuron_deltas": self.neuron_deltas,
            "layer_deltas": self.layer_deltas,
            "pool_indices": self.pool_indices,
        }

    # Reset all stored activations and deltas
    def _reset_stats(self):
        self.activations = {}
        self.neuron_deltas = {}
        self.layer_deltas = {}
        self.pool_indices = {}