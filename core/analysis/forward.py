# core/analysis/forward.py

import torch
import torch.nn as nn
from utils.tensor_utils import ensure_tensor_batch


class ForwardAnalyzer:
    def __init__(self, model: nn.Module, profiler_result):
        self.model = model
        self.hooks = []
        self.layer_types = {}

        # Set means from profiler result
        self.neuron_means = profiler_result["neuron_means"]
        self.layer_means = profiler_result["layer_means"]
        self.channel_means = profiler_result["channel_means"]

        # Store deltas (sample activations - profiler means)
        self.activations = {}
        self.pool_indices = {}
        self.neuron_deltas = {}
        self.layer_deltas = {}
        self.channel_deltas = {}

    # Function that registers hook function to each layer
    def _register_hooks(self):
        for name, layer in self.model.named_modules():

            # Skip layer with children (like sequentials or blocks) for now
            if len(list(layer.children())) > 0:
                continue

            self.layer_types[name] = layer
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
        self.layer_deltas["input"] = self.input_delta.abs().mean()

    # Compute channel deltas for a given layer (mean over neuron deltas)
    def _compute_channel_deltas(self, layer_name):
        neuron_deltas = self.neuron_deltas[layer_name]
        self.channel_deltas[layer_name] = neuron_deltas.abs().mean(dim=(-2, -1))

    # Compute neuron deltas for a given layer
    def _compute_neuron_deltas(self, layer_name):
        activation = self.activations[layer_name]
        mean = self.neuron_means[layer_name].to(activation.device)

        delta = activation - mean
        self.neuron_deltas[layer_name] = delta

    # Compute layer delta for a given layer
    def _compute_layer_delta(self, layer_name):
        if layer_name in self.channel_deltas:
            self.layer_deltas[layer_name] = self.channel_deltas[layer_name].mean()
        else:
            self.layer_deltas[layer_name] = self.neuron_deltas[layer_name].abs().mean()

    # Execute forward analysis for a given sample
    def execute(self, sample):
        self._reset_stats()
        self._register_hooks()
        self.model.eval()

        sample = ensure_tensor_batch(sample)

        with torch.no_grad():
            _ = self.model(sample)

        # Compute input delta
        self._compute_input_delta(sample)

        # Compute deltas for each layer
        for layer_name in self.activations:
            self._compute_neuron_deltas(layer_name)

            # If convolutional layer, compute channel deltas
            if isinstance(self.layer_types[layer_name], nn.Conv2d):
                self._compute_channel_deltas(layer_name)

            self._compute_layer_delta(layer_name)

        self._remove_hooks()

        return {
            "activations": self.activations,
            "neuron_deltas": self.neuron_deltas,
            "layer_deltas": self.layer_deltas,
            "channel_deltas": self.channel_deltas,
            "pool_indices": self.pool_indices,
        }

    # Reset all stored activations and deltas
    def _reset_stats(self):
        self.activations = {}
        self.neuron_deltas = {}
        self.layer_deltas = {}
        self.channel_deltas = {}
        self.pool_indices = {}
