import torch
import torch.nn as nn
from utils.tensor import ensure_tensor_batch

from core.tracing.base import BaseAnalyzer


class ForwardAnalyzer(BaseAnalyzer):
    """Computes activation deltas (activation - mean)"""

    def __init__(self, model, profiler_result):
        super().__init__(model)

        self.neuron_means = profiler_result["neuron_means"]
        self.layer_means = profiler_result["layer_means"]
        self.channel_means = profiler_result["channel_means"]

        self.activations = {}
        self.pool_inputs = {}
        self.neuron_deltas = {}
        self.layer_deltas = {}
        self.channel_deltas = {}
        self.block_deltas = {}

    # Register forward hook for each layer to collect activation information
    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            self.activations[layer_name] = output.detach()
            # Store input activation for maxpool layers (needed for pool_indices in backward)
            if isinstance(module, nn.MaxPool2d):
                self.pool_inputs[layer_name] = input[0].detach()

        return hook

    # Compute input delta by subtracting mean input from current input
    def _compute_input_delta(self, sample):
        mean = self.neuron_means["input"].to(sample.device)
        self.input_delta = sample - mean

        self.neuron_deltas["input"] = self.input_delta
        self.layer_deltas["input"] = self.input_delta.abs().mean()

    # Compute channel delta by averaging absolute neuron deltas over spatial dimensions
    def _compute_channel_deltas(self, layer_name):
        neuron_deltas = self.neuron_deltas[layer_name]
        self.channel_deltas[layer_name] = neuron_deltas.abs().mean(dim=(-2, -1))

    # Compute neuron delta by subtracting mean activation from current activation
    def _compute_neuron_deltas(self, layer_name):
        activation = self.activations[layer_name]
        mean = self.neuron_means[layer_name].to(activation.device)

        delta = activation - mean
        self.neuron_deltas[layer_name] = delta

    # Compute layer delta by averaging channel deltas (if conv) or neuron deltas (if linear)
    def _compute_layer_delta(self, layer_name):
        if layer_name in self.channel_deltas:
            self.layer_deltas[layer_name] = self.channel_deltas[layer_name].mean()
        else:
            self.layer_deltas[layer_name] = self.neuron_deltas[layer_name].abs().mean()

    # Compute block deltas by averaging layer deltas within each block
    def _compute_block_deltas(self):
        for block_name, layer_names in self.blocks.items():
            deltas = []

            for layer in layer_names:
                if layer in self.layer_deltas:
                    delta = self.layer_deltas[layer]
                    if isinstance(delta, torch.Tensor):
                        deltas.append(delta.abs())
                    else:
                        deltas.append(torch.tensor(abs(delta)))

            if deltas:
                # Stack deltas and compute mean
                stacked_deltas = torch.stack(deltas)
                self.block_deltas[block_name] = stacked_deltas.mean()

    def execute(self, sample):
        """Run forward analysis for the given sample"""
        self._reset_stats()
        self._register_hooks()
        self.model.eval()

        sample = ensure_tensor_batch(sample)

        with torch.no_grad():
            _ = self.model(sample)

        self._compute_input_delta(sample)

        for layer_name in self.activations:
            self._compute_neuron_deltas(layer_name)

            if isinstance(self.layer_types[layer_name], nn.Conv2d):
                self._compute_channel_deltas(layer_name)

            self._compute_layer_delta(layer_name)

        self._identify_blocks(self.layer_deltas.keys())
        self._compute_block_deltas()

        self._remove_hooks()

        return {
            "activations": self.activations,
            "pool_inputs": self.pool_inputs,
            "neuron_deltas": self.neuron_deltas,
            "layer_deltas": self.layer_deltas,
            "channel_deltas": self.channel_deltas,
            "block_deltas": self.block_deltas,
            "blocks": self.blocks,
        }

    def _reset_stats(self):
        self.activations = {}
        self.pool_inputs = {}
        self.neuron_deltas = {}
        self.layer_deltas = {}
        self.channel_deltas = {}
        self.block_deltas = {}
        self.blocks = {}
