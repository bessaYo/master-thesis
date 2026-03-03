# core/tracing/profiler.py

import torch
import torch.nn as nn

from core.tracing.base import BaseAnalyzer


class Profiler(BaseAnalyzer):
    """Collects activations for each layer and computes mean activation across a given dataset"""

    def __init__(self, model):
        super().__init__(model)

        self.input_sum = 0.0
        self.input_count = 0

        self.activation_sums = {}
        self.activation_counts = {}

        self.neuron_means = {}
        self.channel_means = {}
        self.layer_means = {}
        self.block_means = {}

    # Register forward hook for each layer to collect activation information
    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            # Sum over batch dimension to get total activation for this layer
            batch_size = output.size(0)
            batch_sum = output.sum(dim=0)

            if layer_name not in self.activation_sums:
                self.activation_sums[layer_name] = batch_sum
                self.activation_counts[layer_name] = batch_size
            else:
                self.activation_sums[layer_name] += batch_sum
                self.activation_counts[layer_name] += batch_size

        return hook

    # Compute input mean by summing over all samples and dividing by total count
    def _compute_input_mean(self, samples: torch.Tensor, batch_size=128):
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            self.input_sum += batch.sum(dim=0)
            self.input_count += batch.size(0)

        self.neuron_means["input"] = self.input_sum / self.input_count

    # Compute mean for each neuron by dividing total sum of activations by count of samples
    def _compute_neuron_mean(self):
        for layer, total_sum in self.activation_sums.items():
            count = self.activation_counts[layer]
            self.neuron_means[layer] = total_sum / count

    # Compute mean for channels by averaging over spatial dims
    def _compute_channel_mean(self):
        for layer_name, neuron_means in self.neuron_means.items():
            layer = self.layer_types.get(layer_name)

            if isinstance(layer, nn.Conv2d):
                self.channel_means[layer_name] = neuron_means.mean(dim=(1, 2))

    # Compute mean for each layer by averaging channel means (if conv) or neuron means (if linear)
    def _compute_layer_mean(self):
        for layer_name, neuron_means in self.neuron_means.items():
            layer = self.layer_types.get(layer_name)

            if isinstance(layer, nn.Conv2d):
                self.layer_means[layer_name] = self.channel_means[layer_name].mean()
            else:
                self.layer_means[layer_name] = neuron_means.mean()

    # Compute mean for each block by averaging layer means
    def _compute_block_mean(self):
        for block_name, layer_names in self.blocks.items():
            deltas = []

            # If layer in block, add to delta list
            for layer in layer_names:
                if layer in self.layer_means:
                    deltas.append(self.layer_means[layer].abs())

            if deltas:
                # Stack delta tensors to one big tensor
                stacked_deltas = torch.stack(deltas)
                self.block_means[block_name] = stacked_deltas.mean()

    def execute(self, samples, batch_size=128):
        """Profile over the provided samples"""
        self._reset_stats()
        self._register_hooks()
        self.model.eval()

        self._compute_input_mean(samples, batch_size=batch_size)

        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                self.model(batch)

        self._remove_hooks()

        # Compute mean activations at all levels
        self._compute_neuron_mean()
        self._compute_channel_mean()
        self._compute_layer_mean()

        self._identify_blocks(self.layer_means.keys())
        self._compute_block_mean()

        return {
            "neuron_means": self.neuron_means,
            "channel_means": self.channel_means,
            "layer_means": self.layer_means,
            "block_means": self.block_means,
            "blocks": self.blocks,
        }

    def _reset_stats(self):
        self.input_sum = 0.0
        self.input_count = 0
        self.activation_sums = {}
        self.activation_counts = {}
        self.neuron_means = {}
        self.channel_means = {}
        self.layer_means = {}
        self.block_means = {}
