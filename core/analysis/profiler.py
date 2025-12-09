# core/analysis/profiler.py

import torch
import torch.nn as nn
from utils.tensor_utils import ensure_tensor_batch

#! TODO: Multiple neuron activations (CNNs) need to be averaged over spatial dimensions

class Profiler:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []

        # Input stats
        self.input_sum = 0.0
        self.input_count = 0

        # Per layer neuron stats
        self.activation_sums = {}
        self.activation_counts = {}

        # Mean values for each level
        self.neuron_means = {}
        self.layer_means = {}
        self.block_means = {}

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
                output = output[0]  

            batch_size = output.size(0)
            batch_sum = output.sum(dim=0)

            if layer_name not in self.activation_sums:
                self.activation_sums[layer_name] = batch_sum
                self.activation_counts[layer_name] = batch_size
            else:
                self.activation_sums[layer_name] += batch_sum
                self.activation_counts[layer_name] += batch_size

        return hook

    # Remove all registered hooks from the model
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    # Compute mean over all inputs
    def _compute_input_mean(self, samples: torch.Tensor):
        for sample in samples:
            sample = ensure_tensor_batch(sample)

            self.input_sum += sample.sum(dim=0)
            self.input_count += sample.size(0)

        # Final mean
        self.neuron_means["input"] = self.input_sum / self.input_count

    # Compute mean activations for each neuron
    def _compute_neuron_mean(self):
        for layer, total_sum in self.activation_sums.items():
            count = self.activation_counts[layer]
            self.neuron_means[layer] = total_sum / count

    # Compute mean activations for each layer
    def _compute_layer_mean(self):
        for layer, neuron_means in self.neuron_means.items():
            self.layer_means[layer] = neuron_means.mean()

    # Compute mean activations for each block
    def compute_block_mean(self):
        pass

    # Execute profiling over the provided samples
    def execute(self, samples):
        self._reset_stats()
        self._register_hooks()
        self.model.eval()

        self._compute_input_mean(samples)

        with torch.no_grad():
            for sample in samples:
                sample = ensure_tensor_batch(sample)
                self.model(sample)

        self._remove_hooks()
        self._compute_neuron_mean()
        self._compute_layer_mean()

        return {
            "neuron_means": self.neuron_means,
            "layer_means": self.layer_means,
        }

    # Reset all stored statistics
    def _reset_stats(self):
        self.input_sum = 0.0
        self.input_count = 0

        self.activation_sums = {}
        self.activation_counts = {}

        self.neuron_means = {}
        self.layer_means = {}
        self.block_means = {}
