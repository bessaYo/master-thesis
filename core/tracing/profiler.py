# core/tracing/profiler.py

import torch
import torch.nn as nn
from utils.tensor_utils import ensure_tensor_batch
from models.resnet import BasicBlock

#! TODO: Multiple neuron activations (CNNs) need to be averaged over spatial dimensions

class Profiler:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_types = {}
        self.blocks = {}

        # Input stats
        self.input_sum = 0.0
        self.input_count = 0

        # Per layer neuron stats
        self.activation_sums = {}
        self.activation_counts = {}

        # Mean values for each level
        self.neuron_means = {}
        self.channel_means = {}
        self.layer_means = {}
        self.block_means = {}

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
    def _compute_input_mean(self, samples: torch.Tensor, batch_size=128):
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            self.input_sum += batch.sum(dim=0)
            self.input_count += batch.size(0)

            # Final mean
            self.neuron_means["input"] = self.input_sum / self.input_count

    # Compute mean activations for each neuron
    def _compute_neuron_mean(self):
        for layer, total_sum in self.activation_sums.items():
            count = self.activation_counts[layer]
            self.neuron_means[layer] = total_sum / count

    # Compute mean activations for each channel
    def _compute_channel_mean(self):
        for layer_name, neuron_means in self.neuron_means.items():
            layer = self.layer_types.get(layer_name)

            if isinstance(layer, nn.Conv2d):
                # neuron_means: [C,H,W]
                self.channel_means[layer_name] = neuron_means.mean(dim=(1, 2))

    # Compute mean activations for each layer
    def _compute_layer_mean(self):
        for layer_name, neuron_means in self.neuron_means.items():
            layer = self.layer_types.get(layer_name)

            if isinstance(layer, nn.Conv2d):
                self.layer_means[layer_name] = self.channel_means[layer_name].mean()
            else:
                self.layer_means[layer_name] = neuron_means.mean()

    # Compute mean activations for each block
    def compute_block_mean(self):
        for block_name, layer_names in self.blocks.items():
            deltas = []

            for layer in layer_names:
                if layer in self.layer_means:
                    deltas.append(self.layer_means[layer].abs())

            # Average over layers in the block if delta list is not empty
            if deltas:
                self.block_means[block_name] = torch.stack(deltas).mean()

    # Identify basic blocks (residual blocks) in the model
    def _identify_blocks(self):
        self.blocks = {}

        # Find blocks
        for name, module in self.model.named_modules():
            if isinstance(module, BasicBlock):
                self.blocks[name] = []

        # Add layers to blocks
        for block_name in self.blocks:
            for layer_name in self.layer_means.keys():
                if layer_name.startswith(block_name + "."):
                    self.blocks[block_name].append(layer_name)

    # Execute profiling over the provided samples
    def execute(self, samples, batch_size=128):
        self._reset_stats()
        self._register_hooks()
        self.model.eval()

        self._compute_input_mean(samples, batch_size=batch_size)

        with torch.no_grad():
            # Batched forward passes
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                self.model(batch)

        self._remove_hooks()
        self._compute_neuron_mean()
        self._compute_channel_mean()
        self._compute_layer_mean()

        # For Residual Blocks
        self._identify_blocks()
        self.compute_block_mean()

        return {
            "neuron_means": self.neuron_means,
            "channel_means": self.channel_means,
            "layer_means": self.layer_means,
            "block_means": self.block_means,
            "blocks": self.blocks,
        }

    # Reset all stored statistics
    def _reset_stats(self):
        self.input_sum = 0.0
        self.input_count = 0

        self.activation_sums = {}
        self.activation_counts = {}

        self.neuron_means = {}
        self.channel_means = {}
        self.layer_means = {}
        self.block_means = {}
