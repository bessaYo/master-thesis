import torch
import torch.nn.functional as F

from core.tracing.operations.ops_vec import BackwardOperations


class BackwardOperationsLoop(BackwardOperations):
    """Loop based backward operations for linear and conv layers"""

    def linear(self, module, CONTRIB_n, delta_n, delta_i, activation_n):
        """Loop based backward contribution for linear layer"""
        original_shape = delta_i.shape

        delta_i = delta_i.squeeze(0).flatten()
        CONTRIB_n = CONTRIB_n.squeeze(0).flatten()
        delta_n = delta_n.squeeze(0).flatten()
        activation_n = activation_n.squeeze(0).flatten()

        weight = module.weight.detach()
        out_features, in_features = weight.shape

        # Theta filtering
        magnitude = weight * delta_i
        if self.theta > 0:
            keep_neurons = self.theta_filter.filter_linear(magnitude, activation_n)
        else:
            keep_neurons = torch.ones_like(magnitude, dtype=torch.bool)

        synapse_contrib = torch.zeros(out_features, in_features)
        neuron_contrib = torch.zeros(in_features)

        # Loop over output and input neurons
        for out_neuron in range(out_features):
            if CONTRIB_n[out_neuron] == 0:
                continue
            for i in range(in_features):
                if not keep_neurons[out_neuron, i]:
                    continue
                local = CONTRIB_n[out_neuron] * delta_n[out_neuron] * magnitude[out_neuron, i]
                sign = torch.sign(local)
                synapse_contrib[out_neuron, i] = sign
                neuron_contrib[i] += sign

        neuron_contrib = neuron_contrib.reshape(original_shape)
        return neuron_contrib, synapse_contrib

    def conv2d(self, module, CONTRIB_n, delta_n, delta_i, activation_n):
        """Loop based backward contribution for convolutional layer"""
        weight = module.weight.detach()
        out_channels, in_channels, kH, kW = weight.shape
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        # Unfold into patches
        patches = F.unfold(delta_i, kernel_size, stride=stride, padding=padding)
        num_positions = patches.shape[2]

        weight_flat = weight.flatten(1)

        magnitude_full = weight_flat.unsqueeze(2) * patches

        # Theta filtering
        if self.theta > 0:
            act = activation_n.squeeze(0)
            magnitude_full = self.theta_filter.filter_conv(magnitude_full, act)

        CONTRIB_flat = CONTRIB_n.flatten(2).squeeze(0)
        delta_n_flat = delta_n.flatten(2).squeeze(0)

        # Accumulate contributions per neuron and synapse
        synapse_accum = torch.zeros_like(weight)
        neuron_accum = torch.zeros(in_channels * kH * kW, num_positions)

        # Loop over output channels and spatial positions of kernel
        for out_channel in range(out_channels):
            for position in range(num_positions):
                # Skip if no contribution for this output channel and position
                if CONTRIB_flat[out_channel, position] == 0:
                    continue

                # Loop over every kernel element and apply contribution formula
                for k_pos in range(in_channels * kH * kW):
                    local = CONTRIB_flat[out_channel, position] * delta_n_flat[out_channel, position] * magnitude_full[out_channel, k_pos, position] # local contribution formula
                    sign = torch.sign(local) # sign of local contribution
                    neuron_accum[k_pos, position] += sign # accumulate contribution to input neuron
                    synapse_accum.view(out_channels, -1)[out_channel, k_pos] += sign # accumulate contribution to synapse

        synapse_contrib = torch.sign(synapse_accum)

        neuron_contrib = neuron_accum.unsqueeze(0)

        # Fold back from patches to original dimension
        neuron_contrib = F.fold(neuron_contrib, delta_i.shape[2:], kernel_size, stride=stride, padding=padding)
        return neuron_contrib, synapse_contrib
