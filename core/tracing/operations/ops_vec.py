import torch
import torch.nn.functional as F

from core.filtering import ThetaSlicer


class BackwardOperations:
    """Defines contribution operations for different layer types / modules"""

    def __init__(self, theta=0.0):
        self.theta_filter = ThetaSlicer(theta=theta)
        self.theta = theta

    def linear(self, module, CONTRIB_n, delta_n, delta_i, activation_n):
        """Backward contribution for linear layer"""
        original_shape = delta_i.shape

        # Remove batch dimension and flatten for matrix operations
        delta_i = delta_i.squeeze(0).flatten()
        CONTRIB_n = CONTRIB_n.squeeze(0).flatten()
        delta_n = delta_n.squeeze(0).flatten()
        activation_n = activation_n.squeeze(0).flatten()

        # Step 1: Filter neurons by magnitude (weight * delta_i) with theta threshold
        weight = module.weight.detach()
        magnitude = weight * delta_i
        if self.theta > 0:
            keep_neurons = self.theta_filter.filter_linear(magnitude, activation_n)
        else:
            keep_neurons = torch.ones_like(magnitude, dtype=torch.bool)

        # Step 2: Compute local contribution for each input neuron, mask with keep_neurons
        active_outputs = (CONTRIB_n != 0).unsqueeze(1)
        local_contrib = CONTRIB_n.unsqueeze(1) * delta_n.unsqueeze(1) * magnitude

        # keep only sign as NNSlicer does
        local_contrib = torch.sign(local_contrib) * keep_neurons.float() * active_outputs.float()

        # Synapse contributions: [out_features, in_features] — sign per weight
        synapse_contrib = local_contrib.clone()

        # Step 3: For every input neuron, accumulate local contributions
        contrib = local_contrib.sum(dim=0)  # sum over output neurons → one value per input neuron

        # Reshape to match original delta_i shape
        neuron_contrib = contrib.reshape(original_shape)

        return neuron_contrib, synapse_contrib

    def conv2d(self, module, CONTRIB_n, delta_n, delta_i, activation_n):
        """Backward contribution for convolutional layer"""
        # Get weights, kernel size, stride and padding from current module
        weight = module.weight.detach()  # [out_channels, in_channels, kH, kW]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        # Step 1: Create patches of each kernel window with unfold
        patches = F.unfold(delta_i, kernel_size, stride=stride, padding=padding)

        # Step 2: Compute magnitude per position
        weight_flat = weight.flatten(1).unsqueeze(2)
        magnitude = weight_flat * patches

        # Step 3: Apply theta filter per output channel
        if self.theta > 0:
            activation_n = activation_n.squeeze(0)  # [out_ch, H, W]
            magnitude = self.theta_filter.filter_conv(magnitude, activation_n)

        # Step 4: Compute local contributions
        CONTRIB_flat = CONTRIB_n.flatten(2).squeeze(0).unsqueeze(1)  # flatten spatial dims, expand for kernel broadcasting
        delta_n_flat = delta_n.flatten(2).squeeze(0).unsqueeze(1)

        active = CONTRIB_flat != 0
        local = CONTRIB_flat * delta_n_flat * magnitude
        local = torch.sign(local) * active.float()

        # Synapse contributions: sum over spatial positions → [out_ch, in_ch*kH*kW]
        # Then reshape to weight shape [out_ch, in_ch, kH, kW]
        synapse_contrib = local.sum(dim=2).reshape(weight.shape)

        # Step 5: Accumulate over output channels
        neuron_contrib = local.sum(dim=0).unsqueeze(0)  # sum over output channels, add batch dim
        neuron_contrib = F.fold(neuron_contrib, delta_i.shape[2:], kernel_size, stride=stride, padding=padding)
        return neuron_contrib, synapse_contrib

    def maxpool(self, module, CONTRIB_n, delta_n, delta_i, pool_input):
        """Backward contribution for max pooling"""
        # Get kernel size, stride and padding from current module
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        # Get positions of max indices from forward pass
        max_indices = F.max_pool2d(pool_input, kernel_size, stride, padding, return_indices=True)[1]

        # Gather positions of max indices in the input delta, reshape to flat to apply gather function
        flat_delta_i = delta_i.flatten(2)
        flat_max_indices = max_indices.flatten(2)
        max_delta_i = torch.gather(flat_delta_i, 2, flat_max_indices)
        max_delta_i = max_delta_i.view_as(CONTRIB_n)  # Reshape back to output shape

        # Compute local contribution
        contrib = torch.sign(CONTRIB_n * delta_n * max_delta_i)
        mask = (CONTRIB_n != 0).float()
        contrib = contrib * mask  # Zero out contributions where CONTRIB_n is zero

        # Scatter contributions back to the positions of the max values in the input
        out = torch.zeros_like(flat_delta_i)
        out.scatter_(2, flat_max_indices, contrib.flatten(2))
        return out.view_as(delta_i)

    def avgpool(self, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for average pooling layers"""
        return self._passthrough(CONTRIB_n, delta_n, delta_i)

    def relu(self, activation, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for ReLU activation layers"""
        # Only neurons that were active in the forward pass can contribute
        positive_activation = activation > 0
        mask = positive_activation.float()
        return self._passthrough(CONTRIB_n, delta_n, delta_i * mask)

    def batchnorm2d(self, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for batch normalization layers"""
        # BN is treated as passthrough. Values are scaled but it doesn't change which neurons matter
        return self._passthrough(CONTRIB_n, delta_n, delta_i)

    def add(self, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for add operation in residual blocks"""
        return self._passthrough(CONTRIB_n, delta_n, delta_i)

    def flatten(self, CONTRIB_n, delta_n, delta_i):
        """Backward contribution for flatten operation"""
        flatten_contrib = CONTRIB_n.reshape(delta_i.shape)
        return flatten_contrib

    def _passthrough(self, CONTRIB_n, delta_n, delta_i):
        """Generic passthrough contribution function"""
        local_contrib = CONTRIB_n * delta_n * delta_i
        return torch.sign(local_contrib)
