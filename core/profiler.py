import torch
import torch.nn as nn


class Profiler:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.layer_sums = {}
        self.input_sum = None
        self.input_count = 0

    def register_hooks(self):

        self._register_input_hook()
        for name, module in self.model.named_modules():

            if len(list(module.children())) > 0:
                continue

            hook = module.register_forward_hook(self._hook_fn(name, module))
            self.hooks.append(hook)

    def _register_input_hook(self):
        def pre_hook(module, input):
            x = input[0]
            batch_sum = x.sum(dim=0)
            batch_size = x.size(0)

            if self.input_sum is None:
                self.input_sum = batch_sum
                self.input_count = batch_size
            else:
                self.input_sum += batch_sum
                self.input_count += batch_size

        hook = self.model.register_forward_pre_hook(pre_hook)
        self.hooks.append(hook)
    

    def _hook_fn(self, name: str, module: nn.Module):
        def hook(module, input, output):

            # First we aggregate over the batch dimension
            batch_sum, batch_size = self.aggregate_batch(output, module)

            # If first time seeing this layer, initialize storage
            if name not in self.layer_sums:
                self.layer_sums[name] = {"sum": batch_sum, "count": batch_size}

            # If not first time seeing this layer, accumulate values
            else:
                self.layer_sums[name]["sum"] += batch_sum
                self.layer_sums[name]["count"] += batch_size

        return hook

    # If we have batch of samples, we aggregate values accordingly
    def aggregate_batch(self, output: torch.Tensor, module: nn.Module):
        batch_size = output.size(0)

        # Convolutional layers / normalization / pooling (mean over batch and spatial dimensions)
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d)):
            batch_sum = (
                output.sum(dim=(0, 2, 3)) if output.dim() == 4 else output.sum(dim=0)
            )

        # Fully connected layers (mean over batch dimension)
        elif isinstance(module, nn.Linear):
            batch_sum = output.sum(dim=0)

        # Activation layers
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
            batch_sum = (
                output.sum(dim=(0, 2, 3)) if output.dim() == 4 else output.sum(dim=0)
            )

        # Flatten layers
        elif isinstance(module, nn.Flatten):
            batch_sum = output.sum(dim=0)

        # Regularization layers
        elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
            batch_sum = (
                output.sum(dim=(0, 2, 3)) if output.dim() == 4 else output.sum(dim=0)
            )

        # Default fallback
        else:
            batch_sum = (
                output.sum(dim=(0, 2, 3)) if output.dim() == 4 else output.sum(dim=0)
            )

        return batch_sum, batch_size

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def compute_means(self):
        means = {
            name: data["sum"] / data["count"]
            for name, data in self.layer_sums.items()
        }

        if self.input_sum is not None:
            means["input"] = self.input_sum / self.input_count

        return means
