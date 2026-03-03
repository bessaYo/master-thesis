# core/tracing/base.py

from models.resnet import BasicBlock as CIFARBlock
from models.imagenet import BasicBlock, Bottleneck

BLOCK_TYPES = (CIFARBlock, BasicBlock, Bottleneck)


class BaseAnalyzer:
    """Base analyzer class shared by Profiler and Forward Analyzer"""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_types = {}
        self.blocks = {}

    # Register forward hook for each layer
    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            if len(list(layer.children())) > 0:
                continue
            self.layer_types[name] = layer
            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)

    # Remove all registered hooks
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    # Identify residual blocks based on layer name
    def _identify_blocks(self, layer_keys):
        self.blocks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BLOCK_TYPES):
                self.blocks[name] = []

        # Add layers to block based on name prefix matching
        for block_name in self.blocks:
            for layer_name in layer_keys:
                if layer_name.startswith(block_name + "."):
                    self.blocks[block_name].append(layer_name)
