# evaluation/counterfactual.py

import torch
import torch.nn as nn


class CounterfactualEvaluator:
    """Class to evaluate counterfactual reasoning"""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.final_layer = self._find_final_layer(model)

    def _get_scaled_mask(self, contrib, scale, target="slice"):
        """Return a mask that scales neurons"""
        in_slice = (contrib != 0).float()
        if target == "slice":
            return in_slice * scale + (1.0 - in_slice)
        else:
            return in_slice + (1.0 - in_slice) * scale

    def _register_hooks(self, contributions, scale=None, scale_target="slice"):
        """Register forward hooks to scale neuron activations"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if name not in contributions:
                continue

            # Skip final layer since it directly produces logits
            if name == self.final_layer:
                continue

            # Scale neurons by factor based on whether they are in the slice or not
            mask = self._get_scaled_mask(contributions[name], scale, target=scale_target)

            def make_hook(mask):
                def hook(_, __, output):
                    return output * mask

                return hook

            self.hooks.append(module.register_forward_hook(make_hook(mask)))

    def evaluate(self, input, contributions, target_class):
        """Evaluate necessity and sufficiency of slice neurons by scaling them down"""
        scales = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        self.model.eval()

        results = {"necessity": {}, "sufficiency": {}}

        with torch.no_grad():
            results["original"] = self._forward_pass(input, target_class)

            for s in scales:
                # Necessity: scale down slice neurons
                self._register_hooks(contributions, scale=s, scale_target="slice")
                results["necessity"][s] = self._forward_pass(input, target_class)
                self._remove_hooks()

                # Sufficiency: scale down non-slice neurons
                self._register_hooks(contributions, scale=s, scale_target="nonslice")
                results["sufficiency"][s] = self._forward_pass(input, target_class)
                self._remove_hooks()

        return results

    def _forward_pass(self, input, target_class):
        """Run a forward pass and return predicted class and target class probability"""
        logits = self.model(input)[0]
        probs = torch.softmax(logits, dim=-1)
        return {
            "pred": logits.argmax().item(),
            "target_prob": probs[target_class].item(),
        }

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _find_final_layer(self, model):
        """Find the name of the final linear layer in the model"""
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                return name
        return None
