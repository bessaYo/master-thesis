import torch
import torch.nn as nn


class CounterfactualEvaluator:
    """Evaluate necessity and sufficiency of slice neurons by scaling activations"""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.final_layer = self._find_final_layer(model)

    def _get_scaled_mask(self, contrib, scale, target="slice"):
        in_slice = (contrib != 0).float()
        if target == "slice":
            return in_slice * scale + (1.0 - in_slice)
        else:
            return in_slice + (1.0 - in_slice) * scale

    def _register_hooks(self, contributions, scale=None, scale_target="slice"):
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if name not in contributions or name == self.final_layer:
                continue

            mask = self._get_scaled_mask(contributions[name], scale, target=scale_target)

            def make_hook(mask):
                def hook(_, __, output):
                    return output * mask
                return hook

            self.hooks.append(module.register_forward_hook(make_hook(mask)))

    def evaluate(self, input, contributions, target_class):
        """Run necessity and sufficiency tests at multiple scaling factors"""
        scales = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        self.model.eval()
        results = {"necessity": {}, "sufficiency": {}}

        with torch.no_grad():
            results["original"] = self._forward_pass(input, target_class)

            for s in scales:
                self._register_hooks(contributions, scale=s, scale_target="slice")
                results["necessity"][s] = self._forward_pass(input, target_class)
                self._remove_hooks()

                self._register_hooks(contributions, scale=s, scale_target="nonslice")
                results["sufficiency"][s] = self._forward_pass(input, target_class)
                self._remove_hooks()

        return results

    def _forward_pass(self, input, target_class):
        logits = self.model(input)[0]
        probs = torch.softmax(logits, dim=-1)
        return {
            "pred": logits.argmax().item(),
            "target_prob": probs[target_class].item(),
        }

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _find_final_layer(self, model):
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                return name
        return None
