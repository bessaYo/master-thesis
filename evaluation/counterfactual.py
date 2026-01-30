import torch
import torch.nn as nn


class CounterfactualEvaluator:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    # --------------------------------------------------
    def _get_mask(self, contrib, keep_slice):
        if keep_slice:
            return (contrib != 0).float()
        else:
            return (contrib == 0).float()

    def _register_hooks(self, contributions, keep_slice):
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if name not in contributions:
                continue

            mask = self._get_mask(contributions[name], keep_slice)

            def make_hook(mask):
                def hook(_, __, output):
                    return output * mask
                return hook

            self.hooks.append(module.register_forward_hook(make_hook(mask)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    # --------------------------------------------------
    @staticmethod
    def _entropy(logits):
        p = torch.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-8)).sum().item()

    @staticmethod
    def _margin(logits):
        vals, _ = torch.topk(logits, 2)
        return (vals[0] - vals[1]).item()

    def _run(self, input, target_class):
        logits = self.model(input)[0]
        return {
            "logits": logits,
            "pred": logits.argmax().item(),
            "target_logit": logits[target_class].item(),
            "margin": self._margin(logits),
            "entropy": self._entropy(logits),
        }

    # --------------------------------------------------
    def evaluate(self, input, contributions, target_class):
        self.model.eval()

        with torch.no_grad():
            original = self._run(input, target_class)

            self._register_hooks(contributions, keep_slice=True)
            keep = self._run(input, target_class)
            self._remove_hooks()

            self._register_hooks(contributions, keep_slice=False)
            remove = self._run(input, target_class)
            self._remove_hooks()

        return {
            "original": original,
            "keep": keep,
            "remove": remove,
        }