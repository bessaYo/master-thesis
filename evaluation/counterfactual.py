import torch
import torch.nn as nn


class CounterfactualEvaluator:
    def __init__(self, model, soft_mask: bool = False, min_mask_value: float = 0.1):
        """
        Args:
            model: The neural network model
            soft_mask: If True, use soft masking (keep min_mask_value for non-slice neurons)
            min_mask_value: Minimum mask value for non-slice neurons (only used if soft_mask=True)
        """
        self.model = model
        self.hooks = []
        self.soft_mask = soft_mask
        self.min_mask_value = min_mask_value
        self.final_layer = self._find_final_layer(model)

    def _find_final_layer(self, model):
        """Find the last Linear layer (classifier output)."""
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                return name
        return None

    # --------------------------------------------------
    def _get_mask(self, contrib, keep_slice):
        if keep_slice:
            if self.soft_mask:
                # Soft mask: non-slice neurons keep min_mask_value instead of 0
                mask = (contrib != 0).float()
                mask = torch.clamp(mask, min=self.min_mask_value)
                return mask
            else:
                # Hard mask: binary 0 or 1
                return (contrib != 0).float()
        else:
            # Remove slice: always binary
            return (contrib == 0).float()

    def _register_hooks(self, contributions, keep_slice):
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if name not in contributions:
                continue
            
            # Skip final layer - it defines the slicing criterion
            if name == self.final_layer:
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
        probs = torch.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "probs": probs,
            "pred": logits.argmax().item(),
            "target_logit": logits[target_class].item(),
            "target_prob": probs[target_class].item(),
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