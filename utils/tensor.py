import torch


def ensure_tensor_batch(x):
    """Ensure input is a tensor with batch dimension: (C,H,W) -> (1,C,H,W)"""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    if x.dim() in (1, 3):
        x = x.unsqueeze(0)
    return x
