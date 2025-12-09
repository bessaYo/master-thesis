# utils/tensor_utils.py

import torch


# Ensure that input is a tensor with batch dimension
def ensure_tensor_batch(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Add batch dimension if missing (C, H, W) -> (1, C, H, W) and (N,) -> (1, N)
    if x.dim() in (1, 3):
        x = x.unsqueeze(0)
    return x
