# scripts/test_slicer_simple_nn.py

import torch
from models.simple import SimpleNN
from core.slicer import Slicer

# Load sample Model
model = SimpleNN()
model.eval()
 
x = torch.tensor([[1.0, 2.0]])  # shape (1, 2)

y = model(x)

# Execute Slicer
slicer = Slicer(
    model,
    profiling_samples=torch.zeros_like(x),  # Baseline = 0 as in paper
    input_sample=x
)

result = slicer.execute(
    target_index=0,
    theta=0.3
)

assert result["slice"] is not None