# Quick debug
import torch
from models import get_model

model = get_model('resnet18', pretrained=True)
fc = model.fc  # oder model.linear, je nach Implementation

print("Linear layer bias:")
for i, (name, bias) in enumerate(zip(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'], fc.bias)):
    print(f"  {name}: {bias.item():.4f}")