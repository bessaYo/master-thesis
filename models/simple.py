# models/simple.py
import torch
import torch.nn as nn


# Same model of paper with fixed weights for test purposes
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)
        self.fc3 = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.fc1.weight[:] = torch.tensor([[3, 1], [-2, 2], [0, 1]])
            self.fc2.weight[:] = torch.tensor([[1, 0.0, 2], [-1, 3, -2]])
            self.fc3.weight[:] = torch.tensor([[2, 1], [1, 3]])

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Convolutional Neural Network for test purposes
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv: 1 input channel → 2 feature maps
        self.conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Pooling: 2x2 Avg Pool
        self.pool = nn.AvgPool2d(kernel_size=2)

        # Fully connected layer: 2 inputs (from 2 feature maps) → 1 output
        self.fc = nn.Linear(2, 1, bias=False)

        with torch.no_grad():
            # conv.weight shape: (out_channels, in_channels, kH, kW)
            self.conv.weight[:] = torch.tensor(
                [
                    [[[1.0, -1.0], [0.0, 2.0]]],  # Feature map 1
                    [[[0.0, 1.0], [1.0, -1.0]]],  # Feature map 2
                ]
            )

            # fc.weight shape = (1, 2)
            self.fc.weight[:] = torch.tensor([[4.0, -2.0]])

    def forward(self, x):
        x = self.conv(x)  # (B, 2, 2, 2)
        x = self.relu(x)  # ReLU
        x = self.pool(x)  # (B, 2, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten → (B, 2)
        x = self.fc(x)  # Final output
        return x
