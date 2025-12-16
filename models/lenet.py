# models/lenet.py

import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2, return_indices=True)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2, return_indices=True)

        # Fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x, indices1 = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x, indices2 = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)
        
        x = self.fc3(x)
        return x