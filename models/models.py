import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# Same model of paper with fixed weights for test purposes
class PaperNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)
        self.fc3 = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.fc1.weight[:] = torch.tensor([
                [3, 1],
                [-2, 2],
                [0, 1]
            ])
            self.fc2.weight[:] = torch.tensor([
                [1, 0., 2],
                [-1, 3, -2]
            ])
            self.fc3.weight[:] = torch.tensor([
                [2, 1],
                [1, 3]
            ])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    

# Feedforward Neural Network for test purposes
class SimpleNN(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list = [128, 64],
        output_size: int = 10,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Convolutional Neural Network for test purposes
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))   # now an actual module
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Factory method to get models by name
def get_model(name: str):
    name = name.lower()
    if name == "resnet18":
        return models.resnet18(weights="IMAGENET1K_V1")
    if name == "lenet":
        return LeNet()
    if name == "simplenn":
        return SimpleNN()
    if name == "simplecnn":
        return SimpleCNN()
    if name == "papernetwork":
        return PaperNetwork()
    raise ValueError(f"Unknown model: {name}")
