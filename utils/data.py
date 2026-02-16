# utils/data.py

from torch.utils.data import Subset
from torchvision import datasets, transforms
from collections import defaultdict
from typing import List


MNIST_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGENETTE_CLASSES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]


# mean/std computed over the training sets (values from PyTorch)
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

CIFAR10_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# ImageNet preprocessing used by torchvision pretrained models
IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

IMAGENETTE_TRANSFORM = IMAGENET_TRANSFORM  # same preprocessing


# ImageNette local index (0-9) → ImageNet class index (for pretrained ResNet with 1000 classes)
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def imagenette_local_to_imagenet(local_idx):
    """Convert local ImageNette index (0-9) to ImageNet class index"""
    return IMAGENETTE_TO_IMAGENET[local_idx]


def load_mnist(train: bool = False):
    return datasets.MNIST(root="data", train=train, transform=MNIST_TRANSFORM, download=True)


def load_cifar10(train: bool = False):
    return datasets.CIFAR10(root="data", train=train, transform=CIFAR10_TRANSFORM, download=True)


def load_imagenet_val():
    """Load ImageNet ILSVRC2012 validation set (50k images, 1000 classes)"""
    return datasets.ImageFolder(root="data/imagenet/val", transform=IMAGENET_TRANSFORM)


def load_imagenette(train: bool = False):
    """Load ImageNette dataset (10 ImageNet classes)"""
    split = "train" if train else "val"
    return datasets.ImageFolder(root=f"data/imagenette2/{split}", transform=IMAGENETTE_TRANSFORM)


def get_samples_for_classes(dataset, classes: List[int], per_class: int) -> List[tuple]:
    """Get samples for specified classes"""
    samples = defaultdict(list)
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label in classes and len(samples[label]) < per_class:
            samples[label].append((img, label, idx))
        if all(len(samples[c]) >= per_class for c in classes):
            break
    return [s for c in classes for s in samples[c]]


def stratified_subset(dataset, num_classes: int, per_class: int) -> Subset:
    """Return a Subset with at most `per_class` samples per class"""
    counts = defaultdict(int)
    indices = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if counts[label] < per_class:
            indices.append(idx)
            counts[label] += 1
        if all(counts[c] >= per_class for c in range(num_classes)):
            break
    return Subset(dataset, indices)
