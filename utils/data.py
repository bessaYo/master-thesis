from torchvision import datasets, transforms
from collections import defaultdict


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


# mean/std computed over the training sets (values from PyTorch)
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


def load_cifar10(train=False):
    return datasets.CIFAR10(root="data", train=train, transform=CIFAR10_TRANSFORM, download=True)


def load_imagenet_val():
    """Load ImageNet ILSVRC2012 validation set"""
    return datasets.ImageFolder(root="data/imagenet/val", transform=IMAGENET_TRANSFORM)


def get_samples_for_classes(dataset, classes, per_class):
    """Get samples for specified classes"""
    samples = defaultdict(list)
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label in classes and len(samples[label]) < per_class:
            samples[label].append((img, label, idx))
        if all(len(samples[c]) >= per_class for c in classes):
            break
    return [s for c in classes for s in samples[c]]
