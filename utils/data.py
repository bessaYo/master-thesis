from torchvision import datasets, transforms


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
    import os
    import sys

    if os.path.isdir("data/imagenet/val"):
        root = "data/imagenet/val"
    elif os.path.isdir("data/imagenet"):
        root = "data/imagenet"
    else:
        print("Error: ImageNet dataset not found at data/imagenet/")
        print("Download the validation set from https://image-net.org/ and place it in data/imagenet/")
        sys.exit(1)
    return datasets.ImageFolder(root=root, transform=IMAGENET_TRANSFORM)


def get_samples_for_classes(dataset, classes, per_class):
    """Get samples for specified classes"""
    samples = {c: [] for c in classes}
    collected = 0
    for i, (img, label) in enumerate(dataset):
        if label in classes and len(samples[label]) < per_class:
            samples[label].append((img, label, i))
            collected += 1
        if collected >= len(classes) * per_class:
            break
    result = []
    for c in classes:
        result.extend(samples[c])
    return result
