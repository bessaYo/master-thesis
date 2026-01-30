"""
Training Utilities for Master Thesis

Contains:
- Device handling (CPU/GPU)
- Data loading (CIFAR-10, MNIST)
- Training and evaluation functions
- Checkpoint saving/loading
- Profile computation

Usage:
    from utils.training import (
        get_device,
        get_dataloaders,
        train_model,
        evaluate,
        save_checkpoint,
        load_checkpoint,
        compute_profile
    )
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm


# =============================================================================
# Device Handling
# =============================================================================

def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (no GPU available)")
    
    return device


# =============================================================================
# Data Loading
# =============================================================================

# Standard normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """Get CIFAR-10 transforms with optional augmentation."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])


def get_mnist_transforms(train: bool = True) -> transforms.Compose:
    """Get MNIST transforms."""
    # MNIST doesn't benefit much from augmentation
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])


def get_dataloaders(
    dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test dataloaders.
    
    Args:
        dataset: "cifar10" or "mnist"
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        data_root: Root directory for datasets
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset.lower() == "cifar10":
        train_transform = get_cifar10_transforms(train=True)
        test_transform = get_cifar10_transforms(train=False)
        
        trainset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        testset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )
        
    elif dataset.lower() == "mnist":
        train_transform = get_mnist_transforms(train=True)
        test_transform = get_mnist_transforms(train=False)
        
        trainset = datasets.MNIST(
            root=data_root, train=True, download=True, transform=train_transform
        )
        testset = datasets.MNIST(
            root=data_root, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'cifar10' or 'mnist'.")
    
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers
    )
    
    print(f"✓ Loaded {dataset.upper()}")
    print(f"  Train: {len(trainset):,} samples")
    print(f"  Test:  {len(testset):,} samples")
    
    return train_loader, test_loader


def get_test_dataset(dataset: str = "cifar10", data_root: str = "./data"):
    """Get test dataset (not dataloader) for evaluation."""
    if dataset.lower() == "cifar10":
        transform = get_cifar10_transforms(train=False)
        return datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
    elif dataset.lower() == "mnist":
        transform = get_mnist_transforms(train=False)
        return datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if criterion:
                loss = criterion(outputs, targets)
                running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(test_loader) if criterion else 0.0
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    dataset: str = "cifar10",
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    milestones: list = [80, 120],
    gamma: float = 0.1,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Complete training pipeline.
    
    Args:
        model: Model to train
        dataset: Dataset name ("cifar10" or "mnist")
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Initial learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization
        milestones: Epochs for LR decay
        gamma: LR decay factor
        save_path: Path to save best checkpoint
        device: Device to use (auto-detect if None)
        
    Returns:
        Dictionary with training history and best accuracy
    """
    # Setup device
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {n_params:,} parameters")
    
    # Data
    train_loader, test_loader = get_dataloaders(dataset, batch_size)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }
    best_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"LR: {current_lr:.4f} | "
              f"Train Loss: {train_loss:.3f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Test Acc: {test_acc:.1f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            if save_path:
                save_checkpoint(model, save_path, {
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'dataset': dataset
                })
                print(f"  → Saved best model (acc: {test_acc:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best accuracy: {best_acc:.1f}%")
    print(f"{'='*60}")
    
    return {
        'history': history,
        'best_acc': best_acc
    }


# =============================================================================
# Checkpoint Handling
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict] = None
):
    """Save model checkpoint with optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Returns:
        Metadata dictionary
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    # Handle both old (state_dict only) and new format
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        metadata = checkpoint.get('metadata', {})
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    
    print(f"✓ Loaded checkpoint from {path}")
    if metadata:
        print(f"  Metadata: {metadata}")
    
    return metadata


# =============================================================================
# Profile Computation
# =============================================================================

def compute_profile(
    model: nn.Module,
    dataset: str = "cifar10",
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute activation profile (mean activations over training set).
    
    This is used as baseline for the forward analysis phase.
    
    Args:
        model: Trained model
        dataset: Dataset to profile on
        device: Device to use
        save_path: Path to save profile
        
    Returns:
        Dictionary mapping layer names to mean activations
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    model.eval()
    
    # Get training data (we profile on train set like the paper)
    train_loader, _ = get_dataloaders(dataset, batch_size=128)
    
    # Storage for activations
    activation_sums = {}
    activation_counts = {}
    
    # Register hooks to capture activations
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activation_sums:
                activation_sums[name] = torch.zeros_like(output[0]).to(device)
                activation_counts[name] = 0
            
            # Sum activations (mean over batch)
            activation_sums[name] += output.mean(dim=0)
            activation_counts[name] += 1
        return hook
    
    # Register hooks for all relevant layers
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    # Run through dataset
    print(f"Computing profile over {len(train_loader.dataset):,} samples...")
    
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader, desc="Profiling"):
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute means
    profile = {}
    for name in activation_sums:
        profile[name] = activation_sums[name] / activation_counts[name]
    
    print(f"✓ Profile computed for {len(profile)} layers")
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(profile, save_path)
        print(f"✓ Profile saved to {save_path}")
    
    return profile


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Quick test
    device = get_device()
    train_loader, test_loader = get_dataloaders("cifar10", batch_size=64)
    
    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels[:10]}")