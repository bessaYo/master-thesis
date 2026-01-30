import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import sys
import csv
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from core.slicer import Slicer
from evaluation.counterfactual import CounterfactualEvaluator


# =============================================================================
# Constants
# =============================================================================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ConfidenceMetrics:
    """Confidence-based metrics for a single evaluation."""
    target_confidence: float  # Softmax probability for target class
    target_logit: float       # Raw logit for target class
    predicted_class: int      # argmax prediction
    predicted_confidence: float  # Confidence for predicted class
    logit_gap: float          # target_logit - second_highest_logit
    entropy: float            # Entropy of softmax distribution
    
    @classmethod
    def from_logits(cls, logits: torch.Tensor, target_class: int) -> 'ConfidenceMetrics':
        """Compute all metrics from raw logits."""
        probs = F.softmax(logits, dim=-1).squeeze()
        logits = logits.squeeze()
        
        target_conf = probs[target_class].item()
        target_logit = logits[target_class].item()
        pred_class = probs.argmax().item()
        pred_conf = probs[pred_class].item()
        
        # Logit gap: difference to second highest
        sorted_logits, _ = torch.sort(logits, descending=True)
        if sorted_logits[0].item() == target_logit:
            gap = target_logit - sorted_logits[1].item()
        else:
            gap = target_logit - sorted_logits[0].item()
        
        # Entropy of distribution
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        return cls(
            target_confidence=target_conf,
            target_logit=target_logit,
            predicted_class=pred_class,
            predicted_confidence=pred_conf,
            logit_gap=gap,
            entropy=entropy
        )


@dataclass
class CounterfactualResult:
    """Results from a single counterfactual evaluation."""
    sample_idx: int
    true_label: int
    slice_size_ratio: float  # Fraction of neurons in slice
    
    # Metrics for each condition
    original: ConfidenceMetrics
    keep_slice: ConfidenceMetrics
    remove_slice: ConfidenceMetrics
    
    @property
    def confidence_drop(self) -> float:
        """How much confidence drops when slice is removed."""
        return self.original.target_confidence - self.remove_slice.target_confidence
    
    @property
    def confidence_retention(self) -> float:
        """How much confidence is retained with only slice."""
        return self.keep_slice.target_confidence / (self.original.target_confidence + 1e-10)
    
    @property
    def keep_correct(self) -> bool:
        """Does keep-slice predict correctly?"""
        return self.keep_slice.predicted_class == self.true_label
    
    @property
    def remove_flipped(self) -> bool:
        """Does remove-slice flip the prediction?"""
        return self.remove_slice.predicted_class != self.true_label
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        return {
            'sample_idx': self.sample_idx,
            'true_label': self.true_label,
            'slice_size_ratio': self.slice_size_ratio,
            # Original
            'orig_confidence': self.original.target_confidence,
            'orig_logit': self.original.target_logit,
            'orig_entropy': self.original.entropy,
            # Keep slice
            'keep_confidence': self.keep_slice.target_confidence,
            'keep_logit': self.keep_slice.target_logit,
            'keep_pred': self.keep_slice.predicted_class,
            'keep_correct': self.keep_correct,
            'keep_entropy': self.keep_slice.entropy,
            # Remove slice
            'remove_confidence': self.remove_slice.target_confidence,
            'remove_logit': self.remove_slice.target_logit,
            'remove_pred': self.remove_slice.predicted_class,
            'remove_flipped': self.remove_flipped,
            'remove_entropy': self.remove_slice.entropy,
            # Derived metrics
            'confidence_drop': self.confidence_drop,
            'confidence_retention': self.confidence_retention,
        }


@dataclass
class PartialRemovalResult:
    """Results from partial slice removal test."""
    sample_idx: int
    true_label: int
    removal_fractions: List[float]  # [0.1, 0.2, ..., 1.0]
    confidences: List[float]        # Confidence at each removal level
    predictions: List[int]          # Prediction at each removal level
    
    def to_dict(self) -> dict:
        return {
            'sample_idx': self.sample_idx,
            'true_label': self.true_label,
            'removal_fractions': self.removal_fractions,
            'confidences': self.confidences,
            'predictions': self.predictions,
        }


# =============================================================================
# Enhanced Counterfactual Evaluator
# =============================================================================

class EnhancedCounterfactualEvaluator:
    """
    Enhanced evaluator with confidence-based metrics.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hooks = []
        self.device = next(model.parameters()).device
    
    def _create_mask(self, contrib: torch.Tensor, keep_slice: bool) -> torch.Tensor:
        """Create binary mask from contributions."""
        if keep_slice:
            return (contrib != 0).float()
        else:
            return (contrib == 0).float()
    
    def _register_hooks(self, contributions: Dict[str, torch.Tensor], keep_slice: bool):
        """Register forward hooks for masking."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if name not in contributions:
                continue
            
            contrib = contributions[name]
            mask = self._create_mask(contrib, keep_slice).to(self.device)
            
            def make_hook(m):
                def hook(module, input, output):
                    # Handle different output shapes
                    if m.dim() == 1:
                        # FC layer: [batch, features]
                        return output * m.unsqueeze(0)
                    elif m.dim() == 3:
                        # Conv layer: [channels, H, W] -> [1, channels, H, W]
                        return output * m.unsqueeze(0)
                    else:
                        return output * m
                return hook
            
            hook = module.register_forward_hook(make_hook(mask))
            self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return logits."""
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            return self.model(input_tensor)
    
    def evaluate(
        self,
        input_tensor: torch.Tensor,
        contributions: Dict[str, torch.Tensor],
        target_class: int,
        sample_idx: int = 0
    ) -> CounterfactualResult:
        """
        Run full counterfactual evaluation with confidence metrics.
        """
        self.model.eval()
        
        # Calculate slice size ratio
        total_neurons = 0
        slice_neurons = 0
        for name, contrib in contributions.items():
            total_neurons += contrib.numel()
            slice_neurons += (contrib != 0).sum().item()
        slice_ratio = slice_neurons / total_neurons if total_neurons > 0 else 0
        
        # 1. Original forward
        original_logits = self._forward_pass(input_tensor)
        original_metrics = ConfidenceMetrics.from_logits(original_logits, target_class)
        
        # 2. Keep-slice forward
        self._register_hooks(contributions, keep_slice=True)
        keep_logits = self._forward_pass(input_tensor)
        keep_metrics = ConfidenceMetrics.from_logits(keep_logits, target_class)
        self._remove_hooks()
        
        # 3. Remove-slice forward
        self._register_hooks(contributions, keep_slice=False)
        remove_logits = self._forward_pass(input_tensor)
        remove_metrics = ConfidenceMetrics.from_logits(remove_logits, target_class)
        self._remove_hooks()
        
        return CounterfactualResult(
            sample_idx=sample_idx,
            true_label=target_class,
            slice_size_ratio=slice_ratio,
            original=original_metrics,
            keep_slice=keep_metrics,
            remove_slice=remove_metrics
        )
    
    def evaluate_partial_removal(
        self,
        input_tensor: torch.Tensor,
        contributions: Dict[str, torch.Tensor],
        target_class: int,
        sample_idx: int = 0,
        removal_steps: int = 10
    ) -> PartialRemovalResult:
        """
        Incrementally remove top contributors and measure confidence degradation.
        
        This shows how "concentrated" the important features are in the slice.
        """
        self.model.eval()
        
        # Collect all contributions with their locations
        all_contribs = []
        for name, contrib in contributions.items():
            flat = contrib.flatten()
            for i, val in enumerate(flat):
                if val != 0:
                    all_contribs.append((name, i, abs(val.item())))
        
        # Sort by contribution magnitude (highest first)
        all_contribs.sort(key=lambda x: x[2], reverse=True)
        
        removal_fractions = []
        confidences = []
        predictions = []
        
        total_slice_size = len(all_contribs)
        
        for step in range(removal_steps + 1):
            frac = step / removal_steps
            removal_fractions.append(frac)
            
            # Number of top contributors to remove
            n_remove = int(frac * total_slice_size)
            
            # Create modified contributions
            modified_contribs = {
                name: contrib.clone() for name, contrib in contributions.items()
            }
            
            # Zero out top-n contributors
            for name, idx, _ in all_contribs[:n_remove]:
                flat = modified_contribs[name].flatten()
                flat[idx] = 0
                modified_contribs[name] = flat.reshape(contributions[name].shape)
            
            # Run keep-slice with modified contributions
            self._register_hooks(modified_contribs, keep_slice=True)
            logits = self._forward_pass(input_tensor)
            self._remove_hooks()
            
            probs = F.softmax(logits, dim=-1).squeeze()
            confidences.append(probs[target_class].item())
            predictions.append(probs.argmax().item())
        
        return PartialRemovalResult(
            sample_idx=sample_idx,
            true_label=target_class,
            removal_fractions=removal_fractions,
            confidences=confidences,
            predictions=predictions
        )


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def load_dataset(dataset_name: str, train: bool = False):
    """Load dataset with appropriate transforms."""
    if dataset_name == 'cifar10':
        return datasets.CIFAR10(
            root='data', train=train, transform=CIFAR10_TRANSFORM, download=True
        )
    elif dataset_name == 'mnist':
        return datasets.MNIST(
            root='data', train=train, transform=MNIST_TRANSFORM, download=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_samples_by_class(
    dataset,
    samples_per_class: int,
    num_classes: int = 10
) -> Dict[int, List[Tuple[torch.Tensor, int, int]]]:
    """
    Get balanced samples for each class.
    Returns dict: class_id -> [(image, label, dataset_idx), ...]
    """
    samples = defaultdict(list)
    
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if len(samples[label]) < samples_per_class:
            samples[label].append((image, label, idx))
        
        # Check if we have enough samples
        if all(len(samples[c]) >= samples_per_class for c in range(num_classes)):
            break
    
    return samples


def compute_slice(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    profile: dict,
    theta: float = 0.3,
    channel_mode: bool = True,
    block_mode: bool = False,
    filter_percent: float = 0.2,
    block_threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """Compute slice for a single sample."""
    device = next(model.parameters()).device
    
    slicer = Slicer(
        model=model,
        input_sample=image.unsqueeze(0).to(device),
        precomputed_profile=profile
    )
    slicer.profile()
    slicer.forward()
    slicer.backward(
        target_index=target_class,
        theta=theta,
        channel_mode=channel_mode,
        block_mode=block_mode,
        filter_percent=filter_percent,
        block_threshold=block_threshold,
        debug=False
    )
    
    return slicer.backward_result["neuron_contributions"]


def run_evaluation(
    model_name: str,
    dataset_name: str,
    profile_path: str,
    samples_per_class: int = 10,
    theta: float = 0.3,
    channel_mode: bool = True,
    block_mode: bool = False,
    filter_percent: float = 0.2,
    block_threshold: float = 0.5,
    output_dir: str = "results/slice_quality"
):
    """
    Run complete slice quality evaluation.
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"{model_name}_fp{filter_percent}_{'block' if block_mode else 'channel'}"
    
    print("\n" + "=" * 70)
    print("SLICE QUALITY EVALUATION")
    print("=" * 70)
    print(f"Model:          {model_name}")
    print(f"Dataset:        {dataset_name}")
    print(f"Samples/class:  {samples_per_class}")
    print(f"Filter percent: {filter_percent}")
    print(f"Channel mode:   {channel_mode}")
    print(f"Block mode:     {block_mode}")
    print("=" * 70)
    
    # Load model and profile
    print("\nLoading model and profile...")
    model = get_model(model_name, pretrained=True)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    profile = torch.load(profile_path)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, train=False)
    samples = get_samples_by_class(dataset, samples_per_class)
    
    # Initialize evaluator
    evaluator = EnhancedCounterfactualEvaluator(model)
    
    # Storage for results
    all_results: List[CounterfactualResult] = []
    partial_results: List[PartialRemovalResult] = []
    
    # Run evaluation
    total_samples = sum(len(v) for v in samples.values())
    
    print(f"\nEvaluating {total_samples} samples...")
    
    with tqdm(total=total_samples, desc="Processing") as pbar:
        for class_id, class_samples in samples.items():
            for image, label, dataset_idx in class_samples:
                # Compute slice
                contributions = compute_slice(
                    model=model,
                    image=image,
                    target_class=label,
                    profile=profile,
                    theta=theta,
                    channel_mode=channel_mode,
                    block_mode=block_mode,
                    filter_percent=filter_percent,
                    block_threshold=block_threshold
                )
                
                # Standard counterfactual evaluation
                result = evaluator.evaluate(
                    input_tensor=image.unsqueeze(0),
                    contributions=contributions,
                    target_class=label,
                    sample_idx=dataset_idx
                )
                all_results.append(result)
                
                # Partial removal evaluation (for a subset)
                if len(partial_results) < 20:  # Only do this for 20 samples
                    partial = evaluator.evaluate_partial_removal(
                        input_tensor=image.unsqueeze(0),
                        contributions=contributions,
                        target_class=label,
                        sample_idx=dataset_idx,
                        removal_steps=10
                    )
                    partial_results.append(partial)
                
                pbar.update(1)
    
    # ==========================================================================
    # Compute Aggregate Statistics
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Basic accuracy metrics
    keep_correct = sum(1 for r in all_results if r.keep_correct)
    remove_flipped = sum(1 for r in all_results if r.remove_flipped)
    
    print(f"\n--- Binary Metrics (n={len(all_results)}) ---")
    print(f"Keep-Slice Accuracy:    {keep_correct}/{len(all_results)} ({100*keep_correct/len(all_results):.1f}%)")
    print(f"Remove-Slice Flip Rate: {remove_flipped}/{len(all_results)} ({100*remove_flipped/len(all_results):.1f}%)")
    
    # Confidence metrics
    conf_drops = [r.confidence_drop for r in all_results]
    conf_retentions = [r.confidence_retention for r in all_results]
    slice_ratios = [r.slice_size_ratio for r in all_results]
    
    print(f"\n--- Confidence Metrics ---")
    print(f"Confidence Drop (ΔC):     mean={np.mean(conf_drops):.3f}, std={np.std(conf_drops):.3f}")
    print(f"Confidence Retention:     mean={np.mean(conf_retentions):.3f}, std={np.std(conf_retentions):.3f}")
    print(f"Slice Size Ratio:         mean={np.mean(slice_ratios):.3f}, std={np.std(slice_ratios):.3f}")
    
    # Original vs Keep vs Remove confidence comparison
    orig_confs = [r.original.target_confidence for r in all_results]
    keep_confs = [r.keep_slice.target_confidence for r in all_results]
    remove_confs = [r.remove_slice.target_confidence for r in all_results]
    
    print(f"\n--- Confidence Comparison ---")
    print(f"Original:     mean={np.mean(orig_confs):.3f}")
    print(f"Keep-Slice:   mean={np.mean(keep_confs):.3f} ({100*np.mean(keep_confs)/np.mean(orig_confs):.1f}% of original)")
    print(f"Remove-Slice: mean={np.mean(remove_confs):.3f} ({100*np.mean(remove_confs)/np.mean(orig_confs):.1f}% of original)")
    
    # Entropy comparison
    orig_entropy = [r.original.entropy for r in all_results]
    keep_entropy = [r.keep_slice.entropy for r in all_results]
    remove_entropy = [r.remove_slice.entropy for r in all_results]
    
    print(f"\n--- Entropy (higher = more confused) ---")
    print(f"Original:     mean={np.mean(orig_entropy):.3f}")
    print(f"Keep-Slice:   mean={np.mean(keep_entropy):.3f}")
    print(f"Remove-Slice: mean={np.mean(remove_entropy):.3f}")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    
    # Save detailed results as CSV
    csv_path = output_path / f"{config_str}_{timestamp}_detailed.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].to_dict().keys())
        writer.writeheader()
        for result in all_results:
            writer.writerow(result.to_dict())
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Save summary statistics as JSON
    summary = {
        'config': {
            'model': model_name,
            'dataset': dataset_name,
            'samples_per_class': samples_per_class,
            'filter_percent': filter_percent,
            'channel_mode': channel_mode,
            'block_mode': block_mode,
            'theta': theta,
            'block_threshold': block_threshold,
        },
        'binary_metrics': {
            'keep_slice_accuracy': keep_correct / len(all_results),
            'remove_slice_flip_rate': remove_flipped / len(all_results),
            'n_samples': len(all_results),
        },
        'confidence_metrics': {
            'confidence_drop_mean': float(np.mean(conf_drops)),
            'confidence_drop_std': float(np.std(conf_drops)),
            'confidence_retention_mean': float(np.mean(conf_retentions)),
            'confidence_retention_std': float(np.std(conf_retentions)),
            'slice_size_ratio_mean': float(np.mean(slice_ratios)),
            'slice_size_ratio_std': float(np.std(slice_ratios)),
        },
        'confidence_comparison': {
            'original_mean': float(np.mean(orig_confs)),
            'keep_slice_mean': float(np.mean(keep_confs)),
            'remove_slice_mean': float(np.mean(remove_confs)),
        },
        'entropy_comparison': {
            'original_mean': float(np.mean(orig_entropy)),
            'keep_slice_mean': float(np.mean(keep_entropy)),
            'remove_slice_mean': float(np.mean(remove_entropy)),
        }
    }
    
    json_path = output_path / f"{config_str}_{timestamp}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")
    
    # Save partial removal results
    if partial_results:
        partial_path = output_path / f"{config_str}_{timestamp}_partial_removal.json"
        with open(partial_path, 'w') as f:
            json.dump([r.to_dict() for r in partial_results], f, indent=2)
        print(f"Partial removal results saved to: {partial_path}")
    
    return summary, all_results, partial_results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate slice quality with confidence-based metrics"
    )
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['lenet', 'resnet10', 'resnet18', 'resnet34'],
        help='Model architecture'
    )
    parser.add_argument(
        '--dataset', type=str, default='cifar10',
        choices=['cifar10', 'mnist'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--profile', type=str, default=None,
        help='Path to precomputed profile (default: profiles/{dataset}_{model}.pt)'
    )
    parser.add_argument(
        '--samples', type=int, default=10,
        help='Number of samples per class'
    )
    parser.add_argument(
        '--filter_percent', type=float, default=0.2,
        help='Filter percent for channel filtering (0.05-0.5)'
    )
    parser.add_argument(
        '--theta', type=float, default=0.3,
        help='Theta threshold for contribution filtering'
    )
    parser.add_argument(
        '--block_mode', action='store_true',
        help='Enable block-level skipping'
    )
    parser.add_argument(
        '--channel_mode', action='store_true',
        help='Enable channel-level skipping'
    )
    parser.add_argument(
        '--block_threshold', type=float, default=0.5,
        help='Threshold for block skipping'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/slice_quality',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Default profile path
    if args.profile is None:
        args.profile = f"profiles/{args.dataset}_{args.model}.pt"
    
    # Adjust dataset for lenet
    if args.model == 'lenet' and args.dataset == 'cifar10':
        print("Note: LeNet typically uses MNIST. Switching to MNIST.")
        args.dataset = 'mnist'
    
    # Run evaluation
    run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        profile_path=args.profile,
        samples_per_class=args.samples,
        theta=args.theta,
        channel_mode=args.channel_mode,
        block_mode=args.block_mode,
        filter_percent=args.filter_percent,
        block_threshold=args.block_threshold,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()