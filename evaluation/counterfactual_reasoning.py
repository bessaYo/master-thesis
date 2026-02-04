import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
from multiprocessing import Pool
from functools import partial

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from core.slicer import Slicer
from evaluation.counterfactual import CounterfactualEvaluator

# =============================================================================
# Constants
# =============================================================================

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
MNIST_CLASSES = [str(i) for i in range(10)]

CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# =============================================================================
# Worker Function
# =============================================================================

def process_sample(
    sample_data: Tuple[torch.Tensor, int, int],
    model_name: str,
    profile_path: str,
    theta: float,
    channel_mode: bool,
    block_mode: bool,
    channel_alpha: float,
    block_beta: float,
    soft_mask: bool,
    min_mask_value: float,
    baseline: str = None,
) -> dict:
    """Process a single sample."""
    image, label, dataset_idx = sample_data
    
    model = get_model(model_name, pretrained=True)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    profile = torch.load(profile_path, map_location=device)
    
    # Compute slice
    slicer = Slicer(
        model=model,
        input_sample=image.unsqueeze(0).to(device),
        precomputed_profile=profile,
        debug=False
    )
    slicer.profile()
    slicer.forward()
    slicer.backward(
        target_index=label,
        theta=theta,
        channel_mode=channel_mode,
        block_mode=block_mode,
        channel_alpha=channel_alpha,
        block_beta=block_beta,
    )
    
    contributions = slicer.backward_result["neuron_contributions"]

    if baseline == 'random':
        for name in contributions:
            orig = contributions[name]
            ratio = (orig != 0).float().mean().item()
            # Zufällige Neuronen statt die wichtigsten
            contributions[name] = (torch.rand_like(orig.float()) < ratio).float()
    
    # Calculate slice size
    total = sum(c.numel() for c in contributions.values())
    in_slice = sum((c != 0).sum().item() for c in contributions.values())
    slice_ratio = in_slice / total if total > 0 else 0
    
    # Evaluate
    evaluator = CounterfactualEvaluator(model, soft_mask=soft_mask, min_mask_value=min_mask_value)
    result = evaluator.evaluate(image.unsqueeze(0).to(device), contributions, label)
    
    return {
        'sample_idx': dataset_idx,
        'true_label': label,
        'slice_size_ratio': slice_ratio,
        # Original
        'orig_prob': result['original']['target_prob'],
        'orig_logits': result['original']['logits'].tolist(),
        'orig_probs': result['original']['probs'].tolist(),
        'orig_entropy': result['original']['entropy'],
        # Keep
        'keep_prob': result['keep']['target_prob'],
        'keep_pred': result['keep']['pred'],
        'keep_logits': result['keep']['logits'].tolist(),
        'keep_probs': result['keep']['probs'].tolist(),
        'keep_entropy': result['keep']['entropy'],
        'keep_correct': result['keep']['pred'] == label,
        # Remove
        'remove_prob': result['remove']['target_prob'],
        'remove_pred': result['remove']['pred'],
        'remove_logits': result['remove']['logits'].tolist(),
        'remove_probs': result['remove']['probs'].tolist(),
        'remove_entropy': result['remove']['entropy'],
        'remove_flipped': result['remove']['pred'] != label,
    }


def _worker(task, **kwargs):
    sample_data, = task
    return process_sample(sample_data, **kwargs)


# =============================================================================
# Dataset Utils
# =============================================================================

def load_dataset(name: str):
    if name == 'cifar10':
        return datasets.CIFAR10(root='data', train=False, transform=CIFAR10_TRANSFORM, download=True)
    elif name == 'mnist':
        return datasets.MNIST(root='data', train=False, transform=MNIST_TRANSFORM, download=True)
    raise ValueError(f"Unknown dataset: {name}")


def get_samples(dataset, per_class: int, num_classes: int = 10):
    samples = defaultdict(list)
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if len(samples[label]) < per_class:
            samples[label].append((img, label, idx))
        if all(len(samples[c]) >= per_class for c in range(num_classes)):
            break
    return [s for c in sorted(samples.keys()) for s in samples[c]]

def run_cross_image_evaluation(
    model_name: str,
    dataset_name: str,
    profile_path: str,
    slices_per_class: int = 1,
    test_per_class: int = 10,
    theta: float = 0.3,
    channel_mode: bool = False,
    channel_alpha: float = 0.8,
    block_mode: bool = False,
    block_beta: float = 0.9,
    **kwargs
):
    """Cross-image evaluation: compute slice on one image, test on others."""
    
    class_names = CIFAR10_CLASSES if dataset_name == 'cifar10' else MNIST_CLASSES
    num_classes = len(class_names)
    
    # Setup
    model = get_model(model_name, pretrained=True)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    profile = torch.load(profile_path, map_location=device)
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Get samples
    samples_needed = slices_per_class + test_per_class
    all_samples = get_samples(dataset, samples_needed, num_classes)
    
    # Split in slice samples und test samples
    samples_by_class = defaultdict(list)
    for img, label, idx in all_samples:
        samples_by_class[label].append((img, label, idx))
    
    slice_samples = {c: samples_by_class[c][:slices_per_class] for c in range(num_classes)}
    test_samples = {c: samples_by_class[c][slices_per_class:] for c in range(num_classes)}
    
    results = []
    
    # Für jede Klasse: berechne Slice und teste
    for slice_class in tqdm(range(num_classes), desc="Processing classes"):
        slice_img, slice_label, slice_idx = slice_samples[slice_class][0]
        
        # Compute slice
        slicer = Slicer(
            model=model,
            input_sample=slice_img.unsqueeze(0).to(device),
            precomputed_profile=profile,
            debug=False
        )
        slicer.profile()
        slicer.forward()
        slicer.backward(
            target_index=slice_label,
            theta=theta,
            channel_mode=channel_mode,
            channel_alpha=channel_alpha,
            block_mode=block_mode,
            block_beta=block_beta,
        )
        contributions = slicer.backward_result["neuron_contributions"]
        
        # Slice size
        total = sum(c.numel() for c in contributions.values())
        in_slice = sum((c != 0).sum().item() for c in contributions.values())
        slice_ratio = in_slice / total
        
        evaluator = CounterfactualEvaluator(model)
        
        # Test auf ALLEN Klassen
        for test_class in range(num_classes):
            for test_img, test_label, test_idx in test_samples[test_class]:
                result = evaluator.evaluate(
                    test_img.unsqueeze(0).to(device),
                    contributions,
                    test_label
                )
                
                results.append({
                    'slice_class': slice_class,
                    'slice_class_name': class_names[slice_class],
                    'test_class': test_class,
                    'test_class_name': class_names[test_class],
                    'same_class': slice_class == test_class,
                    'slice_size': slice_ratio,
                    # Accuracy: predictet es die echte Klasse des Test-Bildes?
                    'keep_correct': result['keep']['pred'] == test_label,
                    'keep_prob': result['keep']['target_prob'],
                    # Was predictet der Slice tatsächlich?
                    'keep_pred': result['keep']['pred'],
                    'keep_pred_prob': max(result['keep']['probs']),
                    'pred_is_slice_class': result['keep']['pred'] == slice_class,
                    'orig_prob': result['original']['target_prob'],
                    'orig_pred': result['original']['probs'].argmax().item(),
                })
    
    # === Print Results ===
    print("\n" + "=" * 90)
    print("CROSS-IMAGE EVALUATION RESULTS")
    print("=" * 90)
    
    # Same-class results
    same_class = [r for r in results if r['same_class']]
    cross_class = [r for r in results if not r['same_class']]
    
    print(f"\n--- Same Class (slice and test from same class) ---")
    print(f"N = {len(same_class)}")
    print(f"Keep Accuracy:   {100 * np.mean([r['keep_correct'] for r in same_class]):.1f}%")
    print(f"Keep Confidence: {100 * np.mean([r['keep_prob'] for r in same_class]):.1f}%")
    
    print(f"\n--- Cross Class (slice from class A, test on class B) ---")
    print(f"N = {len(cross_class)}")
    print(f"Keep Accuracy:   {100 * np.mean([r['keep_correct'] for r in cross_class]):.1f}%")
    print(f"Keep Confidence: {100 * np.mean([r['keep_prob'] for r in cross_class]):.1f}%")
    
    # Wie oft predictet der Slice seine eigene Klasse?
    pred_slice_class = np.mean([r['pred_is_slice_class'] for r in results])
    print(f"\n--- Slice Dominance ---")
    print(f"Prediction = Slice Class: {100 * pred_slice_class:.1f}% (of all {len(results)} tests)")
    
    # === Matrix 1: Keep Accuracy ===
    print(f"\n--- Keep Accuracy by Slice/Test Class ---")
    print(f"{'Slice↓ Test→':<12}", end="")
    for c in range(num_classes):
        print(f"{class_names[c][:6]:>8}", end="")
    print()
    
    for slice_c in range(num_classes):
        print(f"{class_names[slice_c]:<12}", end="")
        for test_c in range(num_classes):
            subset = [r for r in results if r['slice_class'] == slice_c and r['test_class'] == test_c]
            if subset:
                acc = 100 * np.mean([r['keep_correct'] for r in subset])
                print(f"{acc:>7.0f}%", end="")
            else:
                print(f"{'N/A':>8}", end="")
        print()
    
    # === Matrix 2: Keep Confidence ===
    print(f"\n--- Keep Confidence (for true class) by Slice/Test Class ---")
    print(f"{'Slice↓ Test→':<12}", end="")
    for c in range(num_classes):
        print(f"{class_names[c][:6]:>8}", end="")
    print()
    
    for slice_c in range(num_classes):
        print(f"{class_names[slice_c]:<12}", end="")
        for test_c in range(num_classes):
            subset = [r for r in results if r['slice_class'] == slice_c and r['test_class'] == test_c]
            if subset:
                conf = 100 * np.mean([r['keep_prob'] for r in subset])
                print(f"{conf:>7.1f}%", end="")
            else:
                print(f"{'N/A':>8}", end="")
        print()
    
    # === Matrix 3: Prediction Distribution ===
    print(f"\n--- Prediction Distribution (what does the slice predict?) ---")
    print(f"{'Slice↓ Pred→':<12}", end="")
    for c in range(num_classes):
        print(f"{class_names[c][:6]:>8}", end="")
    print()
    
    for slice_c in range(num_classes):
        print(f"{class_names[slice_c]:<12}", end="")
        subset = [r for r in results if r['slice_class'] == slice_c]
        for pred_c in range(num_classes):
            count = sum(1 for r in subset if r['keep_pred'] == pred_c)
            pct = 100 * count / len(subset) if subset else 0
            print(f"{pct:>7.0f}%", end="")
        print()
    
    # === Matrix 4: Original Model Accuracy (Sanity Check) ===
    print(f"\n--- Original Model Accuracy (sanity check) ---")
    print(f"{'Test Class':<12} {'Accuracy':>10}")
    for test_c in range(num_classes):
        subset = [r for r in results if r['test_class'] == test_c]
        acc = 100 * np.mean([r['orig_pred'] == r['test_class'] for r in subset])
        print(f"{class_names[test_c]:<12} {acc:>9.0f}%")
    
    return results

# =============================================================================
# Print Functions
# =============================================================================

def print_results(results: List[dict], class_names: List[str], show_examples: int = 3):
    n = len(results)
    
    # Binary metrics
    keep_acc = sum(r['keep_correct'] for r in results) / n
    flip_rate = sum(r['remove_flipped'] for r in results) / n
    
    # Confidence metrics
    orig_probs = [r['orig_prob'] for r in results]
    keep_probs = [r['keep_prob'] for r in results]
    remove_probs = [r['remove_prob'] for r in results]
    slice_sizes = [r['slice_size_ratio'] for r in results]
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n--- Binary Metrics (n={n}) ---")
    print(f"Keep-Slice Accuracy:    {sum(r['keep_correct'] for r in results)}/{n} ({100*keep_acc:.1f}%)")
    print(f"Remove-Slice Flip Rate: {sum(r['remove_flipped'] for r in results)}/{n} ({100*flip_rate:.1f}%)")
    
    print(f"\n--- Confidence (Target Class) ---")
    print(f"Original:     {np.mean(orig_probs):.3f}")
    print(f"Keep-Slice:   {np.mean(keep_probs):.3f} ({100*np.mean(keep_probs)/np.mean(orig_probs):.1f}% of original)")
    print(f"Remove-Slice: {np.mean(remove_probs):.3f} ({100*np.mean(remove_probs)/np.mean(orig_probs):.1f}% of original)")
    
    print(f"\n--- Slice Size ---")
    print(f"Mean: {np.mean(slice_sizes):.3f}, Std: {np.std(slice_sizes):.3f}")
    
    print(f"\n--- Entropy (0=certain, 2.3=random) ---")
    print(f"Original:     {np.mean([r['orig_entropy'] for r in results]):.3f}")
    print(f"Keep-Slice:   {np.mean([r['keep_entropy'] for r in results]):.3f}")
    print(f"Remove-Slice: {np.mean([r['remove_entropy'] for r in results]):.3f}")
    
    # Detailed examples
    print("\n" + "=" * 90)
    print(f"DETAILED EXAMPLES (First {show_examples} samples)")
    print("=" * 90)
    
    for i, r in enumerate(results[:show_examples]):
        label = r['true_label']
        print(f"\n{'─' * 90}")
        print(f"Sample {i+1}: True class = {label} ({class_names[label]}), Slice size = {r['slice_size_ratio']*100:.1f}%")
        print(f"{'─' * 90}")
        
        print(f"{'Class':<12} {'Orig Prob':>10} {'Keep Prob':>10} {'Rem Prob':>10} │ {'Orig Logit':>11} {'Keep Logit':>11} {'Rem Logit':>11}")
        print(f"{'─'*12} {'─'*10} {'─'*10} {'─'*10} ┼ {'─'*11} {'─'*11} {'─'*11}")
        
        for c in range(len(class_names)):
            marker = " ◄── TARGET" if c == label else ""
            print(f"{class_names[c]:<12} {r['orig_probs'][c]:>10.4f} {r['keep_probs'][c]:>10.4f} {r['remove_probs'][c]:>10.4f} │ {r['orig_logits'][c]:>11.3f} {r['keep_logits'][c]:>11.3f} {r['remove_logits'][c]:>11.3f}{marker}")
        
        print(f"{'─'*12} {'─'*10} {'─'*10} {'─'*10} ┼ {'─'*11} {'─'*11} {'─'*11}")
        print(f"{'Prediction':<12} {class_names[r['orig_probs'].index(max(r['orig_probs']))]:>10} {class_names[r['keep_pred']]:>10} {class_names[r['remove_pred']]:>10} │")


# =============================================================================
# Main
# =============================================================================

def run_evaluation(
    model_name: str,
    dataset_name: str,
    profile_path: str,
    samples_per_class: int = 1,
    theta: float = 0.3,
    channel_mode: bool = False,
    block_mode: bool = False,
    channel_alpha: float = 0.8,
    block_beta: float = 0.9,
    soft_mask: bool = False,
    min_mask_value: float = 0.1,
    num_workers: int = 5,
    output_dir: str = "results/slice_quality",
    show_examples: int = 3,
    baseline: str = None,  # <-- NEU
):
    class_names = CIFAR10_CLASSES if dataset_name == 'cifar10' else MNIST_CLASSES
    
    print("\n" + "=" * 70)
    print("SLICE QUALITY EVALUATION")
    print("=" * 70)
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"Channel mode: {channel_mode} (α={channel_alpha}), Block mode: {block_mode} (α={block_beta})")
    print(f"Soft mask: {soft_mask} (min={min_mask_value})")
    print("=" * 70)
    
    # Load data
    dataset = load_dataset(dataset_name)
    samples = get_samples(dataset, samples_per_class)
    print(f"\nProcessing {len(samples)} samples...")
    
    # Process
    worker_fn = partial(
        process_sample,
        model_name=model_name,
        profile_path=profile_path,
        theta=theta,
        channel_mode=channel_mode,
        block_mode=block_mode,
        channel_alpha=channel_alpha,
        block_beta=block_beta,
        soft_mask=soft_mask,
        min_mask_value=min_mask_value,
        baseline=baseline,  # <-- NEU
    )
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(worker_fn, samples), total=len(samples)))
    
    # Print results
    print_results(results, class_names, show_examples)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = output_path / f"{model_name}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'model': model_name, 'dataset': dataset_name,
                'channel_mode': channel_mode, 'channel_alpha': channel_alpha,
                'block_mode': block_mode, 'block_beta': block_beta,
                'soft_mask': soft_mask, 'min_mask_value': min_mask_value,
            },
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--channel_mode', action='store_true')
    parser.add_argument('--channel_alpha', type=float, default=0.8)
    parser.add_argument('--block_mode', action='store_true')
    parser.add_argument('--block_beta', type=float, default=0.9)
    parser.add_argument('--soft_mask', action='store_true')
    parser.add_argument('--min_mask_value', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='results/slice_quality')
    parser.add_argument('--show_examples', type=int, default=3)
    parser.add_argument('--baseline', type=str, default=None, choices=[None, 'random'])
    parser.add_argument('--cross_image', action='store_true', help='Run cross-image evaluation (test slice on other images)')
    parser.add_argument('--test_per_class', type=int, default=10,  help='Number of test images per class for cross-image evaluation')
    
    args = parser.parse_args()
    
    if args.profile is None:
        args.profile = f"profiles/{args.dataset}_{args.model}.pt"
    
    if args.model == 'lenet':
        args.dataset = 'mnist'
    
    if args.cross_image:
        run_cross_image_evaluation(
            model_name=args.model,
            dataset_name=args.dataset,
            profile_path=args.profile,
            slices_per_class=1,
            test_per_class=args.test_per_class,
            theta=0.3,
            channel_mode=args.channel_mode,
            channel_alpha=args.channel_alpha,
            block_mode=args.block_mode,
            block_beta=args.block_beta,
        )
    else:
        run_evaluation(
            model_name=args.model,
            dataset_name=args.dataset,
            profile_path=args.profile,
            samples_per_class=args.samples,
            channel_mode=args.channel_mode,
            channel_alpha=args.channel_alpha,
            block_mode=args.block_mode,
            block_beta=args.block_beta,
            soft_mask=args.soft_mask,
            min_mask_value=args.min_mask_value,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            show_examples=args.show_examples,
            baseline=args.baseline,
        )


if __name__ == "__main__":
    main()