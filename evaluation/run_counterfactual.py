"""Single-config counterfactual evaluation.

Computes slices for one config and evaluates via keep/remove counterfactual.

Usage:
    python evaluation/run_counterfactual.py --model resnet50 --target 7 --num_images 10 --num_workers 6
    python evaluation/run_counterfactual.py --model resnet50 --target 7 --block_mode --block_beta 0.7
    python evaluation/run_counterfactual.py --model resnet18 --target 3 --channel_mode --num_workers 4
"""

import argparse
import torch
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from models import get_model
from core.slicer import Slicer
from evaluation.counterfactual import CounterfactualEvaluator
from utils.evaluation_utils import CIFAR10_CLASSES, load_cifar10, get_samples_for_classes


# =============================================================================
# Worker
# =============================================================================

def process_sample(sample_data, model_name, profile_path, theta,
                   channel_mode, channel_alpha, block_mode, block_beta):
    """Compute slice and run counterfactual evaluation for one image."""
    image, label, idx = sample_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(model_name, pretrained=True).to(device).eval()
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    # Compute slice
    slicer = Slicer(model=model, input_sample=image.unsqueeze(0).to(device),
                    precomputed_profile=profile)
    slicer.profile()
    slicer.forward()
    slicer.backward(target_index=label, theta=theta,
                    channel_mode=channel_mode, channel_alpha=channel_alpha,
                    block_mode=block_mode, block_beta=block_beta)

    contributions = slicer.backward_result["neuron_contributions"]

    # Slice size
    total = sum(c.numel() for c in contributions.values())
    in_slice = sum((c != 0).sum().item() for c in contributions.values())
    slice_ratio = in_slice / total if total > 0 else 0

    # Counterfactual evaluation
    evaluator = CounterfactualEvaluator(model)
    result = evaluator.evaluate(image.unsqueeze(0).to(device), contributions, label)

    return {
        'label': label,
        'slice_size': slice_ratio,
        'orig_prob': result['original']['target_prob'],
        'orig_pred': result['original']['pred'],
        'keep_prob': result['keep']['target_prob'],
        'keep_pred': result['keep']['pred'],
        'keep_correct': result['keep']['pred'] == label,
        'remove_prob': result['remove']['target_prob'],
        'remove_pred': result['remove']['pred'],
        'remove_flipped': result['remove']['pred'] != label,
        'orig_entropy': result['original']['entropy'],
        'keep_entropy': result['keep']['entropy'],
        'remove_entropy': result['remove']['entropy'],
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual Slice Evaluation")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--channel_mode", action="store_true")
    parser.add_argument("--channel_alpha", type=float, default=0.8)
    parser.add_argument("--block_mode", action="store_true")
    parser.add_argument("--block_beta", type=float, default=0.7)
    args = parser.parse_args()

    profile_path = f"profiles/cifar10_{args.model}.pt"
    target_name = CIFAR10_CLASSES[args.target]

    # Header
    print(f"\n{'=' * 60}")
    print("COUNTERFACTUAL EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Model:   {args.model}")
    print(f"  Target:  {target_name} (class {args.target})")
    print(f"  Images:  {args.num_images}, theta={args.theta}")
    if args.channel_mode:
        print(f"  Channel filter: ON (alpha={args.channel_alpha})")
    if args.block_mode:
        print(f"  Block filter: ON (alpha={args.block_beta})")

    # Get samples
    test_set = load_cifar10(train=False)
    samples = get_samples_for_classes(test_set, [args.target], args.num_images)

    # Process in parallel
    print(f"\nComputing slices & evaluating...")
    worker_fn = partial(process_sample, model_name=args.model,
                        profile_path=profile_path, theta=args.theta,
                        channel_mode=args.channel_mode, channel_alpha=args.channel_alpha,
                        block_mode=args.block_mode, block_beta=args.block_beta)

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(worker_fn, samples),
                           total=len(samples), desc="  Evaluating", leave=False))

    # Aggregate
    n = len(results)
    keep_acc = sum(r['keep_correct'] for r in results) / n
    flip_rate = sum(r['remove_flipped'] for r in results) / n
    avg_slice = np.mean([r['slice_size'] for r in results])

    orig_probs = [r['orig_prob'] for r in results]
    keep_probs = [r['keep_prob'] for r in results]
    remove_probs = [r['remove_prob'] for r in results]

    # Results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    print(f"\n  Slice size:  {100*avg_slice:.1f}%")

    print(f"\n  --- Accuracy ---")
    print(f"  Keep-Slice correct:     {sum(r['keep_correct'] for r in results)}/{n}  ({100*keep_acc:.1f}%)")
    print(f"  Remove-Slice flipped:   {sum(r['remove_flipped'] for r in results)}/{n}  ({100*flip_rate:.1f}%)")

    print(f"\n  --- Target Probability ---")
    print(f"  Original:      {np.mean(orig_probs):.3f}")
    print(f"  Keep-Slice:    {np.mean(keep_probs):.3f}  ({100*np.mean(keep_probs)/np.mean(orig_probs):.1f}% of original)")
    print(f"  Remove-Slice:  {np.mean(remove_probs):.3f}  ({100*np.mean(remove_probs)/np.mean(orig_probs):.1f}% of original)")

    print(f"\n  --- Entropy (0=certain, 2.3=random) ---")
    print(f"  Original:      {np.mean([r['orig_entropy'] for r in results]):.3f}")
    print(f"  Keep-Slice:    {np.mean([r['keep_entropy'] for r in results]):.3f}")
    print(f"  Remove-Slice:  {np.mean([r['remove_entropy'] for r in results]):.3f}")

    # Per-sample detail
    print(f"\n  --- Per Sample ---")
    print(f"  {'#':<4} {'Orig':>8} {'Keep':>8} {'Remove':>8} {'Slice':>8} {'Keep OK':>8} {'Flipped':>8}")
    print("  " + "-" * 56)
    for i, r in enumerate(results):
        print(f"  {i+1:<4} {r['orig_prob']:>7.3f} {r['keep_prob']:>8.3f} {r['remove_prob']:>8.3f} "
              f"{100*r['slice_size']:>7.1f}% {'✓' if r['keep_correct'] else '✗':>7} "
              f"{'✓' if r['remove_flipped'] else '✗':>7}")

    print(f"{'=' * 60}")
