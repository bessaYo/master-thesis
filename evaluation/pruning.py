"""Single-config pruning evaluation.

Computes slices for one config and evaluates pruning quality.

Usage:
    python evaluation/pruning.py --model resnet50 --target 7 --num_images 10 --num_workers 6
    python evaluation/pruning.py --model resnet50 --target 7 --block_mode --block_beta 0.7
    python evaluation/pruning.py --model resnet18 --target 3 --channel_mode --channel_alpha 0.8 --debug
"""

import argparse
import torch
from models import get_model
from utils.evaluation_utils import (
    CIFAR10_CLASSES, load_cifar10, get_samples_for_classes,
    compute_slices, aggregate_slices, compute_slice_size,
    prune_model, evaluate_per_class,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruning-based Slice Quality Evaluation")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--channel_mode", action="store_true")
    parser.add_argument("--channel_alpha", type=float, default=0.8)
    parser.add_argument("--block_mode", action="store_true")
    parser.add_argument("--block_beta", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    profile_path = f"profiles/cifar10_{args.model}.pt"
    target_name = CIFAR10_CLASSES[args.target]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    non_target_classes = [c for c in range(10) if c != args.target]

    # Header
    print(f"\n{'=' * 60}")
    print("PRUNING EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Model:   {args.model}")
    print(f"  Target:  {target_name} (class {args.target})")
    print(f"  Slices:  {args.num_images} images, theta={args.theta}")
    if args.channel_mode:
        print(f"  Channel filter: ON (alpha={args.channel_alpha})")
    if args.block_mode:
        print(f"  Block filter: ON (alpha={args.block_beta})")

    # Load model & data
    model = get_model(args.model, pretrained=True).to(device).eval()
    test_set = load_cifar10(train=False)
    train_set = load_cifar10(train=True)
    samples = get_samples_for_classes(train_set, [args.target], args.num_images)

    # Baseline
    print("\nComputing baseline...")
    base_per_class, base_overall = evaluate_per_class(model, test_set, device)

    # Compute slices
    print("Computing slices...")
    slices = compute_slices(
        samples, model_name=args.model, profile_path=profile_path,
        theta=args.theta,
        channel_mode=args.channel_mode, channel_alpha=args.channel_alpha,
        block_mode=args.block_mode, block_beta=args.block_beta,
        num_workers=args.num_workers,
    )

    aggregated = aggregate_slices(slices)
    slice_size = compute_slice_size(aggregated)

    # Debug: per-layer activity
    if args.debug:
        print(f"\n{'Layer':<30} {'Active Ch':>10}")
        print("-" * 42)
        for name, tensor in sorted(aggregated.items()):
            if tensor.dim() == 4:
                ch = tensor.abs().sum(dim=(2, 3)).squeeze(0)
                print(f"  {name:<28} {int((ch > 0).sum())}/{ch.numel()}")

    # Prune & evaluate
    print("\nPruning model...")
    pruned_model, active_frac = prune_model(model, aggregated, device)

    print("Evaluating pruned model...")
    pruned_per_class, pruned_overall = evaluate_per_class(pruned_model, test_set, device)

    # Aggregate target / non-target
    base_target = base_per_class[args.target]
    pruned_target = pruned_per_class[args.target]
    base_nontarget = sum(base_per_class[c] for c in non_target_classes) / len(non_target_classes)
    pruned_nontarget = sum(pruned_per_class[c] for c in non_target_classes) / len(non_target_classes)

    # Results
    print(f"\n{'=' * 60}")
    print("SLICE INFO")
    print(f"  Active channels: {100*slice_size:.1f}%")

    print("\nRESULTS")
    print(f"  Target class accuracy:     {100*pruned_target:.1f}%  (baseline: {100*base_target:.1f}%)")
    print(f"  Non-target class accuracy: {100*pruned_nontarget:.1f}%  (baseline: {100*base_nontarget:.1f}%)")
    print(f"  Overall accuracy:          {100*pruned_overall:.1f}%  (baseline: {100*base_overall:.1f}%)")

    print(f"\nPER-CLASS BREAKDOWN")
    print(f"  {'Class':<12} {'Pruned':>8} {'Baseline':>10}")
    print("  " + "-" * 34)
    for c in range(10):
        marker = "  < TARGET" if c == args.target else ""
        print(f"  {CIFAR10_CLASSES[c]:<12} {100*pruned_per_class[c]:>7.1f}% {100*base_per_class[c]:>9.1f}%{marker}")

    print(f"{'=' * 60}")
