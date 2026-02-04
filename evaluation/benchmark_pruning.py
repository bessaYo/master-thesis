"""Benchmark: Compare pruning quality across slicing configurations.

Runs 3 configs (baseline theta-only, block, block+channel), computes
slices in parallel, prunes, and compares accuracy side by side.

Usage:
    python evaluation/benchmark_pruning.py --model resnet50 --target 7 --num_images 10 --num_workers 6
    python evaluation/benchmark_pruning.py --model resnet18 --target 3 --channel_alpha 0.8
"""

import argparse
from models import get_model
from utils.evaluation_utils import (
    CIFAR10_CLASSES, load_cifar10, get_samples_for_classes,
    compute_slices, aggregate_slices, compute_slice_size,
    prune_model, evaluate_per_class,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pruning quality across configs")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--channel_alpha", type=float, default=0.8)
    parser.add_argument("--block_beta", type=float, default=0.7)
    args = parser.parse_args()

    import torch
    profile_path = f"profiles/cifar10_{args.model}.pt"
    target_name = CIFAR10_CLASSES[args.target]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'=' * 70}")
    print(f"PRUNING BENCHMARK")
    print(f"{'=' * 70}")
    print(f"  Model:    {args.model}")
    print(f"  Target:   {target_name} (class {args.target})")
    print(f"  Images:   {args.num_images}, Workers: {args.num_workers}")
    print(f"  Theta:    {args.theta}")
    print(f"  Channel:  α={args.channel_alpha}")
    print(f"  Block:    α={args.block_beta}")

    # Load data
    model = get_model(args.model, pretrained=True).to(device).eval()
    test_set = load_cifar10(train=False)
    train_set = load_cifar10(train=True)
    samples = get_samples_for_classes(train_set, [args.target], args.num_images)

    # Baseline accuracy
    print("\nComputing baseline accuracy...")
    base_per_class, base_overall = evaluate_per_class(model, test_set, device)
    base_target = base_per_class[args.target]
    non_target_classes = [c for c in range(10) if c != args.target]
    base_nontarget = sum(base_per_class[c] for c in non_target_classes) / len(non_target_classes)

    # Define configs
    common = dict(model_name=args.model, profile_path=profile_path, theta=args.theta)

    configs = [
        ("NNSlicer (theta only)", dict(
            channel_mode=False, channel_alpha=args.channel_alpha,
            block_mode=False, block_beta=args.block_beta)),
        (f"Block (α={args.block_beta})", dict(
            channel_mode=False, channel_alpha=args.channel_alpha,
            block_mode=True, block_beta=args.block_beta)),
        (f"Block+Channel (α_b={args.block_beta}, α_c={args.channel_alpha})", dict(
            channel_mode=True, channel_alpha=args.channel_alpha,
            block_mode=True, block_beta=args.block_beta)),
    ]

    # Run each config
    results = []
    for name, cfg in configs:
        print(f"\n--- {name} ---")
        slices = compute_slices(samples, num_workers=args.num_workers,
                                desc=name, **common, **cfg)
        aggregated = aggregate_slices(slices)
        slice_size = compute_slice_size(aggregated)

        pruned_model, active_frac = prune_model(model, aggregated, device)
        pruned_per_class, pruned_overall = evaluate_per_class(pruned_model, test_set, device)

        pruned_target = pruned_per_class[args.target]
        pruned_nontarget = sum(pruned_per_class[c] for c in non_target_classes) / len(non_target_classes)

        results.append({
            "name": name,
            "slice_size": slice_size,
            "target_acc": pruned_target,
            "nontarget_acc": pruned_nontarget,
            "overall_acc": pruned_overall,
            "per_class": pruned_per_class,
        })

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"RESULTS  (baseline: target={100*base_target:.1f}%, "
          f"non-target={100*base_nontarget:.1f}%, overall={100*base_overall:.1f}%)")
    print(f"{'=' * 70}")

    header = f"  {'Config':<45} {'Slice':>6} {'Target':>8} {'Non-Tgt':>8} {'Overall':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        print(f"  {r['name']:<45} {100*r['slice_size']:>5.1f}% "
              f"{100*r['target_acc']:>7.1f}% {100*r['nontarget_acc']:>7.1f}% "
              f"{100*r['overall_acc']:>7.1f}%")

    # Per-class breakdown
    print(f"\nPER-CLASS BREAKDOWN")
    class_header = f"  {'Class':<12}" + "".join(f" {r['name'][:15]:>15}" for r in results) + f" {'Baseline':>10}"
    print(class_header)
    print("  " + "-" * (len(class_header) - 2))
    for c in range(10):
        marker = " <" if c == args.target else ""
        row = f"  {CIFAR10_CLASSES[c]:<12}"
        for r in results:
            row += f" {100*r['per_class'][c]:>14.1f}%"
        row += f" {100*base_per_class[c]:>9.1f}%{marker}"
        print(row)

    print(f"{'=' * 70}")
