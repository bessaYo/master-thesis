"""Benchmark: Compare slicing speed across configurations.

Usage:
    python evaluation/benchmark_time.py --model resnet50 --target 7 --num_images 3 --num_workers 6
    python evaluation/benchmark_time.py --model resnet18 --target 3 --channel_alpha 0.8
"""

import argparse
import time
import torch
import torchvision.datasets as datasets
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from utils.evaluation_utils import CIFAR10_CLASSES, CIFAR10_TRANSFORM, compute_single_slice_timed


def run_config(name, samples, num_workers, **kwargs):
    worker = partial(compute_single_slice_timed, **kwargs)
    t0 = time.perf_counter()
    with Pool(num_workers) as pool:
        backward_times = list(tqdm(pool.imap(worker, samples),
                                   total=len(samples), desc=f"  {name}",
                                   leave=False))
    wall_time = time.perf_counter() - t0
    avg_backward = sum(backward_times) / len(backward_times)
    return name, wall_time, avg_backward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark slicing speed")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--channel_alpha", type=float, default=0.8)
    parser.add_argument("--block_beta", type=float, default=0.7)
    args = parser.parse_args()

    profile_path = f"profiles/cifar10_{args.model}.pt"
    target_name = CIFAR10_CLASSES[args.target]

    test_set = datasets.CIFAR10(root="data", train=False, transform=CIFAR10_TRANSFORM, download=False)
    samples = [(img, args.target, i)
               for i, (img, label) in enumerate(test_set)
               if label == args.target][:args.num_images]

    print(f"\nBenchmark: {args.model}, target={target_name}, "
          f"{args.num_images} images, {args.num_workers} workers")
    print("=" * 80)

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

    results = []
    for name, cfg in configs:
        print(f"\nRunning: {name} ...")
        name, wall, avg = run_config(name, samples, args.num_workers, **common, **cfg)
        results.append((name, wall, avg))

    # Print results
    baseline_wall = results[0][1]
    print(f"\n{'Config':<50} {'Wall time':>10} {'Avg backward':>14} {'Speedup':>10}")
    print("-" * 85)
    for name, wall, avg in results:
        speedup = baseline_wall / wall
        print(f"{name:<50} {wall:>9.1f}s {avg:>13.1f}s {speedup:>9.2f}x")
