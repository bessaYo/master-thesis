# evaluation/pruning_curve.py

import json
import numpy as np
import torch
from pathlib import Path

from models import get_model
from utils.data import CIFAR10_CLASSES, load_cifar10, get_samples_for_classes
from utils.evaluation import compute_slices, aggregate_slices, evaluate_per_class
from utils.pruning import get_weight_contributions, prune_by_ratio, prune_random

# --- Config ---
MODEL_NAME = "resnet-cifar"
TARGET_CLASS = 3
NUM_IMAGES = 10
NUM_WORKERS = 6
THETA = 0.3
EVAL_SAMPLES = None
RANDOM_SEEDS = 3
PRUNE_RATIOS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile_path = f"pretrained/profiles/cifar10_{MODEL_NAME}.pt"
    target_name = CIFAR10_CLASSES[TARGET_CLASS]

    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    test_set = load_cifar10(train=False)
    train_set = load_cifar10(train=True)

    print(f"Pruning curve: {MODEL_NAME}, target={target_name}")

    # Baseline
    base_per_class, base_overall = evaluate_per_class(model, test_set, device, num_classes=10, eval_samples=EVAL_SAMPLES)
    base_target = base_per_class[TARGET_CLASS]
    print(f"Baseline: {100*base_target:.1f}%")

    # Slices for target class
    print("Computing target slices...")
    target_samples = get_samples_for_classes(train_set, [TARGET_CLASS], NUM_IMAGES)
    target_agg = aggregate_slices(compute_slices(
        target_samples, MODEL_NAME, profile_path, THETA, num_workers=NUM_WORKERS,
    ))
    target_contribs, weight_map, total_weights = get_weight_contributions(target_agg, model)

    # Slices for all classes
    print("Computing all-class slices...")
    all_samples = get_samples_for_classes(train_set, list(range(10)), NUM_IMAGES)
    all_agg = aggregate_slices(compute_slices(
        all_samples, MODEL_NAME, profile_path, THETA, num_workers=NUM_WORKERS,
    ))
    all_contribs, _, _ = get_weight_contributions(all_agg, model)

    # Evaluate at each ratio
    results_targeted, results_all, results_random = [], [], []

    for ratio in PRUNE_RATIOS:
        print(f"\nRatio {ratio:.2f}:")

        # Targeted pruning
        pc_t, _ = evaluate_per_class(
            prune_by_ratio(model, target_contribs, weight_map, ratio, device),
            test_set, device, num_classes=10, eval_samples=EVAL_SAMPLES,
        )
        results_targeted.append(round(pc_t[TARGET_CLASS], 4))
        print(f"  Targeted: {100*pc_t[TARGET_CLASS]:.1f}%")

        # All-class pruning
        pc_a, _ = evaluate_per_class(
            prune_by_ratio(model, all_contribs, weight_map, ratio, device),
            test_set, device, num_classes=10, eval_samples=EVAL_SAMPLES,
        )
        results_all.append(round(pc_a[TARGET_CLASS], 4))
        print(f"  All:      {100*pc_a[TARGET_CLASS]:.1f}%")

        # Random pruning (averaged over seeds)
        rand_accs = []
        for seed in range(RANDOM_SEEDS):
            pc_r, _ = evaluate_per_class(
                prune_random(model, weight_map, ratio, device, seed),
                test_set, device, num_classes=10, eval_samples=EVAL_SAMPLES,
            )
            rand_accs.append(pc_r[TARGET_CLASS])
        results_random.append(round(float(np.mean(rand_accs)), 4))
        print(f"  Random:   {100*np.mean(rand_accs):.1f}%")

    # Save results
    output_dir = Path("evaluation/results/weight_pruning")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"weight_pruning_{MODEL_NAME}_class{TARGET_CLASS}.json"

    with open(output_path, "w") as f:
        json.dump({
            "baseline": round(base_target, 4),
            "prune_ratios": PRUNE_RATIOS,
            "targeted": results_targeted,
            "all_classes": results_all,
            "random": results_random,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")
