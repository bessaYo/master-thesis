# evaluation/pruning.py

import json
import torch
from pathlib import Path

from models import get_model
from utils.data import (
    CIFAR10_CLASSES, IMAGENETTE_CLASSES,
    load_cifar10, load_imagenet_val, load_imagenette, get_samples_for_classes,
    imagenette_local_to_imagenet,
)
from utils.evaluation import (
    compute_slices, aggregate_slices, compute_slice_size,
    evaluate_per_class, evaluate_per_class_imagenette,
)
from utils.pruning import prune_model

# --- Config ---
MODEL_NAME = "resnet18"
TARGET_CLASS = 3
NUM_IMAGES = 10
NUM_WORKERS = 4
THETA = 0.3
CHANNEL_MODE = True
CHANNEL_ALPHA = 0.8
BLOCK_MODE = True
BLOCK_BETA = 0.7
EVAL_SAMPLES = 100  # samples per class for evaluation, None = full test set

IMAGENET_MODELS = {"resnet18", "resnet34", "resnet50", "resnet101"}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_imagenet = MODEL_NAME in IMAGENET_MODELS

    # Setup
    if is_imagenet:
        num_classes = 10
        class_names = IMAGENETTE_CLASSES
        profile_path = f"pretrained/profiles/imagenet_{MODEL_NAME}.pt"
    else:
        num_classes = 10
        class_names = CIFAR10_CLASSES
        profile_path = f"pretrained/profiles/cifar10_{MODEL_NAME}.pt"

    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    print(f"Pruning evaluation: {MODEL_NAME}, class {TARGET_CLASS} ({class_names[TARGET_CLASS]})")

    # Load data
    if is_imagenet:
        imagenet_target = imagenette_local_to_imagenet(TARGET_CLASS)
        samples = get_samples_for_classes(load_imagenet_val(), [imagenet_target], NUM_IMAGES)
        test_set = load_imagenette(train=False)
    else:
        samples = get_samples_for_classes(load_cifar10(train=True), [TARGET_CLASS], NUM_IMAGES)
        test_set = load_cifar10(train=False)

    # Baseline accuracy
    print("Computing baseline...")
    if is_imagenet:
        base_per_class, base_overall = evaluate_per_class_imagenette(model, test_set, device, eval_samples=EVAL_SAMPLES)
    else:
        base_per_class, base_overall = evaluate_per_class(model, test_set, device, num_classes=num_classes, eval_samples=EVAL_SAMPLES)

    # Compute slices
    print("Computing slices...")
    slices = compute_slices(
        samples, model_name=MODEL_NAME, profile_path=profile_path,
        theta=THETA, channel_mode=CHANNEL_MODE, channel_alpha=CHANNEL_ALPHA,
        block_mode=BLOCK_MODE, block_beta=BLOCK_BETA, num_workers=NUM_WORKERS,
    )
    aggregated = aggregate_slices(slices)
    slice_size = compute_slice_size(aggregated, model=model)

    # Prune and evaluate
    print("Pruning and evaluating...")
    pruned_model, active_frac = prune_model(model, aggregated, device)

    if is_imagenet:
        pruned_per_class, pruned_overall = evaluate_per_class_imagenette(pruned_model, test_set, device, eval_samples=EVAL_SAMPLES)
    else:
        pruned_per_class, pruned_overall = evaluate_per_class(pruned_model, test_set, device, num_classes=num_classes, eval_samples=EVAL_SAMPLES)

    # Print summary
    non_target = [c for c in range(num_classes) if c != TARGET_CLASS]
    base_nontarget = sum(base_per_class[c] for c in non_target) / len(non_target)
    pruned_nontarget = sum(pruned_per_class[c] for c in non_target) / len(non_target)

    print(f"\nSlice size: {100*slice_size:.1f}%")
    print(f"Target:     {100*pruned_per_class[TARGET_CLASS]:.1f}% (baseline: {100*base_per_class[TARGET_CLASS]:.1f}%)")
    print(f"Non-target: {100*pruned_nontarget:.1f}% (baseline: {100*base_nontarget:.1f}%)")
    print(f"Overall:    {100*pruned_overall:.1f}% (baseline: {100*base_overall:.1f}%)")

    # Save results
    output_dir = Path("evaluation/results/channel_pruning")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"channel_pruning_{MODEL_NAME}_class{TARGET_CLASS}.json"

    result = {
        "slice_size": round(slice_size, 4),
        "baseline": {
            "target": round(base_per_class[TARGET_CLASS], 4),
            "non_target": round(base_nontarget, 4),
            "overall": round(base_overall, 4),
        },
        "pruned": {
            "target": round(pruned_per_class[TARGET_CLASS], 4),
            "non_target": round(pruned_nontarget, 4),
            "overall": round(pruned_overall, 4),
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_path}")
