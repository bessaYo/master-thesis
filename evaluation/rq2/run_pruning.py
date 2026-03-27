import argparse
import json
import torch
from pathlib import Path

from models import get_model
from utils.data import CIFAR10_CLASSES, load_cifar10, get_samples_for_classes
from utils.evaluation import (
    compute_slices,
    aggregate_slices,
    aggregate_synapse_contribs,
    compute_slice_size,
    evaluate_per_class,
)
from evaluation.rq2.pruning import prune_model

MODEL_NAME = "resnet_cifar"
NUM_IMAGES = 10
THETA = 0.2
EVAL_SAMPLES = 100
NUM_WORKERS = 4


def build_config_name(alpha, beta):
    parts = []
    if alpha is not None:
        parts.append(f"a{alpha}")
    if beta is not None:
        parts.append(f"b{beta}")
    if not parts:
        return "baseline"
    return "_".join(parts)


def run(alpha, beta):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile_path = f"pretrained/profiles/cifar10_{MODEL_NAME}.pt"

    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    train_set = load_cifar10(train=True)
    test_set = load_cifar10(train=False)

    print("Computing baseline accuracy...")
    base_per_class, base_overall = evaluate_per_class(
        model, test_set, device, num_classes=10, eval_samples=EVAL_SAMPLES
    )
    print(f"Baseline overall: {100 * base_overall:.1f}%")

    config_name = build_config_name(alpha, beta)
    print(f"\nConfig: {config_name} (alpha={alpha}, beta={beta})")
    print("=" * 60)

    per_class_results = {}

    for cls in range(10):
        class_name = CIFAR10_CLASSES[cls]
        samples = get_samples_for_classes(train_set, [cls], NUM_IMAGES)
        print(f"\n  Class {cls} ({class_name}): {len(samples)} samples")

        slices = compute_slices(
            samples,
            model_name=MODEL_NAME,
            profile_path=profile_path,
            theta=THETA,
            channel_alpha=alpha,
            block_beta=beta,
            num_workers=NUM_WORKERS,
        )
        aggregated = aggregate_slices(slices)
        synapse_agg = aggregate_synapse_contribs(slices)
        slice_size = compute_slice_size(aggregated, model=model)

        total_blocks = slices[0]["total_blocks"] if slices else 0
        skipped_counts = [s["skipped_blocks"] for s in slices]
        avg_skipped = sum(skipped_counts) / len(skipped_counts) if skipped_counts else 0

        pruned_model, stats = prune_model(model, aggregated, device, synapse_contributions=synapse_agg)
        pruned_per_class, pruned_overall = evaluate_per_class(
            pruned_model, test_set, device, num_classes=10, eval_samples=EVAL_SAMPLES
        )

        non_target = [c for c in range(10) if c != cls]
        non_target_acc = sum(pruned_per_class[c] for c in non_target) / len(non_target)

        per_class_results[cls] = {
            "class_name": class_name,
            "target": round(pruned_per_class[cls], 4),
            "non_target": round(non_target_acc, 4),
            "slice_size": round(slice_size, 4),
            "channel_ratio": round(stats["channel_ratio"], 4),
            "model_size": round(stats["model_size"], 4),
            "blocks_skipped": round(avg_skipped),
            "total_blocks": total_blocks,
        }

        print(
            f"    target={100 * pruned_per_class[cls]:.1f}%  "
            f"non-target={100 * non_target_acc:.1f}%  "
            f"channels={100 * stats['channel_ratio']:.1f}%  "
            f"model_size={100 * stats['model_size']:.1f}%  "
            f"blocks={round(avg_skipped)}/{total_blocks}"
        )

    targets = [per_class_results[c]["target"] for c in range(10)]
    non_targets = [per_class_results[c]["non_target"] for c in range(10)]
    channel_ratios = [per_class_results[c]["channel_ratio"] for c in range(10)]
    model_sizes = [per_class_results[c]["model_size"] for c in range(10)]

    avg_target = sum(targets) / 10
    avg_non_target = sum(non_targets) / 10
    avg_channels = sum(channel_ratios) / 10
    avg_model_size = sum(model_sizes) / 10
    min_target = min(targets)

    print(f"\n{'='*60}")
    print(f"  Target Acc (avg): {100 * avg_target:.1f}%  (min: {100 * min_target:.1f}%)")
    print(f"  Non-Target Acc:   {100 * avg_non_target:.1f}%")
    print(f"  Channels active:  {100 * avg_channels:.1f}%")
    print(f"  Model size:       {100 * avg_model_size:.1f}%")

    output_dir = Path("evaluation/results/pruning")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": MODEL_NAME,
            "channel_alpha": alpha,
            "block_beta": beta,
            "theta": THETA,
            "num_images": NUM_IMAGES,
            "eval_samples": EVAL_SAMPLES,
        },
        "baseline": {
            "overall": round(base_overall, 4),
            "per_class": {str(c): round(base_per_class[c], 4) for c in range(10)},
        },
        "results": {
            "per_class": {str(c): per_class_results[c] for c in range(10)},
            "average": {
                "target": round(avg_target, 4),
                "target_min": round(min_target, 4),
                "non_target": round(avg_non_target, 4),
                "channel_ratio": round(avg_channels, 4),
                "model_size": round(avg_model_size, 4),
            },
        },
    }

    output_path = output_dir / f"pruning_{MODEL_NAME}_{config_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruning evaluation")
    parser.add_argument("--alpha", type=float, default=None, help="Channel slicing alpha")
    parser.add_argument("--beta", type=float, default=None, help="Block slicing beta")
    args = parser.parse_args()
    run(args.alpha, args.beta)
