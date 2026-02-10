# evaluation/run_counterfactual.py

import json
import torch
from pathlib import Path

from models import get_model
from core.slicer import Slicer
from evaluation.counterfactual import CounterfactualEvaluator
from utils.data import (
    load_cifar10, load_imagenette,
    get_samples_for_classes, imagenette_local_to_imagenet,
)

# --- Config ---
MODEL_NAME = "resnet18"
TARGET_CLASS = 3
NUM_IMAGES = 10
THETA = 0.3
CHANNEL_MODE = True
CHANNEL_ALPHA = 0.8
BLOCK_MODE = True
BLOCK_BETA = 0.7

IMAGENET_MODELS = {"resnet18", "resnet34", "resnet50", "resnet101"}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_imagenet = MODEL_NAME in IMAGENET_MODELS

    # Load model and profile
    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    profile_path = f"pretrained/profiles/{'imagenet' if is_imagenet else 'cifar10'}_{MODEL_NAME}.pt"
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    # Load dataset and get samples
    dataset = load_imagenette(train=False) if is_imagenet else load_cifar10(train=False)
    samples = get_samples_for_classes(dataset, [TARGET_CLASS], NUM_IMAGES)
    print(f"Evaluating {len(samples)} samples for class {TARGET_CLASS}")

    # Evaluate each sample
    evaluator = CounterfactualEvaluator(model)
    all_results = []

    for image, label, idx in samples:
        input_tensor = image.unsqueeze(0).to(device)
        slice_target = imagenette_local_to_imagenet(label) if is_imagenet else label

        # Compute slice
        slicer = Slicer(model=model, input_sample=input_tensor, precomputed_profile=profile)
        slicer.profile()
        slicer.forward()
        slicer.backward(target_index=slice_target, theta=THETA,
                        channel_mode=CHANNEL_MODE, channel_alpha=CHANNEL_ALPHA,
                        block_mode=BLOCK_MODE, block_beta=BLOCK_BETA)

        contributions = slicer.backward_result["neuron_contributions"]
        result = evaluator.evaluate(input_tensor, contributions, slice_target)
        all_results.append(result)

        print(f"  Sample {idx}: orig={result['original']['target_prob']:.3f}")

    # Save results
    output_dir = Path("evaluation/results/counterfactual")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cf_{MODEL_NAME}_class{TARGET_CLASS}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_path}")
