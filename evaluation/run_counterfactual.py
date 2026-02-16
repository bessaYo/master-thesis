# evaluation/run_counterfactual.py

import json
import torch
from pathlib import Path

from models import get_model
from core.slicer import Slicer
from evaluation.counterfactual import CounterfactualEvaluator
from utils.data import load_cifar10, get_samples_for_classes, CIFAR10_CLASSES

# --- Config ---
MODEL_NAME = "resnet_cifar"
TARGET_CLASS = 5
NUM_IMAGES = 10
THETA = 0.3
CHANNEL_ALPHA = 0.8
BLOCK_BETA = 0.7

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and profile
    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    profile_path = f"pretrained/profiles/cifar10_{MODEL_NAME}.pt"
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    # Load dataset and get samples
    dataset = load_cifar10(train=False)
    samples = get_samples_for_classes(dataset, [TARGET_CLASS], NUM_IMAGES)
    print(f"Evaluating {len(samples)} samples for class {TARGET_CLASS} ({CIFAR10_CLASSES[TARGET_CLASS]})")

    # Evaluate each sample
    evaluator = CounterfactualEvaluator(model)
    all_results = []

    for image, label, idx in samples:
        input_tensor = image.unsqueeze(0).to(device)

        # Compute slice
        slicer = Slicer(model=model, input_sample=input_tensor, precomputed_profile=profile)
        slicer.profile()
        slicer.forward()
        slicer.backward(
            target_index=label,
            theta=THETA,
            channel_alpha=CHANNEL_ALPHA,
            block_beta=BLOCK_BETA,
        )

        contributions = slicer.backward_result["neuron_contributions"]
        result = evaluator.evaluate(input_tensor, contributions, label)

        # Add sample info and convert pred indices to class names
        result["sample"] = {"dataset_index": idx, "label": CIFAR10_CLASSES[label]}
        for key in ("necessity", "sufficiency"):
            for scale, data in result[key].items():
                data["pred"] = CIFAR10_CLASSES[data["pred"]]
        result["original"]["pred"] = CIFAR10_CLASSES[result["original"]["pred"]]

        all_results.append(result)
        print(f"  Sample {idx}: orig={result['original']['target_prob']:.3f}, pred={result['original']['pred']}")

    # Save results
    output_dir = Path("evaluation/results/counterfactual")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cf_{MODEL_NAME}_class{TARGET_CLASS}.json"

    output = {
        "config": {
            "model": MODEL_NAME,
            "target_class": TARGET_CLASS,
            "target_class_name": CIFAR10_CLASSES[TARGET_CLASS],
            "num_images": NUM_IMAGES,
            "theta": THETA,
            "channel_alpha": CHANNEL_ALPHA,
            "block_beta": BLOCK_BETA,
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")
