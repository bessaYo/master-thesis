"""Counterfactual evaluation (necessity + sufficiency) for multiple slice configs"""

import json
import torch
from pathlib import Path

from models import get_model
from core.slicer import Slicer
from evaluation.rq2.counterfactual import CounterfactualEvaluator
from utils.data import load_cifar10, get_samples_for_classes, CIFAR10_CLASSES

MODEL_NAME = "resnet_cifar"
NUM_IMAGES = 10
THETA = 0.3

CONFIGS = [
    {"name": "base", "alpha": None, "beta": None},
    {"name": "conservative", "alpha": 0.85, "beta": 0.8},
    {"name": "aggressive", "alpha": 0.5, "beta": 0.5},
]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    profile_path = f"pretrained/profiles/cifar10_{MODEL_NAME}.pt"
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    dataset = load_cifar10(train=False)
    evaluator = CounterfactualEvaluator(model)

    output_dir = Path("evaluation/results/counterfactual")
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        alpha = cfg["alpha"]
        beta = cfg["beta"]
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name} (alpha={alpha}, beta={beta})")

        all_class_results = {}

        for cls in range(10):
            class_name = CIFAR10_CLASSES[cls]
            samples = get_samples_for_classes(dataset, [cls], NUM_IMAGES)
            print(f"\n  Class {cls} ({class_name}): {len(samples)} samples")

            class_results = []
            skipped = 0
            for image, label, idx in samples:
                input_tensor = image.unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(input_tensor).argmax(dim=1).item()
                if pred != label:
                    skipped += 1
                    continue

                slicer = Slicer(model=model, input_sample=input_tensor, precomputed_profile=profile)
                slicer.profile()
                slicer.forward()
                slicer.backward(
                    target_index=label,
                    theta=THETA,
                    channel_alpha=alpha,
                    block_beta=beta,
                )

                contributions = slicer.backward_result["neuron_contributions"]
                result = evaluator.evaluate(input_tensor, contributions, label)
                class_results.append(result)

            print(f"    Correctly classified: {len(class_results)}/{len(samples)} (skipped {skipped})")

            if not class_results:
                print(f"    No correctly classified samples — skipping class")
                continue

            scales = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
            avg_necessity = {}
            avg_sufficiency = {}
            for s in scales:
                nec_probs = [r["necessity"][s]["target_prob"] for r in class_results]
                suf_probs = [r["sufficiency"][s]["target_prob"] for r in class_results]
                avg_necessity[str(s)] = round(sum(nec_probs) / len(nec_probs), 4)
                avg_sufficiency[str(s)] = round(sum(suf_probs) / len(suf_probs), 4)

            orig_probs = [r["original"]["target_prob"] for r in class_results]
            avg_orig = round(sum(orig_probs) / len(orig_probs), 4)

            all_class_results[cls] = {
                "class_name": class_name,
                "num_samples": len(class_results),
                "original": avg_orig,
                "necessity": avg_necessity,
                "sufficiency": avg_sufficiency,
            }
            print(f"    orig={avg_orig:.3f}  nec@0.5={avg_necessity['0.5']:.3f}  suf@0.5={avg_sufficiency['0.5']:.3f}")

        valid_classes = list(all_class_results.keys())
        n_valid = len(valid_classes)
        avg_nec_all = {}
        avg_suf_all = {}
        for s in scales:
            sk = str(s)
            avg_nec_all[sk] = round(sum(all_class_results[c]["necessity"][sk] for c in valid_classes) / n_valid, 4)
            avg_suf_all[sk] = round(sum(all_class_results[c]["sufficiency"][sk] for c in valid_classes) / n_valid, 4)
        avg_orig_all = round(sum(all_class_results[c]["original"] for c in valid_classes) / n_valid, 4)

        print(f"\n  === Average over all classes ===")
        print(f"  Original: {avg_orig_all:.3f}")
        print(f"  Necessity:   ", {sk: f"{v:.3f}" for sk, v in avg_nec_all.items()})
        print(f"  Sufficiency: ", {sk: f"{v:.3f}" for sk, v in avg_suf_all.items()})

        output = {
            "config": {
                "model": MODEL_NAME,
                "name": cfg_name,
                "channel_alpha": alpha,
                "block_beta": beta,
                "theta": THETA,
                "num_images": NUM_IMAGES,
            },
            "per_class": {str(c): all_class_results[c] for c in valid_classes},
            "average": {
                "original": avg_orig_all,
                "necessity": avg_nec_all,
                "sufficiency": avg_suf_all,
            },
        }
        output_path = output_dir / f"cf_{MODEL_NAME}_{cfg_name}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {output_path}")

    print("\nDone.")
