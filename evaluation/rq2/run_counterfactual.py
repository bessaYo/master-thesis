import json
import torch
from pathlib import Path

from models import get_model
from core.slicer import Slicer
from evaluation.rq2.counterfactual import CounterfactualEvaluator
from utils.data import load_cifar10, get_samples_for_classes, CIFAR10_CLASSES

MODEL_NAME = "resnet_cifar"
NUM_IMAGES = 10
THETA = 0.2

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

    for config in CONFIGS:
        config_name = config["name"]
        alpha = config["alpha"]
        beta = config["beta"]
        print(f"\n{'='*60}")
        print(f"Config: {config_name} (alpha={alpha}, beta={beta})")

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
            for scale in scales:
                necessity_probs = [r["necessity"][scale]["target_prob"] for r in class_results]
                sufficiency_probs = [r["sufficiency"][scale]["target_prob"] for r in class_results]
                avg_necessity[str(scale)] = round(sum(necessity_probs) / len(necessity_probs), 4)
                avg_sufficiency[str(scale)] = round(sum(sufficiency_probs) / len(sufficiency_probs), 4)

            original_probs = [r["original"]["target_prob"] for r in class_results]
            avg_original = round(sum(original_probs) / len(original_probs), 4)

            all_class_results[cls] = {
                "class_name": class_name,
                "num_samples": len(class_results),
                "original": avg_original,
                "necessity": avg_necessity,
                "sufficiency": avg_sufficiency,
            }
            print(f"    orig={avg_original:.3f}  nec@0.5={avg_necessity['0.5']:.3f}  suf@0.5={avg_sufficiency['0.5']:.3f}")

        valid_classes = list(all_class_results.keys())
        num_valid = len(valid_classes)
        avg_necessity_all = {}
        avg_sufficiency_all = {}
        for scale in scales:
            scale_key = str(scale)
            nec_sum = 0
            suf_sum = 0
            for c in valid_classes:
                nec_sum += all_class_results[c]["necessity"][scale_key]
                suf_sum += all_class_results[c]["sufficiency"][scale_key]
            avg_necessity_all[scale_key] = round(nec_sum / num_valid, 4)
            avg_sufficiency_all[scale_key] = round(suf_sum / num_valid, 4)

        orig_sum = 0
        for c in valid_classes:
            orig_sum += all_class_results[c]["original"]
        avg_original_all = round(orig_sum / num_valid, 4)

        print(f"\n  === Average over all classes ===")
        print(f"  Original: {avg_original_all:.3f}")
        print(f"  Necessity:    {avg_necessity_all}")
        print(f"  Sufficiency:  {avg_sufficiency_all}")

        per_class_output = {}
        for c in valid_classes:
            per_class_output[str(c)] = all_class_results[c]

        output = {
            "config": {
                "model": MODEL_NAME,
                "name": config_name,
                "channel_alpha": alpha,
                "block_beta": beta,
                "theta": THETA,
                "num_images": NUM_IMAGES,
            },
            "per_class": per_class_output,
            "average": {
                "original": avg_original_all,
                "necessity": avg_necessity_all,
                "sufficiency": avg_sufficiency_all,
            },
        }
        output_path = output_dir / f"cf_{MODEL_NAME}_{config_name}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {output_path}")

    print("\nDone.")
