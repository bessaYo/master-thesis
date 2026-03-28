import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from models import get_model
from utils.data import CIFAR10_CLASSES, load_cifar10
from utils.evaluation import compute_slices

PROFILE_PATH = "pretrained/profiles/cifar10_resnet_cifar.pt"
BLOCKS = [f"layer{lg}.{bi}" for lg in range(1, 4) for bi in range(9)]
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]
VEHICLE_CLASSES = [0, 1, 8, 9]
OUT_DIR = Path("evaluation/results/contrib_similarity")


def collect_images(model, dataset, n=20):
    """Collect n correctly classified images per class"""
    class_images = {c: [] for c in range(10)}
    collected = 0

    for i, (img, label) in enumerate(dataset):
        if len(class_images[label]) >= n:
            continue
        with torch.no_grad():
            if model(img.unsqueeze(0)).argmax(1).item() == label:
                class_images[label].append((img, label, i))
                collected += 1
        if collected >= 10 * n:
            break

    for c in range(10):
        print(f"  {CIFAR10_CLASSES[c]}: {len(class_images[c])} images")
    return class_images


def block_contrib_vectors(class_slices):
    """Create contribution vectors for each block by concatenating conv layer outputs"""
    class_vectors = {}
    for c, slices in class_slices.items():
        class_vectors[c] = []
        for slice_result in slices:
            contribs = slice_result["contributions"]
            block_vectors = {}
            for block in BLOCKS:
                keys = sorted(k for k in contribs if k.startswith(block + ".conv"))
                if keys:
                    block_vectors[block] = np.concatenate(
                        [contribs[k].flatten().float().numpy() for k in keys]
                    )
            class_vectors[c].append(block_vectors)
    return class_vectors


def similarity_matrices(class_vectors):
    """Compute 10x10 cosine similarity matrix per block"""
    results = {}
    for block in BLOCKS:
        matrix = np.zeros((10, 10))

        for ci in range(10):
            sims = []
            for i in range(len(class_vectors[ci])):
                for j in range(i + 1, len(class_vectors[ci])):
                    score = cosine_similarity(
                        [class_vectors[ci][i][block]], [class_vectors[ci][j][block]]
                    )[0][0]
                    sims.append(score)
            if sims:
                matrix[ci][ci] = float(np.mean(sims))

            for cj in range(ci + 1, 10):
                sims = []
                for i in range(len(class_vectors[ci])):
                    for j in range(len(class_vectors[cj])):
                        score = cosine_similarity(
                            [class_vectors[ci][i][block]], [class_vectors[cj][j][block]]
                        )[0][0]
                        sims.append(score)
                avg = float(np.mean(sims))
                matrix[ci][cj] = avg
                matrix[cj][ci] = avg

        results[block] = matrix
    return results


def channel_contribs(class_slices, conv_key="layer3.8.conv1"):
    """Mean absolute contribution per channel per class"""
    per_class = {}
    for c, slices in class_slices.items():
        values = []
        for slice_result in slices:
            contribs = slice_result["contributions"]
            if conv_key in contribs:
                abs_contribs = contribs[conv_key].abs()
                channel_means = abs_contribs.mean(dim=(-2, -1)).squeeze(0).numpy()
                values.append(channel_means)
        per_class[CIFAR10_CLASSES[c]] = np.mean(values, axis=0)
    return per_class


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = get_model("resnet_cifar", pretrained=True)
    model.eval()
    dataset = load_cifar10(train=False)

    print("Collecting 20 images per class...")
    class_images = collect_images(model, dataset, n=20)

    print("Computing slices for each class...")
    class_slices = {}
    for c in range(10):
        class_slices[c] = compute_slices(
            class_images[c],
            model_name="resnet_cifar",
            profile_path=PROFILE_PATH,
            theta=0.2,
            desc=CIFAR10_CLASSES[c],
        )

    # Block similarity
    print("\nComputing block similarity...")
    class_vectors = block_contrib_vectors(class_slices)
    sim_matrices = similarity_matrices(class_vectors)

    # cat=3, dog=5, truck=9, automobile=1
    block_rows = []
    for block in BLOCKS:
        m = sim_matrices[block]
        block_rows.append({
            "block": block,
            "same_class": round(float(np.mean([m[i][i] for i in range(10)])), 4),
            "cat_dog": round(float(m[3][5]), 4),
            "truck_auto": round(float(m[9][1]), 4),
            "cat_truck": round(float(m[3][9]), 4),
        })

    with open(OUT_DIR / "block_similarity.json", "w") as f:
        json.dump(block_rows, f, indent=2)

    print(f"\n{'Block':<12} {'Same-Class':>10} {'Cat-Dog':>10} {'Truck-Auto':>10} {'Cat-Truck':>10}")
    print("-" * 55)
    for row in block_rows:
        print(f"{row['block']:<12} {row['same_class']:>10.4f} {row['cat_dog']:>10.4f} {row['truck_auto']:>10.4f} {row['cat_truck']:>10.4f}")

    # Channel contributions
    print("\nComputing channel contributions...")
    per_class = channel_contribs(class_slices)

    # Find top shared channels for each pair
    pairs = [("cat", "dog"), ("truck", "automobile")]
    animal_names = [CIFAR10_CLASSES[c] for c in ANIMAL_CLASSES]
    vehicle_names = [CIFAR10_CLASSES[c] for c in VEHICLE_CLASSES]

    channel_output = {"per_class": {k: v.tolist() for k, v in per_class.items()}, "top_channels": {}}

    for a, b in pairs:
        overlap = np.minimum(per_class[a], per_class[b])
        top2 = np.argsort(overlap)[-2:][::-1]
        channel_output["top_channels"][f"{a}_{b}"] = [
            {"channel": int(ch), a: round(float(per_class[a][ch]), 4), b: round(float(per_class[b][ch]), 4)}
            for ch in top2
        ]
        print(f"\n  Top shared channels ({a}-{b}):")
        for ch in top2:
            print(f"    Channel {ch}: {a}={per_class[a][ch]:.4f}  {b}={per_class[b][ch]:.4f}")

    for name, group in [("animals", animal_names), ("vehicles", vehicle_names)]:
        arrays = [per_class[c] for c in group]
        overlap = np.min(arrays, axis=0)
        top2 = np.argsort(overlap)[-2:][::-1]
        channel_output["top_channels"][name] = [{"channel": int(ch)} for ch in top2]
        print(f"\n  Top shared channels ({name}):")
        for ch in top2:
            vals = "  ".join(f"{c}={per_class[c][ch]:.4f}" for c in group)
            print(f"    Channel {ch}: {vals}")

    with open(OUT_DIR / "channel_contribs.json", "w") as f:
        json.dump(channel_output, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")
