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
    """Create contribution vectors for each block"""
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
        similarity_matrix = np.zeros((10, 10))

        for ci in range(10):
            # Intra class: pairwise within same class
            similarities = []
            for i in range(len(class_vectors[ci])):
                for j in range(i + 1, len(class_vectors[ci])):
                    score = cosine_similarity(
                        [class_vectors[ci][i][block]], [class_vectors[ci][j][block]]
                    )[0][0]
                    similarities.append(score)
            if similarities:
                similarity_matrix[ci][ci] = float(np.mean(similarities))

            # Inter class: pairwise between different classes
            for cj in range(ci + 1, 10):
                similarities = []
                for i in range(len(class_vectors[ci])):
                    for j in range(len(class_vectors[cj])):
                        score = cosine_similarity(
                            [class_vectors[ci][i][block]], [class_vectors[cj][j][block]]
                        )[0][0]
                        similarities.append(score)
                avg = float(np.mean(similarities))
                similarity_matrix[ci][cj] = avg
                similarity_matrix[cj][ci] = avg

        # Group averages
        animal_sims = [similarity_matrix[i][j] for i in ANIMAL_CLASSES for j in ANIMAL_CLASSES if i < j]
        vehicle_sims = [similarity_matrix[i][j] for i in VEHICLE_CLASSES for j in VEHICLE_CLASSES if i < j]
        cross_sims = [similarity_matrix[i][j] for i in ANIMAL_CLASSES for j in VEHICLE_CLASSES]

        results[block] = {
            "matrix": similarity_matrix.tolist(),
            "intra": float(np.mean([similarity_matrix[i][i] for i in range(10)])),
            "animal_animal": float(np.mean(animal_sims)),
            "vehicle_vehicle": float(np.mean(vehicle_sims)),
            "animal_vehicle": float(np.mean(cross_sims)),
        }

    return results


def channel_contribs(class_slices, conv_key="layer3.8.conv1"):
    """Mean absolute contribution per channel"""
    per_class = {}
    for c, slices in class_slices.items():
        values = []
        for slice_result in slices:
            contribs = slice_result["contributions"]
            if conv_key in contribs:
                abs_contribs = contribs[conv_key].abs()
                channel_means = abs_contribs.mean(dim=(-2, -1)).squeeze(0).numpy()
                values.append(channel_means.tolist())
        per_class[CIFAR10_CLASSES[c]] = np.mean(values, axis=0).tolist()
    return {"conv_key": conv_key, "n_channels": 64, "per_class": per_class}


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = get_model("resnet_cifar", pretrained=True)
    model.eval()
    dataset = load_cifar10(train=False)

    print("Collecting images...")
    class_images = collect_images(model, dataset, n=20)

    # Compute slices for each class
    print("Computing slices...")
    class_slices = {}
    for c in range(10):
        class_slices[c] = compute_slices(
            class_images[c],
            model_name="resnet_cifar",
            profile_path=PROFILE_PATH,
            theta=0.3,
            desc=CIFAR10_CLASSES[c],
        )

    # Compute block similarity
    print("Computing block similarity...")
    class_vectors = block_contrib_vectors(class_slices)
    similarity = similarity_matrices(class_vectors)
    with open(OUT_DIR / "block_similarity.json", "w") as f:
        json.dump({"blocks": BLOCKS, "per_block": similarity}, f, indent=2)
    print(f"  -> {OUT_DIR / 'block_similarity.json'}")

    # Compute channel contributions
    print("Computing channel contributions...")
    channels = channel_contribs(class_slices)
    with open(OUT_DIR / "channel_contribs.json", "w") as f:
        json.dump(channels, f, indent=2)
    print(f"{OUT_DIR / 'channel_contribs.json'}")
