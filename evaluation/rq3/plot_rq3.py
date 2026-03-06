"""RQ3: Block-level interpretability analysis.

Generates 4 plots:
1. Block-delta heatmap — shows forward deltas per block per class (class-independent)
2. Channel-delta heatmap for block 27 (layer3.8) — shows class-specific channel activation
3. PCA of channel deltas for block 27 — shows semantic clustering (animals vs vehicles)
4. Dendrogram of channel deltas for block 27 — shows hierarchical class relationships

Usage:
    python -m evaluation.rq3.plot_rq3
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from pathlib import Path
from matplotlib.patches import Patch
from tqdm import tqdm

from models import get_model
from utils.data import CIFAR10_CLASSES, load_cifar10, get_samples_for_classes
from core.slicer import Slicer

MODEL_NAME = "resnet_cifar"
NUM_IMAGES = 20
RESULTS_DIR = Path("evaluation/results/rq3")
PLOTS_DIR = Path("evaluation/plots/rq3")

# Block 27 = layer3.8, last block with 64 channels
CHANNEL_BLOCK = "layer3.8"
CHANNEL_LAYER = "layer3.8.conv1"


# ── Data: forward block deltas (cached) ─────────────────────────────────────

def compute_forward_deltas():
    """Run forward pass for each image and collect block deltas"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile_path = f"pretrained/profiles/cifar10_{MODEL_NAME}.pt"
    train_set = load_cifar10(train=True)

    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    all_data = {}
    for cls in range(10):
        class_name = CIFAR10_CLASSES[cls]
        samples = get_samples_for_classes(train_set, [cls], NUM_IMAGES)
        print(f"Class {cls} ({class_name}): {len(samples)} images...")

        image_block_deltas = []
        for img, label, idx in tqdm(samples, desc=f"  {class_name}", leave=False):
            slicer = Slicer(
                model=model,
                input_sample=img.unsqueeze(0).to(device),
                precomputed_profile=profile,
            )
            slicer.profile()
            slicer.forward()

            block_deltas = {}
            for bname, delta in slicer.forward_result["block_deltas"].items():
                block_deltas[bname] = delta.item()
            image_block_deltas.append(block_deltas)

        all_data[cls] = {
            "class_name": class_name,
            "image_block_deltas": image_block_deltas,
        }

    block_names = sorted(slicer.forward_result["blocks"].keys())
    return all_data, block_names


def load_or_compute_block_deltas():
    """Load cached block deltas or compute them"""
    cache_path = RESULTS_DIR / "forward_block_deltas.json"
    if cache_path.exists():
        print("Loading cached forward block deltas...")
        with open(cache_path) as f:
            return json.load(f)

    print("Computing forward block deltas...")
    all_data, block_names = compute_forward_deltas()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "block_names": block_names,
        "classes": {str(cls): data for cls, data in all_data.items()},
    }
    with open(cache_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {cache_path}")
    return output


# ── Data: channel deltas for block 27 ───────────────────────────────────────

def compute_channel_deltas():
    """Compute per-class channel delta profiles for block 27"""
    device = "cpu"
    model = get_model(MODEL_NAME, pretrained=True).to(device).eval()
    profile = torch.load(
        f"pretrained/profiles/cifar10_{MODEL_NAME}.pt",
        map_location=device, weights_only=False,
    )
    dataset = load_cifar10(train=False)

    # Build slicer once
    dummy = torch.randn(1, 3, 32, 32).to(device)
    slicer = Slicer(model=model, input_sample=dummy, precomputed_profile=profile)
    slicer.profile()
    slicer.forward()

    n_channels = slicer.forward_result["channel_deltas"][CHANNEL_LAYER].shape[1]
    n_classes = 10

    # Collect image indices per class
    class_indices = {c: [] for c in range(n_classes)}
    for i, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < NUM_IMAGES:
            class_indices[label].append(i)
        if all(len(v) >= NUM_IMAGES for v in class_indices.values()):
            break

    # Compute mean channel deltas per class
    class_channel_deltas = np.zeros((n_classes, n_channels))
    for cls in range(n_classes):
        all_ch = []
        for idx in class_indices[cls]:
            img, _ = dataset[idx]
            slicer.forward(input_sample=img.unsqueeze(0).to(device))
            cd = slicer.forward_result["channel_deltas"][CHANNEL_LAYER]
            all_ch.append(cd.squeeze(0).cpu().numpy())
        class_channel_deltas[cls] = np.mean(all_ch, axis=0)

    return class_channel_deltas, n_channels


# ── Plot 1: Block-delta heatmap ─────────────────────────────────────────────

def plot_block_heatmap(data):
    """Heatmap of mean forward delta per block per class"""
    block_names = data["block_names"]
    n_blocks = len(block_names)
    n_classes = 10
    block_numbers = list(range(1, n_blocks + 1))

    all_deltas = []
    for cls in range(n_classes):
        cls_data = data["classes"][str(cls)]
        cls_deltas = []
        for img_deltas in cls_data["image_block_deltas"]:
            cls_deltas.append([img_deltas[b] for b in block_names])
        all_deltas.append(cls_deltas)
    all_deltas = np.array(all_deltas)

    mean_per_class = all_deltas.mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mean_per_class, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels(block_numbers, fontsize=7)
    ax.set_xlabel("Block")
    ax.set_ylabel("Class")
    ax.set_title("Mean Forward Delta per Block per Class")
    plt.colorbar(im, ax=ax, label="Mean Delta")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "block_deltas_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: block_deltas_heatmap.pdf")


# ── Plot 2: Channel-delta heatmap for block 27 ──────────────────────────────

def plot_channel_heatmap(class_channel_deltas, n_channels):
    """Heatmap of channel deltas in block 27 per class"""
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(class_channel_deltas, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(10))
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Channel")
    ax.set_xticks(range(n_channels))
    ax.set_xticklabels(range(1, n_channels + 1), fontsize=6)
    ax.set_title(f"Channel Deltas — Block 27 ({CHANNEL_LAYER}, {n_channels} channels)")
    plt.colorbar(im, ax=ax, label="Mean Channel Delta")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "channel_deltas_block27.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: channel_deltas_block27.pdf")


# ── Plot 3: PCA of channel deltas ───────────────────────────────────────────

def plot_channel_pca(class_channel_deltas):
    """PCA projection of per-class channel delta profiles"""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(class_channel_deltas)

    animal_classes = {2, 3, 4, 5, 6, 7}  # bird, cat, deer, dog, frog, horse

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(10):
        color = "#e74c3c" if i in animal_classes else "#3498db"
        marker = "o" if i in animal_classes else "s"
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=100, marker=marker, zorder=3)
        ax.annotate(
            CIFAR10_CLASSES[i], (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(8, 5), fontsize=10,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"PCA of Channel Deltas — Block 27 ({CHANNEL_LAYER})")
    ax.legend(handles=[
        Patch(facecolor="#e74c3c", label="Animals"),
        Patch(facecolor="#3498db", label="Vehicles"),
    ])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "channel_pca_block27.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: channel_pca_block27.pdf")
    print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")


# ── Plot 4: Dendrogram ──────────────────────────────────────────────────────

def plot_channel_dendrogram(class_channel_deltas):
    """Hierarchical clustering of per-class channel delta profiles"""
    condensed = pdist(class_channel_deltas, metric="cosine")
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=CIFAR10_CLASSES, ax=ax, leaf_font_size=11)
    ax.set_ylabel("Distance")
    ax.set_title(f"Hierarchical Clustering of Channel Deltas — Block 27 ({CHANNEL_LAYER})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "channel_dendrogram_block27.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: channel_dendrogram_block27.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Block-level heatmap
    block_data = load_or_compute_block_deltas()
    plot_block_heatmap(block_data)

    # Plots 2, 3, 4: Channel-level analysis for block 27
    print("\nComputing channel deltas for block 27...")
    class_channel_deltas, n_channels = compute_channel_deltas()
    plot_channel_heatmap(class_channel_deltas, n_channels)
    plot_channel_pca(class_channel_deltas)
    plot_channel_dendrogram(class_channel_deltas)


if __name__ == "__main__":
    run()
