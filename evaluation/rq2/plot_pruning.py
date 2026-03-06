"""Plot pruning tradeoff: scaling factor vs target accuracy"""

import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
})

RESULTS_DIR = Path("evaluation/results/pruning")
OUTPUT_DIR = Path("evaluation/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_result(filename):
    with open(RESULTS_DIR / filename) as f:
        data = json.load(f)
    avg = data["results"]["average"]
    return avg["model_size"] * 100, avg["target"] * 100


factors = [0.9, 0.8, 0.7, 0.5, 0.3]

channel_targets = [load_result(f"pruning_resnet_cifar_a{a}.json")[1] for a in factors]
block_targets = [load_result(f"pruning_resnet_cifar_b{b}.json")[1] for b in factors]
combined_targets = [load_result(f"pruning_resnet_cifar_a0.85_b{b}.json")[1] for b in factors]
_, baseline_target = load_result("pruning_resnet_cifar_baseline.json")

fig, ax = plt.subplots()
ax.plot(factors, channel_targets, "o-", color="#d62728", label="Channel ($\\alpha$)", markersize=6, linewidth=1.5)
ax.plot(factors, block_targets, "s-", color="#1f77b4", label="Block ($\\beta$)", markersize=6, linewidth=1.5)
ax.plot(factors, combined_targets, "^-", color="#2ca02c", label="Combined ($\\alpha{=}0.85$, $\\beta$)", markersize=6, linewidth=1.5)

ax.set_xlabel("Scaling Factor")
ax.set_ylabel("Target Accuracy (%)")
ax.set_xlim(0.25, 0.95)
ax.set_ylim(0, 105)
ax.set_xticks(factors)
ax.invert_xaxis()
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pruning_tradeoff.pdf", bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "pruning_tradeoff.png", bbox_inches="tight", dpi=150)
print(f"Saved to {OUTPUT_DIR}/pruning_tradeoff.pdf")
