import json
import os
import glob
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pruning")
OUT_PATH = os.path.join(os.path.dirname(__file__), "pruning_tradeoff.pdf")

channel_only = []
block_only = []
combined = []

for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
    name = os.path.basename(path)
    if "baseline" in name:
        continue

    with open(path) as f:
        data = json.load(f)

    cfg = data["config"]
    avg = data["results"]["average"]
    alpha = cfg["channel_alpha"]
    beta = cfg["block_beta"]

    row = {
        "alpha": alpha,
        "beta": beta,
        "target": avg["target"] * 100,
        "model_size": avg["model_size"] * 100,
    }

    if alpha and not beta:
        channel_only.append(row)
    elif beta and not alpha:
        block_only.append(row)
    elif alpha and beta:
        combined.append(row)

# sort by threshold descending (strict to aggressive)
channel_only.sort(key=lambda x: x["alpha"], reverse=True)
block_only.sort(key=lambda x: x["beta"], reverse=True)
combined.sort(key=lambda x: x["beta"], reverse=True)

# use only combined with alpha=0.85
combined_085 = [r for r in combined if r["alpha"] == 0.85]

fig, ax = plt.subplots(figsize=(7, 4.5))

# channel pruning: x = alpha
ax.plot(
    [r["alpha"] for r in channel_only],
    [r["target"] for r in channel_only],
    marker="o", label="Channel ($\\alpha$)", color="#1f77b4", linewidth=2
)

# block pruning: x = beta
ax.plot(
    [r["beta"] for r in block_only],
    [r["target"] for r in block_only],
    marker="s", label="Block ($\\beta$)", color="#d62728", linewidth=2
)

# combined: x = beta (alpha fixed at 0.85)
ax.plot(
    [r["beta"] for r in combined_085],
    [r["target"] for r in combined_085],
    marker="^", label="Combined ($\\alpha$=0.85, $\\beta$)", color="#2ca02c", linewidth=2
)

ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("Target Accuracy (%)", fontsize=12)
ax.set_xlim(1.0, 0.25)
ax.set_ylim(20, 105)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
