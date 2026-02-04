import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
models = ["ResNet18", "ResNet34", "ResNet50"]
neurons = np.array([0.523, 1.44, 2.13])      # millions
runtime = np.array([7.3, 15.1, 20.7])        # seconds
blocks  = np.array([8, 16, 32])              # relative block count

# ------------------------------------------------------------
# Reference curves
# ------------------------------------------------------------
x = np.linspace(0.35, 6.0, 200)

linear_ref = runtime[0] * (x / neurons[0])
alpha = 0.75
sublinear_ref = runtime[0] * (x / neurons[0]) ** alpha

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(9, 5.5))

plt.fill_between(
    x, sublinear_ref, linear_ref,
    alpha=0.2, color="green", label="Time saved"
)

plt.plot(
    x, linear_ref,
    "--", color="gray", linewidth=2, label="Linear $O(n)$"
)

plt.plot(
    x, sublinear_ref,
    color="blue", linewidth=3, label=r"Observed $O(n^{0.75})$"
)

# Single scatter (size encodes block count)
marker_sizes = 80 + 5 * blocks
plt.scatter(
    neurons, runtime,
    s=marker_sizes,
    color="blue",
    edgecolor="black",
    zorder=3
)

# Labels BELOW the points (robust)
for n, t, m in zip(neurons, runtime, models):
    plt.annotate(
        f"{m}\n({t:.1f}s)",
        xy=(n, t),
        xytext=(0, -9),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9
    )

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
plt.xlabel("Neurons in Millions (increasing number of blocks)", fontsize=12)
plt.ylabel("Backward Analysis Time (seconds)", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)

plt.xlim(0.3, 4.0)
plt.ylim(0, 90)

plt.tight_layout()
plt.savefig("scalability_runtime.jpg", dpi=300, bbox_inches="tight")
plt.show()