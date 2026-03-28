import json
from pathlib import Path

SLICES_DIR = Path("evaluation/slices")
MODELS = ["resnet18", "resnet34", "resnet101"]
TARGETS = [0, 1, 2]
CONFIGS = [
    ("Baseline", ""),
    ("Channel 0.8", "_ch0.8"),
    ("Channel 0.5", "_ch0.5"),
    ("Channel 0.2", "_ch0.2"),
    ("Block 0.8", "_bl0.8"),
    ("Block 0.5", "_bl0.5"),
    ("Block 0.2", "_bl0.2"),
    ("Combined", "_ch0.8_bl0.8"),
]

for model in MODELS:
    print(f"\n  {model} (mean over targets {TARGETS})")
    print(f"  {'Config':<16} {'Time':>7} {'N%':>7} {'S%':>7} {'Skipped':>9}")

    for label, suffix in CONFIGS:
        results = []
        for target in TARGETS:
            path = SLICES_DIR / f"{model}_t{target}_img0{suffix}.json"
            if path.exists():
                with open(path) as f:
                    results.append(json.load(f)["results"])

        if not results:
            print(f"  {label:<16} {'--':>7} {'--':>7} {'--':>7} {'--':>9}")
            continue

        count = len(results)
        avg_time = sum(r["backward_time"] for r in results) / count
        neurons = sum(r["neurons_ratio"] for r in results) / count
        synapses = sum(r["synapses_ratio"] for r in results) / count
        skipped = round(sum(r["skipped_blocks"] for r in results) / count)
        total_blocks = results[0]["total_blocks"]

        print(f"  {label:<16} {avg_time:>6.2f}s {neurons:>6.1f} {synapses:>6.1f} {skipped}/{total_blocks:>7}")
