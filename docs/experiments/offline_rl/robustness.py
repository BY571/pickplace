"""Plot the observation-robustness sweep of the stage-2.1 students (BC, IQL, TD3+BC).

One panel per perturbation family, one line per algorithm, success rate against severity. Every family
shares the same clean control (severity 0), so each panel starts from the same three points and the panel
shows only how fast each policy falls away from it.

Usage (on the Spark, where the artifacts live):
    python docs/experiments/offline_rl/robustness.py <label>=<results.json> ... [out=<png>]
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

#: family -> (panel title, x-axis label, x of the clean control)
FAMILIES = OrderedDict([
    ("image_noise", ("Image noise (additive Gaussian)", "Noise sigma (of 255)", 0.0)),
    ("brightness", ("Brightness (additive offset)", "Offset (of 255)", 0.0)),
    ("contrast", ("Contrast (multiplicative gain)", "1 - gain", 0.0)),
    ("blur", ("Defocus (Gaussian blur)", "Blur sigma (pixels of 84)", 0.0)),
    ("occlusion", ("Occlusion (fixed black patch, both cameras)", "Fraction of each frame occluded", 0.0)),
    ("proprio", ("Proprioception noise", "Joint-position sigma (rad)", 0.0)),
])

args = [a for a in sys.argv[1:] if not a.startswith("out=")]
out = next((a[len("out="):] for a in sys.argv[1:] if a.startswith("out=")), "robustness.png")

runs: "OrderedDict[str, list]" = OrderedDict()
for arg in args:
    label, path = arg.split("=", 1)
    runs[label] = json.loads(Path(path).read_text())["results"]

num_envs = {r["num_envs"] for rows in runs.values() for r in rows}.pop()

sns.set()
fig, axes = plt.subplots(2, 3, figsize=(15, 11))
colors = sns.color_palette(n_colors=len(runs))
for ax, (family, (title, xlabel, clean_x)) in zip(axes.flat, FAMILIES.items()):
    for (label, rows), color in zip(runs.items(), colors):
        clean = next(r["success_rate"] for r in rows if r["family"] == "clean")
        points = sorted([(r["severity"], r["success_rate"]) for r in rows if r["family"] == family]
                        + [(clean_x, clean)])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        sns.lineplot(x=x, y=y, label=f"{label} ({y[-1]:.0%})", linewidth=2.5, color=color, ax=ax)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(f"Online success rate\n({num_envs} evaluation episodes)", fontsize=13)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc="best")

fig.suptitle("How the offline-RL students degrade under degraded observations (no retraining)", fontsize=17)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(out, dpi=150)
print(f"WROTE {out}")
