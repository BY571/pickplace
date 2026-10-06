"""Plot the three-algorithm, three-seed comparison once per demonstration tier.

One panel per tier (expert, medium), one line per algorithm: the mean success rate across seeds with a
shaded min-max band, against the reference line that matters for that panel — the teacher for the expert
tier, the medium shard's own success rate for the medium tier. Deployable observations throughout (both
cameras + proprioception).

Usage (on the Spark, where the artifacts live):
    python docs/experiments/offline_rl/by_data_quality.py \
        "Expert demonstrations (teacher at 98%)|0.984|BC=<dir>,<dir>,<dir>|IQL=..." \
        "Medium demonstrations (demonstrator at 67%)|0.674|BC=..." [out=<png>]

Each positional argument is one panel: ``<title>|<reference success rate>|<label>=<run dirs>|...``.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("out=")]
out = next((a[len("out="):] for a in sys.argv[1:] if a.startswith("out=")), "by_data_quality.png")

panels = []
for arg in args:
    title, reference, *groups = arg.split("|")
    series = []
    for group in groups:
        label, run_dirs = group.split("=", 1)
        curves = []
        for run_dir in run_dirs.split(","):
            history = json.loads((Path(run_dir) / "manifest.json").read_text())["eval_history"]
            curves.append(([e["step"] for e in history], [e["success_rate"] for e in history]))
        length = min(len(steps) for steps, _ in curves)  # a stopped-short seed must not stretch the others
        series.append((label, curves[0][0][:length], np.array([v[:length] for _, v in curves])))
    panels.append((title, float(reference), series))

sns.set()
fig, axes = plt.subplots(len(panels), 1, figsize=(15, 5.5 * len(panels)), sharex=True)
axes = axes if len(panels) > 1 else [axes]
for ax, (title, reference, series) in zip(axes, panels):
    for (label, steps, values), color in zip(series, sns.color_palette(n_colors=len(series))):
        mean = values.mean(axis=0)
        sns.lineplot(x=steps, y=mean, label=f"{label} ({mean[-1]:.0%})", linewidth=2.5, color=color, ax=ax)
        # min-max envelope over seeds: with three seeds a standard deviation is noisier than the spread
        ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=0.15, linewidth=0)
    ax.axhline(reference, color="black", linestyle=":", linewidth=2.5)
    ax.text(ax.get_xlim()[1], reference, f" data {reference:.0%}", va="center", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Online success rate\n(128 evaluation episodes)", fontsize=15)
    floor = max(0.0, min(v.min() for _, _, v in series) - 0.05)
    ax.set_ylim(floor, 1.02)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc="lower right")
axes[-1].set_xlabel("Gradient steps", fontsize=15)
fig.suptitle("Offline RL by demonstration quality: mean over 3 seeds, min-max band", fontsize=17)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(out, dpi=150, bbox_inches="tight")
print("WROTE", out)
