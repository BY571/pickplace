"""Plot the four-algorithm, three-seed comparison of the stage-2.1 students.

One line per algorithm: the mean success rate across seeds, with a shaded min-max band over the seeds,
against the teacher's reference line. Deployable observations only (both cameras + proprioception).

Usage (on the Spark, where the artifacts live):
    python docs/experiments/offline_rl/algorithms.py <label>=<run_dir>,<run_dir>,... ... [out=<png>]
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

TEACHER_SUCCESS = 0.984  # pipeline/0_state_teacher evaluation of ppo_teacher_final.pt, same protocol

args = [a for a in sys.argv[1:] if not a.startswith("out=")]
out = next((a[len("out="):] for a in sys.argv[1:] if a.startswith("out=")), "algorithms.png")

series: "OrderedDict[str, tuple]" = OrderedDict()
for arg in args:
    label, run_dirs = arg.split("=", 1)
    curves = []
    for run_dir in run_dirs.split(","):
        history = json.loads((Path(run_dir) / "manifest.json").read_text())["eval_history"]
        curves.append(([e["step"] for e in history], [e["success_rate"] for e in history]))
    length = min(len(steps) for steps, _ in curves)  # a stopped-short seed must not stretch the others
    steps = curves[0][0][:length]
    values = np.array([v[:length] for _, v in curves])
    series[label] = (steps, values)

sns.set()
fig, ax = plt.subplots(figsize=(15, 8))
for (label, (steps, values)), color in zip(series.items(), sns.color_palette(n_colors=len(series))):
    mean = values.mean(axis=0)
    sns.lineplot(x=steps, y=mean, label=f"{label} ({mean[-1]:.0%})", linewidth=2.5, color=color, ax=ax)
    # The band is the min-max envelope over seeds: with three seeds a standard deviation would be noisier
    # than the spread it summarises.
    ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=0.15, linewidth=0)
ax.axhline(TEACHER_SUCCESS, color="black", linestyle=":", linewidth=2.5)
ax.text(ax.get_xlim()[1], TEACHER_SUCCESS, f" teacher {TEACHER_SUCCESS:.0%}", va="center", fontsize=13)
ax.set_title("Offline-RL algorithms on deployable observations: mean over 3 seeds, min-max band",
             fontsize=15)
ax.set_xlabel("Gradient steps", fontsize=15)
ax.set_ylabel("Online success rate\n(128 evaluation episodes)", fontsize=15)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.tick_params(labelsize=12)
ax.legend(fontsize=15, loc="lower right")
fig.tight_layout()
fig.savefig(out, dpi=150, bbox_inches="tight")
print("WROTE", out)
