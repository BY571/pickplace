"""Plot online success-rate curves for the stage-2.1 students, one subplot per data tier.

Only deployable observation sets are plotted: everything a real cell can measure (the two cameras and
the robot's own proprioception). Runs that consumed simulator-only privileged state were discarded --
see ../../../pipeline/2_1_offline_rl/README.md.

Usage (on the Spark, where the artifacts live):
    python docs/experiments/offline_rl/plot_curves.py <run_dir>=<label>=<panel> ... [out=<png>]

<panel> is the data tier the runs in that subplot were trained on, e.g. "Expert demonstrations only".
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

TEACHER_SUCCESS = 0.984  # pipeline/0_state_teacher evaluation of ppo_teacher_final.pt, same protocol

args = [a for a in sys.argv[1:] if not a.startswith("out=")]
out = next((a[len("out="):] for a in sys.argv[1:] if a.startswith("out=")), "success_rate.png")

panels: "OrderedDict[str, list]" = OrderedDict()
for arg in args:
    run_dir, label, panel = arg.split("=", 2)
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    history = manifest["eval_history"]
    steps = [e["step"] for e in history]
    values = [e["success_rate"] for e in history]
    panels.setdefault(panel, []).append((f"{label} ({values[-1]:.0%})", steps, values))

sns.set()
fig, axes = plt.subplots(len(panels), 1, figsize=(15, 5.5 * len(panels)), sharex=True, sharey=True)
axes = axes if len(panels) > 1 else [axes]
for ax, (panel, series) in zip(axes, panels.items()):
    for (label, steps, values), color in zip(series, sns.color_palette(n_colors=len(series))):
        sns.lineplot(x=steps, y=values, label=label, linewidth=2.5, color=color, ax=ax)
    ax.axhline(TEACHER_SUCCESS, color="black", linestyle=":", linewidth=2.5)
    ax.text(ax.get_xlim()[1], TEACHER_SUCCESS, f" teacher {TEACHER_SUCCESS:.0%}", va="center", fontsize=13)
    ax.set_title(panel, fontsize=15)
    ax.set_ylabel("Online success rate\n(128 evaluation episodes)", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc="lower right")
axes[-1].set_xlabel("Gradient steps", fontsize=15)
fig.suptitle("Offline-RL students on deployable observations, by demonstration data", fontsize=17)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(out, dpi=150, bbox_inches="tight")
print("WROTE", out)
