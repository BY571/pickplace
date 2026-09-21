"""Plot online success-rate curves for the stage-2.1 students: deployable inputs (cameras only;
cameras+proprio) vs the oracle/upper-bound runs (full teacher-state, simulator-only privileged
observations) vs the teacher reference line. See ../../../pipeline/2_1_offline_rl/README.md.

Usage (on the Spark, where the artifacts live):
    python docs/experiments/offline_rl/plot_curves.py <run_dir>=<label>=<group> ... [out=<png>]

<group> is "deploy" (solid line; real-robot-measurable observations) or "oracle" (dashed line;
includes simulator-only privileged state, not deployable).
"""

import json
import sys
from pathlib import Path

import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

TEACHER_SUCCESS = 0.984  # pipeline/0_state_teacher evaluation of ppo_teacher_final.pt, same protocol

args = [a for a in sys.argv[1:] if not a.startswith("out=")]
out = next((a[len("out="):] for a in sys.argv[1:] if a.startswith("out=")), "success_rate.png")

series = []  # (label, group, steps, values)
for arg in args:
    run_dir, label, group = arg.split("=", 2)
    assert group in ("deploy", "oracle"), f"unknown group {group!r} (want deploy|oracle)"
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    history = manifest["eval_history"]
    final = history[-1]["success_rate"]
    steps = [e["step"] for e in history]
    values = [e["success_rate"] for e in history]
    tag = " — oracle (privileged state)" if group == "oracle" else ""
    series.append((f"{label}{tag} ({final:.0%})", group, steps, values))

all_steps = sorted({s for _, _, steps, _ in series for s in steps})

sns.set()
fig, ax = plt.subplots(figsize=(15, 8))
palette = sns.color_palette(n_colors=len(series))
for (label, group, steps, values), color in zip(series, palette):
    ax.plot(steps, values, linewidth=2.5, linestyle="--" if group == "oracle" else "-",
            label=label, color=color)

ax.plot(all_steps, [TEACHER_SUCCESS] * len(all_steps), linewidth=2.5, linestyle=":", color="black",
        label=f"Teacher, privileged state, cameras off ({TEACHER_SUCCESS:.0%})")

ax.set_title("Offline-RL students: deployable inputs (solid) vs oracle/privileged-state inputs (dashed)",
             fontsize=15)
ax.set_xlabel("Gradient steps", fontsize=15)
ax.set_ylabel("Online success rate (128 evaluation episodes)", fontsize=15)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0.0, 1.05)
ax.tick_params(labelsize=12)
ax.legend(fontsize=13, loc="best", title=None)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"WROTE {out}")
