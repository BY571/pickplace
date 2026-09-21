"""Plot the data-quality comparison: BC/IQL x {expert-only, expert+medium} (see the ablations report).

Top panel: the four raw curves plus the teacher. Bottom panel: the same four curves with a shaded
+-noise band (binomial standard error at n=128 episodes, ~4.3 points around p=0.6) around each point, so a
reader can see which step-to-step differences are real movement and which are within evaluation noise.

Usage (on the Spark, where the artifacts live):
    python docs/experiments/offline_rl/plot_data_quality.py [artifacts=<dir>] [out=<png>]
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

TEACHER_SUCCESS = 0.984  # pipeline/0_state_teacher evaluation of ppo_teacher_final.pt, same protocol
NOISE = 0.043  # binomial SE around p=0.6 at n=128 evaluation episodes (see pipeline/2_1_offline_rl/README.md)

RUNS = [
    ("students/bc_expert_medium_v1", "BC, expert+medium"),
    ("students/bc_expert_only", "BC, expert-only"),
    ("students/iql_expert_medium_v1", "IQL, expert+medium"),
    ("students/iql_expert_only", "IQL, expert-only"),
]

artifacts = Path(next(
    (a[len("artifacts="):] for a in sys.argv[1:] if a.startswith("artifacts=")),
    os.environ.get("FOOD_ROBOT_ARTIFACTS", os.path.expanduser("~/food-robot-artifacts")),
))
out = next((a[len("out="):] for a in sys.argv[1:] if a.startswith("out=")), "data_quality.png")

rows = []
for rel, label in RUNS:
    manifest = json.loads((artifacts / rel / "manifest.json").read_text())
    history = manifest["eval_history"]
    final = history[-1]["success_rate"]
    for entry in history:
        rows.append({"Gradient steps": entry["step"], "Success rate": entry["success_rate"],
                     "Series": f"{label} ({final:.0%})"})

steps = sorted({r["Gradient steps"] for r in rows})
teacher_label = f"Privileged teacher, cameras off ({TEACHER_SUCCESS:.0%})"
for step in steps:
    rows.append({"Gradient steps": step, "Success rate": TEACHER_SUCCESS, "Series": teacher_label})

df = pd.DataFrame(rows)
student_df = df[df["Series"] != teacher_label]

sns.set()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

sns.lineplot(data=df, x="Gradient steps", y="Success rate", hue="Series", linewidth=2.5, ax=ax1)
ax1.set_title("Data-quality ablation: expert-only vs. expert+medium demonstrations", fontsize=15)
ax1.set_ylabel("Online success rate (128 eval episodes)", fontsize=15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax1.set_ylim(0.0, 1.05)
ax1.tick_params(labelsize=12)
ax1.legend(fontsize=13, loc="best", title=None)

palette = dict(zip(sorted(student_df["Series"].unique()), sns.color_palette(n_colors=student_df["Series"].nunique())))
for series, sub in student_df.groupby("Series"):
    sub = sub.sort_values("Gradient steps")
    color = palette[series]
    ax2.plot(sub["Gradient steps"], sub["Success rate"], linewidth=2.5, label=series, color=color)
    ax2.fill_between(sub["Gradient steps"], sub["Success rate"] - NOISE, sub["Success rate"] + NOISE,
                      color=color, alpha=0.15, linewidth=0)
ax2.set_title(f"Same four curves with the per-point evaluation-noise band (binomial SE, ~{NOISE:.0%})", fontsize=15)
ax2.set_xlabel("Gradient steps", fontsize=15)
ax2.set_ylabel("Online success rate (128 eval episodes)", fontsize=15)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax2.set_ylim(0.0, 1.05)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=13, loc="best", title=None)

fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"WROTE {out}")
