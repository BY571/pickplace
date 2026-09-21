"""Plot online success-rate curves for the stage-2.1 baselines + ablations (see the ablations report).

Usage (on the Spark, where the artifacts live):
    python docs/experiments/offline_rl/plot_curves.py <run_dir>=<label> ... [out=<png>]
"""

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

TEACHER_SUCCESS = 0.984  # pipeline/0_state_teacher evaluation of ppo_teacher_final.pt, same protocol

args = [a for a in sys.argv[1:] if not a.startswith("out=")]
out = next((a[len("out="):] for a in sys.argv[1:] if a.startswith("out=")), "success_rate.png")

rows = []
for arg in args:
    run_dir, label = arg.split("=", 1)
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    history = manifest["eval_history"]
    final = history[-1]["success_rate"]
    for entry in history:
        rows.append({"Gradient steps": entry["step"], "Success rate": entry["success_rate"],
                     "Series": f"{label} ({final:.0%})"})

steps = sorted({r["Gradient steps"] for r in rows})
for step in steps:
    rows.append({"Gradient steps": step, "Success rate": TEACHER_SUCCESS,
                 "Series": f"Privileged teacher, ceiling ({TEACHER_SUCCESS:.0%})"})

sns.set()
fig, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(data=pd.DataFrame(rows), x="Gradient steps", y="Success rate", hue="Series", linewidth=2.5, ax=ax)
ax.set_title("Offline-RL students: data-tier and observation-access ablations vs the two baselines", fontsize=15)
ax.set_xlabel("Gradient steps", fontsize=15)
ax.set_ylabel("Online success rate (128 evaluation episodes)", fontsize=15)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0.0, 1.05)
ax.tick_params(labelsize=12)
ax.legend(fontsize=13, loc="best", title=None)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"WROTE {out}")
