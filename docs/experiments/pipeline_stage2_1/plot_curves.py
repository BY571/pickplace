"""Plot the online success-rate curves of the stage 2.1 students from their run manifests.

Usage (on the Spark, where the artifacts live):
    python docs/experiments/pipeline_stage2_1/plot_curves.py <run_dir> [<run_dir> ...] [out=<png>]
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
for run in args:
    manifest = json.loads((Path(run) / "manifest.json").read_text())
    label = manifest["algorithm"].upper()
    history = manifest["eval_history"]
    final = history[-1]["success_rate"]
    for entry in history:
        rows.append({"Gradient steps": entry["step"], "Success rate": entry["success_rate"],
                     "Series": f"{label} ({final:.0%})"})

steps = sorted({r["Gradient steps"] for r in rows})
for step in steps:
    rows.append({"Gradient steps": step, "Success rate": TEACHER_SUCCESS,
                 "Series": f"Privileged state teacher ({TEACHER_SUCCESS:.0%})"})

sns.set()
fig, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(data=pd.DataFrame(rows), x="Gradient steps", y="Success rate", hue="Series", linewidth=2.5, ax=ax)
ax.set_title("Camera-only students on expert + medium shards, online evaluation every 10,000 gradient steps",
             fontsize=15)
ax.set_xlabel("Gradient steps", fontsize=15)
ax.set_ylabel("Success rate (% of 128 evaluation episodes)", fontsize=15)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0.0, 1.05)
ax.tick_params(labelsize=12)
ax.legend(fontsize=15, loc="best", title=None)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"WROTE {out}")
