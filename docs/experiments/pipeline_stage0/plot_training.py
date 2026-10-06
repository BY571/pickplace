"""Plot a state-teacher PPO run from the METRICS lines its training process printed.

Writes two figures:
  <out>_success.png   train and evaluation success rate
  <out>_reward.png    the episode return, and the weighted contribution of each reward term to it

The per-term contributions are the unweighted term sums times the run's reward weights, so they add up to
the episode return exactly.

Usage (on the machine holding the artifacts):
    python docs/experiments/pipeline_stage0/plot_training.py <run_dir> [out=<prefix>] [title=<str>]
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

LABELS = {  # dense task terms in the order the policy has to climb them, then the one-shot bonus
    "reach_food": "reach the item",
    "grasp": "close the gripper on it",
    "grasp_lift": "lift it",
    "transport": "carry it to the bowl",
    "food_in_bowl": "release it in the bowl",
    "success": "success bonus",
    "return_home": "return home",
    "bowl_disturbance": "bowl disturbance",
    "action_rate": "action rate",
    "joint_vel": "joint velocity",
}

run_dir = Path(sys.argv[1])
args = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
out = args.get("out", "training")
title = args.get("title", run_dir.name)

logs = glob.glob(str(run_dir / "logs" / "wandb" / "run-*" / "files" / "output.log"))
if not logs:
    raise SystemExit(f"No W&B output.log under {run_dir}")
rows = [json.loads(line[len("METRICS "):]) for line in open(logs[0]) if line.startswith("METRICS ")]
if not rows:
    raise SystemExit("No METRICS lines found")

manifest = json.loads((run_dir / "manifest.json").read_text())
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pickplace.rewards import resolve_reward_weights
    weights = resolve_reward_weights(manifest["config"]["env"])
except Exception as exc:
    # The run may name a reward set this checkout no longer ships. The logged return is an exact linear
    # combination of the logged term sums, so recover the weights from the run's own numbers instead.
    import numpy as np
    terms = sorted(k[len("train/terms/"):] for k in rows[-1] if k.startswith("train/terms/"))
    A = np.array([[r.get(f"train/terms/{t}", 0.0) for t in terms] for r in rows if "train/episode_return" in r])
    b = np.array([r["train/episode_return"] for r in rows if "train/episode_return" in r])
    fit, *_ = np.linalg.lstsq(A, b, rcond=None)
    weights = {t: float(round(w, 4)) for t, w in zip(terms, fit) if abs(w) > 1e-3}
    residual = float(np.abs(A @ fit - b).max())
    print(f"reward set unavailable ({exc.__class__.__name__}); recovered weights from the run: "
          f"{weights}, max residual {residual:.3f}")


def series(key):
    return [(r["frames"] / 1e6, r[key]) for r in rows if key in r]


sns.set()

# ---------------------------------------------------------------- success
fig, ax = plt.subplots(figsize=(15, 8))
for key, label, color in (("train/success_rate", "training", sns.color_palette()[0]),
                          ("eval/success_rate", "evaluation (deterministic)", sns.color_palette()[1])):
    xs, ys = zip(*series(key))
    sns.lineplot(x=xs, y=ys, label=f"{label} ({ys[-1]:.0%})", linewidth=2.5, color=color, ax=ax)
ax.set_title(title, fontsize=15)
ax.set_xlabel("Environment frames (millions)", fontsize=15)
ax.set_ylabel("Episodes ending in success", fontsize=15)
ax.set_ylim(-0.02, 1.02)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.tick_params(labelsize=12)
ax.legend(fontsize=15, loc="lower right")
fig.tight_layout()
fig.savefig(f"{out}_success.png", dpi=150, bbox_inches="tight")
print("WROTE", f"{out}_success.png")

# ---------------------------------------------------------------- reward
order = list(LABELS)  # task order, not whatever order the weights happen to be in
active = [t for t in order if weights.get(t, 0.0) != 0.0 and series(f"train/terms/{t}")]
active += [t for t, w in weights.items() if w != 0.0 and t not in order and series(f"train/terms/{t}")]
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

xs, ys = zip(*series("train/episode_return"))
sns.lineplot(x=xs, y=ys, label=f"episode return ({ys[-1]:.0f})", linewidth=2.5,
             color=sns.color_palette()[0], ax=ax_top)
ax_top.set_title("Total reward per episode", fontsize=15)
ax_top.set_ylabel("Episode return", fontsize=15)
ax_top.tick_params(labelsize=12)
ax_top.legend(fontsize=13, loc="lower right")

for term, color in zip(active, sns.color_palette(n_colors=len(active))):
    pairs = series(f"train/terms/{term}")
    xs = [x for x, _ in pairs]
    ys = [v * weights[term] for _, v in pairs]
    sns.lineplot(x=xs, y=ys, label=f"{LABELS.get(term, term)} ({ys[-1]:.1f})", linewidth=2.5,
                 color=color, ax=ax_bot)
ax_bot.set_title("Split by reward term, weighted: the stages of the task, learned in order", fontsize=15)
ax_bot.set_xlabel("Environment frames (millions)", fontsize=15)
ax_bot.set_ylabel("Contribution to the return", fontsize=15)
ax_bot.tick_params(labelsize=12)
ax_bot.legend(fontsize=13, loc="upper left")

fig.suptitle(title, fontsize=17)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(f"{out}_reward.png", dpi=150, bbox_inches="tight")
print("WROTE", f"{out}_reward.png")
