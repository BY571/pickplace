"""Plot (and sanity-check) a ppo_pixels.py run from its METRICS lines.

Usage:
    python scripts/plot_training.py <metrics.jsonl> --out <dir> [--title "..."] [--check-sanity]

<metrics.jsonl> may hold raw log lines containing ``METRICS {...}`` (e.g. ``docker logs`` output) or bare
JSON objects. Writes <out>/learning_curves.png, <out>/outcomes.png and <out>/summary.json.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

OUTCOMES = ["success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "food_off_table", "time_out"]
SMOOTH = 10  # iterations


def load_rows(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        index = line.find("METRICS ")
        text = line[index + len("METRICS "):] if index >= 0 else line.strip()
        if text.startswith("{"):
            rows.append(json.loads(text))
    if not rows:
        raise SystemExit(f"No METRICS rows in {path}")
    return sorted(rows, key=lambda r: r["frames"])


def _slope(rows, key) -> float:
    ys = [r[key] for r in rows if key in r and math.isfinite(r[key])]
    if len(ys) < 3:
        return float("nan")
    return float(np.polyfit(np.arange(len(ys)), np.asarray(ys), 1)[0])


def sanity_checks(rows) -> dict[str, bool]:
    finite = all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float)))
    sums = [sum(r[f"train/{t}_rate"] for t in OUTCOMES) for r in rows if "train/success_rate" in r]
    return {
        "no_nan": finite,
        # A strict > 0 comparison is fooled by floating-point noise (~1e-17) from np.polyfit on a perfectly
        # flat series; require a slope clearly distinguishable from that noise.
        "return_trends_up": _slope(rows, "train/episode_return") > 1e-9,
        "reach_food_trends_up": _slope(rows, "episode_reward/reach_food") > 1e-9,
        "outcome_rates_sum_to_one": bool(sums) and all(0.999 <= s <= 1.05 for s in sums),
        "eval_logged": any("eval/success_rate" in r for r in rows),
    }


def _table(rows, series, scale=1.0, smooth=True):
    import pandas as pd

    parts = []
    for key, label, fmt in series:
        points = [(r["frames"] / 1e6, r[key] * scale) for r in rows if key in r]
        if not points:
            continue
        x, y = zip(*points)
        y = pd.Series(y).rolling(SMOOTH if smooth else 1, min_periods=1).mean()
        parts.append(pd.DataFrame({"frames": x, "value": y, "Series": f"{label} ({fmt(y.iloc[-1])})"}))
    return pd.concat(parts, ignore_index=True) if parts else None


def _panel(ax, table, title, ylabel, percent=False, signed=False):
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    if table is not None:
        sns.lineplot(data=table, x="frames", y="value", hue="Series", linewidth=2.5, ax=ax)
        ax.legend(fontsize=13, loc="best")
    if signed:
        ax.axhline(0, color="black", linewidth=0.8)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Environment frames (millions)", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(labelsize=12)


def plot(rows, out: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd

    sns.set()
    number, percent = (lambda v: f"{v:+.1f}"), (lambda v: f"{v:.1f}%")

    def train_and_eval(key, scale, fmt):
        # training curves are smoothed; evaluation points (sparse, deterministic) are not
        parts = [
            _table(rows, [(f"train/{key}", "Training episodes, 10-iteration mean", fmt)], scale=scale),
            _table(rows, [(f"eval/{key}", "Deterministic evaluation", fmt)], scale=scale, smooth=False),
        ]
        parts = [p for p in parts if p is not None]
        return pd.concat(parts, ignore_index=True) if parts else None

    fig, axes = plt.subplots(3, 1, figsize=(15, 16.5), sharex=True)
    _panel(axes[0], train_and_eval("episode_return", 1.0, number), "Episode return", "Return (unscaled reward)",
           signed=True)
    _panel(axes[1], train_and_eval("success_rate", 100.0, percent), "Success rate", "% of finished episodes",
           percent=True)
    _panel(axes[2], _table(rows, [(f"episode_reward/{t}", t.replace("_", " "), number)
                                  for t in ["reach_food", "grasp_lift", "transport", "transport_fine"]]),
           "Reward terms", "Episodic reward per second")
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 8))
    _panel(ax, _table(rows, [(f"train/{t}_rate", t.replace("_", " "), percent) for t in OUTCOMES], scale=100.0),
           "How training episodes end", "% of finished episodes", percent=True)
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "outcomes.png", dpi=150)
    plt.close(fig)


def summarize(rows) -> dict:
    last = [r for r in rows[-SMOOTH:]]
    mean = lambda k: float(np.mean([r[k] for r in last if k in r])) if any(k in r for r in last) else None  # noqa: E731
    evals = [r for r in rows if "eval/success_rate" in r]
    return {
        "iterations": len(rows),
        "frames": rows[-1]["frames"],
        "elapsed_h": rows[-1].get("perf/elapsed_h"),
        "final_train_success_rate": mean("train/success_rate"),
        "final_train_episode_return": mean("train/episode_return"),
        "final_outcome_rates": {t: mean(f"train/{t}_rate") for t in OUTCOMES},
        "best_eval_success_rate": max((r["eval/success_rate"] for r in evals), default=None),
        "last_eval": {k: v for k, v in evals[-1].items() if k.startswith("eval/")} if evals else None,
        "mean_frames_per_hour": mean("perf/frames_per_hour"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Pixel PPO run 1")
    parser.add_argument("--check-sanity", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.metrics)
    plot(rows, out, args.title)
    summary = summarize(rows)
    if args.check_sanity:
        summary["sanity"] = sanity_checks(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.check_sanity:
        print("SANITY_PASS" if all(summary["sanity"].values()) else "SANITY_FAIL " + json.dumps(summary["sanity"]))


if __name__ == "__main__":
    main()
