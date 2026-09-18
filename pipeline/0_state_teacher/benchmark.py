"""Scaling benchmark for the state teacher (cameras off; run on the DGX Spark, inside the container).

Phase A  (env only, random actions): num_envs -> env steps/s of the TorchRL env (``make_env``), peak memory.
Phase A2 (collection split) at 4096 envs and at the fastest Phase-A size: steps/s of (raw) the Isaac Lab gym env,
         (wrapper) the bare ``IsaacLabWrapper``, (env) the ``make_env`` TransformedEnv — Phase A's row — and
         (policy) the TransformedEnv plus the actor forward, as the collector runs it.
Phase B  (full ``train.py`` iterations): the fastest Phase-A sizes within the memory budget x rollout_steps x
         mini_batch_size -> frames/hour, collect/update split, peak memory.
Phase C  (speed options on the Phase-B winner): baseline, compile, compile + cudagraphs, shifted GAE,
         matmul precision "high", and the best combination of the options that helped.
Every configuration runs in its own process (one Isaac Sim per process). Peak memory is the system-wide used
memory, sampled every 0.5 s while the process runs (CPU and GPU share memory on the Spark); a run above
KILL_FRACTION of total memory is killed.

Selection: most env frames/hour among runs with 0 PhysX errors and peak memory below
MEMORY_FRACTION * total - reserve_gb (reserve = the teacher's background checkpoint worker). A Phase-C option is
only eligible with finite metrics and losses in the baseline's range. Writes benchmark.csv, benchmark.png and
summary.json to ``out``; prints ``SELECTED {json}`` and ``BENCHMARK_DONE``.

Usage:
    python pipeline/0_state_teacher/benchmark.py [out=outputs/benchmark_state] [phases=AXBC] [quick=false]
        [reserve_gb=<float>] [plot_only=false]            (phase letters: A, X = A2, B, C)
    python pipeline/0_state_teacher/benchmark.py worker kind=env num_envs=4096 steps=300    (internal)
"""

import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = Path(__file__).resolve().parent / "train.py"
MEMORY_FRACTION = 0.8
KILL_FRACTION = 0.85  # kill a measured run above this share of total memory, before the host OOM killer acts
TIMEOUT_S = 1800
DEFAULT_RESERVE_GB = 20.0  # background worker (eval + render process); pass reserve_gb= from a measurement
FULL = {"num_envs": [4096, 8192, 16384, 32768], "env_steps": 300, "split_envs": 4096,
        "rollout_steps": [16, 24, 32], "mini_batches": [32768, 131072], "ppo_iterations": 4, "phase_b_top": 3,
        "phase_c_iterations": 6}
QUICK = {"num_envs": [64], "env_steps": 50, "split_envs": 64,
         "rollout_steps": [8], "mini_batches": [256], "ppo_iterations": 2, "phase_b_top": 1,
         "phase_c_iterations": 3}
SPLIT_KINDS = ["raw", "wrapper", "env", "policy"]
# Phase C variants (train.py overrides); the combination is built from the ones that beat the baseline.
OPTIONS = {
    "baseline": [],
    "compile": ["compile.compile=true"],
    "compile_cudagraphs": ["compile.compile=true", "compile.cudagraphs=true"],
    "shifted_gae": ["loss.shifted_gae=true"],
    "matmul_high": ["optim.matmul_precision=high"],
}
MIN_GAIN = 1.02  # an option must beat the baseline's frames/hour by 2% to join the combination
LOSS_KEYS = ["train/loss_critic", "train/loss_objective", "train/entropy", "train/kl_approx"]
FIELDS = ["phase", "name", "kind", "options", "num_envs", "rollout_steps", "mini_batch_size", "status", "build_s",
          "env_steps_per_s", "collect_s", "update_s", "update_share", "frames_per_hour", "gradient_steps_per_s",
          "peak_used_gb", "physx_errors", "metrics_finite", "losses_ok", "loss_critic", "loss_objective",
          "entropy", "kl_approx"]


def parse_args(argv):
    positional = [a for a in argv if "=" not in a]
    return positional, dict(a.split("=", 1) for a in argv if "=" in a)


def memory_gb() -> tuple[float, float]:
    """(total, used) system memory in GB."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])
    return info["MemTotal"] / 1024**2, (info["MemTotal"] - info["MemAvailable"]) / 1024**2


def worker(kw):
    """Steps/s of one layer of the collection stack with random actions (``policy``: actor actions)."""
    from food_robot.app import launch_app

    launch_app(headless=True, enable_cameras=False)
    import torch

    n, steps, kind = int(kw["num_envs"]), int(kw["steps"]), kw.get("kind", "env")
    env_cfg = {"num_envs": n, "cameras": False, "privileged_information": True}
    t0 = time.monotonic()
    if kind == "raw":
        import gymnasium as gym

        import food_robot.envs  # noqa: F401  (gym registration)
        from food_robot.config import DEFAULT_ENV, build_cell_env_cfg

        env = gym.make(DEFAULT_ENV["task"], cfg=build_cell_env_cfg(env_cfg))
        env.reset()
        action_dim = env.unwrapped.action_space.shape[-1]
        build_s = time.monotonic() - t0

        def step(state):
            env.step(torch.rand(n, action_dim, device=env.unwrapped.device) * 2 - 1)
            return state

        td = None
    else:
        from food_robot.torchrl_env import make_env

        env = make_env(env_cfg)
        if kind == "wrapper":
            env = env.base_env
        td = env.reset()
        build_s = time.monotonic() - t0
        actor = None
        if kind == "policy":
            import importlib.util

            from omegaconf import OmegaConf

            spec = importlib.util.spec_from_file_location("teacher_utils", Path(__file__).parent / "utils.py")
            tu = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tu)
            network = OmegaConf.load(Path(__file__).parent / "config.yaml").network
            actor, _ = tu.make_teacher_models(env, network, torch.device("cuda:0"))

        def step(td):
            if actor is not None:
                td = actor(td)
            else:
                td.set("action", env.action_spec.rand())
            _, td = env.step_and_maybe_reset(td)
            return td

    with torch.no_grad():
        for _ in range(20):
            td = step(td)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(steps):
            td = step(td)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
    print("BENCH " + json.dumps({"build_s": build_s, "env_steps_per_s": n * steps / dt}), flush=True)
    os._exit(0)


def run_measured(cmd, cwd, log_path):
    """Run cmd, sampling peak system memory; returns (status, stdout lines, peak used GB).

    A run that crosses KILL_FRACTION of total memory is killed here: when the kernel OOM killer acts instead it
    takes down dbus/containerd and the whole machine with it (observed twice). The configuration is recorded
    as aborted_memory.
    """
    total_gb = memory_gb()[0]
    kill_above = KILL_FRACTION * total_gb
    peak = {"used": memory_gb()[1]}
    killed = threading.Event()
    stop = threading.Event()
    proc = None  # assigned below; the sampler thread starts only after Popen returns

    def sample():
        while not stop.is_set():
            used = memory_gb()[1]
            peak["used"] = max(peak["used"], used)
            if used > kill_above:
                killed.set()
                print(f"[benchmark] memory guard: {used:.0f} GB > {kill_above:.0f} GB, killing the run", flush=True)
                proc.kill()
                return
            time.sleep(0.5)

    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    timed_out = threading.Event()
    watchdog = threading.Timer(TIMEOUT_S, lambda: (timed_out.set(), proc.kill()))
    watchdog.start()
    lines = []
    with open(log_path, "w") as log:
        for line in proc.stdout:
            log.write(line)
            lines.append(line.rstrip("\n"))
    proc.wait()
    watchdog.cancel()
    stop.set()
    sampler.join()
    if killed.is_set():
        status = "aborted_memory"
    elif timed_out.is_set():
        status = "timeout"
    else:
        status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    return status, lines, peak["used"]


def json_lines(lines, prefix):
    """Payloads of ``prefix`` lines, matched anywhere in the line (tqdm's stderr bar can share a line)."""
    return [json.loads(line[line.index(prefix) + len(prefix):]) for line in lines if prefix in line]


def physx_errors(lines):
    """PhysX errors and buffer-overflow warnings (``PxgAABBManager``, "exceeding" capacity)."""
    return sum(("PhysX error" in line) or ("PxgAABBManager" in line) or ("exceeding" in line) for line in lines)


def write_csv(rows, csv_path):
    """Rewrite the CSV after every measurement: a crash (or an OOM kill) must not lose finished rows."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class Table:
    """All rows, keyed by name; measured rows of an earlier invocation are kept (resume after a crash)."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.rows = {}
        if csv_path.exists():
            with open(csv_path) as f:
                self.rows = {r["name"]: r for r in csv.DictReader(f)}

    def done(self, name):
        if name in self.rows and self.rows[name]["status"] == "ok":
            print(f"[benchmark] keeping measured {name}", flush=True)
            return True
        return False

    def add(self, row):
        self.rows[row["name"]] = row
        print(f"[benchmark] {json.dumps(row)}", flush=True)
        write_csv(list(self.rows.values()), self.csv_path)

    def phase(self, phase):
        return [r for r in self.rows.values() if r["phase"] == phase]


def env_row(table, out, phase, n, kind, steps):
    name = f"A_n{n}" if kind == "env" else f"X_n{n}_{kind}"
    if table.done(name):
        return
    print(f"[benchmark] {name}", flush=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", f"kind={kind}", f"num_envs={n}", f"steps={steps}"]
    status, lines, peak = run_measured(cmd, REPO, out / "logs" / f"{name}.log")
    bench = json_lines(lines, "BENCH ")
    if status == "ok" and not bench:
        status = "no_result"
    row = {"phase": phase, "name": name, "kind": kind, "num_envs": n, "status": status,
           "peak_used_gb": round(peak, 2), "physx_errors": physx_errors(lines)}
    if bench:
        row.update({"build_s": round(bench[-1]["build_s"], 1), "env_steps_per_s": round(bench[-1]["env_steps_per_s"], 1)})
    table.add(row)


def phase_a(grid, table, out):
    for n in grid["num_envs"]:
        env_row(table, out, "A", n, "env", grid["env_steps"])


def fits(row, budget):
    return row["status"] == "ok" and int(float(row["physx_errors"])) == 0 and float(row["peak_used_gb"]) < budget


def phase_a2(grid, table, out, budget):
    ok = [r for r in table.phase("A") if fits(r, budget)]
    best = max(ok, key=lambda r: float(r["env_steps_per_s"]))["num_envs"] if ok else grid["split_envs"]
    for n in sorted({int(grid["split_envs"]), int(best)}):
        for kind in SPLIT_KINDS:
            env_row(table, out, "A" if kind == "env" else "X", n, kind, grid["env_steps"])


def train_row(table, out, phase, name, n, rollout, mini_batch, iterations, options=(), skip=1):
    if table.done(name):
        return
    print(f"[benchmark] {name}", flush=True)
    cmd = [sys.executable, str(TRAIN), f"env.num_envs={n}", f"collector.rollout_steps={rollout}",
           f"loss.mini_batch_size={mini_batch}", f"max_iterations={iterations}",
           "logger.backend=null", "checkpoint.interval_frames=0", "worker.enabled=false",
           f"run.dir={out / 'runs' / name}", f"hydra.run.dir={out / 'hydra' / name}", *options]
    status, lines, peak = run_measured(cmd, REPO, out / "logs" / f"{name}.log")
    all_metrics = json_lines(lines, "METRICS ")
    metrics = all_metrics[skip:]  # the first iteration(s) include warm-up (and compilation)
    if status == "ok" and ("PPO_DONE" not in "\n".join(lines) or not metrics):
        status = "no_result"
    row = {"phase": phase, "name": name, "options": " ".join(options), "num_envs": n, "rollout_steps": rollout,
           "mini_batch_size": mini_batch, "status": status, "peak_used_gb": round(peak, 2),
           "physx_errors": physx_errors(lines)}
    if metrics:
        mean = lambda k: sum(m[k] for m in metrics) / len(metrics)  # noqa: E731
        collect_s, update_s = mean("perf/collect_s"), mean("perf/update_s")
        finite = all(math.isfinite(v) for m in all_metrics for v in m.values() if isinstance(v, (int, float)))
        row.update({"collect_s": round(collect_s, 3), "update_s": round(update_s, 3),
                    "update_share": round(update_s / (collect_s + update_s), 3),
                    "env_steps_per_s": round(mean("perf/env_steps_per_s"), 1),
                    "frames_per_hour": round(mean("perf/frames_per_hour")),
                    "gradient_steps_per_s": round(mean("perf/gradient_steps_per_s"), 2),
                    "metrics_finite": finite})
        # losses over all iterations (the same seed and data volume in every variant)
        for key in LOSS_KEYS:
            values = [m[key] for m in all_metrics if key in m]
            row[key.split("/", 1)[1]] = round(sum(values) / len(values), 5) if values else ""
    table.add(row)


def phase_b(grid, table, out, budget):
    """Fastest Phase-A sizes within the budget x rollout x mini-batch (mini-batch <= frames per batch)."""
    fitting = sorted([r for r in table.phase("A") if fits(r, budget)], key=lambda r: -float(r["env_steps_per_s"]))
    for base in fitting[: grid["phase_b_top"]]:
        n = int(base["num_envs"])
        for rollout in grid["rollout_steps"]:
            for mini_batch in grid["mini_batches"]:
                if mini_batch > n * rollout:
                    continue
                train_row(table, out, "B", f"B_n{n}_r{rollout}_m{mini_batch}", n, rollout, mini_batch,
                          grid["ppo_iterations"])


def losses_ok(row, base):
    """Losses in the baseline's range: critic loss within 2x, entropy within 10%, KL at most 5x (+0.01)."""
    try:
        critic, b_critic = float(row["loss_critic"]), float(base["loss_critic"])
        entropy, b_entropy = float(row["entropy"]), float(base["entropy"])
        kl, b_kl = float(row["kl_approx"]), float(base["kl_approx"])
    except (KeyError, ValueError):
        return False
    return (0.5 * b_critic <= critic <= 2.0 * b_critic and abs(entropy - b_entropy) <= 0.1 * abs(b_entropy) + 0.05
            and kl <= 5.0 * b_kl + 0.01)


def qualified(row, budget):
    return fits(row, budget) and str(row.get("metrics_finite")) == "True" and str(row.get("losses_ok")) == "True"


def phase_c(grid, table, out, budget, winner):
    n, rollout, mb = int(winner["num_envs"]), int(winner["rollout_steps"]), int(winner["mini_batch_size"])
    iterations = grid["phase_c_iterations"]

    def run(variant, options):
        train_row(table, out, "C", f"C_{variant}", n, rollout, mb, iterations, options, skip=2)

    for variant, options in OPTIONS.items():
        run(variant, options)
    base = table.rows["C_baseline"]
    for row in table.phase("C"):
        row["losses_ok"] = row["status"] == "ok" and losses_ok(row, base)
    write_csv(list(table.rows.values()), table.csv_path)
    if base["status"] != "ok":
        return
    helped = {v for v in OPTIONS if v != "baseline" and qualified(table.rows[f"C_{v}"], budget)
              and float(table.rows[f"C_{v}"]["frames_per_hour"]) > MIN_GAIN * float(base["frames_per_hour"])}
    combo = []
    compiled = [v for v in ("compile", "compile_cudagraphs") if v in helped]
    if compiled:
        combo.append(max(compiled, key=lambda v: float(table.rows[f"C_{v}"]["frames_per_hour"])))
    combo += [v for v in ("shifted_gae", "matmul_high") if v in helped]
    if len(combo) >= 2:
        options = [o for v in combo for o in OPTIONS[v]]
        run("combo", options)
        row = table.rows["C_combo"]
        row["losses_ok"] = row["status"] == "ok" and losses_ok(row, base)
        write_csv(list(table.rows.values()), table.csv_path)


def select(rows, budget, reserve_gb):
    ok = [r for r in rows if fits(r, budget) and (r["phase"] != "C" or qualified(r, budget))]
    if not ok:
        return None
    best = max(ok, key=lambda r: float(r["frames_per_hour"]))
    n, rollout, mb = int(best["num_envs"]), int(best["rollout_steps"]), int(best["mini_batch_size"])
    fph = float(best["frames_per_hour"])
    options = dict(o.split("=", 1) for o in str(best.get("options") or "").split())
    return {"name": best["name"], "num_envs": n, "rollout_steps": rollout, "mini_batch_size": mb,
            "options": options, "frames_per_hour": fph, "peak_used_gb": float(best["peak_used_gb"]),
            "reserve_gb": reserve_gb, "hours_for_200M": round(200_000_000 / fph, 1),
            "suggested": {"env.num_envs": n, "collector.rollout_steps": rollout, "loss.mini_batch_size": mb,
                          **options}}


def plot(rows, out, total_gb, budget):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    sns.set()
    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    a = df[(df.phase == "A")].copy()
    x = df[df.phase.isin(["A", "X"])].copy()
    b = df[df.phase == "B"].copy()
    c = df[df.phase == "C"].copy()
    split_sizes = sorted({int(v) for v in x[x.phase == "X"].num_envs}) if len(x) else []
    panels = 2 + bool(split_sizes) + bool(len(b)) + bool(len(c))
    fig, axes = plt.subplots(panels, 1, figsize=(15, 5.5 * panels))
    thousands = FuncFormatter(lambda v, _: f"{int(v):,}")

    def finish(ax, title, xlabel, ylabel, legend_kwargs=None):
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=13, **(legend_kwargs or {"loc": "best"}))

    a = a.sort_values("num_envs")
    last = a.iloc[-1]
    for ax, column, title, ylabel, label in [
        (axes[0], "env_steps_per_s", "Env-only throughput (TorchRL env, random actions)", "Env steps per second",
         f"env only ({int(last.num_envs):,} envs: {float(last.env_steps_per_s):,.0f} steps/s)"),
        (axes[1], "peak_used_gb", f"Peak memory (budget {budget:.0f} GB = 80% of {total_gb:.0f} GB minus worker reserve)",
         "Used memory (GB)", f"env only ({int(last.num_envs):,} envs: {float(last.peak_used_gb):.1f} GB)"),
    ]:
        sns.lineplot(x=a.num_envs.astype(int), y=a[column].astype(float), label=label, linewidth=2.5, marker="o", ax=ax)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(thousands)
        finish(ax, title, "Parallel envs", ylabel)
    i = 2
    names = {"raw": "Isaac Lab gym env", "wrapper": "+ IsaacLabWrapper", "env": "+ transforms (make_env)",
             "policy": "+ actor forward"}
    if split_sizes:
        ax = axes[i]
        i += 1
        s = x[x.num_envs.astype(int).isin(split_sizes) & x.kind.isin(list(names))].copy()
        s["Stage"] = pd.Categorical(s.kind.map(names), categories=list(names.values()), ordered=True)
        s["Envs"] = s.num_envs.astype(int).map(lambda v: f"{v:,} envs")
        s = s.sort_values(["Stage", "num_envs"])
        s["steps"] = s.env_steps_per_s.astype(float)
        labels = {}
        for envs, group in s.groupby("Envs"):
            raw, full = group[group.kind == "raw"].steps, group[group.kind == "policy"].steps
            tail = f"{float(full.iloc[0]):,.0f}/s with actor" if len(full) else ""
            if len(raw) and len(full):
                tail += f", {float(full.iloc[0]) / float(raw.iloc[0]):.0%} of raw"
            labels[envs] = f"{envs} ({tail})"
        s["Series"] = s.Envs.map(labels)
        sns.barplot(data=s, x="Stage", y="steps", hue="Series", ax=ax, dodge=True)
        ax.yaxis.set_major_formatter(thousands)
        finish(ax, "Collection stack, layer by layer (random actions; last: actor actions)", "Layer",
               "Env steps per second", legend_kwargs={"loc": "center left", "bbox_to_anchor": (1.01, 0.5)})
    if len(b):
        ax = axes[i]
        i += 1
        b["Series"] = b.apply(lambda r: f"{int(r.num_envs):,} envs, rollout {int(r.rollout_steps)}", axis=1)
        for series, group in b.groupby("Series"):
            group = group.sort_values("mini_batch_size")
            last = group.iloc[-1]
            sns.lineplot(x=group.mini_batch_size.astype(int), y=group.frames_per_hour.astype(float) / 1e6,
                         label=f"{series} ({float(last.frames_per_hour) / 1e6:.0f} M/h)", linewidth=2.5, marker="o",
                         ax=ax)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(thousands)
        finish(ax, "Full PPO iterations (4 epochs)", "Minibatch size", "Env frames per hour (millions)",
               legend_kwargs={"loc": "center left", "bbox_to_anchor": (1.01, 0.5)})
    if len(c):
        ax = axes[i]
        c = c.copy()
        c["Variant"] = c.name.str.replace("C_", "", regex=False)
        c["Series"] = c.apply(lambda r: f"{r.Variant} ({float(r.frames_per_hour) / 1e6:.1f} M/h)", axis=1)
        first = c.iloc[0]
        c["mph"] = c.frames_per_hour.astype(float) / 1e6
        sns.barplot(data=c, x="Variant", y="mph", hue="Series", ax=ax, dodge=False)
        finish(ax, f"Speed options at {int(first.num_envs):,} envs, rollout {int(first.rollout_steps)}, "
               f"minibatch {int(first.mini_batch_size):,}", "Variant", "Env frames per hour (millions)",
               legend_kwargs={"loc": "center left", "bbox_to_anchor": (1.01, 0.5)})
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=150, bbox_inches="tight")


def main():
    positional, kw = parse_args(sys.argv[1:])
    if positional[:1] == ["worker"]:
        return worker(kw)
    out = (REPO / kw.get("out", "outputs/benchmark_state")).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    grid = QUICK if kw.get("quick", "false").lower() == "true" else FULL
    phases = kw.get("phases", "AXBC").upper()
    reserve_gb = float(kw.get("reserve_gb", DEFAULT_RESERVE_GB))
    total_gb, _ = memory_gb()
    budget = MEMORY_FRACTION * total_gb - reserve_gb
    table = Table(out / "benchmark.csv")
    if kw.get("plot_only", "false").lower() != "true":
        if "A" in phases:
            phase_a(grid, table, out)
        if "X" in phases:
            phase_a2(grid, table, out, budget)
        if "B" in phases:
            phase_b(grid, table, out, budget)
        if "C" in phases:
            winner = select(table.phase("B"), budget, reserve_gb)
            if winner is not None:
                phase_c(grid, table, out, budget, winner)
    rows = list(table.rows.values())
    write_csv(rows, table.csv_path)
    # Phase C re-measures the Phase-B winner (baseline) next to its options: select within C when it ran.
    pool = [r for r in rows if r["phase"] == "C" and r["status"] == "ok"] or [r for r in rows if r["phase"] == "B"]
    selected = select(pool, budget, reserve_gb)
    summary = {"total_memory_gb": round(total_gb, 1), "budget_gb": round(budget, 1), "reserve_gb": reserve_gb,
               "selected": selected}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(rows, out, total_gb, budget)
    print("SELECTED " + json.dumps(selected), flush=True)
    print("BENCHMARK_DONE", flush=True)


if __name__ == "__main__":
    main()
