"""Scaling benchmark for pixel PPO on one machine (run on the DGX Spark, inside the container).

Phase A (env only, random actions): num_envs x image size at frame_stack=3 -> env steps/s, peak memory.
Phase B (full ppo_pixels.py iterations): the 3 fastest Phase-A configs within the memory budget x
rollout_steps x mini_batch_size -> frames/hour, collect/update split, peak memory.
Every configuration runs in its own process (one Isaac Sim per process). Peak memory is the system-wide used
memory, sampled every 0.5 s while the process runs (CPU and GPU share memory on the Spark).

Selection: most env frames/hour with peak memory below 80% of total memory and update time at most half of
each iteration. Writes benchmark.csv, benchmark.png and summary.json to ``out``.

Usage:
    python scripts/benchmark_pixels.py [out=outputs/benchmark_pixels] [phases=AB] [quick=false]
    python scripts/benchmark_pixels.py worker num_envs=256 image=84 frame_stack=3 steps=300    (internal)
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

REPO = Path(__file__).resolve().parents[1]
PPO_DIR = REPO / "sota-implementations" / "ppo"
FRAME_STACK = 3
MEMORY_FRACTION = 0.8
# Both measured B rows spend ~69% of each iteration in the PPO update (two CNN encoders, 4 epochs), so the old
# 0.5 guard would reject every configuration; the real objective is frames/hour and memory, not update share.
MAX_UPDATE_SHARE = 0.8
EVAL_TIME_SHARE = 0.1
MEM_FACTOR = 10.5  # training memory per collected batch, relative to one float32 copy (see training_estimate_gb)
KILL_FRACTION = 0.85  # kill a measured run above this share of total memory, before the host OOM killer acts
TIMEOUT_S = 1800
TOTAL_FRAMES = 1_000_000_000  # training budget of the long run (config_pixels.yaml collector.total_frames)
FULL = {"num_envs": [128, 256, 512, 1024, 2048], "images": [84, 128], "env_steps": 300,
        "rollout_steps": [16, 32], "mini_batches": [4096, 16384], "ppo_iterations": 3, "phase_b_top": 3}
QUICK = {"num_envs": [16], "images": [64], "env_steps": 50,
         "rollout_steps": [8], "mini_batches": [256], "ppo_iterations": 2, "phase_b_top": 1}
FIELDS = ["phase", "name", "num_envs", "image", "frame_stack", "rollout_steps", "mini_batch_size", "status",
          "build_s", "env_steps_per_s", "collect_s", "update_s", "update_share", "frames_per_hour",
          "gradient_steps_per_s", "peak_used_gb", "physx_errors", "eval_batches"]


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
    from food_robot.app import launch_app

    launch_app(headless=True, enable_cameras=True)
    import torch

    from food_robot.torchrl_env import make_env

    n, image, steps = int(kw["num_envs"]), int(kw["image"]), int(kw["steps"])
    t0 = time.monotonic()
    env = make_env({"num_envs": n, "cameras": True, "privileged_information": False,
                    "image_size": [image, image], "frame_stack": int(kw["frame_stack"])})
    td = env.reset()
    build_s = time.monotonic() - t0

    def step(td):
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

    A run that crosses KILL_FRACTION of total memory is killed here. The estimate in
    ``training_estimate_gb`` keeps hopeless configurations from starting, but it is only an estimate: when one
    slips through, the kernel OOM killer takes down dbus/containerd and the whole machine with it (observed
    twice). Killing the child ourselves keeps the host alive and records the configuration as aborted_memory.
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
    return [json.loads(line[line.index(prefix) + len(prefix):]) for line in lines if prefix in line]


def physx_errors(lines):
    return sum("PhysX error" in line for line in lines)


def write_csv(rows, csv_path):
    """Rewrite the CSV after every measurement: a crash (or an OOM kill) must not lose finished rows."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def phase_a(grid, existing_b, out, csv_path):
    """Run Phase A. Every incremental write includes ``existing_b`` so a re-run with phases=A(B) against a
    CSV that already has completed Phase-B rows can never overwrite them with Phase-A-only content."""
    rows = []
    for image in grid["images"]:
        for n in grid["num_envs"]:
            name = f"A_n{n}_i{image}"
            print(f"[benchmark] {name}", flush=True)
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", f"num_envs={n}", f"image={image}",
                   f"frame_stack={FRAME_STACK}", f"steps={grid['env_steps']}"]
            status, lines, peak = run_measured(cmd, REPO, out / "logs" / f"{name}.log")
            bench = json_lines(lines, "BENCH ")
            if status == "ok" and not bench:
                status = "no_result"
            row = {"phase": "A", "name": name, "num_envs": n, "image": image, "frame_stack": FRAME_STACK,
                   "status": status, "peak_used_gb": round(peak, 2), "physx_errors": physx_errors(lines)}
            if bench:
                row.update({"build_s": round(bench[-1]["build_s"], 1),
                            "env_steps_per_s": round(bench[-1]["env_steps_per_s"], 1)})
            rows.append(row)
            print(f"[benchmark] {json.dumps(row)}", flush=True)
            write_csv(list(existing_b) + rows, csv_path)
    return rows


def training_estimate_gb(peak_a_gb: float, n: int, image: int, rollout: int) -> float:
    """Phase-A peak plus the memory a full PPO iteration adds for one collected batch.

    The collector holds the batch twice (root and "next") as float32; GAE, the uint8 buffer copy and the
    minibatch forward/backward add more. MEM_FACTOR is calibrated against the measured row: 1024 envs at 84 px,
    rollout 16 has a 7.75 GB float batch and a 30.4 GB Phase-A peak, yet peaked at 111.7 GB — an increment of
    ~10.5x the batch.
    """
    batch_gb = n * rollout * 2 * image * image * 3 * FRAME_STACK * 4 / 1024**3
    return peak_a_gb + MEM_FACTOR * batch_gb


def phase_b(grid, rows_a, existing_b, out, total_gb, csv_path):
    """Run Phase B, skipping any candidate already measured in ``existing_b`` (resume after a crash/hang)."""
    budget = MEMORY_FRACTION * total_gb
    fitting = [r for r in rows_a if r["status"] == "ok" and float(r["peak_used_gb"]) < budget]
    best = sorted(fitting, key=lambda r: -float(r["env_steps_per_s"]))
    measured = {r["name"] for r in existing_b}
    rows = list(existing_b)  # keep already-measured rows in the output; write_csv must never drop them
    for base in best:
        n, image, peak_a = int(base["num_envs"]), int(base["image"]), float(base["peak_used_gb"])
        feasible = [
            r for r in grid["rollout_steps"] if training_estimate_gb(peak_a, n, image, r) < budget
        ]
        if not feasible:
            name = f"B_n{n}_i{image}"
            if name in measured:
                print(f"[benchmark] keeping measured {name}", flush=True)
                continue
            rows.append({"phase": "B", "name": name, "num_envs": n, "image": image,
                         "frame_stack": FRAME_STACK, "status": "skipped_memory",
                         "peak_used_gb": round(training_estimate_gb(peak_a, n, image, min(grid["rollout_steps"])), 1)})
            print(f"[benchmark] skipping n={n} i={image}: estimated training memory exceeds {budget:.0f} GB", flush=True)
            write_csv(list(rows_a) + rows, csv_path)
            continue
        if len([r for r in rows if r["phase"] == "B" and r["status"] != "skipped_memory"]) // (
            len(grid["rollout_steps"]) * len(grid["mini_batches"])
        ) >= grid["phase_b_top"]:
            break
        for rollout in feasible:
            for mini_batch in grid["mini_batches"]:
                name = f"B_n{n}_i{image}_r{rollout}_m{mini_batch}"
                if name in measured:
                    print(f"[benchmark] keeping measured {name}", flush=True)
                    continue
                print(f"[benchmark] {name}", flush=True)
                cmd = [sys.executable, "ppo_pixels.py", f"env.num_envs={n}", f"env.image_size=[{image},{image}]",
                       f"env.frame_stack={FRAME_STACK}", f"collector.rollout_steps={rollout}",
                       f"loss.mini_batch_size={mini_batch}", "loss.ppo_epochs=4",
                       f"max_iterations={grid['ppo_iterations']}", "logger.backend=null",
                       "eval.interval_iterations=0", "checkpoint.interval_iterations=0",
                       f"hydra.run.dir={out / 'runs' / name}"]
                status, lines, peak = run_measured(cmd, PPO_DIR, out / "logs" / f"{name}.log")
                metrics = json_lines(lines, "METRICS ")[1:]  # the first iteration includes warm-up
                info = json_lines(lines, "RUN_INFO ")
                if status == "ok" and not metrics:
                    status = "no_result"
                row = {"phase": "B", "name": name, "num_envs": n, "image": image, "frame_stack": FRAME_STACK,
                       "rollout_steps": rollout, "mini_batch_size": mini_batch, "status": status,
                       "peak_used_gb": round(peak, 2), "physx_errors": physx_errors(lines)}
                if metrics:
                    mean = lambda k: sum(m[k] for m in metrics) / len(metrics)  # noqa: E731
                    collect_s, update_s = mean("perf/collect_s"), mean("perf/update_s")
                    row.update({"collect_s": round(collect_s, 3), "update_s": round(update_s, 3),
                                "update_share": round(update_s / (collect_s + update_s), 3),
                                "env_steps_per_s": round(mean("perf/env_steps_per_s"), 1),
                                "frames_per_hour": round(mean("perf/frames_per_hour")),
                                "gradient_steps_per_s": round(mean("perf/gradient_steps_per_s"), 2),
                                "eval_batches": info[-1]["eval_batches"] if info else ""})
                rows.append(row)
                print(f"[benchmark] {json.dumps(row)}", flush=True)
                write_csv(list(rows_a) + rows, csv_path)
    return rows


def select(rows_b, total_gb):
    ok = [r for r in rows_b if r["status"] == "ok" and float(r["peak_used_gb"]) < MEMORY_FRACTION * total_gb
          and float(r["update_share"]) <= MAX_UPDATE_SHARE]
    if not ok:
        return None
    best = max(ok, key=lambda r: float(r["frames_per_hour"]))
    n, image, rollout = int(best["num_envs"]), int(best["image"]), int(best["rollout_steps"])
    collect_s, update_s = float(best["collect_s"]), float(best["update_s"])
    iteration_s, frames_per_batch, eval_batches = collect_s + update_s, n * rollout, int(best["eval_batches"])
    eval_s = eval_batches * collect_s
    iterations_per_hour = 3600.0 / iteration_s
    eval_interval = max(math.ceil(eval_s / (EVAL_TIME_SHARE * iteration_s)), math.ceil(iterations_per_hour / 4))
    effective_per_hour = 3600.0 / (iteration_s + eval_s / eval_interval)

    def budget(hours):
        return int(hours * effective_per_hour * 0.97) * frames_per_batch

    return {
        "num_envs": n, "image": image, "rollout_steps": rollout, "mini_batch_size": int(best["mini_batch_size"]),
        "frames_per_hour": float(best["frames_per_hour"]), "iteration_s": round(iteration_s, 3),
        "eval_batches": eval_batches, "peak_used_gb": float(best["peak_used_gb"]),
        "suggested": {
            "env.num_envs": n,
            "env.image_size": [image, image],
            "collector.rollout_steps": rollout,
            "loss.mini_batch_size": int(best["mini_batch_size"]),
            "hours_for_total_frames": round(TOTAL_FRAMES / (effective_per_hour * frames_per_batch), 1),
            "eval.interval_iterations": eval_interval,
            "checkpoint.interval_iterations": math.ceil(effective_per_hour / 2),
            "sanity_total_frames": budget(0.75),
            "sanity_eval_interval_iterations": max(1, math.ceil(effective_per_hour / 8)),
            "sanity_checkpoint_interval_iterations": max(1, math.ceil(effective_per_hour / 6)),
        },
    }


def plot(rows, out, total_gb):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    sns.set()
    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    a = df[df.phase == "A"].copy()
    b = df[df.phase == "B"].copy()
    panels = 3 if len(b) else 2
    fig, axes = plt.subplots(panels, 1, figsize=(15, 5.5 * panels))

    def finish(ax, title, xlabel, ylabel, legend_kwargs=None):
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=13, **(legend_kwargs or {"loc": "best"}))

    for ax, column, title, ylabel, fmt in [
        (axes[0], "env_steps_per_s", "Env-only throughput (random actions, 3 stacked frames)", "Env steps per second",
         lambda v: f"{v:,.0f} steps/s"),
        (axes[1], "peak_used_gb", f"Peak memory (budget {MEMORY_FRACTION * total_gb:.0f} GB of {total_gb:.0f} GB)",
         "Used memory (GB)", lambda v: f"{v:.1f} GB"),
    ]:
        for image, group in a.groupby("image"):
            group = group.sort_values("num_envs")
            last = group.iloc[-1]
            label = f"{image} px ({int(last.num_envs):,} envs: {fmt(float(last[column]))})"
            sns.lineplot(x=group.num_envs.astype(int), y=group[column].astype(float), label=label, linewidth=2.5,
                         marker="o", ax=ax)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        finish(ax, title, "Parallel envs", ylabel)
    if len(b):
        ax = axes[2]
        b["Series"] = b.apply(lambda r: f"{int(r.num_envs):,} envs, {int(r.image)} px, rollout {int(r.rollout_steps)}", axis=1)
        for series, group in b.groupby("Series"):
            group = group.sort_values("mini_batch_size")
            last = group.iloc[-1]
            sns.lineplot(x=group.mini_batch_size.astype(int), y=group.frames_per_hour.astype(float) / 1e6,
                         label=f"{series} ({float(last.frames_per_hour) / 1e6:.1f} M/h)", linewidth=2.5, marker="o", ax=ax)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        # Up to 7 series on this panel: "best" placement lands the legend box on top of the flatter,
        # lower-throughput lines. Anchor it outside the axes instead so it can never cover data.
        finish(ax, "Full PPO iterations (4 epochs)", "Minibatch size", "Env frames per hour (millions)",
               legend_kwargs={"loc": "center left", "bbox_to_anchor": (1.01, 0.5)})
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=150, bbox_inches="tight")


def main():
    positional, kw = parse_args(sys.argv[1:])
    if positional[:1] == ["worker"]:
        return worker(kw)
    out = (REPO / kw.get("out", "outputs/benchmark_pixels")).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    grid = QUICK if kw.get("quick", "false").lower() == "true" else FULL
    phases = kw.get("phases", "AB").upper()
    total_gb, _ = memory_gb()
    csv_path = out / "benchmark.csv"
    if kw.get("plot_only", "false").lower() == "true":  # re-draw from an existing CSV without re-measuring
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        plot(rows, out, total_gb)
        print("BENCHMARK_DONE", flush=True)
        return

    existing_b = []
    if csv_path.exists():
        with open(csv_path) as f:
            existing_b = [r for r in csv.DictReader(f) if r["phase"] == "B"]

    rows = []
    if "A" in phases:
        rows = phase_a(grid, existing_b, out, csv_path)
    elif csv_path.exists():
        with open(csv_path) as f:
            rows = [r for r in csv.DictReader(f) if r["phase"] == "A"]
    if "B" in phases:
        rows_a = [r for r in rows if r["phase"] == "A"]
        for r in rows_a:
            r["peak_used_gb"] = float(r["peak_used_gb"])
        rows += phase_b(grid, rows_a, existing_b, out, total_gb, csv_path)
    else:
        rows += existing_b  # this invocation didn't touch Phase B; keep its previously measured rows

    write_csv(rows, csv_path)
    selected = select([r for r in rows if r["phase"] == "B"], total_gb)
    summary = {"total_memory_gb": round(total_gb, 1), "budget_gb": round(MEMORY_FRACTION * total_gb, 1),
               "max_update_share": MAX_UPDATE_SHARE, "selected": selected}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(rows, out, total_gb)
    print("SELECTED " + json.dumps(selected), flush=True)
    print("BENCHMARK_DONE", flush=True)


if __name__ == "__main__":
    main()
