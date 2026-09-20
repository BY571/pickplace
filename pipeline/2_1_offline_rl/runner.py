"""Shared trainer for the offline-RL students. Loaded by file path from each algorithm's train.py.

Every algorithm gets the same data mix, the same inputs, the same evaluation protocol and the same budget;
only ``make_algo`` differs. An algorithm module exposes::

    make_algo(cfg, image_shapes, obs_keys, action_dim, device) -> Algo

    class Algo:
        policy: TensorDictModule   # what is evaluated and checkpointed as the student
        update(batch) -> Mapping[str, float]
        state_dict() -> dict

Artifacts go to ``run.dir`` (default ``$FOOD_ROBOT_ARTIFACTS/students/<name>``): ``checkpoints/`` with one
``.pt`` + ``.json`` manifest per checkpoint (git commit, config, shard provenance, the online evaluation at
that checkpoint), and the run's ``manifest.json``. Prints ``RUN_INFO``, ``METRICS``, ``EVAL``,
``CHECKPOINT``, ``STOP_REASON`` as JSON lines, then ``<KIND>_DONE``.
Stop gracefully with SIGTERM: ``docker exec <c> pkill -TERM -f "kit/python/bin/python3.*train.py"``.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pickplace.artifacts import artifacts_root, git_commit, new_run_dir, sha256_file, update_json, write_json
from pickplace.offline import TransitionSampler, evaluate_student, make_student_env, prefetch
from pickplace.system import memory_used_gb


def _shard_paths(cfg) -> list[Path]:
    """Shard names resolve against ``$FOOD_ROBOT_ARTIFACTS/shards``; absolute paths are used as given."""
    root = artifacts_root() / "shards"
    return [Path(s) if Path(s).is_absolute() else root / s for s in cfg.data.shards]


def train(cfg, make_algo, kind: str) -> None:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    name = cfg.run.name or f"{kind}_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(cfg.run.dir) if cfg.run.dir else new_run_dir("students", name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    config = OmegaConf.to_container(cfg, resolve=True)

    obs_keys = [tuple(k) if not isinstance(k, str) else k for k in cfg.data.obs_keys]
    sampler = TransitionSampler(
        _shard_paths(cfg),
        proportions=cfg.data.proportions,
        batch_size=cfg.batch_size,
        obs_keys=obs_keys,
        include_terminals=cfg.data.include_terminals,
        pin_memory=device.type == "cuda",
        seed=cfg.seed,
    )
    provenance = sampler.provenance()
    write_json(manifest_path, {
        "kind": f"offline_{kind}", "algorithm": kind, "git_commit": git_commit(), "config": config,
        "shards": provenance, "transitions": len(sampler),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ended_at": None, "stop_reason": None, "gradient_steps": 0, "checkpoints": [], "best_eval": None,
    })

    probe = sampler.sample()
    shapes = {k: tuple(probe.get(k).shape[1:]) for k in sampler.obs_keys}
    if any(len(s) != 3 for s in shapes.values()):
        raise ValueError(f"The student takes camera observations only (H x W x C); got shapes {shapes}.")
    image_shapes = [shapes[k] for k in sampler.obs_keys]
    action_dim = int(probe.get("action").shape[-1])

    algo = make_algo(cfg, image_shapes, sampler.obs_keys, action_dim, device)
    total_steps = int(cfg.gradient_steps)
    print("RUN_INFO " + json.dumps({
        "run_dir": str(run_dir), "algorithm": kind, "gradient_steps": total_steps,
        "batch_size": cfg.batch_size, "rows_per_batch": sampler.counts,
        "transitions": len(sampler), "obs_keys": [list(k) for k in sampler.obs_keys],
        "image_shapes": [list(s) for s in image_shapes], "action_dim": action_dim,
        "shards": [p["name"] for p in provenance],
        "parameters": sum(p.numel() for p in algo.policy.parameters()),
    }), flush=True)

    logger = None
    if cfg.logger.backend:
        from torchrl.record.loggers import generate_exp_name, get_logger

        logger = get_logger(
            cfg.logger.backend,
            logger_name=str(run_dir / "logs"),
            experiment_name=generate_exp_name(kind.upper(), cfg.logger.exp_name),
            wandb_kwargs={"config": config, "project": cfg.logger.project_name, "group": cfg.logger.group},
        )
        if cfg.logger.backend == "wandb":
            update_json(manifest_path, wandb_url=logger.experiment.url)

    eval_env, eval_interval = None, int(cfg.eval.interval)
    if eval_interval:
        eval_env = make_student_env(sampler.shards[0].manifest, cfg.eval.num_envs, cfg.eval.image_size, cfg.eval.seed)
    latest_eval: dict | None = None
    latest_eval_step = -1
    eval_seconds = 0.0
    best = {"success_rate": -1.0, "step": None}
    checkpoints: list[str] = []
    history: list[dict] = []

    def run_eval(step: int) -> dict:
        nonlocal latest_eval, latest_eval_step, eval_seconds
        t0 = time.monotonic()
        result = evaluate_student(algo.policy, eval_env)
        eval_seconds += time.monotonic() - t0
        latest_eval, latest_eval_step = result, step
        history.append({"step": step, **result})
        print("EVAL " + json.dumps({"step": step, "seconds": round(time.monotonic() - t0, 1), **result}), flush=True)
        if result.get("success_rate", -1.0) > best["success_rate"]:
            best.update(success_rate=result["success_rate"], step=step)
            update_json(manifest_path, best_eval=dict(best))
        return result

    def save(name: str, step: int) -> Path:
        path = run_dir / "checkpoints" / f"{name}.pt"
        tmp = path.with_name(path.name + ".tmp")
        torch.save({**algo.state_dict(), "step": step, "config": config,
                    "obs_keys": [list(k) for k in sampler.obs_keys], "image_shapes": [list(s) for s in image_shapes],
                    "action_dim": action_dim}, tmp)
        tmp.replace(path)
        if eval_env is not None and latest_eval_step != step:
            run_eval(step)
        write_json(path.with_suffix(".json"), {
            "checkpoint": path.name, "algorithm": kind, "gradient_steps": step, "git_commit": git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sha256": sha256_file(path),
            "config": config, "shards": provenance,
            "eval": latest_eval, "eval_step": latest_eval_step if latest_eval else None,
            "eval_num_envs": int(cfg.eval.num_envs) if latest_eval else None,
        })
        checkpoints.append(path.name)
        update_json(manifest_path, checkpoints=checkpoints, gradient_steps=step, eval_history=history)
        print("CHECKPOINT " + json.dumps({"name": path.name, "step": step}), flush=True)
        return path

    stop = {"reason": "gradient_steps"}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(reason="sigterm"))

    batches = prefetch(sampler.sample, depth=int(cfg.prefetch_depth), workers=int(cfg.prefetch_workers))
    log_interval, checkpoint_interval = int(cfg.log_interval), int(cfg.checkpoint.interval)
    reward_scale = float(cfg.reward_scale)
    sums: dict[str, float] = {}
    step, start, last_log = 0, time.monotonic(), time.monotonic()
    sample_s = update_s = 0.0
    try:
        for step in range(1, total_steps + 1):
            if stop["reason"] != "gradient_steps":
                step -= 1
                break
            if cfg.max_hours and time.monotonic() - start > cfg.max_hours * 3600:
                stop["reason"] = "max_hours"
                step -= 1
                break
            t0 = time.perf_counter()
            batch = next(batches).to(device, non_blocking=True)
            if reward_scale != 1.0:
                batch.set(("next", "reward"), batch.get(("next", "reward")) * reward_scale)
            t1 = time.perf_counter()
            losses = algo.update(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            sample_s += t1 - t0
            update_s += t2 - t1
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value)

            if step % log_interval == 0:
                now = time.monotonic()
                metrics = {f"train/{k}": v / log_interval for k, v in sums.items()}
                metrics.update({
                    "perf/steps_per_s": log_interval / (now - last_log),
                    "perf/sample_s_per_step": sample_s / log_interval,
                    "perf/update_s_per_step": update_s / log_interval,
                    "perf/eval_fraction": eval_seconds / max(now - start, 1e-9),
                    "perf/memory_used_gb": memory_used_gb(),
                    "perf/elapsed_h": (now - start) / 3600.0,
                })
                if latest_eval:
                    metrics.update({f"eval/{k}": v for k, v in latest_eval.items()})
                    metrics["eval/step"] = latest_eval_step
                print("METRICS " + json.dumps({"step": step, **metrics}), flush=True)
                if logger is not None:
                    for key, value in metrics.items():
                        logger.log_scalar(key, value, step=step)
                sums, sample_s, update_s, last_log = {}, 0.0, 0.0, now

            if eval_interval and step % eval_interval == 0 and step % checkpoint_interval != 0:
                run_eval(step)
            if checkpoint_interval and step % checkpoint_interval == 0:
                save(f"{kind}_{step}", step)
    except BaseException as error:  # noqa: BLE001 - save the student, then re-raise
        stop["reason"] = f"error: {type(error).__name__}"
        raise
    finally:
        save(f"{kind}_final", step)
        stop_info = {"reason": stop["reason"], "gradient_steps": step, "best_eval": dict(best),
                     "hours": round((time.monotonic() - start) / 3600.0, 3)}
        print("STOP_REASON " + json.dumps(stop_info), flush=True)
        update_json(manifest_path, stop_reason=stop["reason"], gradient_steps=step, best_eval=dict(best),
                    eval_history=history, ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if logger is not None and cfg.logger.backend == "wandb":
            logger.experiment.summary.update({f"stop/{k}": v for k, v in stop_info.items()})
            logger.experiment.finish()  # os._exit below skips wandb's exit hooks
    print(f"{kind.upper()}_DONE", flush=True)
    os._exit(0)  # Isaac Sim shutdown can hang
