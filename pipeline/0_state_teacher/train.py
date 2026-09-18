"""Stage 0: privileged state teacher — PPO from robot state, bowl pose and food pose (cameras off).

Usage: python pipeline/0_state_teacher/train.py [key=value ...]    (config: config.yaml next to this file)

Artifacts go to ``run.dir`` (default ``$FOOD_ROBOT_ARTIFACTS/teachers/<name>``): ``checkpoints/`` with one
``.pt`` + ``.json`` manifest per checkpoint, and the run's ``manifest.json``. Prints ``RUN_INFO``, ``METRICS``
(per iteration), ``CHECKPOINT``, ``STOP_REASON`` lines as JSON, then ``PPO_DONE``.
Stop gracefully with SIGTERM: ``docker exec <container> pkill -TERM -f "kit/python/bin/python3.*train.py"``.
"""

import json
import os
import signal
import time

import hydra
from omegaconf import DictConfig, OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_local_module(name: str):
    """Load a sibling module by file path: Isaac Sim's bundled ``cv2/utils`` can shadow ``import utils``."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(f"teacher_{name}", os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"teacher_{name}"] = module
    spec.loader.exec_module(module)
    return module


@hydra.main(config_path="", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.env.cameras:
        raise ValueError("The state teacher trains with cameras off; set env.cameras=false.")

    from food_robot.app import launch_app

    launch_app(headless=cfg.app.headless, enable_cameras=False, device=cfg.env.device)

    from pathlib import Path

    import warnings

    import torch
    import tqdm
    from tensordict.nn import CudaGraphModule
    from torchrl._utils import compile_with_warmup
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger

    from food_robot.artifacts import git_commit, new_run_dir, update_json, write_json
    from food_robot.system import memory_used_gb
    from food_robot.torchrl_env import make_env
    from food_robot.training import SuccessStreak, episode_metrics

    tu = _load_local_module("utils")

    run_dir = Path(cfg.run.dir) if cfg.run.dir else new_run_dir("teachers", cfg.run.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "state_teacher",
            "git_commit": git_commit(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ended_at": None,
            "stop_reason": None,
            "frames": 0,
            "checkpoints": [],
        },
    )

    torch.manual_seed(cfg.env.seed)
    torch.set_float32_matmul_precision(cfg.optim.matmul_precision)  # "highest" is PyTorch's default
    # Compile wiring follows TorchRL's sota-implementations/ppo/ppo_mujoco.py.
    compile_on, cudagraphs = bool(cfg.compile.compile), bool(cfg.compile.cudagraphs)
    compile_mode = None
    graphed = compile_on or cudagraphs  # outputs live in reused graph buffers: mark steps, clone outputs
    if compile_on:
        compile_mode = cfg.compile.compile_mode or ("default" if cudagraphs else "reduce-overhead")
    device = torch.device(cfg.env.device)
    env = make_env(OmegaConf.to_container(cfg.env, resolve=True))
    num_envs = env.batch_size[0]
    frames_per_batch = num_envs * int(cfg.collector.rollout_steps)
    if cfg.max_iterations:
        total_iterations = int(cfg.max_iterations)
    else:
        total_iterations = int(cfg.collector.total_frames) // frames_per_batch
    if total_iterations < 1:
        raise ValueError(
            f"No iteration would run: collector.total_frames={cfg.collector.total_frames} is below frames_per_batch="
            f"{frames_per_batch} (env.num_envs x collector.rollout_steps). Set max_iterations or raise total_frames."
        )
    mini_batch_size = min(int(cfg.loss.mini_batch_size), frames_per_batch)
    print(
        "RUN_INFO "
        + json.dumps(
            {
                "run_dir": str(run_dir),
                "num_envs": num_envs,
                "frames_per_batch": frames_per_batch,
                "total_iterations": total_iterations,
                "max_episode_length": int(env.base_env._env.unwrapped.max_episode_length),
            }
        ),
        flush=True,
    )

    try:
        actor, critic = tu.make_teacher_models(env, cfg.network, device)
        collector = Collector(
            env,
            actor,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=device,
            no_cuda_sync=True,
            trust_policy=True,
            **(
                {
                    "compile_policy": {"mode": compile_mode, "warmup": 1} if compile_on else False,
                    "cudagraph_policy": {"warmup": 10} if cudagraphs else False,
                }
                if compile_on or cudagraphs
                else {}
            ),
        )
        buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch, device=device, compilable=compile_on),
            sampler=SamplerWithoutReplacement(),
            batch_size=mini_batch_size,
            compilable=compile_on,
        )
        # shifted: one critic call over [obs_0..obs_T] instead of two (obs and ("next", obs)). Both paths first
        # replace NaN ("next", obs) on done rows (native auto-reset) with the root obs, see
        # ValueEstimatorBase._sanitize_next_obs_nan; truncations beyond the one-slot budget are masked
        # ("shifted_valid"), and ClipPPOLoss drops masked samples.
        gae_kwargs = {"vectorized": False} if compile_on else {}
        if cfg.loss.shifted_gae:
            gae_kwargs["shifted"] = True
        adv_module = GAE(
            gamma=cfg.loss.gamma,
            lmbda=cfg.loss.gae_lambda,
            value_network=critic,
            average_gae=False,
            device=device,
            **gae_kwargs,
        )
        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=cfg.loss.clip_epsilon,
            entropy_coeff=cfg.loss.entropy_coeff,
            critic_coeff=cfg.loss.critic_coeff,
            normalize_advantage=True,
        )
        if graphed:
            # A tensor lr (updated in place) keeps the annealed lr from triggering recompiles; a tensor lr
            # needs capturable=True in eager mode (compile_with_warmup runs the first call eagerly).
            optim = torch.optim.Adam(
                loss_module.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5, capturable=True
            )
        else:
            optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr, eps=1e-5)

        max_grad_norm = cfg.optim.max_grad_norm

        def update(batch):
            loss = loss_module(batch)
            total = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
            optim.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            return loss.detach()

        if compile_on:
            update = compile_with_warmup(update, mode=compile_mode, warmup=1)
            adv_module = compile_with_warmup(adv_module, mode=compile_mode, warmup=1)
        if cudagraphs:
            warnings.warn(
                "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
                category=UserWarning,
            )
            update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
            adv_module = CudaGraphModule(adv_module)

        logger = None
        if cfg.logger.backend:
            logger = get_logger(
                cfg.logger.backend,
                logger_name=str(run_dir / "logs"),
                experiment_name=generate_exp_name("StateTeacher", cfg.logger.exp_name),
                wandb_kwargs={
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "project": cfg.logger.project_name,
                    "group": cfg.logger.group,
                },
            )
            if cfg.logger.backend == "wandb":
                update_json(manifest_path, wandb_url=logger.experiment.url)
    except BaseException as error:  # noqa: BLE001 - a failed setup must not leave the manifest open
        # Without this the run's manifest would keep stop_reason/ended_at null forever (no final
        # checkpoint either), and the worker tooling would read the run as still training.
        update_json(
            manifest_path,
            stop_reason=f"error: {type(error).__name__}",
            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        raise

    worker = tu.CheckpointWorker(run_dir, cfg.worker) if cfg.worker.enabled else None

    def log_worker_results(metrics: dict, step: int) -> None:
        """Merge finished checkpoint evaluations into ``metrics``; upload finished videos."""
        if worker is None:
            return
        for kind, manifest in worker.new_results():
            if kind == "eval":
                metrics.update({f"eval/{k}": v for k, v in manifest["eval"].items()})
                metrics["eval/checkpoint_frames"] = manifest["frames"]
                print("EVAL_LOGGED " + json.dumps({"checkpoint_frames": manifest["frames"], **manifest["eval"]}), flush=True)
            elif logger is not None and cfg.logger.backend == "wandb":
                import wandb

                video = run_dir / "checkpoints" / manifest["video"]
                logger.experiment.log(
                    {"eval/video": wandb.Video(str(video), format="mp4"), "eval/video_checkpoint_frames": manifest["frames"]},
                    step=step,
                )

    checkpoints: list[str] = []

    def save(name: str, frames: int, iteration: int):
        path = tu.save_teacher_checkpoint(run_dir, name, actor, critic, optim, cfg, frames, iteration)
        checkpoints.append(name)
        update_json(manifest_path, checkpoints=checkpoints, frames=frames)
        print("CHECKPOINT " + json.dumps({"name": name, "path": str(path), "frames": frames}), flush=True)
        return path

    stop = {"reason": "total_frames"}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(reason="sigterm"))
    early_stop_on = cfg.early_stop.success_rate is not None
    success_streak = SuccessStreak(cfg.early_stop.success_rate or 0.0, cfg.early_stop.consecutive_iterations)
    interval = int(cfg.checkpoint.interval_frames)

    data_iter = iter(collector)
    frames_total, iteration, start = 0, -1, time.monotonic()
    pbar = tqdm.tqdm(total=total_iterations, desc="iterations")
    try:
        for iteration in range(total_iterations):
            if stop["reason"] != "total_frames":
                break
            if cfg.max_hours and time.monotonic() - start > cfg.max_hours * 3600:
                stop["reason"] = "max_hours"
                break
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            data = next(data_iter)
            torch.cuda.synchronize(device)
            collect_s = time.perf_counter() - t0
            frames_before, frames_total = frames_total, frames_total + data.numel()
            metrics = {"iteration": iteration}
            metrics.update(episode_metrics(data, "train"))
            early_stop_reached = success_streak.update(metrics.get("train/success_rate"))
            metrics["train/success_streak"] = success_streak.count

            t1 = time.perf_counter()
            data.set(("next", "reward"), data.get(("next", "reward")) * cfg.reward_scale)
            loss_sums, num_updates = {}, 0
            for _ in range(cfg.loss.ppo_epochs):
                with torch.no_grad():
                    if graphed:
                        torch.compiler.cudagraph_mark_step_begin()
                    data = adv_module(data)
                    if graphed:
                        data = data.clone()
                buffer.extend(data.exclude("next").reshape(-1))
                for batch in buffer:
                    if graphed:
                        torch.compiler.cudagraph_mark_step_begin()
                    loss = update(batch)
                    if graphed:
                        loss = loss.clone()
                    for key in ("loss_objective", "loss_critic", "loss_entropy", "entropy", "kl_approx", "clip_fraction"):
                        if key in loss.keys():
                            loss_sums[key] = loss_sums.get(key, 0.0) + loss[key].detach().float().mean()
                    num_updates += 1
            torch.cuda.synchronize(device)
            update_s = time.perf_counter() - t1

            metrics.update({f"train/{k}": (v / num_updates).item() for k, v in loss_sums.items()})
            metrics["train/lr"] = float(optim.param_groups[0]["lr"])
            metrics.update(
                {
                    "perf/collect_s": collect_s,
                    "perf/update_s": update_s,
                    "perf/env_steps_per_s": frames_per_batch / collect_s,
                    "perf/frames_per_hour": frames_per_batch / (collect_s + update_s) * 3600.0,
                    "perf/gradient_steps_per_s": num_updates / update_s,
                    "perf/memory_used_gb": memory_used_gb(),
                    "perf/cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "perf/elapsed_h": (time.monotonic() - start) / 3600.0,
                }
            )
            if cfg.optim.anneal_lr:
                for group in optim.param_groups:
                    lr = cfg.optim.lr * (1.0 - (iteration + 1) / total_iterations)
                    if torch.is_tensor(group["lr"]):
                        group["lr"].fill_(lr)
                    else:
                        group["lr"] = lr
            collector.update_policy_weights_()

            log_worker_results(metrics, frames_total)
            print("METRICS " + json.dumps({"frames": frames_total, **metrics}), flush=True)
            if logger is not None:
                for key, value in metrics.items():
                    logger.log_scalar(key, value, step=frames_total)
            pbar.update(1)
            if interval and frames_total // interval > frames_before // interval:
                save(f"ppo_teacher_{frames_total}", frames_total, iteration)
                if worker is not None and not worker.launch():
                    print("[worker] previous pass still running; this checkpoint is picked up by the next pass", flush=True)
            if early_stop_on and early_stop_reached:
                stop["reason"] = "early_stop"
                break
    except BaseException as error:  # noqa: BLE001 - save the policy, then re-raise
        stop["reason"] = f"error: {type(error).__name__}"
        raise
    finally:
        save("ppo_teacher_final", frames_total, iteration)
        stop_info = {"reason": stop["reason"], "frames": frames_total, "success_streak": success_streak.count}
        print("STOP_REASON " + json.dumps(stop_info), flush=True)
        collector.shutdown()
        if worker is not None and cfg.worker.on_exit:
            worker.wait(cfg.worker.exit_timeout_s)  # finish the running pass, then one pass for the final checkpoint
            worker.launch()
            worker.wait(cfg.worker.exit_timeout_s)
            final = {}
            log_worker_results(final, frames_total)
            if logger is not None:
                for key, value in final.items():
                    logger.log_scalar(key, value, step=frames_total)
        update_json(
            manifest_path,
            stop_reason=stop["reason"],
            frames=frames_total,
            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        if logger is not None and cfg.logger.backend == "wandb":
            logger.experiment.summary.update({f"stop/{k}": v for k, v in stop_info.items()})
            logger.experiment.finish()  # os._exit below skips wandb's exit hooks
    print("PPO_DONE", flush=True)
    os._exit(0)  # Isaac Sim shutdown can hang


if __name__ == "__main__":
    main()
