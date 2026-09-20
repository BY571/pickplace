"""PPO from camera images + robot state on the food cell env (no privileged state, no bowl position).

Usage: python ppo_pixels.py [key=value ...]    (config: config_pixels.yaml)

Prints ``RUN_INFO {json}`` once and ``METRICS {json}`` every iteration (training, evaluation and perf/*
metrics); scripts/benchmark_pixels.py and scripts/plot_training.py parse these lines.
"""

import json
import math
import os
import signal
import time

import hydra
from omegaconf import DictConfig, OmegaConf


def _load_local_module(name: str):
    """Load a sibling module (``utils``, ``utils_pixels``) by file path, not by a bare ``import``.

    Isaac Sim's camera extensions put a bundled OpenCV ``cv2/utils`` package where it can shadow a plain
    ``import utils`` (see ``ppo.py::_load_local_utils``).
    """
    import importlib.util
    import sys

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _memory_used_gb() -> float:
    """System-wide used memory [GB]; on the DGX Spark CPU and GPU share this unified memory."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])  # kB
    return (info["MemTotal"] - info["MemAvailable"]) / 1024**2


@hydra.main(config_path="", config_name="config_pixels", version_base="1.3")
def main(cfg: DictConfig):
    if not cfg.env.cameras:
        raise ValueError("ppo_pixels.py trains from camera observations; set env.cameras=true.")

    from pickplace.app import launch_app

    launch_app(headless=cfg.app.headless, enable_cameras=True, device=cfg.env.device)

    import torch
    import tqdm
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger

    from pickplace.torchrl_env import make_env, reward_term_stats

    save_checkpoint = _load_local_module("utils").save_checkpoint
    up = _load_local_module("utils_pixels")

    torch.manual_seed(cfg.env.seed)
    device = torch.device(cfg.env.device)
    env = make_env(OmegaConf.to_container(cfg.env, resolve=True))
    unwrapped = env.base_env._env.unwrapped
    num_envs = env.batch_size[0]
    rollout_steps = int(cfg.collector.rollout_steps)
    frames_per_batch = num_envs * rollout_steps
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
    max_episode_length = int(unwrapped.max_episode_length)
    eval_batches = math.ceil((max_episode_length + 1) / rollout_steps)
    print(
        "RUN_INFO "
        + json.dumps(
            {
                "num_envs": num_envs,
                "frames_per_batch": frames_per_batch,
                "total_iterations": total_iterations,
                "max_episode_length": max_episode_length,
                "eval_batches": eval_batches,
                "step_dt": unwrapped.step_dt,
            }
        ),
        flush=True,
    )

    actor, critic = up.make_ppo_models(env, cfg.network, device)
    pixel_keys = up.image_keys(env.observation_spec, env.batch_size)
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=-1,  # endless: training iterations and evaluation batches both draw from it
        device=device,
        no_cuda_sync=True,
        trust_policy=True,
    )
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=mini_batch_size,
    )
    adv_module = GAE(
        gamma=cfg.loss.gamma, lmbda=cfg.loss.gae_lambda, value_network=critic, average_gae=False, device=device
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_coeff,
        critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr, eps=1e-5)

    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            cfg.logger.backend,
            logger_name="logs",
            experiment_name=generate_exp_name("PPO_pixels", cfg.logger.exp_name),
            wandb_kwargs={
                "config": OmegaConf.to_container(cfg, resolve=True),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group,
            },
        )

    data_iter = iter(collector)

    def evaluate(with_video: bool):
        """Deterministic policy from a fresh reset of every env; each env's first finished episode is scored."""
        previous = collector.exploration_type
        collector.reset()
        collector.exploration_type = ExplorationType.DETERMINISTIC
        slims, frames = [], []
        try:
            for _ in range(eval_batches):
                slim, frame = up.slim_eval_batch(next(data_iter), pixel_keys, with_video)
                slims.append(slim)
                if frame is not None:
                    frames.append(frame)
        finally:
            collector.exploration_type = previous
            collector.reset()
        rollout = torch.cat(slims, dim=1)
        results = up.first_episode_metrics(rollout, "eval")
        video = None
        if with_video:
            done0 = rollout["next", "done"][0].reshape(-1)
            length = int(done0.nonzero()[0, 0]) + 1 if bool(done0.any()) else done0.numel()
            video = up.policy_view_video(frames, pixel_keys, cfg.eval.video_upscale)[:length]
        return results, video

    # Every way of ending the run goes through the final checkpoint below. `docker exec <container> pkill -TERM
    # -f "kit/python/bin/python3.* ppo_pixels.py"` sends SIGTERM: the current iteration finishes, then the loop
    # exits and saves. `docker stop` does not reach this process through Isaac Sim's python.sh wrapper.
    stop = {"reason": "total_frames"}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(reason="sigterm"))
    early_stop_on = cfg.early_stop.success_rate is not None
    success_streak = up.SuccessStreak(cfg.early_stop.success_rate or 0.0, cfg.early_stop.consecutive_iterations)

    frames_total, evals, start = 0, 0, time.monotonic()
    pbar = tqdm.tqdm(total=total_iterations, desc="iterations")
    # The policy is always saved as the last checkpoint, whatever ends the run (budget, early stop, time,
    # SIGTERM, or an exception) — the loop body is wrapped so the finally below always runs.
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
            frames_total += data.numel()
            metrics = {"iteration": iteration}
            metrics.update(up.episode_metrics(data, "train"))
            early_stop_reached = success_streak.update(metrics.get("train/success_rate"))
            metrics["train/success_streak"] = success_streak.count
            metrics.update({f"episode_reward/{k}": v for k, v in reward_term_stats(env).items()})

            t1 = time.perf_counter()
            data.set(("next", "reward"), data.get(("next", "reward")) * cfg.reward_scale)
            loss_sums, num_updates = {}, 0
            for _ in range(cfg.loss.ppo_epochs):
                with torch.no_grad():
                    data = up.compute_advantage(adv_module, data, cfg.loss.gae_env_chunk)
                # ClipPPOLoss reads root observations, action, log-prob, advantage and value_target only: drop
                # "next" and store camera frames as uint8 (together ~8x smaller than the collected float batch).
                buffer.extend(up.compress_pixels(data.exclude("next").reshape(-1), pixel_keys))
                for batch in buffer:
                    loss = loss_module(batch)
                    total = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                    optim.zero_grad(set_to_none=True)
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)
                    optim.step()
                    for key in ("loss_objective", "loss_critic", "loss_entropy", "entropy", "kl_approx", "clip_fraction"):
                        if key in loss.keys():
                            loss_sums[key] = loss_sums.get(key, 0.0) + loss[key].detach().float().mean()
                    num_updates += 1
            torch.cuda.synchronize(device)
            update_s = time.perf_counter() - t1

            metrics.update({f"train/{k}": (v / num_updates).item() for k, v in loss_sums.items()})
            metrics["train/lr"] = optim.param_groups[0]["lr"]
            metrics.update(
                {
                    "perf/collect_s": collect_s,
                    "perf/update_s": update_s,
                    "perf/env_steps_per_s": frames_per_batch / collect_s,
                    "perf/frames_per_hour": frames_per_batch / (collect_s + update_s) * 3600.0,
                    "perf/gradient_steps_per_s": num_updates / update_s,
                    "perf/memory_used_gb": _memory_used_gb(),
                    "perf/cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "perf/elapsed_h": (time.monotonic() - start) / 3600.0,
                }
            )
            if cfg.optim.anneal_lr:
                for group in optim.param_groups:
                    group["lr"] = cfg.optim.lr * (1.0 - (iteration + 1) / total_iterations)
            collector.update_policy_weights_()

            video = None
            if cfg.eval.interval_iterations and (iteration + 1) % cfg.eval.interval_iterations == 0:
                with_video = bool(cfg.eval.video_interval_evals) and evals % cfg.eval.video_interval_evals == 0
                t2 = time.perf_counter()
                eval_metrics, video = evaluate(with_video)
                metrics.update(eval_metrics)
                metrics["perf/eval_s"] = time.perf_counter() - t2
                evals += 1

            print("METRICS " + json.dumps({"frames": frames_total, **metrics}), flush=True)
            if logger is not None:
                for key, value in metrics.items():
                    logger.log_scalar(key, value, step=frames_total)
                if video is not None and cfg.logger.backend == "wandb":
                    logger.log_video(
                        "eval/policy_view", video, step=frames_total, fps=round(1.0 / unwrapped.step_dt), format="mp4"
                    )
            pbar.update(1)
            if cfg.checkpoint.interval_iterations and (iteration + 1) % cfg.checkpoint.interval_iterations == 0:
                save_checkpoint(f"checkpoints/ppo_pixels_{frames_total}.pt", actor, critic, optim, cfg, frames_total)
            if early_stop_on and early_stop_reached:
                stop["reason"] = "early_stop"
                break
    except BaseException as error:  # noqa: BLE001 - save the policy, then re-raise
        stop["reason"] = f"error: {type(error).__name__}"
        raise
    finally:
        save_checkpoint("checkpoints/ppo_pixels_final.pt", actor, critic, optim, cfg, frames_total)
        stop_info = {"reason": stop["reason"], "frames": frames_total, "success_streak": success_streak.count}
        print("STOP_REASON " + json.dumps(stop_info), flush=True)
        collector.shutdown()
        if logger is not None and cfg.logger.backend == "wandb":
            logger.experiment.summary.update({f"stop/{k}": v for k, v in stop_info.items()})
            logger.experiment.finish()  # flush the run: os._exit below skips wandb's exit hooks
    print("PPO_DONE", flush=True)
    os._exit(0)  # Isaac Sim shutdown can hang


if __name__ == "__main__":
    main()
