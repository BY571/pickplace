"""State-teacher models and checkpoints. Loaded by file path from this folder's scripts (see train.py)."""

from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

from food_robot.artifacts import git_commit, read_json, sha256_file, write_json
from food_robot.keys import expand_in_keys


class StateNet(nn.Module):
    """Flattens and concatenates vector observations -> MLP (ELU, orthogonal init)."""

    def __init__(self, shapes, hidden, out_dim: int, out_gain: float):
        super().__init__()
        self.shapes = shapes
        self.mlp = MLP(
            in_features=sum(math.prod(s) for s in shapes),
            out_features=out_dim,
            num_cells=list(hidden),
            activation_class=nn.ELU,
        )
        linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for layer in linears:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, out_gain)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        flat = [x.float().reshape(*x.shape[: x.dim() - len(s)], -1) for x, s in zip(xs, self.shapes)]
        return self.mlp(torch.cat(flat, dim=-1))


class ActorNet(nn.Module):
    def __init__(self, body: StateNet, action_dim: int):
        super().__init__()
        self.body = body
        self.scale = AddStateIndependentNormalScale(action_dim, scale_lb=1e-4)

    def forward(self, *xs: torch.Tensor):
        return self.scale(self.body(*xs))


def make_teacher_models(env, network_cfg, device: torch.device):
    """Separate MLP actor (TanhNormal on [-1, 1], state-independent scale) and MLP critic."""
    spec = env.observation_spec
    ndim = len(env.batch_size)
    action_dim = env.action_spec.shape[-1]

    def shapes(keys):
        return [tuple(spec[k].shape[ndim:]) for k in keys]

    actor_keys = expand_in_keys(spec, list(network_cfg.actor_in_keys))
    critic_keys = expand_in_keys(spec, list(network_cfg.critic_in_keys))
    for key in actor_keys + critic_keys:
        if len(spec[key].shape[ndim:]) > 1:
            raise ValueError(f"The state teacher takes vector observations only; {key} has shape {spec[key].shape}.")
    actor = ProbabilisticActor(
        TensorDictModule(
            ActorNet(StateNet(shapes(actor_keys), network_cfg.hidden, action_dim, out_gain=0.01), action_dim),
            in_keys=actor_keys,
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -torch.ones(action_dim, device=device),
            "high": torch.ones(action_dim, device=device),
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    critic = ValueOperator(StateNet(shapes(critic_keys), network_cfg.hidden, 1, out_gain=1.0), in_keys=critic_keys)
    return actor.to(device), critic.to(device)


def load_teacher_weights(checkpoint_path, actor, critic, device: torch.device) -> None:
    """Warm-start freshly built actor/critic from an existing checkpoint (fine-tuning; fresh optimizer).

    Raises (via ``load_state_dict``'s strict mode) if the checkpoint's network shapes don't match this run's,
    naming the checkpoint so the mismatch is easy to place.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
    except RuntimeError as error:
        raise RuntimeError(f"init_checkpoint {checkpoint_path} does not match this run's network config: {error}") from error


def load_teacher_actor(checkpoint_path, env, device: torch.device):
    """Rebuild the actor from a teacher checkpoint (its own network config and weights)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor, _ = make_teacher_models(env, OmegaConf.create(checkpoint["config"]["network"]), device)
    actor.load_state_dict(checkpoint["actor"])
    return actor


def save_teacher_checkpoint(run_dir, name: str, actor, critic, optim, cfg, frames: int, iteration: int) -> Path:
    """``<run_dir>/checkpoints/<name>.pt`` plus the sidecar manifest ``<name>.json``."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pt"
    config = OmegaConf.to_container(cfg, resolve=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optim": optim.state_dict(),
            "frames": frames,
            "iteration": iteration,
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)
    write_json(
        path.with_suffix(".json"),
        {
            "checkpoint": path.name,
            "frames": frames,
            "iteration": iteration,
            "git_commit": git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": sha256_file(path),
            "config": config,
            "eval": None,
            "video": None,
        },
    )
    return path


HERE = Path(__file__).resolve().parent


def frames_of(path) -> float:
    """Sort key for ``ppo_teacher_<frames>.pt``; ``ppo_teacher_final`` sorts last."""
    stem = Path(path).stem
    return float("inf") if stem.endswith("_final") else int(stem.rsplit("_", 1)[-1])


class CheckpointWorker:
    """Evaluates and renders a run's checkpoints in a background process while training continues.

    One pass = ``evaluate.py run=<run_dir>`` then ``render.py all=<run_dir>``; both skip checkpoints whose
    manifest already has an ``eval`` / ``video``, so a pass started later catches up on everything missed while
    the previous pass was busy. Output is appended to ``<run_dir>/worker.log``.
    """

    GRACE_S = 30.0  # between SIGTERM and SIGKILL when a pass overruns its timeout

    def __init__(self, run_dir, worker_cfg):
        self.run_dir = Path(run_dir)
        self.cfg = worker_cfg
        self.proc: subprocess.Popen | None = None
        self._reported: set[tuple[str, str]] = set()
        self.failed_passes = 0
        self._pass_checked = True  # nothing to check until a pass has been launched

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def check_failure(self) -> None:
        """Report a finished pass's non-zero exit code loudly, once.

        Without this a worker crash (e.g. "No CUDA GPUs are available" in a subprocess) is silent: no
        exception, training just never gains an eval/video for any checkpoint again.
        """
        if self.proc is None or self.running() or self._pass_checked:
            return
        self._pass_checked = True
        if self.proc.returncode == 0:
            return
        self.failed_passes += 1
        tail = []
        try:
            tail = (self.run_dir / "worker.log").read_text().splitlines()[-15:]
        except OSError:
            pass
        print(
            "WORKER_FAILED "
            + json.dumps(
                {
                    "exit_code": self.proc.returncode,
                    "failed_passes_total": self.failed_passes,
                    "unfinished": self.unfinished(),
                    "log_tail": tail,
                }
            ),
            flush=True,
        )

    def launch(self) -> bool:
        """Start a pass unless one is running; returns whether a pass was started."""
        self.check_failure()
        if self.running():
            return False
        python = shlex.quote(sys.executable)
        run = shlex.quote(str(self.run_dir))
        commands = [f"{python} {shlex.quote(str(HERE / 'evaluate.py'))} run={run} num_envs={int(self.cfg.eval_num_envs)}"]
        if self.cfg.render:
            commands.append(
                f"{python} {shlex.quote(str(HERE / 'render.py'))} all={run} "
                f"seconds={float(self.cfg.render_seconds)} image={int(self.cfg.render_image)}"
            )
        log = open(self.run_dir / "worker.log", "a")  # noqa: SIM115 - handed to the child process
        # start_new_session: `bash -c "a; b"` does not exec, so the Isaac Sim children are only reachable
        # through the process group -- see _signal_group(), used when a pass overruns its timeout.
        self.proc = subprocess.Popen(
            ["bash", "-c", "; ".join(commands)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self._pass_checked = False
        log.close()
        return True

    def unfinished(self) -> list[str]:
        """``<checkpoint>(eval,video)`` for every checkpoint still missing a result, oldest first."""
        kinds = ("eval", "video") if self.cfg.render else ("eval",)
        out = []
        for path in sorted((self.run_dir / "checkpoints").glob("ppo_teacher_*.json"), key=frames_of):
            manifest = read_json(path)
            missing = [kind for kind in kinds if manifest.get(kind) is None]
            if missing:
                out.append(f"{manifest['checkpoint']}({','.join(missing)})")
        return out

    def _signal_group(self, sig) -> None:
        """Signal the whole pass (bash plus its Isaac Sim children); never raises."""
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except OSError:  # already gone
            pass

    def wait(self, timeout_s: float) -> None:
        """Wait for the running pass; terminate it if it overruns. Never raises.

        An overrunning pass must not be left behind: an orphaned Isaac Sim process outlives the trainer and
        collides with the next run (a stale ``/dev/shm`` carb semaphore alone can wedge ``launch_app()`` for
        tens of minutes). Whatever it did not finish is listed so it can be caught up by hand.
        """
        if not self.running():
            return
        try:
            self.proc.wait(timeout=timeout_s)
            self.check_failure()
            return
        except subprocess.TimeoutExpired:
            pass
        print(
            f"[worker] still running after {timeout_s:.0f} s; terminating it. Not finished: "
            f"{', '.join(self.unfinished()) or 'nothing'}. Catch up with: "
            f"evaluate.py run={self.run_dir} && render.py all={self.run_dir}",
            flush=True,
        )
        self._signal_group(signal.SIGTERM)
        try:
            self.proc.wait(timeout=self.GRACE_S)
        except subprocess.TimeoutExpired:
            self._signal_group(signal.SIGKILL)
            try:
                self.proc.wait(timeout=self.GRACE_S)
            except subprocess.TimeoutExpired:
                print("[worker] the pass survived SIGKILL; leaving it", flush=True)

    def new_results(self) -> list[tuple[str, dict]]:
        """(``"eval"`` | ``"video"``, checkpoint manifest) for results not reported before, oldest first."""
        results = []
        manifests = [read_json(p) for p in (self.run_dir / "checkpoints").glob("ppo_teacher_*.json")]
        for m in sorted(manifests, key=lambda m: m["frames"]):
            for kind in ("eval", "video"):
                if m.get(kind) is not None and (m["checkpoint"], kind) not in self._reported:
                    self._reported.add((m["checkpoint"], kind))
                    results.append((kind, m))
        return results
