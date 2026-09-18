"""Continuous-operation demo and evaluation of a state teacher (1 env, cameras on, deterministic policy).

A production line instead of episodes: ``bowls`` pallets circulate on the belt (a pallet that reaches the end is
written back to the entry with its bowl emptied), the supply tray is refilled from a pool of spare food items, and
the robot is never reset. The trained teacher runs unchanged: before every policy call its bowl and food
observations are rewritten to show exactly one bowl (the target) and one food item (the active one), computed
by the env's own observation terms, as in training.

Counts: ``placed`` (the success termination's condition, in an open bowl), ``missed`` (a bowl left the reach
zone empty), ``dropped`` (the active food fell off the table or rode off the belt end outside a bowl),
``misplaced`` (food settled in a bowl that was already filled or already missed).

Usage:
    python pipeline/0_state_teacher/demo.py checkpoint=<path.pt>
    options: [seconds=120] [bowls=3] [spacing=<m>] [total_bowls=null] [home_between=false] [home_seconds=1.0]
             [food_pool=<bowls+2>] [out=<mp4>] [image=128] [seed=0]
Writes <out>.mp4, <out>.json (summary) and <out>_frame.png; prints ``DEMO {json}`` and ``DEMO_DONE``.
"""

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
cli = OmegaConf.from_dotlist(sys.argv[1:])
if not cli.get("checkpoint"):
    raise SystemExit("Pass checkpoint=<path.pt>.")
CHECKPOINT = Path(cli.checkpoint)
SECONDS = float(cli.get("seconds", 120.0))
BOWLS = int(cli.get("bowls", 3))
SPACING = None if cli.get("spacing") is None else float(cli.spacing)
TOTAL_BOWLS = None if cli.get("total_bowls") is None else int(cli.total_bowls)
HOME_BETWEEN = bool(cli.get("home_between", False))
HOME_SECONDS = float(cli.get("home_seconds", 1.0))
FOOD_POOL = int(cli.get("food_pool", BOWLS + 2))
IMAGE = int(cli.get("image", 128))
SEED = int(cli.get("seed", 0))
OUT = Path(cli.get("out") or CHECKPOINT.with_name(f"{CHECKPOINT.stem}_demo{'_home' if HOME_BETWEEN else ''}.mp4"))
TAIL_SECONDS = 1.0  # video kept running after the last bowl of a finite (total_bowls) run was resolved
HOME_GAIN = 0.2  # fraction of the remaining TCP pose error commanded per step while homing

from food_robot.system import memory_used_gb  # noqa: E402

baseline_gb = memory_used_gb()

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from food_robot.carousel import PALLET_GAP, DemoTally, park_position, pick_target  # noqa: E402
from food_robot.envs import mdp  # noqa: E402
from food_robot.torchrl_env import make_env  # noqa: E402

# Loaded by file path: with cameras enabled, Isaac Sim's bundled cv2/utils shadows `import utils`.
_spec = importlib.util.spec_from_file_location("teacher_utils", os.path.join(HERE, "utils.py"))
tu = importlib.util.module_from_spec(_spec)
sys.modules["teacher_utils"] = tu
_spec.loader.exec_module(tu)

SCENE_HW = (720, 1280)
PANEL = 640


# --- video layout (as render.py) ---------------------------------------------------------------------------
def to_uint8(img: torch.Tensor) -> np.ndarray:
    return img[..., -3:].float().clamp(0, 255).to(torch.uint8).cpu().numpy()


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
    x0, y0, w, h = img.shape[1] - 520, 44, 510, 14 + 32 * len(lines)
    box = img[y0:y0 + h, x0:x0 + w]
    img[y0:y0 + h, x0:x0 + w] = (0.35 * box).astype(np.uint8)
    for k, text in enumerate(lines):
        cv2.putText(img, text, (x0 + 12, y0 + 34 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return img


def compose(scene, overview, wrist, caption: str, lines: list[str]) -> np.ndarray:
    def panel(img, name):
        big = cv2.resize(to_uint8(img), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        return label(big, f"student camera: {name} ({IMAGE}x{IMAGE})")

    top = overlay(label(to_uint8(scene), f"scene camera - {caption}"), lines)
    return np.concatenate([top, np.concatenate([panel(overview, "overview_rgb"), panel(wrist, "wrist_rgb")], axis=1)])


# --- the line ------------------------------------------------------------------------------------------------
class Line:
    """Carousel of pallets, the food pool, the policy's observation rewrite and the bookkeeping."""

    def __init__(self, u, total_bowls: int | None):
        cfg = self.cfg = u.cfg
        if not cfg.demo:
            raise RuntimeError("the env was not built in demo mode")
        self.u, self.device = u, u.device
        self.env_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self.origin = u.scene.env_origins[0]
        k = cfg.demo_bowls
        self.pallets = ["pallet"] + [f"pallet_{i}" for i in range(1, k)]
        self.bowls = ["bowl"] + [f"bowl_{i}" for i in range(1, k)]
        self.foods = ["food"] + [f"food_{j}" for j in range(1, cfg.demo_food_pool + 1)]
        self.total_bowls = total_bowls
        self.tally = DemoTally()

        terms = cfg.demo_termination_params  # the original (now disabled) termination params
        self.exit_params = dict(terms["bowl_exited_zone"])
        self.zone_end_x = float(self.exit_params["zone_end_x"])
        self.drop_params = dict(terms["food_off_table"])
        self.settle = dict(terms["success"])
        self.settle_steps = int(self.settle.pop("settle_steps"))
        self.settle["robot_cfg"] = SceneEntityCfg("robot", joint_names=cfg.arm.gripper_joint_names)
        self.settle["robot_cfg"].resolve(u.scene)

        # one reset_belt term per pallet+bowl pair: re-seating a pallet uses exactly the training reset
        self.belt_params = {k_: v for k_, v in cfg.events.reset_belt.params.items() if not k_.endswith("_cfg")}
        self.belt_params["speed_noise"] = 0.0  # every pallet at the nominal speed keeps the spacing exact
        self.belt_terms = [
            mdp.reset_belt(
                EventTermCfg(func=mdp.reset_belt, mode="reset",
                             params={**self.belt_params, "pallet_cfg": SceneEntityCfg(p), "bowl_cfg": SceneEntityCfg(b)}),
                u,
            )
            for p, b in zip(self.pallets, self.bowls)
        ]
        self.food_params = dict(cfg.events.reset_food.params)
        self.food_params.pop("food_cfg")
        self.ground_z = cfg.scene.plane.init_state.pos[2]
        self.pitch = cfg.belt.pallet.size[0] + PALLET_GAP

        # the policy's observation terms whose input is the (single) bowl or food: rewritten per step
        om = u.observation_manager
        self.rewrites = []
        for group in ("belt", "privileged"):
            if group not in om.active_terms:
                continue
            for name, term in zip(om.active_terms[group], om._group_obs_term_cfgs[group]):
                self.rewrites.append((group, name, term))

        self.status = ["idle"] * k  # idle (not counted) | open | filled | missed | retired
        self.food_in: list[list[str]] = [[] for _ in range(k)]
        self.settle_count = [0] * k
        self.active: str | None = None
        self.target: int | None = None

    # -- geometry --
    def pallet_q(self, i: int) -> float:
        term = self.belt_terms[i]
        return float(self.u.scene[self.pallets[i]].data.joint_pos.torch[0, term.joint_ids[0]])

    def bowl_x(self, i: int) -> float:
        return float(mdp.asset_pos_cell(self.u, SceneEntityCfg(self.bowls[i]))[0, 0])

    # -- actions on the scene --
    def seat(self, i: int, q: float) -> None:
        """Pallet ``i`` to joint position ``q`` at belt speed, its bowl re-seated empty (as ``reset_belt``)."""
        self.belt_terms[i](self.u, self.env_ids, **{**self.belt_params, "pallet_start_range": (q, q),
                                                    "pallet_cfg": SceneEntityCfg(self.pallets[i]),
                                                    "bowl_cfg": SceneEntityCfg(self.bowls[i])})
        self.settle_count[i] = 0

    def stop(self, i: int) -> None:
        term, zero = self.belt_terms[i], torch.zeros(1, 1, device=self.device)
        pallet = self.u.scene[self.pallets[i]]
        pallet.write_joint_velocity_to_sim_index(velocity=zero, joint_ids=term.joint_ids, env_ids=self.env_ids)
        pallet.set_joint_velocity_target_index(target=zero, joint_ids=term.joint_ids, env_ids=self.env_ids)

    def park(self, food: str) -> None:
        j = self.foods.index(food)
        pose = torch.zeros(1, 7, device=self.device)
        pose[0, :3] = self.origin + torch.tensor(park_position(j, self.ground_z, self.cfg.food.item_radius),
                                                 device=self.device)
        pose[0, 6] = 1.0
        asset = self.u.scene[food]
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=self.env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.device), env_ids=self.env_ids)

    def spawn_food(self) -> None:
        """Next spare food item into the tray, at a random spot (the tray reset's randomization)."""
        busy = {f for held in self.food_in for f in held}
        free = [f for f in self.foods if f not in busy]
        if not free:
            return  # every item is riding in a bowl; retry next step
        self.active = free[0]
        mdp.reset_food_in_bowl(self.u, self.env_ids, **self.food_params, food_cfg=SceneEntityCfg(self.active))
        self.settle_count = [0] * len(self.bowls)

    # -- episode-free start --
    def start(self) -> None:
        for i in range(len(self.bowls)):
            self.seat(i, i * self.cfg.demo_spacing)
        for food in self.foods[1:]:
            self.park(food)
        self.active = "food"  # placed in the tray by the env's own reset
        for i in range(len(self.bowls)):
            if self.bowl_x(i) <= self.zone_end_x:
                self.status[i] = "open"
                self.tally.bowl_entered()

    # -- observation rewrite --
    def rewrite_obs(self, td) -> None:
        xs = [self.bowl_x(i) for i in range(len(self.bowls))]
        self.target = pick_target(xs, [s == "open" for s in self.status], self.zone_end_x)
        # no open bowl in reach: point the policy at the entry, where the next bowl will appear
        bowl = self.bowls[self.target] if self.target is not None else None
        for group, name, term in self.rewrites:
            params = dict(term.params)
            if group == "belt":
                if bowl is None:
                    value = mdp.asset_pos_cell(self.u, SceneEntityCfg(params["asset_cfg"].name)).clone()
                    value[:, 0] = self.belt_params["entry_x"]
                    value[:, 1] = self.belt_params["belt_y"]
                    td.set((group, name), value.to(td.device))
                    continue
                params["asset_cfg"] = SceneEntityCfg(bowl)
            else:
                if self.active is None:
                    continue  # keep showing the last item until a spare one is free
                if "asset_cfg" in params:
                    params["asset_cfg"] = SceneEntityCfg(self.active)
                elif "food_cfg" in inspect.signature(term.func).parameters:
                    params["food_cfg"] = SceneEntityCfg(self.active)
            td.set((group, name), term.func(self.u, **params).to(td.device).reshape(td.get((group, name)).shape))

    # -- bookkeeping after a step; returns True on a placement, miss or drop --
    def update(self) -> bool:
        event, u = False, self.u
        if self.active is not None:
            food_cfg = SceneEntityCfg(self.active)
            for i, bowl in enumerate(self.bowls):
                inside = bool(mdp.settled_in_bowl_mask(u, **self.settle, food_cfg=food_cfg,
                                                       bowl_cfg=SceneEntityCfg(bowl))[0])
                self.settle_count[i] = self.settle_count[i] + 1 if inside else 0
                if self.settle_count[i] >= self.settle_steps:
                    if self.status[i] == "open":
                        self.status[i] = "filled"
                        self.tally.placed += 1
                    else:
                        self.tally.misplaced += 1
                    self.food_in[i].append(self.active)
                    self.active, event = None, True
                    break
        if self.active is not None and bool(mdp.food_off_table(u, **self.drop_params,
                                                                 food_cfg=SceneEntityCfg(self.active))[0]):
            self.tally.dropped += 1
            self.park(self.active)
            self.active, event = None, True
        for i, bowl in enumerate(self.bowls):
            if self.status[i] == "open" and bool(mdp.bowl_exited_zone(u, **self.exit_params,
                                                                        bowl_cfg=SceneEntityCfg(bowl))[0]):
                self.status[i] = "missed"
                self.tally.missed += 1
                event = True
        event |= self.carousel()
        if self.active is None:
            self.spawn_food()
        return event

    def exhausted(self) -> bool:
        return self.total_bowls is not None and self.tally.bowls_seen >= self.total_bowls

    def carousel(self) -> bool:
        event, recycle_q = False, self.cfg.demo_recycle_q
        retired = [self.pallet_q(i) for i, s in enumerate(self.status) if s == "retired"]
        # finite run: pallets queue up at the belt end instead of colliding with the ones already stopped there
        limit = min([recycle_q] + [q - self.pitch for q in retired]) if self.exhausted() else recycle_q
        for i in range(len(self.bowls)):
            q = self.pallet_q(i)
            if self.status[i] == "retired" or q < limit:
                continue
            if self.status[i] == "open":  # knocked off the pallet and never crossed the zone end
                self.status[i] = "missed"
                self.tally.missed += 1
                event = True
            for food in self.food_in[i]:
                self.park(food)
            self.food_in[i] = []
            if self.active is not None and self._on_pallet(self.active, q):
                self.tally.dropped += 1  # rode off the belt end next to the bowl
                self.park(self.active)
                self.active, event = None, True
            if self.exhausted():
                self.stop(i)
                self.status[i] = "retired"
                limit = min(limit, q - self.pitch)
            else:
                self.seat(i, q - recycle_q)
                self.status[i] = "open"
                self.tally.bowl_entered()
        return event

    def _on_pallet(self, food: str, q: float) -> bool:
        pos = mdp.asset_pos_cell(self.u, SceneEntityCfg(food))[0]
        belt = self.cfg.belt
        return (abs(float(pos[0]) - (self.belt_params["entry_x"] + q)) < 0.5 * belt.pallet.size[0]
                and abs(float(pos[1]) - belt.belt_y) < belt.belt_half_width and float(pos[2]) > 0.0)

    def finished(self) -> bool:
        return self.exhausted() and self.tally.bowls_seen == self.tally.placed + self.tally.missed


class Homing:
    """Scripted return of the TCP to its pose after the reset (the arm's default joint pose), gripper unchanged."""

    def __init__(self, u):
        self.u, self.cfg = u, u.cfg
        self.home_pos, self.home_quat = self.tcp()
        self.steps_left = 0

    def tcp(self):
        data = self.u.scene["ee_frame"].data
        return data.target_pos_w.torch[:, 0, :].clone(), data.target_quat_w.torch[:, 0, :].clone()

    def action(self, gripper: torch.Tensor) -> torch.Tensor:
        if self.cfg.action_mode == "joint_pos":  # use_default_offset: a zero action holds the default pose
            arm = torch.zeros(1, len(self.cfg.arm.arm_joint_names), device=self.u.device)
        else:
            pos, quat = self.tcp()
            error = torch.cat([self.home_pos - pos, math_utils.quat_box_minus(self.home_quat, quat)], dim=-1)
            arm = (HOME_GAIN * error / self.cfg.arm.ik_action_scale).clamp(-1.0, 1.0)
        return torch.cat([arm, gripper.reshape(1, 1).to(arm.device)], dim=-1)


# --- run ------------------------------------------------------------------------------------------------------
env_cfg = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["config"]["env"]
env_cfg = {**env_cfg, "num_envs": 1, "cameras": True, "render_camera": True, "seed": SEED,
           "render_image_size": list(SCENE_HW), "image_size": [IMAGE, IMAGE],
           "demo": {"bowls": BOWLS, "food_pool": FOOD_POOL, "spacing": SPACING}}
env = make_env(env_cfg)
u = env.base_env._env.unwrapped
dt = u.step_dt
actor = tu.load_teacher_actor(CHECKPOINT, env, env.device)

td = env.reset()
line = Line(u, TOTAL_BOWLS)
line.start()
homing = Homing(u)
home_steps = max(1, round(HOME_SECONDS / dt))
max_steps = int(round(SECONDS / dt))
caption = f"{CHECKPOINT.parent.parent.name}/{CHECKPOINT.stem} - continuous demo"
OUT.parent.mkdir(parents=True, exist_ok=True)
still = OUT.with_name(f"{OUT.stem}_frame.png")
still_step, first_place_step = max_steps // 2, None
gripper = torch.ones(1, device=env.device)  # open
episode_ends = robot_resets = 0
n = tail = 0
last_length = int(u.episode_length_buf[0])

with imageio.get_writer(OUT, fps=round(1.0 / dt), codec="libx264", quality=8, macro_block_size=1) as writer, \
        torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    while n < max_steps:
        line.rewrite_obs(td)
        if homing.steps_left > 0 or line.active is None:
            td["action"] = homing.action(gripper).to(td.device)
            homing.steps_left = max(0, homing.steps_left - 1)
        else:
            td = actor(td)
        gripper = td["action"][..., -1].clone()
        stepped, td = env.step_and_maybe_reset(td)
        n += 1
        episode_ends += int(stepped["next", "done"].any())
        length = int(u.episode_length_buf[0])
        robot_resets += int(length <= last_length)
        last_length = length
        placed_before = line.tally.placed
        if line.update() and HOME_BETWEEN:
            homing.steps_left = home_steps
        if line.tally.placed > placed_before and first_place_step is None:
            first_place_step = n
            still_step = n + round(0.5 / dt)
        t = n * dt
        s = line.tally.summary(t)
        lines = [
            f"placed {s['placed']}   missed {s['missed']}   dropped {s['dropped']}",
            f"{s['placements_per_min']:.2f} placements/min   t = {t:5.1f} s",
            f"bowls {s['bowls_seen']}   {'homing' if homing.steps_left > 0 else 'policy'}"
            f"   target {'-' if line.target is None else line.bowls[line.target]}",
        ]
        frame = compose(u.scene["render_cam"].data.output["rgb"][0], td["pixels", "overview_rgb"][0],
                        td["pixels", "wrist_rgb"][0], caption, lines)
        writer.append_data(frame)
        if n == still_step:
            imageio.imwrite(still, frame)
        if line.finished():
            tail += 1
            if tail >= round(TAIL_SECONDS / dt):
                break

if not still.exists():  # the run ended before the chosen still
    imageio.imwrite(still, frame)
summary = {
    **line.tally.summary(n * dt),
    "checkpoint": str(CHECKPOINT),
    "video": str(OUT),
    "frame": str(still),
    "steps": n,
    "robot_resets": robot_resets,
    "episode_ends": episode_ends,
    "episode_length_steps": last_length,
    "first_placement_s": None if first_place_step is None else round(first_place_step * dt, 3),
    "settings": {
        "bowls": BOWLS, "spacing": round(float(u.cfg.demo_spacing), 4), "recycle_q": round(float(u.cfg.demo_recycle_q), 4),
        "food_pool": FOOD_POOL, "total_bowls": TOTAL_BOWLS, "home_between": HOME_BETWEEN,
        "home_seconds": HOME_SECONDS, "belt_speed": float(u.cfg.belt.speed), "image": IMAGE, "seed": SEED,
        "requested_seconds": SECONDS, "step_dt": dt,
    },
}
OUT.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print("DEMO " + json.dumps(summary), flush=True)
print("MEMORY " + json.dumps({"used_gb": round(memory_used_gb(), 2), "delta_gb": round(memory_used_gb() - baseline_gb, 2)}), flush=True)
print("DEMO_DONE", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
