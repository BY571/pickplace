"""Spike: a velocity-driven prismatic pallet must carry a free bowl at the commanded speed."""

from _common import finish

from pickplace.app import launch_app

app = launch_app(headless=True)

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from pickplace.assets.scene_assets import make_bowl_cfg, make_pallet_cfg  # noqa: E402
from pickplace.assets.usd_builders import BowlGeometry, PalletGeometry  # noqa: E402

PALLET = PalletGeometry()
Z0 = 0.5


@configclass
class SpikeSceneCfg(InteractiveSceneCfg):
    pallet = make_pallet_cfg(PALLET, "{ENV_REGEX_NS}/Pallet", pos=(0.0, 0.0, Z0), damping=1e4)
    bowl = make_bowl_cfg(BowlGeometry(), "{ENV_REGEX_NS}/Bowl", pos=(0.0, 0.0, Z0 + PALLET.size[2] + 0.002), kinematic=False)


def main():
    dt, duration = 0.01, 2.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))
    scene = InteractiveScene(SpikeSceneCfg(num_envs=2, env_spacing=2.0))
    sim.reset()
    pallet, bowl = scene["pallet"], scene["bowl"]
    joint_ids, _ = pallet.find_joints("slider")
    env_ids = torch.arange(2, device=sim.device)

    def run(steps):
        for _ in range(steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)

    run(50)  # let the bowl settle on the plate
    plate_x0 = pallet.data.joint_pos.torch[:, joint_ids[0]].clone()
    bowl_p0 = bowl.data.root_pos_w.torch.clone()
    base_z0 = pallet.data.root_pos_w.torch[:, 2].clone()

    speeds = torch.tensor([[0.08], [0.12]], device=sim.device)
    pallet.set_joint_velocity_target_index(target=speeds, joint_ids=joint_ids, env_ids=env_ids)
    run(int(duration / dt))

    plate_dx = pallet.data.joint_pos.torch[:, joint_ids[0]] - plate_x0
    bowl_p1 = bowl.data.root_pos_w.torch
    payload = dict(
        speeds=speeds.squeeze(-1).tolist(),
        duration=duration,
        plate_dx=plate_dx.tolist(),
        bowl_dx=(bowl_p1[:, 0] - bowl_p0[:, 0]).tolist(),
        bowl_z_drop=(bowl_p0[:, 2] - bowl_p1[:, 2]).tolist(),
        base_dz=(pallet.data.root_pos_w.torch[:, 2] - base_z0).tolist(),
    )
    expected = (speeds.squeeze(-1) * duration)
    ok = bool(
        torch.all((plate_dx - expected).abs() / expected < 0.05)
        and torch.all(((bowl_p1[:, 0] - bowl_p0[:, 0]) - expected).abs() / expected < 0.10)
        and torch.all(bowl_p0[:, 2] - bowl_p1[:, 2] < 0.01)
    )
    finish(ok, **payload)


try:
    main()
except Exception as exc:  # report instead of hanging
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
