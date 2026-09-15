import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives.value.advantages import GAE

REPO = Path(__file__).resolve().parents[2]
N, H, K = 4, 64, 3


def _load(name, module_name=None):
    path = REPO / "sota-implementations" / "ppo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name or name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name or name] = module
    spec.loader.exec_module(module)
    return module


up = _load("utils_pixels")

NETWORK = OmegaConf.create(
    {
        "actor_in_keys": ["proprio", "pixels"],
        "critic_in_keys": ["proprio", "pixels"],
        "cnn_channels": [32, 64, 64],
        "cnn_kernels": [8, 4, 3],
        "cnn_strides": [4, 2, 1],
        "image_embed": 256,
        "proprio_embed": 128,
        "fusion": [512, 256],
    }
)


def fake_env():
    spec = Composite(
        {
            "proprio": Composite(
                {"joint_pos": Unbounded(shape=(N, 9)), "ee_pos": Unbounded(shape=(N, 3))}, shape=(N,)
            ),
            "pixels": Composite(
                {"wrist_rgb": Unbounded(shape=(N, H, H, 3 * K)), "overview_rgb": Unbounded(shape=(N, H, H, 3 * K))},
                shape=(N,),
            ),
        },
        shape=(N,),
    )
    return SimpleNamespace(
        observation_spec=spec, batch_size=torch.Size([N]), action_spec=Bounded(-1.0, 1.0, shape=(N, 7))
    )


def fake_obs(dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    pix = lambda: torch.randint(0, 256, (N, H, H, 3 * K), generator=g).to(dtype)  # noqa: E731
    return TensorDict(
        {
            "proprio": {"joint_pos": torch.randn(N, 9, generator=g), "ee_pos": torch.randn(N, 3, generator=g)},
            "pixels": {"wrist_rgb": pix(), "overview_rgb": pix()},
        },
        batch_size=[N],
    )


def test_models_output_shapes_and_bounded_actions():
    actor, critic = up.make_ppo_models(fake_env(), NETWORK, torch.device("cpu"))
    td = actor(fake_obs())
    assert td["action"].shape == (N, 7)
    assert td["action"].abs().max() <= 1.0
    assert critic(fake_obs())["state_value"].shape == (N, 1)


def test_actor_and_critic_do_not_share_parameters():
    actor, critic = up.make_ppo_models(fake_env(), NETWORK, torch.device("cpu"))
    actor_ids = {id(p) for p in actor.parameters()}
    assert not any(id(p) in actor_ids for p in critic.parameters())


def test_uint8_and_float_pixels_give_identical_outputs():
    actor, critic = up.make_ppo_models(fake_env(), NETWORK, torch.device("cpu"))
    as_float, as_uint8 = fake_obs(torch.float32), fake_obs(torch.float32)
    up.compress_pixels(as_uint8, [("pixels", "wrist_rgb"), ("pixels", "overview_rgb")])
    assert as_uint8["pixels", "wrist_rgb"].dtype == torch.uint8
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        torch.testing.assert_close(actor(as_float.clone())["loc"], actor(as_uint8.clone())["loc"])
    torch.testing.assert_close(critic(as_float)["state_value"], critic(as_uint8)["state_value"])


def test_load_actor_restores_weights_from_a_checkpoint(tmp_path):
    ppo_utils = _load("utils", module_name="ppo_utils_for_checkpoint_test")
    actor, critic = up.make_ppo_models(fake_env(), NETWORK, torch.device("cpu"))
    optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    cfg = OmegaConf.create({"network": NETWORK})
    path = tmp_path / "checkpoint.pt"
    ppo_utils.save_checkpoint(path, actor, critic, optim, cfg, frames=0)

    loaded_actor = up.load_actor(path, fake_env(), torch.device("cpu"))
    obs = fake_obs()
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        torch.testing.assert_close(actor(obs.clone())["loc"], loaded_actor(obs.clone())["loc"])


def test_compute_advantage_in_env_chunks_matches_single_pass():
    T = 5
    _, critic = up.make_ppo_models(fake_env(), NETWORK, torch.device("cpu"))
    obs = fake_obs().unsqueeze(1).expand(N, T).clone()
    next_obs = fake_obs().unsqueeze(1).expand(N, T).clone()
    done = torch.zeros(N, T, 1, dtype=torch.bool)
    done[:, -1] = True
    data = obs
    data["next"] = next_obs
    data["next", "reward"] = torch.randn(N, T, 1)
    data["next", "done"] = done
    data["next", "terminated"] = done.clone()

    adv_module = GAE(gamma=0.99, lmbda=0.95, value_network=critic, average_gae=False)
    full = up.compute_advantage(adv_module, data.clone(True), env_chunk=0)
    chunked = up.compute_advantage(adv_module, data.clone(True), env_chunk=3)

    torch.testing.assert_close(full["advantage"], chunked["advantage"])
    torch.testing.assert_close(full["value_target"], chunked["value_target"])


def test_image_keys_finds_only_images():
    env = fake_env()
    assert sorted(up.image_keys(env.observation_spec, env.batch_size)) == [
        ("pixels", "overview_rgb"),
        ("pixels", "wrist_rgb"),
    ]


def _rollout(done, success, reward, episode_reward, step_count):
    t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt).unsqueeze(-1)  # noqa: E731
    outcome = {term: torch.zeros_like(t(done, torch.bool)) for term in up.OUTCOME_TERMS}
    outcome["success"] = t(success, torch.bool)
    outcome["bowl_exited_zone"] = t(done, torch.bool) & ~t(success, torch.bool)
    return TensorDict(
        {
            "episode_reward": t(episode_reward),
            "step_count": t(step_count, torch.int64),
            "next": {"done": t(done, torch.bool), "reward": t(reward), "outcome": outcome},
        },
        batch_size=[2, 3],
    )


def test_episode_metrics_uses_completed_return_length_and_rates():
    data = _rollout(
        done=[[False, True, False], [False, False, True]],
        success=[[False, True, False], [False, False, False]],
        reward=[[1.0, 150.0, 1.0], [1.0, 1.0, -150.0]],
        episode_reward=[[5.0, 6.0, 0.0], [1.0, 2.0, 3.0]],
        step_count=[[3, 4, 0], [7, 8, 9]],
    )
    m = up.episode_metrics(data, "train")
    assert m["train/episode_return"] == pytest.approx(((6.0 + 150.0) + (3.0 - 150.0)) / 2)
    assert m["train/episode_length"] == pytest.approx((5 + 10) / 2)
    assert m["train/episodes"] == 2
    assert m["train/success_rate"] == 0.5
    assert m["train/bowl_exited_zone_rate"] == 0.5
    assert sum(v for k, v in m.items() if k.endswith("_rate")) == pytest.approx(1.0)


def test_episode_metrics_empty_without_finished_episodes():
    data = _rollout([[False] * 3] * 2, [[False] * 3] * 2, [[0.0] * 3] * 2, [[0.0] * 3] * 2, [[0] * 3] * 2)
    assert up.episode_metrics(data, "train") == {}


def test_first_episode_metrics_counts_one_episode_per_env():
    data = _rollout(
        done=[[True, False, True], [False, False, False]],
        success=[[False, False, True], [False, False, False]],
        reward=[[-150.0, 0.0, 150.0], [0.0, 0.0, 0.0]],
        episode_reward=[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        step_count=[[0, 0, 1], [0, 1, 2]],
    )
    m = up.first_episode_metrics(data, "eval")
    assert m["eval/episodes"] == 1
    assert m["eval/success_rate"] == 0.0  # the second (successful) episode of env 0 is not counted
    assert m["eval/finished_fraction"] == 0.5


def test_success_streak_needs_consecutive_iterations_at_threshold():
    streak = up.SuccessStreak(0.85, 3)
    assert not streak.update(0.9) and not streak.update(0.86)
    assert not streak.update(0.5)  # a miss resets the streak
    assert not streak.update(0.9)
    assert not streak.update(None)  # no episode finished this iteration: streak unchanged
    assert not streak.update(0.85)
    assert streak.update(0.95) and streak.count == 3


def test_policy_view_video_places_newest_frames_side_by_side():
    obs = fake_obs()
    keys = [("pixels", "overview_rgb"), ("pixels", "wrist_rgb")]
    batch = TensorDict({"pixels": obs["pixels"].unsqueeze(1).expand(N, 5)}, batch_size=[N, 5])
    _, frames = up.slim_eval_batch(batch, keys, video=True)
    video = up.policy_view_video([frames, frames], keys, upscale=2)
    assert video.shape == (10, 3, 2 * H, 2 * 2 * H)
    assert video.dtype.name == "uint8"
    newest_overview = obs["pixels", "overview_rgb"][0, ..., -3:].to(torch.uint8)
    assert (torch.from_numpy(video[0, :, ::2, : 2 * H : 2]).permute(1, 2, 0) == newest_overview).all()
