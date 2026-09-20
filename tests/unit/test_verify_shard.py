import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_shard", REPO / "scripts" / "verify_shard.py")
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)


def test_within_tol():
    assert vs.within_tol(1.0, 1.02, rel=0.03)
    assert not vs.within_tol(1.0, 1.5, rel=0.03)
    assert vs.within_tol(0.0, 1e-9)


def test_out_of_bounds_count():
    actions = torch.tensor([[0.5, -1.0], [1.2, -0.999], [1.0001, 0.0]])
    assert vs.out_of_bounds_count(actions, bound=1.0) == 1  # only the 1.2


def test_all_finite():
    ok = torch.tensor([1.0, 2.0, -3.0])
    bad = torch.tensor([1.0, float("nan"), 3.0])
    assert vs.all_finite(ok)
    assert not vs.all_finite(ok, bad)


def test_contiguous_runs_violations_flags_a_reused_id():
    # 2 envs, 4 timesteps, time-major flat layout (t * num_envs + e).
    # env 0: ids 1,1,2,2 (fine, non-decreasing); env 1: ids 5,6,5,7 (id 5 reused -> violation).
    grid = torch.tensor([[1, 5], [1, 6], [2, 5], [2, 7]])
    flat = grid.reshape(-1)
    assert vs.contiguous_runs_violations(flat, num_envs=2) == 1


def test_contiguous_runs_violations_clean_case():
    grid = torch.tensor([[1, 5], [1, 6], [2, 6], [3, 7]])
    flat = grid.reshape(-1)
    assert vs.contiguous_runs_violations(flat, num_envs=2) == 0


def test_outcome_exclusive_violations():
    done = torch.tensor([True, True, False, True])
    a = torch.tensor([True, False, False, False])
    b = torch.tensor([False, False, False, False])
    c = torch.tensor([False, True, True, False])
    # row0: done, exactly one (a) -> ok. row1: done, exactly one (c) -> ok.
    # row2: not done, irrelevant. row3: done, none set -> violation.
    assert vs.outcome_exclusive_violations(done, {"a": a, "b": b, "c": c}) == 1


def test_successor_stride_violations_detects_broken_step_count():
    # 2 envs, 3 timesteps. Row i's successor is row i+2.
    traj_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    step_count = torch.tensor([0, 0, 1, 1, 5, 2])
    done = torch.tensor([False] * 6)
    idx = torch.tensor([0, 1, 2, 3])
    bad, checked = vs.successor_stride_violations(traj_ids, step_count, done, stride=2, idx=idx)
    # pairs checked: (0,2) 0->1 ok, (1,3) 0->1 ok, (2,4) 1->5 BROKEN, (3,5) 1->2 ok.
    assert checked == 4
    assert bad == 1


def test_successor_stride_violations_catches_a_break():
    traj_ids = torch.tensor([0, 1, 0, 1])
    step_count = torch.tensor([0, 0, 5, 1])  # row0's successor (row2) should have step_count 1, but has 5
    done = torch.tensor([False, False, False, False])
    idx = torch.tensor([0, 1])
    bad, checked = vs.successor_stride_violations(traj_ids, step_count, done, stride=2, idx=idx)
    assert checked == 2
    assert bad == 1


def test_episode_stats_from_rows_recomputes_return_and_length():
    # Two trajectories: id 0 has 3 rows (reward 1,1,1, done on last), id 1 has 2 rows (reward 2,2, done on last).
    traj_ids = torch.tensor([0, 0, 0, 1, 1])
    done = torch.tensor([False, False, True, False, True])
    reward = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0])
    success = torch.tensor([False, False, True, False, False])
    stats = vs.episode_stats_from_rows(traj_ids, done, reward, {"success": success})
    assert stats["episodes"] == 2
    assert stats["episode_return"] == 3.5  # mean(3.0, 4.0)
    assert stats["episode_length"] == 2.5  # mean(3, 2)
    assert stats["success_rate"] == 0.5


def test_all_float32_leaves():
    from tensordict import TensorDict

    good = TensorDict({"a": torch.zeros(2, dtype=torch.float32), "b": {"c": torch.zeros(2, dtype=torch.float32)}}, batch_size=[2])
    bad = TensorDict({"a": torch.zeros(2, dtype=torch.float64)}, batch_size=[2])
    assert vs.all_float32_leaves(good)
    assert not vs.all_float32_leaves(bad)
