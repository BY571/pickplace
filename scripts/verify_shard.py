"""Verify offline-RL dataset shards written by ``pipeline/1_collect_data/collect.py`` before publishing them.

Checks, per shard: structure (keys/dtypes/shapes match the README's documented schema), value sanity
(no NaN/inf, actions in bounds, images not constant/blank), episode structure (trajectory-id runs,
episode-length plausibility, done/outcome exclusivity, the successor-stride invariant), the shard's own
recomputed statistics against its manifest, and the source checkpoint's continued existence/hash. With
more than one shard path, also prints a cross-tier comparison of success rates and failure mixes.

Simulator-free: loads the shard's memmap via ``pickplace.datasets.load_shard`` (no Isaac Sim, no GPU).
Reads pixels only for a spread sample of rows -- everything else is read in full (it's tiny per row).

Usage:
    python scripts/verify_shard.py <shard_dir> [<shard_dir> ...] [--image-samples 200]
        [--stride-samples 5000] [--tol 0.03]

Prints a compact per-shard report, then ``VERIFY_OK`` or ``VERIFY_FAILED <json list of failures>`` per shard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pickplace.artifacts import sha256_file
from pickplace.datasets import load_shard, shard_manifest
from pickplace.metrics import OUTCOME_TERMS
from pickplace.rewards import REWARD_TERMS

CAMERAS = ("overview_rgb", "wrist_rgb")
ACTION_BOUND = 1.0  # the teacher's TanhNormal support; collect.py clips executed actions (incl. noise) to this
LIGHT_KEYS = [
    "proprio",
    "belt",
    "privileged",
    "action",
    "loc",
    "scale",
    "step_count",
    ("collector", "traj_ids"),
    ("next", "reward"),
    ("next", "reward_terms"),
    ("next", "terminated"),
    ("next", "truncated"),
    ("next", "done"),
    ("next", "outcome"),
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested with small synthetic tensors; no shard I/O)
# ---------------------------------------------------------------------------


def all_finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(t.float()).all()) for t in tensors)


def out_of_bounds_count(t: torch.Tensor, bound: float, tol: float = 1e-4) -> int:
    return int((t.abs() > bound + tol).sum())


def all_float32_leaves(td) -> bool:
    """True if every leaf tensor of a (possibly nested) TensorDict is float32."""
    return all(v.dtype == torch.float32 for v in td.values(True, True))


def contiguous_runs_violations(traj_ids: torch.Tensor, num_envs: int) -> int:
    """Count of sub-envs whose trajectory-id stream is not non-decreasing over time.

    Storage is time-major: row t * num_envs + e is env e's step t, so column e of
    ``traj_ids.reshape(-1, num_envs)`` is env e's id stream over time. Every new episode gets a fresh,
    strictly larger id (the collector's global counter), so within one sub-env the stream is non-decreasing
    iff every id occupies exactly one contiguous run (never resumes an earlier one).
    """
    grid = traj_ids.reshape(-1, num_envs)
    if grid.shape[0] < 2:
        return 0
    non_decreasing = (grid[1:] >= grid[:-1]).all(dim=0)
    return int((~non_decreasing).sum())


def outcome_exclusive_violations(done: torch.Tensor, outcome_flags: dict[str, torch.Tensor]) -> int:
    """Count of done rows where the number of true outcome flags isn't exactly one."""
    done = done.reshape(-1).bool()
    stacked = torch.stack([flags.reshape(-1).bool() for flags in outcome_flags.values()], dim=0)
    n_true = stacked.sum(0)
    return int(((n_true != 1) & done).sum())


def successor_stride_violations(
    traj_ids: torch.Tensor, step_count: torch.Tensor, done: torch.Tensor, stride: int, idx: torch.Tensor
) -> tuple[int, int]:
    """Among sampled row indices ``idx``, how many break the successor-stride invariant.

    A row ``i`` that isn't done and has a successor in range must have row ``i + stride`` as the same
    trajectory's next step: same traj id, ``step_count`` incremented by exactly 1.
    """
    n = traj_ids.numel()
    idx = idx[(idx + stride) < n]
    idx = idx[~done.reshape(-1).bool()[idx]]
    if idx.numel() == 0:
        return 0, 0
    succ = idx + stride
    step_flat = step_count.reshape(-1)
    bad = (traj_ids[succ] != traj_ids[idx]) | (step_flat[succ] != step_flat[idx] + 1)
    return int(bad.sum()), int(idx.numel())


def episode_stats_from_rows(
    traj_ids: torch.Tensor, done: torch.Tensor, reward: torch.Tensor, outcomes: dict[str, torch.Tensor]
) -> dict:
    """Recompute per-completed-episode return/length/outcome mix straight from stored rows -- independent
    of collect.py's own online ``Stats`` (whose numbers land in the manifest)."""
    uniq, inverse, counts = torch.unique(traj_ids, return_inverse=True, return_counts=True)
    done_flat = done.reshape(-1).bool()
    completed = torch.zeros(len(uniq), dtype=torch.bool)
    completed.scatter_(0, inverse[done_flat], True)
    n = int(completed.sum())
    returns = torch.zeros(len(uniq), dtype=torch.float64).scatter_add_(0, inverse, reward.reshape(-1).double())
    result = {
        "episodes": n,
        "episode_return": float(returns[completed].mean()) if n else float("nan"),
        "episode_length": float(counts[completed].double().mean()) if n else float("nan"),
        "lengths": counts[completed],
    }
    for name, flags in outcomes.items():
        hit = flags.reshape(-1).bool() & done_flat
        per_traj = torch.zeros(len(uniq), dtype=torch.bool)
        per_traj.scatter_(0, inverse[hit], True)
        result[f"{name}_rate"] = float(per_traj[completed].double().mean()) if n else float("nan")
    return result


def within_tol(a: float, b: float, rel: float = 0.03, abs_: float = 1e-6) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b), 1e-9))


def percentiles(values: torch.Tensor, qs=(0.0, 0.5, 0.9, 0.99, 1.0)) -> dict[str, float]:
    values = values.double().sort().values
    n = values.numel()
    return {f"p{int(q * 100)}": float(values[min(int(q * (n - 1)), n - 1)]) for q in qs}


# ---------------------------------------------------------------------------
# Shard verification
# ---------------------------------------------------------------------------


def verify_shard(path: Path, image_samples: int, stride_samples: int, tol: float) -> dict:
    manifest = shard_manifest(path)
    frames = manifest["frames"]
    num_envs = manifest["num_envs"]
    stride = manifest["successor_stride"]
    image_size = manifest["image_size"]
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    buffer = load_shard(path, batch_size=64)
    check(len(buffer) == frames, f"len(buffer)={len(buffer)} != manifest frames={frames}")

    # --- 1. structure --------------------------------------------------------------------------------
    head = buffer[: min(64, len(buffer))]
    for cam in manifest.get("cameras", CAMERAS):
        img = head["pixels", cam]
        check(img.dtype == torch.uint8, f"pixels/{cam} dtype {img.dtype} != uint8")
        check(tuple(img.shape[1:]) == (image_size, image_size, 3), f"pixels/{cam} shape {tuple(img.shape[1:])} != ({image_size},{image_size},3)")
    check(head["action"].shape[-1] == head["loc"].shape[-1] == head["scale"].shape[-1], "action/loc/scale dims disagree")
    check(head["action"].dtype == torch.float32, "action dtype != float32")
    check(head["step_count"].dtype == torch.int64, "step_count dtype != int64")
    check(head["collector", "traj_ids"].dtype == torch.int64, "traj_ids dtype != int64")
    check(head["next", "reward"].dtype == torch.float32, "reward dtype != float32")
    check(head["next", "reward_terms"].shape[-1] == len(REWARD_TERMS), f"reward_terms width {head['next','reward_terms'].shape[-1]} != {len(REWARD_TERMS)}")
    check(head["next", "reward_terms"].dtype == torch.float32, "reward_terms dtype != float32")
    for k in ("terminated", "truncated", "done"):
        check(head["next", k].dtype == torch.bool, f"next/{k} dtype != bool")
    outcome_keys = set(head["next", "outcome"].keys())
    check(outcome_keys == set(OUTCOME_TERMS), f"outcome keys {sorted(outcome_keys)} != {sorted(OUTCOME_TERMS)}")
    for term in OUTCOME_TERMS:
        check(head["next", "outcome", term].dtype == torch.bool, f"outcome/{term} dtype != bool")
    check(all_float32_leaves(head["proprio"]), "proprio has a non-float32 leaf")
    check(all_float32_leaves(head["belt"]), "belt has a non-float32 leaf")
    check(all_float32_leaves(head["privileged"]), "privileged has a non-float32 leaf")

    # --- load the light (non-pixel) columns in full; they're a few hundred bytes/row -----------------
    light = buffer[:].select(*LIGHT_KEYS, strict=True)
    action, loc, scale = light["action"], light["loc"], light["scale"]
    step_count = light["step_count"]
    traj_ids = light["collector", "traj_ids"]
    reward, reward_terms = light["next", "reward"], light["next", "reward_terms"]
    terminated, truncated, done = light["next", "terminated"], light["next", "truncated"], light["next", "done"]
    outcomes = {t: light["next", "outcome", t] for t in OUTCOME_TERMS}
    obs_tensors = list(light["proprio"].values(True, True)) + list(light["belt"].values(True, True)) + list(light["privileged"].values(True, True))

    # --- 2. sanity of values ---------------------------------------------------------------------------
    check(all_finite(*obs_tensors), "NaN/inf in an observation (proprio/belt/privileged)")
    check(all_finite(action, loc, scale), "NaN/inf in action/loc/scale")
    check(all_finite(reward, reward_terms), "NaN/inf in reward or reward_terms")
    n_oob = out_of_bounds_count(action, ACTION_BOUND)
    check(n_oob == 0, f"{n_oob} action components outside [-{ACTION_BOUND}, {ACTION_BOUND}]")

    image_stats = {}
    idx = torch.linspace(0, frames - 1, min(image_samples, frames)).round().long().unique()
    sample = buffer[:].select(*[("pixels", cam) for cam in manifest.get("cameras", CAMERAS)])[idx]
    for cam in manifest.get("cameras", CAMERAS):
        imgs = sample["pixels", cam].float()
        zero_frac = float((imgs.reshape(imgs.shape[0], -1).amax(dim=1) == 0).float().mean())
        stats = {
            "min": float(imgs.min()),
            "max": float(imgs.max()),
            "mean": float(imgs.mean()),
            "std": float(imgs.std()),
            "zero_frame_fraction": zero_frac,
            "n_sampled": int(imgs.shape[0]),
        }
        image_stats[cam] = stats
        check(stats["std"] > 0.0, f"pixels/{cam} is constant across the sample (std=0)")
        check(stats["max"] > 0.0, f"pixels/{cam} is all-zero across the sample")

    # --- 3. episode structure -------------------------------------------------------------------------
    run_violations = contiguous_runs_violations(traj_ids, num_envs)
    check(run_violations == 0, f"{run_violations}/{num_envs} sub-envs have a non-contiguous traj_id stream")

    outcome_violations = outcome_exclusive_violations(done, outcomes)
    check(outcome_violations == 0, f"{outcome_violations} done rows without exactly one outcome flag set")

    gen = torch.Generator().manual_seed(0)
    sample_idx = torch.randint(0, frames, (stride_samples,), generator=gen)
    stride_bad, stride_checked = successor_stride_violations(traj_ids, step_count, done, stride, sample_idx)
    check(stride_bad == 0, f"{stride_bad}/{stride_checked} sampled rows break the successor-stride (step_count+1) invariant")

    recomputed = episode_stats_from_rows(traj_ids, done, reward, outcomes)
    ep_lengths = recomputed.pop("lengths")
    length_pctl = percentiles(ep_lengths.float()) if ep_lengths.numel() else {}

    # --- 4. statistics vs the manifest ------------------------------------------------------------------
    manifest_stats = manifest.get("stats", {})
    check(recomputed["episodes"] == manifest_stats.get("episodes"), f"recomputed episodes {recomputed['episodes']} != manifest {manifest_stats.get('episodes')}")
    for key in ("episode_return", "episode_length"):
        check(within_tol(recomputed[key], manifest_stats.get(key, float("nan")), tol), f"recomputed {key}={recomputed[key]:.3f} vs manifest {manifest_stats.get(key)}")
    for term in OUTCOME_TERMS:
        key = "success_rate" if term == "success" else f"{term}_rate"
        check(within_tol(recomputed[f"{term}_rate"], manifest_stats.get(key, float("nan")), tol), f"recomputed {key}={recomputed[f'{term}_rate']:.4f} vs manifest {manifest_stats.get(key)}")

    # --- 5. provenance -----------------------------------------------------------------------------------
    for field in ("checkpoint", "checkpoint_sha256", "checkpoint_eval", "env", "git_commit"):
        check(manifest.get(field) not in (None, {}, ""), f"manifest missing provenance field {field!r}")
    ckpt = Path(manifest["checkpoint"])
    ckpt_exists = ckpt.is_file()
    check(ckpt_exists, f"checkpoint no longer exists at {ckpt}")
    ckpt_sha_ok = None
    if ckpt_exists:
        actual_sha = sha256_file(ckpt)
        ckpt_sha_ok = actual_sha == manifest["checkpoint_sha256"]
        check(ckpt_sha_ok, f"checkpoint sha256 {actual_sha} != manifest {manifest['checkpoint_sha256']}")

    report = {
        "name": manifest.get("name", path.name),
        "path": str(path),
        "frames": frames,
        "recomputed": {**recomputed, "length_percentiles": length_pctl},
        "manifest_stats": manifest_stats,
        "image_stats": image_stats,
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt_exists,
        "checkpoint_sha_ok": ckpt_sha_ok,
        "reward_set": manifest.get("reward_set"),
        "noise_sigma": manifest.get("noise_sigma"),
        "failures": failures,
    }
    return report


def print_report(report: dict) -> None:
    r = report["recomputed"]
    print(f"=== {report['name']} ({report['path']}) ===")
    print(f"frames={report['frames']}  noise_sigma={report['noise_sigma']}  reward_set={report['reward_set']}")
    print(
        f"recomputed: episodes={r['episodes']} success_rate={r.get('success_rate', float('nan')):.4f} "
        f"episode_return={r['episode_return']:.2f} episode_length={r['episode_length']:.2f}"
    )
    if r.get("length_percentiles"):
        p = r["length_percentiles"]
        print(f"episode length percentiles: p0={p['p0']:.0f} p50={p['p50']:.0f} p90={p['p90']:.0f} p99={p['p99']:.0f} p100={p['p100']:.0f}")
    for term in OUTCOME_TERMS:
        if term == "success":
            continue
        print(f"  {term}_rate={r.get(f'{term}_rate', float('nan')):.4f}", end="")
    print()
    for cam, s in report["image_stats"].items():
        print(f"pixels/{cam}: min={s['min']:.0f} max={s['max']:.0f} mean={s['mean']:.1f} std={s['std']:.1f} zero_frame_fraction={s['zero_frame_fraction']:.4f} (n={s['n_sampled']})")
    print(f"checkpoint: {report['checkpoint']} exists={report['checkpoint_exists']} sha256_ok={report['checkpoint_sha_ok']}")
    if report["failures"]:
        print(f"VERIFY_FAILED {json.dumps(report['failures'])}")
    else:
        print("VERIFY_OK")
    print()


def cross_tier_report(reports: list[dict]) -> None:
    print("=== cross-tier ===")
    by_name = {r["name"]: r["recomputed"]["success_rate"] for r in reports}
    print("success rates: " + ", ".join(f"{name}={rate:.4f}" for name, rate in by_name.items()))
    order_note = []
    expert = next((r for r in reports if "expert" in r["name"]), None)
    medium = next((r for r in reports if "medium" in r["name"]), None)
    noisy = next((r for r in reports if "noisy" in r["name"]), None)
    if expert and noisy and medium:
        se, sn, sm = (x["recomputed"]["success_rate"] for x in (expert, noisy, medium))
        ok = se > sn > sm
        order_note.append(f"expert({se:.4f}) > noisy({sn:.4f}) > medium({sm:.4f}): {'yes' if ok else 'NO'}")
    for r in reports:
        rec = r["recomputed"]
        mix = {t: rec.get(f"{t}_rate", 0.0) for t in OUTCOME_TERMS if t != "success"}
        dominant = max(mix, key=mix.get) if mix else None
        print(f"{r['name']}: dominant failure = {dominant} ({mix.get(dominant, 0.0):.4f})" if dominant else f"{r['name']}: no failures recomputed")
    for note in order_note:
        print(note)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", help="One or more shard directories")
    parser.add_argument("--image-samples", type=int, default=200)
    parser.add_argument("--stride-samples", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=0.03)
    args = parser.parse_args()

    reports = []
    any_failed = False
    for shard in args.shards:
        report = verify_shard(Path(shard), args.image_samples, args.stride_samples, args.tol)
        print_report(report)
        reports.append(report)
        any_failed = any_failed or bool(report["failures"])

    if len(reports) > 1:
        cross_tier_report(reports)

    raise SystemExit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
