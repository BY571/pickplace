# Stage 0 · privileged state teacher

PPO from robot state (`proprio`), bowl pose (`belt`) and food pose + grasp flag (`privileged`), cameras off,
with a dense simulator-only reward (`env.reward_set`, default `simple_v2`: reach, grasp, lift, transport,
food released in the bowl and success, no penalties). Its checkpoints are the experts stage 1 records camera datasets from.

## Run (DGX Spark)

    ./scripts/spark.sh --detach python pipeline/0_state_teacher/train.py
    ./scripts/spark.sh --detach python pipeline/0_state_teacher/train.py env.reward_set=staged_v1 +env.reward_weights.grasp=4.0

Stop gracefully (final checkpoint, evaluation and video are still written):

    ssh spark 'docker exec <container> pkill -TERM -f "kit/python/bin/python3.*train.py"'

Early stopping (`early_stop.success_rate`) reads the background worker's **evaluation** success rate, not the
training one: it stops once `early_stop.consecutive_evals` (4) evaluations in a row reach the threshold, so it
needs `worker.enabled=true` (a warning is logged and early stopping is disabled otherwise).

## v3 (return home, smoother motion)

`config_v3.yaml` = `config.yaml` plus: `env.success_requires_home: true` — success (and the episode end) needs the
food settled in the bowl **and** the arm joints within `env.home_tolerance` (0.15 rad, L2 over the 7 arm joints) of
the default pose, so the teacher learns to finish the job instead of wandering after the release; reward set
`simple_v3` — `return_home` (new dense term, `1 - tanh(d / 1 rad)` while the food lies released in the bowl),
success 150, and small motion penalties (`action_rate` -0.1, `joint_vel` -0.01, `bowl_disturbance` -10) against
jerky actions and bumped bowls (reasoning in `food_robot/reward_sets/simple_v3.yaml`); `total_frames` 1 B with early
stopping (eval success now means placed AND home); W&B name `state_teacher_v3`. The v2 defaults are unchanged.

    ./scripts/launch_teacher_v3.sh     # one detached container: v3 tests, then (only if they pass) training
    python pipeline/0_state_teacher/train.py --config-name config_v3 [key=value ...]   # directly

With a v3 checkpoint, try the continuous demo without the scripted return: `demo.py ... home_between=false`.

## Outputs (`$FOOD_ROBOT_ARTIFACTS/teachers/<run>/`)

- `manifest.json` — git commit, resolved config, W&B URL, start/end, stop reason, checkpoint list.
- `checkpoints/ppo_teacher_<frames>.pt` every `checkpoint.interval_frames` (10 M) and `ppo_teacher_final.pt`.
- `checkpoints/<name>.json` — frames, iteration, git commit, sha256, config, and once the background worker has
  run: `eval` (deterministic, `worker.eval_num_envs` envs, each env's first episode: success rate, return,
  length, outcome rates, `terms/<term>` episodic sums) and `video`.
- `checkpoints/<name>.mp4` / `<name>_frame.png` — scene camera plus the two student camera views.
- `worker.log` — output of the background evaluation/render passes.

## By hand

    python pipeline/0_state_teacher/evaluate.py run=<run_dir>            # checkpoints without an eval
    python pipeline/0_state_teacher/render.py all=<run_dir>              # checkpoints without a video
    python pipeline/0_state_teacher/render.py checkpoint=<path.pt> force=true seconds=20

## Scaling benchmark

    ./scripts/spark.sh --detach python pipeline/0_state_teacher/benchmark.py

See `docs/experiments/pipeline_stage0/benchmark/`.
Selected (2026-09-18): 32,768 envs, rollout 16, minibatch 32,768, `loss.shifted_gae=true` -> ~270 M frames/hour
(minibatch 131,072 was ~2% faster but gives 4x fewer gradient steps)
(4,096 envs x 24: ~160 M/h).
