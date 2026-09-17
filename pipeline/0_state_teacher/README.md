# Stage 0 · privileged state teacher

PPO from robot state (`proprio`), bowl pose (`belt`) and food pose + grasp flag (`privileged`), cameras off,
with a dense simulator-only reward (`env.reward_set`, default `staged_v1`). Its checkpoints are the experts
stage 1 records camera datasets from.

## Run (DGX Spark)

    ./scripts/spark.sh --detach python pipeline/0_state_teacher/train.py
    ./scripts/spark.sh --detach python pipeline/0_state_teacher/train.py env.reward_set=staged_v1 env.reward_weights.grasp=4.0

Stop gracefully (final checkpoint, evaluation and video are still written):

    ssh spark 'docker exec <container> pkill -TERM -f "kit/python/bin/python3.*train.py"'

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
