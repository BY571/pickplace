# Stage 0 · privileged state teacher

PPO from robot state (`proprio`), bowl pose (`belt`) and food pose + grasp flag (`privileged`), cameras off,
with a dense simulator-only reward (`env.reward_set`, default and only set `simple_v3b`: reach, grasp, lift,
transport, food released in the bowl, return home, a success bonus and small motion penalties — see
`docs/environment.md`). Its checkpoints are the experts stage 1 records camera datasets from.

## Run

    python pipeline/0_state_teacher/train.py
    python pipeline/0_state_teacher/train.py +env.reward_weights.grasp=4.0   # override single weights

Stop gracefully (final checkpoint, evaluation and video are still written):

    docker exec <container> pkill -TERM -f "kit/python/bin/python3.*train.py"

Early stopping (`early_stop.success_rate`) reads the background worker's **evaluation** success rate, not the
training one: it stops once `early_stop.consecutive_evals` (4) evaluations in a row reach the threshold, so it
needs `worker.enabled=true` (a warning is logged and early stopping is disabled otherwise).

## config_v3c — the config behind the published teacher

`config_v3c.yaml` = `config.yaml` plus `env.success_requires_home: true` (success, and the episode end, needs
the food settled in the bowl **and** the TCP within `env.home_tolerance` = 0.05 m of its home position) and
randomized start poses (`robot_reset`: ±0.25 rad, ±0.1 rad/s). It warm-starts from an earlier checkpoint via
`init_checkpoint`; set `init_checkpoint=null` to train from scratch.

    ./scripts/launch_teacher.sh                                      # tests, then training, in one container
    python pipeline/0_state_teacher/train.py --config-name config_v3c [key=value ...]

With such a checkpoint, try the continuous demo without the scripted return: `demo.py ... home_between=false`.

**How the reward got here.** Earlier reward sets (`simple_v1`/`v2`, `staged_v1`) and their configs have been
removed now that the pipeline has settled on one; the history is in the git log and worth one summary. The
first teachers ended the episode the moment the food landed, so the arm had no trained behaviour afterwards
and froze in an extrapolated pose as soon as the line kept running. `return_home` plus
`success_requires_home` fixed that. The motion penalties were initially 10x larger; from scratch they
collapsed policy entropy before the lift was ever discovered, because a random policy's action-rate and
joint-velocity sums dominate the return long before any shaping term pays out — hence the gentle −0.01 /
−0.001 in `simple_v3b`.

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

## Continuous demo

A production line instead of episodes, as a video and an evaluation (1 env, cameras on, deterministic policy):

    python pipeline/0_state_teacher/demo.py checkpoint=<path.pt> [seconds=120] [bowls=3] [spacing=<m>] \
        [total_bowls=null] [home_between=true] [home_seconds=1.0] [out=<mp4>] [image=128]

- Scene (`env.demo`, demo-only; the training scene is unchanged): `bowls` pallets `spacing` apart circulate on the
  belt — a pallet reaching the belt end is written back upstream (the training `reset_belt`) with its bowl emptied.
  Default spacing is one reach-zone length (5 s of belt travel per bowl); the belt and table are extended upstream
  so the queued bowls fit. Arm, supply tray, reach zone, belt end and belt speed are the training ones.
  Spare food items wait under the table; after a placement or drop the next one goes into the tray (the training
  food reset). No termination and no time-out: the robot is never reset.
- The teacher runs unchanged. A bowl becomes the target once it is where a training episode can start one; the
  target is the most downstream open bowl in reach. Before every policy call the `belt`/`privileged` observations
  are recomputed by the env's own observation terms for the target bowl and the active food. With no target the
  arm returns to its default joint pose and waits; `home_between` (default true, the demo standard) also does
  that for `home_seconds` after every placement, miss or drop.
- Counts: `placed` (the success termination's condition), `missed` (a bowl left the zone empty), `dropped` (food
  off the table), `misplaced` (food settled in a bowl already filled or missed), placements per minute.
  `total_bowls=N` stops sending bowls after N and ends the video once they are all resolved.
- Writes `<out>.mp4` (scene camera + the two student views, counters overlaid), `<out>.json` (counters, event log,
  settings) and `<out>_frame.png`; prints `DEMO {json}` and `DEMO_DONE`.

## Scaling benchmark

    python pipeline/0_state_teacher/benchmark.py

See `docs/experiments/pipeline_stage0/benchmark/`.
Selected (2026-09-18): 32,768 envs, rollout 16, minibatch 32,768, `loss.shifted_gae=true` -> ~270 M frames/hour
(minibatch 131,072 was ~2% faster but gives 4x fewer gradient steps)
(4,096 envs x 24: ~160 M/h).
