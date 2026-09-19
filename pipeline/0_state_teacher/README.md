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
food settled in the bowl **and** the TCP within `env.home_tolerance` (0.05 m) of its home position (the TCP at
the default joint pose; end-effector space because the IK policy cannot steer the redundant elbow), so the teacher
learns to finish the job instead of wandering after the release; reward set
`simple_v3` — `return_home` (new dense term, `1 - tanh(d / 0.2 m)` while the food lies released in the bowl),
success 150, and small motion penalties (`action_rate` -0.1, `joint_vel` -0.01, `bowl_disturbance` -10) against
jerky actions and bumped bowls (reasoning in `food_robot/reward_sets/simple_v3.yaml`); `total_frames` 600 M with early
stopping (eval success now means placed AND home); W&B name `state_teacher_v3`. The v2 defaults are unchanged.

    ./scripts/launch_teacher_v3.sh     # one detached container: v3 tests, then (only if they pass) training
    python pipeline/0_state_teacher/train.py --config-name config_v3 [key=value ...]   # directly

With a v3 checkpoint, try the continuous demo without the scripted return: `demo.py ... home_between=false`.

## v3b (fine-tune v2 with the v3 reward)

`config_v3b.yaml` fine-tunes the v2 checkpoint (`init_checkpoint`, fresh optimizer) instead of training v3 from
scratch, with reward set `simple_v3b` (`simple_v3`'s `action_rate`/`joint_vel` 10x smaller — full size collapsed
entropy before the lift was ever discovered) and a lower `optim.lr`. Launch: `CONFIG=config_v3b ./scripts/launch_teacher_v3.sh`.

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
        [total_bowls=null] [home_between=false] [home_seconds=1.0] [out=<mp4>] [image=128]

- Scene (`env.demo`, demo-only; the training scene is unchanged): `bowls` pallets `spacing` apart circulate on the
  belt — a pallet reaching the belt end is written back upstream (the training `reset_belt`) with its bowl emptied.
  Default spacing is one reach-zone length (5 s of belt travel per bowl); the belt and table are extended upstream
  so the queued bowls fit. Arm, supply tray, reach zone, belt end and belt speed are the training ones.
  Spare food items wait under the table; after a placement or drop the next one goes into the tray (the training
  food reset). No termination and no time-out: the robot is never reset.
- The teacher runs unchanged. A bowl becomes the target once it is where a training episode can start one; the
  target is the most downstream open bowl in reach. Before every policy call the `belt`/`privileged` observations
  are recomputed by the env's own observation terms for the target bowl and the active food. With no target the
  arm returns to its default joint pose and waits; `home_between=true` also does that for `home_seconds` after
  every placement, miss or drop.
- Counts: `placed` (the success termination's condition), `missed` (a bowl left the zone empty), `dropped` (food
  off the table), `misplaced` (food settled in a bowl already filled or missed), placements per minute.
  `total_bowls=N` stops sending bowls after N and ends the video once they are all resolved.
- Writes `<out>.mp4` (scene camera + the two student views, counters overlaid), `<out>.json` (counters, event log,
  settings) and `<out>_frame.png`; prints `DEMO {json}` and `DEMO_DONE`.

## Scaling benchmark

    ./scripts/spark.sh --detach python pipeline/0_state_teacher/benchmark.py

See `docs/experiments/pipeline_stage0/benchmark/`.
Selected (2026-09-18): 32,768 envs, rollout 16, minibatch 32,768, `loss.shifted_gae=true` -> ~270 M frames/hour
(minibatch 131,072 was ~2% faster but gives 4x fewer gradient steps)
(4,096 envs x 24: ~160 M/h).
