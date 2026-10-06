# Robustness of the stage-2.1 students to degraded observations

The full sweep behind the one-paragraph summary in
[`pipeline/2_1_offline_rl/README.md`](../../../pipeline/2_1_offline_rl/README.md). 17 conditions per
algorithm, 256 episodes each, no retraining: the perturbation rewrites the observation after the
environment produced it and before the policy sees it.

**The question.** All three students sit near the teacher on clean observations (1000 episodes: BC 0.983,
IQL 0.947, teacher 0.982), yet the continuous production demo inverts them — IQL 24 placements / 0 misses
against BC's 12 / 11 with 9 drops. One plausible explanation is robustness: IQL's advantage-weighted policy
might generalise, while BC is a sharper fit to the exact training distribution. This experiment tests that
directly, by degrading what a real deployment degrades and watching who falls off fastest.

**What was measured.** `pickplace.offline.ObservationPerturbation` rewrites the observation *after* the env
produced it and *before* the policy sees it: the policy acts on the degraded copy while the env keeps
stepping the clean tensordict, so the simulation is bit-for-bit the same run and only the policy's input is
damaged. No retraining, the same `<algo>_final.pt` checkpoints the videos used (`bc_expert_proprio_s0`,
`iql_expert_proprio_s1`, `td3_bc_expert_proprio_s2`), and the usual evaluation protocol — deterministic
policy, `max_episode_length + 1` steps, each env's first finished episode scored — at **256 episodes per
condition**. `pipeline/2_1_offline_rl/robustness.py` runs all 17 conditions of one algorithm in a single
Isaac process (the env is built once; Isaac startup costs more than the whole sweep) and reseeds the global
RNG before each condition, so every condition scores the *same* set of initial states; the perturbation
draws its own noise from a private generator that never touches the global RNG. Severity 0 is an exact
no-op — the clean control returns the very same tensors, not "noise with sigma 0" — which
`tests/unit/test_offline.py` pins down along with the shapes, the dtypes, the [0, 255] image range and the
fact that proprio noise lands on exactly the sensor entries.

    python pipeline/2_1_offline_rl/robustness.py \
        checkpoint=/workspace/artifacts/students/bc_expert_proprio_s0/checkpoints/bc_final.pt num_envs=256

**The conditions** (severities in the units the sensor has, not abstract levels):

| Family | Severities | What it models |
|---|---|---|
| Image noise | additive Gaussian, sigma **2 / 5 / 10 / 20** of 255, clipped | sensor read noise (0.8%–7.8% of range) |
| Brightness | additive offset **−25 / +25** of 255 | an exposure or ambient-light shift of ±10% of range |
| Contrast | multiplicative gain **0.9 / 0.8** | a dirty lens or a stopped-down aperture |
| Defocus | separable Gaussian blur, sigma **0.8 px (3×3)** and **1.5 px (5×5)** on the 84 px frame | a ~1 px and ~2 px circle of confusion |
| Occlusion | one black square per camera covering **5%** and **15%** of the frame, **both cameras**, position drawn once per sub-env and then fixed | dirt on the lens, a fixture in the way |
| Proprio noise | joint positions and gripper opening **0.005 / 0.01 / 0.02 rad**, joint velocities **0.05 / 0.1 / 0.2 rad/s**, end-effector position **2 / 5 / 10 mm** | encoder noise |
| Combined | image sigma 5 + the mid proprio level | the "realistic sensor" |

Proprio noise is scaled per entry type, one severity index across all three: the velocity sigma is the
joint-position sigma differenced over a 100 ms filter window (5 control steps at the 50 Hz control rate),
i.e. 10× the position sigma in rad/s, and the end-effector sigma is the position error those joint-angle
errors produce over a ~0.5 m arm. `ee_quat` (a unit quaternion — perturbing it needs a rotation, not an
additive sigma) and `last_action` (the policy's own previous output, not a sensor reading) are left alone.

![Offline-RL students under degraded observations, 256 episodes per condition](robustness.png)

**Success rate per condition, 256 episodes each.** The binomial 95% CI is **±2.7%** near 0.95 and ±5.6%
near 0.70, so a two-algorithm gap has to clear roughly **6 points** near the ceiling to be a gap at all.

| Condition | BC | IQL | TD3+BC |
|---|---|---|---|
| **clean** | **0.973** | 0.934 | 0.930 |
| image noise sigma 2 | 0.977 | 0.945 | 0.941 |
| image noise sigma 5 | 0.961 | 0.945 | 0.895 |
| image noise sigma 10 | 0.941 | 0.891 | 0.941 |
| image noise sigma 20 | 0.691 | 0.688 | **0.914** |
| brightness −25 | 0.977 | 0.961 | 0.922 |
| brightness +25 | **0.973** | 0.895 | 0.879 |
| contrast gain 0.9 | 0.984 | 0.938 | 0.914 |
| contrast gain 0.8 | **0.953** | 0.910 | 0.840 |
| blur sigma 0.8 | 0.953 | 0.926 | 0.875 |
| blur sigma 1.5 | **0.938** | 0.781 | 0.750 |
| occlusion 5% | 0.234 | 0.246 | 0.160 |
| occlusion 15% | 0.016 | 0.043 | 0.035 |
| proprio 0.005 rad / 2 mm | 0.941 | 0.926 | 0.941 |
| proprio 0.01 rad / 5 mm | 0.965 | 0.891 | 0.938 |
| proprio 0.02 rad / 10 mm | **0.902** | 0.812 | 0.855 |
| combined (sigma 5 + proprio mid) | 0.973 | 0.902 | 0.941 |

Occlusion at 5% was the one condition that looked decisive but borderline, so it was re-run on its own at
**512 episodes**: BC **0.248**, IQL **0.262**, TD3+BC **0.154** (127 / 134 / 79 successes). BC and IQL are
indistinguishable there (z = 0.4); TD3+BC is genuinely worse than both (z = 3.8 vs BC, 4.3 vs IQL).

**Severity at which each algorithm first drops below 0.80:**

| Family | BC | IQL | TD3+BC |
|---|---|---|---|
| Image noise | sigma 20 | sigma 20 | **never** (0.914 at sigma 20) |
| Brightness (±25) | never | never | never |
| Contrast (gain ≥ 0.8) | never | never | never (0.840 at gain 0.8) |
| Defocus | **never** (0.938 at sigma 1.5) | sigma 1.5 | sigma 1.5 |
| Occlusion | 5% | 5% | 5% |
| Proprio noise | never | never (0.812 at 0.02 rad) | never |
| Combined | never | never | never |

**Do the differences beat the CI?** Yes, four of them, and they do not point the way the hypothesis did.

* **Image noise, sigma 20 — TD3+BC is the outlier, and it is the robust one.** BC 0.691 and IQL 0.688 both
  collapse; TD3+BC holds 0.914, which is 0.983 of its own clean rate. z = 6.6 against either. BC and IQL
  are identical here (z = 0.07).
* **Defocus, sigma 1.5 — BC is the robust one.** BC 0.938 against IQL 0.781 (z = 5.3) and TD3+BC 0.750
  (z = 6.1). The ranking is the exact reverse of the noise panel.
* **Photometric shifts — BC again.** At gain 0.8, BC 0.953 vs TD3+BC 0.840 (z = 4.3); at offset +25, BC
  0.973 vs IQL 0.895 (z = 3.6) and vs TD3+BC 0.879 (z = 4.1). All three are nevertheless above 0.84
  everywhere in this family: brightness and contrast are the perturbations this task cares least about.
* **Proprio noise — BC again, over IQL.** At 0.02 rad / 0.2 rad/s / 10 mm, BC 0.902 vs IQL 0.812
  (z = 2.9, p = 0.003); BC vs TD3+BC (0.855) is *not* significant (z = 1.6).
* **Occlusion — no algorithm survives it, and the differences are small.** 5% of both frames blacked out
  takes every student from ~0.95 to 0.15–0.26, and 15% to 0.02–0.04. This is by far the most damaging
  perturbation tested, and it is the one where the three are closest together in relative terms.
* **Everything else is inside the CI**, including the whole combined "realistic sensor" condition: BC
  0.973, IQL 0.902, TD3+BC 0.941 — each within noise of its own clean rate. A mid image noise plus a mid
  proprio noise costs none of the three anything measurable.

**Verdict: robustness does not explain the production-mode inversion.** The hypothesis was that IQL is the
robust policy and BC the brittle over-fit one. The measurement says the opposite or nothing: BC is the most
robust student in four of the six families (defocus, brightness, contrast, proprio noise), tied with IQL in
the fifth (image noise, where both collapse and *TD3+BC* is the robust one), and statistically tied with
IQL in the sixth (occlusion, where all three fail). On no condition in this sweep does IQL beat BC by more
than the confidence interval — at its single best showing, occlusion 5% at 512 episodes, IQL leads BC by
0.014 with z = 0.4. The clean control of this very sweep already reproduces the 1000-episode ranking
(BC 0.973 > IQL 0.934 ≈ TD3+BC 0.930), so sensor degradation cannot be what flips the two in production.
What the sweep *does* show is that the three objectives have genuinely different failure modes — TD3+BC's
Q-gradient-shaped policy is nearly immune to pixel noise and the weakest under blur, occlusion and contrast
loss, i.e. it leans on sharp local image structure that noise leaves intact and blur destroys, whereas the
cloning-shaped policies lean on the smoothed appearance that noise destroys and blur preserves — and that
the whole family is unusable with even 5% of each camera occluded, which is the one finding here with a
direct deployment consequence. The production inversion has to be explained by something the perturbation
sweep does not touch: the continuous demo is a *state*-distribution shift (repeated placements, homing
between items, whatever pose the cell is left in after the previous object) rather than a sensor shift, and
robustness to noise on the input is not robustness to being started somewhere the demonstrations never
visited. That is the next thing to measure.
