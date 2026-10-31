# Guide 1: Real rollout data for training

Owner: Brooklyn. Executor: Codex. Source of truth for everything this guide does not restate: the Brier build plan, `AGENTS.md` and `SCOPE.md`. Where this guide is more specific than the plan, this guide wins, and Brooklyn authorizes every file, flag and dependency it names.

When you finish this guide, these exist and pass their checks:

1. Frozen scenario sets for PushT and ALOHA transfer cube (16 each).
2. A change library of real policy perturbations (80 changes per env, about 35% benign).
3. Real simulation episodes in GCS for every (change, scenario) pair: cheap tier (3 episodes) and expensive tier (20 episodes), plus a 50-episode baseline per scenario.
4. Keyframes for every failed expensive episode, in the public prefix of the bucket.
5. `ledger.sqlite` with changes, scenarios, runs and labels, built by `brier label`.
6. `compiled.jsonl` with the exact training prompts and labels, built by `brier compile`, split into train, calibration, test and OOD.
7. A dataset card and a label report that pass the quality gates in section 9.

Guide 2 (Glimmer training on Modal) starts from item 6.

---

## 0. How to run this guide

- Work through the task cards in section 6 in order, **one Codex session per card**, exactly like every other card in this repo. The runbook steps in section 7 are run after the cards they depend on have landed and passed `make check`.
- If a card's target file already exists from an earlier session (T03, T04, T05, T06, T07, T08, T09, T10, T11), do not rewrite it. Diff its behavior against the card's spec. Fix only the gaps, inside the card's file list and line budget, and list every gap you fixed in your final message.
- Stop and report instead of improvising when: a credential is missing, a quota blocks a step, a gate in section 9 fails twice, or a library API differs from what a card describes and the installed source does not make the right call obvious.
- Never fake data. No mock rollouts, no synthetic episodes, no copied numbers. Every row in GCS must come from a real `env.step` loop with the real pretrained policy.

### Card to plan mapping (use the plan id in commit messages)

| This guide | Plan task |
| --- | --- |
| D1 env and policy runner | T03 |
| D2 change library | T04 |
| D3 scenarios | T05 |
| D4 rollout and storage | T03 and T07 (shared) |
| D5 timing | T06 |
| D6 shard runner, D7 rollouts.yml | T07 |
| D8 GCP spot runner | T08 |
| Runbook R6 baselines | T09 |
| D9 aggregate, D10 brier label | T10 |
| D11 brier compile | T11 |

Commit as, for example, `T05: scenarios (guide 1, D3)`.

### Run Codex where the network works

Rollouts need GCS, the Hugging Face Hub and `gh`. Run Codex locally (Codex CLI on Brooklyn's machine) with network access, not in a sandbox that blocks outbound traffic. The heavy compute runs on GitHub Actions and GCP spot VMs; the local machine only launches and checks.

---

## 1. Credentials and accounts Brooklyn provides

Export these in the shell Codex uses. Never write them into the repo, a log or a commit.

| Name | What it is | Used by |
| --- | --- | --- |
| `GCP_PROJECT` | GCP project id, billing account upgraded from free trial | everything on GCP |
| `GCP_REGION` / `GCP_ZONE` | for example `us-central1` / `us-central1-a` | spot VMs |
| `BRIER_BUCKET` | bucket name only, for example `brier-data-<suffix>` (bucket names are global, so `brier-data` is likely taken) | all data paths |
| `BRIER_DATA` | `gs://$BRIER_BUCKET` | `AGENTS.md` data root |
| `GOOGLE_APPLICATION_CREDENTIALS` | path to the service account key JSON on the local machine | local `gcloud`, Python GCS client |
| GitHub secret `GCP_SA_KEY` | the same key JSON, stored as a repo secret | `rollouts.yml` |
| GitHub variable `BRIER_DATA` | `gs://$BRIER_BUCKET` | `rollouts.yml` |
| `HF_TOKEN` | Hugging Face token. Read scope is enough for rollouts; write scope for the dataset publish in step R12 | policy downloads, dataset publish |
| `HF_USER` | Hugging Face username or org for the dataset repo | step R12 |
| `gh` | authenticated GitHub CLI with admin on the `brier` repo | secrets, workflow dispatch |

Service account roles (Brooklyn creates the account `brier-rollouts@$GCP_PROJECT.iam.gserviceaccount.com`):

- `roles/storage.objectAdmin` on the bucket.
- `roles/compute.instanceAdmin.v1` and `roles/iam.serviceAccountUser` on the project (for spot VMs).

These env vars are allowed in addition to the `AGENTS.md` list: `GCP_PROJECT`, `GCP_REGION`, `GCP_ZONE`, `BRIER_BUCKET`, `GOOGLE_APPLICATION_CREDENTIALS`, `HF_TOKEN`, `HF_USER`, `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `SDL_VIDEODRIVER`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`.

### One-time GCP setup (runbook R0, run by Codex with Brooklyn's credentials)

```bash
gcloud config set project "$GCP_PROJECT"
gcloud services enable compute.googleapis.com storage.googleapis.com
gcloud storage buckets create "gs://$BRIER_BUCKET" --location="$GCP_REGION" --uniform-bucket-level-access
# public prefix for keyframes shown on the platform
gcloud storage buckets add-iam-policy-binding "gs://$BRIER_BUCKET" \
  --member=allUsers --role=roles/storage.objectViewer \
  --condition='expression=resource.name.startsWith("projects/_/buckets/'"$BRIER_BUCKET"'/objects/public/"),title=public-prefix'
# budget alert at $100 is set by Brooklyn in the console (plan rule), confirm it exists
gcloud compute regions describe "$GCP_REGION" --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -E "CPUS|PREEMPTIBLE|C2D" || true
gh secret set GCP_SA_KEY < "$GOOGLE_APPLICATION_CREDENTIALS"
gh variable set BRIER_DATA --body "gs://$BRIER_BUCKET"
gh secret set HF_TOKEN --body "$HF_TOKEN"
```

If the bucket-level IAM condition is refused (conditions need uniform access, which the create command sets), stop and report. Do not make the whole bucket public.

---

## 2. What the data is (the contract)

### 2.1 Environments and policies

| env id (in data) | gym id | LeRobot env config | Policy (HF Hub) | Horizon (expensive) |
| --- | --- | --- | --- | --- |
| `pusht` | `gym_pusht/PushT-v0` | `PushtEnv` (obs `pixels_agent_pos`) | `lerobot/diffusion_pusht` | the LeRobot config `episode_length` (300) |
| `aloha` | `gym_aloha/AlohaTransferCube-v0` | `AlohaEnv(task="AlohaTransferCube-v0")` (obs `pixels_agent_pos`) | `lerobot/act_aloha_sim_transfer_cube_human` | the LeRobot config `episode_length` (400) |

`AlohaInsertion` is out unless the timing benchmark (card D5) shows ALOHA transfer cube under 6 s per expensive episode **and** Brooklyn says go. Do not build it otherwise.

Facts from the installed environment sources that the cards rely on (verify them in the installed versions in card D1 and fail loudly if any is false):

- gym-pusht: `info["is_success"]` is `coverage > 0.95`, `info["coverage"]` is reported every step, the block is `env.unwrapped.block` (pymunk body, `.position`, `.angle`), the workspace is 512 x 512 with the goal near the center, and `reset(seed=...)` samples the start state.
- gym-aloha: `info["is_success"]` is `reward == 4`. The transfer cube reward is 1 (right gripper touches the box), 2 (lifted), 3 (left gripper touches, transfer attempted), 4 (transferred). Physics is `env.unwrapped._env.physics` (dm_control). The box start pose is set in `reset(seed)` through `gym_aloha.tasks.sim.BOX_POSE[0] = sample_box_pose(seed)`; the first 3 values are x, y, z.

### 2.2 Pinning

- Python 3.11 (plan). If the newest `lerobot` that supports these two policies requires a newer Python, use the lowest version it accepts and record the reason in `SCOPE.md`'s "written reason" line. That is the only edit to `SCOPE.md` this guide allows, and only Brooklyn commits it.
- Pin `lerobot[pusht,aloha]` to one exact version in `py/pyproject.toml` (or the repo's existing Python dependency file). Pick the newest release on PyPI on the day you start whose `lerobot-eval` runs both policies (card D1 proves it). Record the version in every episode row (`lerobot_version`).
- Pin the policy repos to a commit: resolve `huggingface_hub.model_info(repo).sha` once in card D1, write both shas to `py/brier_sim/pins.json`, and load policies with `revision=<sha>`.
- Torch: CPU wheels only for rollouts (`--index-url https://download.pytorch.org/whl/cpu`), at whatever version the pinned `lerobot` requires.

### 2.3 Scenarios (16 per env, frozen)

A scenario is a fixed seed set plus environment parameters. Scenario ids follow `AGENTS.md`: `pusht-s01` to `pusht-s16`, `aloha-s01` to `aloha-s16`.

| Dimension | PushT | ALOHA transfer cube |
| --- | --- | --- |
| Start pose bin (4) | quadrant of the block's start position relative to (256, 256): bin 1 = x < 256 and y < 256, bin 2 = x >= 256 and y < 256, bin 3 = x < 256 and y >= 256, bin 4 = x >= 256 and y >= 256 | quadrant of the box's start (x, y) relative to the midpoints of the `sample_box_pose` ranges (read the ranges from the installed `gym_aloha/utils.py`; at the time of writing x in [0.0, 0.2], y in [0.4, 0.6], so midpoints 0.1 and 0.5) |
| Lighting (2) | `normal` or `low`: low multiplies every image pixel by 0.55 before the policy sees it | `normal` or `low`: low multiplies `physics.model.light_diffuse` and `light_specular` by 0.45 (restore originals for `normal`) |
| Camera jitter (on/off) | on: a per-episode random image translation of up to 3 px (of 96) in x and y, constant through the episode | on: a per-episode random image translation of up to 12 px (of 640 x 480) in x and y, constant through the episode |
| Physics jitter (on/off) | not a scenario dimension | on: per-episode multipliers U(0.7, 1.3) on box friction and box mass (restore originals at every reset) |

Grids:

- PushT: pose (4) x lighting (2) x camera jitter (2) = 16.
- ALOHA: pose (4) x lighting (2) x physics jitter (2) = 16. Camera jitter is on when `(pose_bin + lighting_index) % 2 == 1`, so half the scenarios have it.

Seed selection (card D3): scan seeds 0, 1, 2, ... with `env.reset(seed=s)`, read the start pose, put the seed into its pose bin. Each scenario gets a disjoint list of 50 seeds from its bin, never shared with another scenario. Scenario k uses:

- baseline: its 50 seeds,
- expensive tier: the first 20 of those 50 (paired with the baseline, which lowers label noise),
- cheap tier: the first 3 of those 50.

All per-episode randomness (camera jitter offsets, physics multipliers, observation noise, frame drops, bias direction) comes from `numpy.random.default_rng(hash64(scenario_id, seed, tier_salt))` where `hash64` is the first 8 bytes of `sha256(f"{scenario_id}:{seed}:{salt}")` as an unsigned int. The same (scenario, seed) therefore sees the same world in the baseline and in every change run. That pairing is what makes a 15-point drop meaningful with 20 episodes.

### 2.4 Evaluation tiers

| Tier | Episodes | Horizon | Extra randomization |
| --- | --- | --- | --- |
| `cheap` | 3 | 50% of the expensive horizon (floor) | none |
| `expensive` (hardware stand-in) | 20 | 100% | observation noise (Gaussian, sigma 0.02 on images in [0, 1], sigma 0.002 on the state vector in its raw units) plus physics jitter (ALOHA: box friction and box mass x U(0.85, 1.15) on top of any scenario jitter; PushT: the block's pymunk mass x U(0.85, 1.15) and the env's PD gains `k_p`, `k_v` x U(0.9, 1.1); verify these attribute names in the installed source and fail loudly if absent) |
| `baseline` | 50 | as expensive | as expensive, with the unchanged base policy, once per scenario |

An episode is a success if `info["is_success"]` is true at any step within the tier's horizon. The cheap tier stops at its horizon even if the env would continue.

### 2.5 Change library (the "policy PRs")

A change wraps the base policy with one or two perturbation families. Each change has a change card: what a real engineer would write in a policy PR.

| Family | Parameters (sample from these) | How it is applied |
| --- | --- | --- |
| `obs_noise` | sigma in {0.01, 0.03, 0.05, 0.1} | Gaussian noise on every image (values in [0, 1], clipped) before the policy |
| `action_latency` | k in {1, 2, 4} steps | the env receives the action from k steps earlier. Until the buffer fills, send a hold action (the current measured agent position from the state vector) |
| `action_scale` | s in {0.8, 0.9, 1.1, 1.2} | `a' = p + s * (a - p)` where p is the current measured agent position. Both envs use absolute position actions, so never scale raw action values |
| `action_bias` | b in {1, 2, 4} units per 100 steps | `a' = a + (b / 100) * t * u` where u is a fixed random unit vector per change and t the step index. Units: pixels for PushT, radians for ALOHA joints, gripper dims excluded |
| `camera_shift` | PushT: shift in {2, 4, 6} px, rotation in {0, 3} deg. ALOHA: shift in {8, 16, 32} px, rotation in {0, 3} deg | fixed affine transform on every image, same for the whole change |
| `occlusion` | patch side as a fraction of the image short side in {0.1, 0.2, 0.3}; location fixed per change | mid-gray square pasted on every image |
| `brightness` | gain in {0.6, 0.8, 1.25, 1.5}, contrast in {0.8, 1.0, 1.2} | `x' = clip((x - 0.5) * contrast + 0.5) * gain`, clipped to [0, 1] |
| `frame_drop` | p in {0.1, 0.2, 0.4} | with probability p the policy receives the previous step's observation instead of the current one |
| `chunk_len` | PushT diffusion: `n_action_steps` in {4, 12, 16} (base is the checkpoint's value, read it). ACT: `n_action_steps` in {25, 50} and never above `chunk_size` | set `policy.config.n_action_steps` before `policy.reset()` |
| `weight_noise` | sigma in {0.01, 0.03, 0.05} | for every float parameter tensor w: `w += N(0, sigma * w.std())`, seeded by the change id, applied once when the policy loads |
| `benign` | one of: refactor, logging, metadata, config rename, sigma 0 | no-op. Behavior must be bit-identical to the base policy |

Rules:

- 80 changes per env. Ids: PushT `c001` to `c080`, ALOHA `c081` to `c160`.
- 28 benign changes per env (35%). The other 52 have one family (70% of them) or two families (30%).
- Every non-benign family appears at least 4 times per env.
- Magnitudes come from the pilot sweep (runbook R4): sample magnitudes so that roughly half of the non-benign changes regress on some scenarios but not all. Scenario-specific failure is the whole point.
- Change ids, families, parameters and descriptions are generated once with a fixed seed (`2026`) and written to `gs://$BRIER_BUCKET/manifests/changes_<env>.jsonl`. They never change after the first expensive rollout lands.
- `desc` is a one-line PR title a robotics engineer would write, drawn from a fixed list per family. Examples: `obs_noise`: "Switch to new camera driver", "Use cheaper USB camera"; `action_latency`: "Move inference to edge box", "Run policy server over the network"; `action_scale`: "Retune gripper gains"; `action_bias`: "Apply calibration offset from new fixture"; `camera_shift`: "Re-mount wrist camera"; `occlusion`: "Route cable across camera view"; `brightness`: "Swap lighting rig"; `frame_drop`: "Raise camera FPS under load"; `chunk_len`: "Shorten action chunks for responsiveness"; `weight_noise`: "Checkpoint from overnight run"; `benign`: "Bump logging, no policy change", "Refactor obs pipeline", "Rename config keys". At least 4 titles per family, picked by the generator's seeded RNG.

Change card JSON (stored in `changes_<env>.jsonl` and later in the ledger `card_json`):

```json
{"change_id": "c042", "env": "aloha", "families": [{"family": "action_latency", "params": {"k": 2}}],
 "desc": "Move inference to edge box", "seed": 1042}
```

`cause_family` for a change is the first family in `families` (the "dominant" one); for two-family changes the generator puts the larger expected effect first using the pilot table.

### 2.6 Episode record (Parquet, one row per episode)

| Column | Type | Notes |
| --- | --- | --- |
| `env` | string | `pusht` or `aloha` |
| `policy_repo`, `policy_revision` | string | from `pins.json` |
| `change_id` | string | `base` for baselines |
| `scenario_id` | string | |
| `tier` | string | `cheap`, `expensive`, `baseline`, `pilot` |
| `episode_idx` | int32 | 0-based within the key |
| `seed` | int64 | |
| `success` | bool | `is_success` seen within the horizon |
| `steps` | int32 | steps actually executed |
| `max_steps` | int32 | the tier's horizon |
| `reward_sum`, `max_reward`, `final_reward` | float64 | |
| `coverage_final`, `coverage_max` | float64, nullable | PushT only |
| `block_disp_px` | float64, nullable | PushT only: distance between block start and final position |
| `touched_right`, `touched_left`, `lifted` | bool, nullable | ALOHA only: reward reached 1, 3, 2 at any step |
| `table_contact_steps` | int32, nullable | ALOHA only: steps where any gripper finger geom touches the `table` geom |
| `action_flip_rate` | float64 | fraction of steps t >= 2 where `dot(a_t - a_{t-1}, a_{t-1} - a_{t-2}) < 0` and `norm(a_t - a_{t-1}) > 1e-6` |
| `terminal_state_json` | string | PushT: final agent and block pose; ALOHA: final box xyz and both gripper qpos |
| `keyframes_uri` | string, nullable | set only for failed expensive episodes |
| `wall_seconds` | float64 | |
| `lerobot_version`, `code_commit`, `host` | string | provenance |

One Parquet file per key, written only after the whole key finishes:

```
gs://$BRIER_BUCKET/episodes/env=<env>/tier=<tier>/change=<change_id>/<scenario_id>.parquet
gs://$BRIER_BUCKET/episodes/env=<env>/tier=baseline/change=base/<scenario_id>.parquet
```

A key exists if and only if its Parquet object exists. That is the idempotency rule: every runner skips existing keys, so any runner can be killed and restarted at any time.

### 2.7 Keyframes

For every failed expensive episode, save 8 frames from the policy camera (`top` for ALOHA, the rendered 96 x 96 image upscaled to 384 x 384 with nearest neighbour for PushT). Capture at steps `round(i * (max_steps - 1) / 7)` for i in 0..7, replacing the last one with the terminal frame if the episode ended early. Save the frame the env rendered, before any change or scenario perturbation is applied, so the explainer sees the world, not the corrupted input. Write JPEG quality 90:

```
gs://$BRIER_BUCKET/public/keyframes/<env>/<change_id>/<scenario_id>/<episode_idx>/f1.jpg ... f8.jpg
```

`keyframes_uri` holds the folder URI. Keep the 8 frames in memory during the episode and upload only if the episode failed.

### 2.8 Labels (computed only by `brier label`)

Pair (change, scenario):

- `regressed` = expensive success rate < baseline success rate - 0.15.
- `failure_mode` = `none` if not regressed, otherwise the most common episode failure mode among the failed expensive episodes, ties broken by the order in the rules below.
- `success_bucket` = expensive success rate in fifths: 1 for [0, 0.2), 2 for [0.2, 0.4), 3 for [0.4, 0.6), 4 for [0.6, 0.8), 5 for [0.8, 1.0].
- `cause_family` = the change's first family. Used only to score the explainer, never shown to any model.

Episode failure mode (first rule that matches):

ALOHA

1. success: `none`
2. `action_flip_rate > 0.35`: `oscillation`
3. `table_contact_steps >= 20`: `collision`
4. `max_reward >= 2 and final_reward < 2`, or `max_reward == 1`: `grasp_or_drop`
5. `max_reward == 3`: `overshoot_or_misalignment`
6. otherwise: `stall_or_timeout`

PushT

1. success: `none`
2. `action_flip_rate > 0.35`: `oscillation`
3. `coverage_max >= 0.5`: `overshoot_or_misalignment`
4. `block_disp_px < 20`: `stall_or_timeout`
5. otherwise: `overshoot_or_misalignment`

The 0.35 and 20 thresholds may be adjusted once, after the pilot (runbook R4), if the pilot shows that more than 30% of base-policy failures land in `oscillation` or `collision`. Record the final thresholds in the dataset card.

### 2.9 Compiled states (computed only by `brier compile`)

`compiled.jsonl`, one line per (change, scenario, question, variant):

```json
{"id": "c042:aloha-s07:regressed:ev", "pair_id": "c042:aloha-s07", "change_id": "c042", "scenario_id": "aloha-s07",
 "env": "aloha", "variant": "ev", "split": "train", "question": "regressed",
 "options": ["no", "yes"], "prompt": "<system> You are Brier ...\n<answer>", "label": "B"}
```

- `variant`: `ev` includes the cheap-tier evidence line; `noev` replaces it with `CHEAP_TIER: not run`. Every pair yields both.
- `question` and options (canonical order):
  - `regressed`: "Will the expensive evaluation regress by more than 15 points?" options `yes`, `no`.
  - `failure_mode`: "If the expensive evaluation regresses, what is the main failure mode?" options `none`, `stall or timeout`, `overshoot or misalignment`, `grasp or drop`, `collision`, `oscillation`.
  - `success_bucket`: "What will the expensive success rate be?" options `0-19%`, `20-39%`, `40-59%`, `60-79%`, `80-100%`.
- `options` lists the option texts in the order shown in the prompt. The letter for position i is `"ABCDEF"[i]`. `label` is the letter of the correct option.
- Shuffling: train split only. Permute the options with a seeded RNG (`hash64(id)`), rewrite the `<options>` line and the label. Calibration, test and OOD keep canonical order.
- Prompt template, byte for byte (the plan's format; `\n` is a newline, nothing follows `<answer>`):

```
<system> You are Brier, a calibrated decision model for robot policy CI. Answer with one option letter.
<state>
CHANGE: id=c042 family=action_latency k=2 desc="Move inference to edge box"
SCENARIO: env=aloha_transfer_cube pose_bin=3 lighting=low physics_jitter=on camera_jitter=off
BASELINE: success=0.86 (50 eps)
CHEAP_TIER: success=1/3 mean_reward=0.41 (3 eps, 50% horizon)
HISTORY: scenario_fragility=0.31; similar_changes(k=5): 3 regressed, 2 passed
</state>
<question> Will the expensive evaluation regress by more than 15 points?
<options> A) yes  B) no
<answer>
```

  For two-family changes the CHANGE line lists both: `family=action_latency k=2 + obs_noise sigma=0.03`. The SCENARIO line lists only the parameters the scenario has (PushT has no `physics_jitter`), in the order env, pose_bin, lighting, physics_jitter, camera_jitter. Numbers use 2 decimals.
- Splits are by change, decided once: a change whose families include `occlusion` or `chunk_len` goes to `ood`. Otherwise `n = u32(first 4 bytes of sha256(change_id))`, and `n % 10` in 0..5 is `train`, 6..7 is `calibration`, 8..9 is `test`. A change never appears in two splits.
- Leakage rule for HISTORY: `scenario_fragility` (regression rate of this scenario across changes) and `similar_changes` use **train-split labels only**, and a train change never counts itself. Similarity: same env, then highest family-set Jaccard, then smallest normalized magnitude distance, ties by change id. If fewer than 5 neighbours exist, print the count found (`similar_changes(k=3)`).

---

## 3. Sizing: how much data and how long

Target per env (plan): 80 changes x 16 scenarios = 1,280 pairs, 23 episodes per pair (3 cheap + 20 expensive) plus 16 x 50 baseline episodes. That is 30,240 episodes per env.

Minimum viable (plan): 40 changes x 12 scenarios per env.

Card D5 measures seconds per episode per env per tier on a GitHub Actions runner and on one GCP spot VM. Then:

```
cpu_hours(env) = (pairs * (3 * t_cheap + 20 * t_exp) + 16 * 50 * t_exp) / 3600
wall_hours = sum over envs of cpu_hours / total_parallel_processes
```

Parallel processes: GitHub Actions public repo, 20 concurrent jobs x 4 processes = 80. Each GCP spot VM adds `nproc` processes (start with 4 x `c2d-highcpu-32` = 128 if the quota allows; check `PREEMPTIBLE_CPUS` or `CPUS` quota in R0).

Decision rule (write the chosen numbers into `gs://$BRIER_BUCKET/manifests/sizing.json` and the dataset card):

| Projected wall time for all target data | Action |
| --- | --- |
| <= 10 h | Full target: 80 changes x 16 scenarios per env |
| 10 to 20 h | Keep 16 scenarios, cut changes per env to the largest multiple of 4 that fits (not below 48) |
| > 20 h | Minimum viable: 40 changes x 12 scenarios (drop scenarios s13 to s16), plus tell Brooklyn |

If PushT diffusion is the bottleneck (above 30 s per expensive episode on one CPU core), first lower the diffusion policy's `num_inference_steps` for **all** runs including baselines (record it in `pins.json`), re-time, and only then cut data. Do not mix settings between baselines and change runs.

---

## 4. Code layout (all inside existing directories)

```
py/brier_sim/
  pins.json            # policy repo shas, lerobot version, num_inference_steps overrides
  envs.py              # make env + policy exactly like lerobot-eval; scenario hooks
  scenarios.py         # seed selection, scenario params, writes scenarios_<env>.json
  changes.py           # change library generator + perturbation wrappers
  rollout.py           # run one key (change, scenario, tier) -> rows + keyframes
  storage.py           # GCS read/write, key existence, atomic upload
  run_shard.py         # CLI: python -m brier_sim.run_shard ... (work queue + process pool)
  pilot.py             # magnitude sweep for the change library
  aggregate.py         # Parquet -> episodes.jsonl, changes.jsonl, scenarios.jsonl for brier label
  gcp_spot.sh          # up | status | down for spot VMs
  tests/               # pytest for each public function
.github/workflows/rollouts.yml
crates/brier-core/src/label.rs, compile.rs (+ tests)
```

No new top-level files or directories. If `py/brier_sim` already uses different file names for the same jobs (from T03 to T09), keep the existing names and map this guide's specs onto them.

Dependencies allowed for this guide (Python): `lerobot[pusht,aloha]` (pinned), `gymnasium` (as pinned by lerobot), `numpy`, `pyarrow`, `google-cloud-storage`, `pillow`, `huggingface_hub`, `pytest`. Rust: no new crates beyond what `crates/brier-core` already uses plus `sha2` if it is not there yet (needed for the split hash).

---

## 5. Hard rules for this guide

1. Never write a key twice. Check existence before running and write the Parquet object only after all episodes of the key are done.
2. Never change scenarios, seeds, the change library, tiers or pins after the first expensive rollout lands. If something must change, delete the affected keys and say so in the final message.
3. Baselines and change runs use identical env construction, horizon, randomization and policy settings, apart from the change's own perturbation.
4. One thread per process: set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `torch.set_num_threads(1)` in every rollout process, and parallelize with processes.
5. Headless rendering: `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa` (install `libosmesa6-dev`), and `SDL_VIDEODRIVER=dummy` for pygame. If osmesa is too slow on a VM with a GPU you may use `egl`, but GitHub runners and CPU VMs use osmesa.
6. Fail loudly. A missing `info["is_success"]`, a policy that fails to load, an env attribute that does not exist: raise with the exact name. No silent fallbacks, no retries except the single GCS upload retry in `storage.py`.
7. Logging, not print, in library code. `run_shard.py` logs one line per finished key: key, successes/episodes, wall seconds.

---

## 6. Task cards

Each card follows the repo template. Line budgets exclude tests.

### D1: Env and policy runner exactly like lerobot-eval

```
TASK D1: brier_sim env + policy runner
Goal: Build env and policy for pusht and aloha the same way the pinned lerobot-eval does, and run N plain episodes.
Files (only these): py/brier_sim/envs.py, py/brier_sim/pins.json, py/brier_sim/tests/test_envs.py, py/pyproject.toml (dependency pins only)
Dependencies allowed: lerobot[pusht,aloha]==<pinned>, numpy, pytest
Line budget: 220
```

Spec:

- Read the installed `lerobot/scripts/lerobot_eval.py` (`rollout()` and `eval_main()`) and mirror its call sequence exactly for one environment: `make_env` with `n_envs=1` and a synchronous vector env, `make_policy`, `make_pre_post_processors` (and `make_env_pre_post_processors` if the installed version has it), then per step: `preprocess_observation`, any `add_envs_task`, the env preprocessor, the policy preprocessor, `policy.select_action` under `torch.inference_mode()`, the policy postprocessor, the env postprocessor, then `env.step`. Call `policy.reset()` at every episode start. If the installed `make_env` returns a nested dict of suites, take the single vector env for the task.
- Assert the vector env is a `SyncVectorEnv` with one sub-env, and expose `raw_env = vec_env.envs[0].unwrapped` for scenario hooks.
- Expose two hook points the later cards use: `obs_hook(obs_dict) -> obs_dict` applied after `preprocess_observation` and before the env and policy preprocessors (images are float tensors in [0, 1] there, keys like `observation.image`, `observation.images.top`, `observation.state`; assert the keys you touch exist), and `action_hook(action_np, raw_env) -> action_np` applied right before `env.step`.
- Policy loading takes `revision` from `pins.json`. Resolve the shas the first time with `huggingface_hub.model_info(repo).sha` and write them into `pins.json` in this card.
- `run_plain_episodes(env_name, n, seeds) -> list[dict]` returns success and steps per episode. Success is `info["is_success"]` at any step; for vector envs read it the way `lerobot_eval.py` does (the per-env info or `final_info`).

Acceptance:

```bash
pytest -q py/brier_sim/tests/test_envs.py
python -c "from brier_sim.envs import run_plain_episodes as r; print(r('pusht', 2, [0, 1])); print(r('aloha', 2, [0, 1]))"
lerobot-eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=1 --eval.n_episodes=2 --policy.device=cpu
```

The test runs 1 episode per env with a 20-step cap (smoke) and checks the hook points fire. The manual commands are run once; the `lerobot-eval` run must succeed, which proves the pinned version works with the checkpoint.

### D2: Change library and perturbation wrappers

```
TASK D2: change library
Goal: Generate the change library and apply each family as obs/action/policy perturbations.
Files (only these): py/brier_sim/changes.py, py/brier_sim/tests/test_changes.py
Dependencies allowed: numpy, pillow
Line budget: 300
```

Spec: section 2.5 in full. Public functions:

- `generate_changes(env, n=80, n_benign=28, seed=2026, magnitudes=None) -> list[dict]` producing change cards. `magnitudes` is the per-family allowed list from the pilot (default: the full grid in 2.5).
- `apply_change(policy, change) -> ChangeState` loads policy-level effects (`weight_noise`, `chunk_len`) and returns a small state object holding per-episode buffers (latency queue, previous obs for frame drop, step counter).
- `obs_hook(change_state, obs, rng)` and `action_hook(change_state, action, raw_env, rng)` implement the image and action families.
- `benign` must be bit-identical: a test runs 1 episode of 20 steps with and without a benign change on the same seed and compares actions exactly.

Acceptance: `pytest -q py/brier_sim/tests/test_changes.py` (one test per family: the wrapper changes what it should and nothing else, with a fixed RNG).

### D3: Scenarios and seed selection

```
TASK D3: scenarios
Goal: Pick seeds per pose bin, define the 16 scenarios per env, apply scenario + tier randomization hooks.
Files (only these): py/brier_sim/scenarios.py, py/brier_sim/tests/test_scenarios.py
Dependencies allowed: numpy
Line budget: 220
```

Spec: sections 2.3 and 2.4. Public functions:

- `select_seeds(env, per_scenario=50) -> dict[bin, list[int]]` scans seeds until every bin has `per_scenario x (scenarios per bin)` seeds, assigned disjointly.
- `build_scenarios(env) -> list[dict]` writes `scenarios_<env>.json` with `scenario_id`, `params` (pose_bin, lighting, camera_jitter, physics_jitter) and `seeds` (50).
- `scenario_reset_hook(raw_env, scenario, seed, tier, rng)` applies lighting and physics settings after `env.reset` and before the first policy step, restoring originals first. `scenario_obs_hook` applies image-side lighting and camera jitter (PushT lighting is image side).
- `tier_obs_noise_hook` and the expensive-tier physics jitter.
- The per-episode RNG follows the hash rule in 2.3.

Acceptance: `pytest -q py/brier_sim/tests/test_scenarios.py` checks: bins are disjoint, 50 seeds each, the reset actually lands in the bin (re-reset 5 random seeds and recompute), restoring physics returns the original arrays exactly.

### D4: Rollout of one key with keyframes and stats

```
TASK D4: rollout + storage
Goal: Run one (change, scenario, tier) key and produce Parquet rows and keyframes; GCS helpers.
Files (only these): py/brier_sim/rollout.py, py/brier_sim/storage.py, py/brier_sim/tests/test_rollout.py
Dependencies allowed: pyarrow, google-cloud-storage, pillow, numpy
Line budget: 300
```

Spec: sections 2.6 and 2.7.

- `run_key(env, change, scenario, tier, policy_cache) -> (rows, keyframes)` runs the tier's episodes for that key. Hook order per step: scenario obs hook, tier obs-noise hook, change obs hook, then lerobot preprocessors and the policy, then change action hook, then `env.step`. Reset order per episode: `env.reset(seed)`, scenario reset hook, tier physics jitter, `policy.reset()`, change state reset.
- Stats come from `info` and `raw_env` each step: coverage (PushT), reward, contacts (ALOHA: walk `physics.data.contact[:physics.data.ncon]` and map geom ids to names with `physics.model.id2name(id, "geom")`; count a step if a pair is (`table`, any geom whose name contains `gripper_finger`)).
- `storage.key_uri(...)`, `storage.exists(uri)`, `storage.write_parquet(rows, uri)` (write to a local temp file, upload with `if_generation_match=0` so two runners cannot both write the key, one retry on transient errors), `storage.upload_keyframes(...)`.
- `policy_cache` keeps one loaded policy per (env, change) in the process so `weight_noise` and `chunk_len` are applied once.

Acceptance: `pytest -q py/brier_sim/tests/test_rollout.py` runs a cheap-tier key with a 30-step horizon on PushT against a local `file://` style temp directory (`BRIER_DATA=/tmp/brier-test`), checks the row schema exactly, and checks that a second call skips the existing key.

### D5: Timing benchmark

```
TASK D5: timing benchmark
Goal: Measure seconds per episode per env per tier and write sizing.json.
Files (only these): py/brier_sim/run_shard.py (subcommand "timing" only in this card), py/brier_sim/tests/test_timing.py
Line budget: 120
```

Spec: `run_shard timing --env E` runs 3 expensive and 3 cheap episodes on scenario s01 with the base policy, single process, records median seconds and `nproc`, and writes `timing_<env>_<host>.json` to `gs://$BRIER_BUCKET/timing/`. `run_shard sizing` reads all timing files, applies section 3 (GitHub capacity 80 processes; GCP capacity from the VM count Brooklyn approves, default 4 VMs) and writes `manifests/sizing.json` with the chosen changes and scenarios per env plus the projection. The gcp_spot startup script runs `timing` instead of `run` when `phase=timing`.

Acceptance: `python -m brier_sim.run_shard timing --env pusht --out /tmp/t.json` prints the medians.

### D6: Shard runner (work queue)

```
TASK D6: run_shard
Goal: Run every missing key of a plan, sharded, with a process pool.
Files (only these): py/brier_sim/run_shard.py, py/brier_sim/tests/test_run_shard.py
Line budget: 200
```

Spec:

- `python -m brier_sim.run_shard run --env pusht --phase {baselines,pairs,pilot} --shard I --n-shards N --procs P [--max-hours H]`.
- The plan is derived deterministically from the manifests in GCS (`scenarios_<env>.json`, `changes_<env>.jsonl`, `sizing.json`): baselines are 16 keys per env; pairs are changes x scenarios x {cheap, expensive}.
- Sharding: group keys by change (so a process loads each changed policy once), order groups by change id, assign group g to shard `g % N`. Baselines: group by scenario.
- Skip keys whose Parquet exists. Before running a key, re-check existence (another runner may have finished it).
- Pairs phase refuses to start for an env until all 16 baseline keys of that env exist (labels need them).
- `--max-hours` stops taking new keys after H hours so jobs end cleanly before the 6-hour GitHub limit (use 5.5).
- Exit code 0 when the shard has no missing keys left, 3 when it stopped at `--max-hours` with work left.

Acceptance: `pytest -q py/brier_sim/tests/test_run_shard.py` (plan determinism, sharding covers every key exactly once across shards, skip logic).

### D7: rollouts.yml (GitHub Actions matrix)

```
TASK D7: rollouts.yml
Goal: Matrix workflow that runs shards on GitHub runners and writes to GCS.
Files (only these): .github/workflows/rollouts.yml
Line budget: 90
```

Spec:

- `on: workflow_dispatch` with inputs `env` (pusht or aloha), `phase` (timing, baselines, pairs, pilot), `n_shards` (default 40), `shard_offset` (default 0), `shard_count` (default 20). `phase=timing` runs `run_shard timing` and uploads the result instead of `run_shard run`.
- Matrix `shard: [offset .. offset + count - 1]` built with `fromJSON` from a small setup job. `max-parallel: 20`, `fail-fast: false`, `timeout-minutes: 350`.
- Steps: checkout; `sudo apt-get install -y libosmesa6-dev libgl1 libglib2.0-0 ffmpeg`; set up Python (3.11 or the pinned version); install `uv`; `uv pip install --system` the pinned deps with torch from the CPU index; cache `~/.cache/huggingface` keyed on `pins.json`; `google-github-actions/auth@v2` with `credentials_json: ${{ secrets.GCP_SA_KEY }}`; run `python -m brier_sim.run_shard run --env ... --phase ... --shard ${{ matrix.shard }} --n-shards ... --procs 4 --max-hours 5.5`.
- Env: `BRIER_DATA: ${{ vars.BRIER_DATA }}`, `HF_TOKEN: ${{ secrets.HF_TOKEN }}`, `MUJOCO_GL: osmesa`, `PYOPENGL_PLATFORM: osmesa`, `SDL_VIDEODRIVER: dummy`, `OMP_NUM_THREADS: 1`, `MKL_NUM_THREADS: 1`.
- Treat exit code 3 as success for the job (work left is picked up by the next dispatch).

Acceptance: `gh workflow run rollouts.yml -f env=pusht -f phase=pilot -f n_shards=2 -f shard_count=2` finishes green and writes pilot keys.

### D8: GCP spot runner

```
TASK D8: gcp_spot.sh
Goal: Start, watch and stop spot VMs that run shards of the same plan.
Files (only these): py/brier_sim/gcp_spot.sh
Line budget: 120
```

Spec:

- `gcp_spot.sh up <env> <phase> <first_shard> <n_vms> <n_shards> [machine]` creates `n_vms` instances `brier-roll-<env>-<shard>` with `--provisioning-model=SPOT --instance-termination-action=DELETE --machine-type ${machine:-c2d-highcpu-32} --image-family debian-12 --image-project debian-cloud --boot-disk-size 50GB --service-account brier-rollouts@$GCP_PROJECT.iam.gserviceaccount.com --scopes cloud-platform`, with metadata `env`, `phase`, `shard`, `n_shards`, `commit` (current `git rev-parse HEAD`) and a startup script.
- The startup script installs apt deps and `uv`, clones the public repo at `commit`, installs pinned deps with CPU torch, exports the env vars from rule 5, reads `HF_TOKEN` from Secret Manager **only if** Brooklyn created a secret named `hf-token` (otherwise runs without it: the policies are public), runs `run_shard run ... --procs $(nproc) --max-hours 20`, and powers the VM off when it exits.
- `gcp_spot.sh status` lists `brier-roll-*` instances and counts Parquet keys per env and phase in GCS.
- `gcp_spot.sh down` deletes all `brier-roll-*` instances.
- Preemption is fine: rerun `up` for the same shards; finished keys are skipped.

Acceptance: `bash -n py/brier_sim/gcp_spot.sh` and one real VM started with `up pusht pilot 1 1 2`, which writes at least one pilot key, then `down`.

### D9: Aggregate for the ledger

```
TASK D9: aggregate
Goal: Turn Parquet episodes and manifests into the three JSONL inputs brier label reads.
Files (only these): py/brier_sim/aggregate.py, py/brier_sim/tests/test_aggregate.py
Line budget: 150
```

Spec: download all episode Parquet for both envs (or read from GCS), validate that every planned key exists, and write to `data/`:

- `episodes.jsonl`: one line per episode with exactly the section 2.6 columns.
- `changes.jsonl`: change cards.
- `scenarios.jsonl`: scenario params plus `baseline_success` (successes / 50 from the baseline keys).

Fail with the list of missing keys if any planned key is absent. Acceptance: test on a tiny fixture of 2 changes x 2 scenarios.

### D10: brier label (Rust)

```
TASK D10: brier label
Goal: Load changes, scenarios and episodes into the ledger and compute labels.
Files (only these): crates/brier-core/src/label.rs, crates/brier-core/src/main.rs, crates/brier-core/tests/label.rs, crates/brier-core/fixtures/label/*
Dependencies allowed: uuid (features v4 and v5) only if the crate does not already have it
Line budget: 250
```

Spec: section 2.8. `brier label --ledger data/ledger.sqlite --changes data/changes.jsonl --scenarios data/scenarios.jsonl --episodes data/episodes.jsonl`:

- Inserts `changes`, `scenarios` (with `baseline_success`), one `runs` row per (change, scenario, tier) with `episodes`, `successes`, `mean_reward`, `uri`, `wall_seconds` (sum), and one `labels` row per pair.
- Idempotent: running twice gives the same database (use `INSERT OR REPLACE` keyed on the primary keys). Run ids are UUIDv5 of the key string `<change>:<scenario>:<tier>` so a re-run produces the same ids. `AGENTS.md` says UUIDv4; Brooklyn approves UUIDv5 for `runs` rows only, because idempotent labelling needs stable ids. Decisions keep UUIDv4.
- Prints a label summary: pairs, regressed rate overall, per family, per scenario, per env, benign false-regression rate, failure mode counts.

If T10 already exists, keep its CLI shape if it differs only in flag names, and make the rules match 2.8 exactly.

Acceptance: `cargo test label`, plus a fixture where every rule branch fires once.

### D11: brier compile (Rust)

```
TASK D11: brier compile
Goal: Emit compiled.jsonl with the exact prompt text, options, labels and splits.
Files (only these): crates/brier-core/src/compile.rs, crates/brier-core/src/main.rs, crates/brier-core/tests/compile.rs
Dependencies allowed: sha2 (only if not present)
Line budget: 280
```

Spec: section 2.9. `brier compile --ledger data/ledger.sqlite --out data/compiled.jsonl`.

- Golden test: for a fixture pair, the emitted prompt equals a checked-in expected string byte for byte, for each question and both variants.
- Test that no calibration, test or OOD change contributes to any HISTORY value.
- Test that train options are shuffled for some lines and that the label letter always points at the right text.
- Prints counts per split, question and variant.

If T11 already exists, add the missing fields (`pair_id`, `change_id`, `scenario_id`, `env`, `variant`), shuffling, OOD split and the leakage rule, without changing the prompt text it already emits unless it differs from 2.9.

---

## 7. Runbook (run in order after the cards land)

Each step lists its exit condition. Log what you ran and what came back.

**R0 GCP and GitHub setup.** Section 1. Exit: bucket exists, secrets set, quota printed.

**R1 Install and smoke locally.** `uv pip install -e py` (or the repo's way), then D1's acceptance commands. Exit: both envs run, `lerobot-eval` succeeds.

Before any `gh workflow run`: the cards' commits must be on the default branch of the public `brier` repo, because `workflow_dispatch` runs the workflow file and code from the branch you dispatch on (use `--ref main`). Spot VMs clone the commit passed in their metadata, so push first there too.

**R2 Scenarios.** `python -m brier_sim.scenarios --env pusht --write` and the same for `aloha` (D3 exposes this entry point). Upload `scenarios_<env>.json` to `gs://$BRIER_BUCKET/manifests/`. Exit: 16 scenarios per env, 50 seeds each.

**R3 Timing.** Run D5 timing locally first. Then on one GitHub runner per env: `gh workflow run rollouts.yml -f env=pusht -f phase=timing -f n_shards=1 -f shard_count=1` (and `env=aloha`). Then on one spot VM per env: `bash py/brier_sim/gcp_spot.sh up pusht timing 0 1 1`, wait for the timing file in `gs://$BRIER_BUCKET/timing/`, then `gcp_spot.sh down`. Then `python -m brier_sim.run_shard sizing` writes `manifests/sizing.json`. Exit: sizing decided per section 3, numbers shown to Brooklyn in the final message.

**R4 Pilot magnitude sweep.** For each family and each magnitude in 2.5, make a pilot change and run 10 episodes on 4 scenarios (one per pose bin, lighting normal, jitter off) with the expensive settings. That is at most about 30 magnitudes x 4 x 10 = 1,200 episodes per env; run it through `rollouts.yml` with `phase=pilot`. Then compute each magnitude's mean success drop versus the baseline seeds (run the 4 baselines first with the same 10 seeds). Keep magnitudes whose mean drop is between 0.05 and 0.6; drop magnitudes that never move success (< 0.05) or always destroy it (> 0.6 on all 4 scenarios). Write the kept grid to `manifests/magnitudes_<env>.json`. Also look at failure modes of failed pilot episodes and apply the one-time threshold rule in 2.8 if needed. Exit: every non-benign family keeps at least one magnitude; if a family keeps none, report it and drop that family from the library.

**R5 Change library.** `python -m brier_sim.changes --env pusht --n <from sizing> --magnitudes manifests/magnitudes_pusht.json --write` and the same for aloha. Upload `changes_<env>.jsonl`. Exit: counts match 2.5 rules (print them).

**R6 Baselines.** Dispatch `rollouts.yml` with `phase=baselines` for each env (16 keys, use `n_shards=16`, `shard_count=16`). Exit: 16 baseline keys per env exist. Check gate G1 (section 9) now.

**R7 Pairs at scale.** Dispatch `phase=pairs` for both envs. Split shards between GitHub and GCP so they do not collide much: for `n_shards=64`, GitHub takes shards 0 to 39 (two dispatches of 20), GCP spot VMs take 40 to 63. Example:

```bash
gh workflow run rollouts.yml -f env=pusht -f phase=pairs -f n_shards=64 -f shard_offset=0  -f shard_count=20
gh workflow run rollouts.yml -f env=pusht -f phase=pairs -f n_shards=64 -f shard_offset=20 -f shard_count=20
bash py/brier_sim/gcp_spot.sh up pusht pairs 40 24 64
```

Then the same for aloha once PushT GitHub jobs finish (the free plan runs 20 jobs at a time across the account). Re-dispatch any shard whose job exited 3 or failed. Watch progress with `gcp_spot.sh status` every 30 minutes. Exit: every planned key exists for both envs. Run `gcp_spot.sh down` at the end and confirm no `brier-roll-*` instance remains.

**R8 Aggregate.** `python -m brier_sim.aggregate --out data/`. Exit: no missing keys.

**R9 Label.** `cargo run -p brier-core -- ledger init --ledger data/ledger.sqlite` (if not already), then `brier label ...` from D10. Exit: summary printed; gates G2 to G4 pass.

**R10 Compile.** `brier compile --ledger data/ledger.sqlite --out data/compiled.jsonl`. Exit: counts printed; gate G5 passes.

**R11 Upload outputs.** Copy `data/ledger.sqlite`, `data/compiled.jsonl`, `data/episodes.jsonl`, `data/changes.jsonl`, `data/scenarios.jsonl` and the label summary to `gs://$BRIER_BUCKET/release/v1/`. Write `release/v1/SHA256SUMS`.

**R12 Dataset card and publish.** Write the card (section 8) and publish to `https://huggingface.co/datasets/$HF_USER/brier-robot-ci` (public, Apache 2.0, matching the repo) with `huggingface_hub.upload_folder`: the compiled JSONL, the three manifests, the label summary, `episodes.jsonl` and the card. Keyframes stay in GCS (public prefix); the card links to them. Exit: the dataset page opens logged out.

---

## 8. Dataset card (README.md of the HF dataset)

Sections, in this order, each filled with real numbers from the label summary:

1. What it is: real simulation rollouts of pretrained LeRobot policies under policy-change perturbations, labelled for regression prediction.
2. Environments, policies and pinned revisions; `lerobot` version.
3. Scenarios (the table from 2.3) and tiers (2.4); state plainly that the expensive tier stands in for hardware.
4. Change families and the magnitudes kept after the pilot.
5. Counts: changes, pairs, episodes per tier, per env; GPU and CPU hours spent; where it ran.
6. Labels: rules from 2.8 including the final thresholds; regression rate per env, family and split; failure-mode distribution; benign false-regression rate.
7. Splits and the OOD families; the leakage rule for HISTORY.
8. Files and schema (2.6, 2.9).
9. Known limits: simulation only; failure modes are rule-derived; 20 episodes per expensive key gives about plus or minus 0.1 on a success rate.

---

## 9. Quality gates

| Gate | Check | If it fails |
| --- | --- | --- |
| G1 baselines usable | Every scenario's baseline success >= 0.30 (a scenario below that cannot show a 15-point drop) | Replace the failing scenario's pose bin seeds with seeds from the same bin whose start pose is closer to the bin's inner corner (re-run D3 for that scenario only), re-run its baseline. If more than 4 scenarios per env fail, stop and report |
| G2 labels not too easy | Overall regressed rate between 10% and 45%; no family with 0% or 100% | Too few: raise magnitudes one step for families under 5%; too many: lower one step for families over 80%. This needs new changes, so do it only before R7 finishes, otherwise report |
| G3 labels not too noisy | Benign false-regression rate <= 12% | If above: report to Brooklyn with the per-scenario breakdown. The fix (more expensive episodes) changes the budget, so it is Brooklyn's call |
| G4 coverage | Every planned key exists; every pair has a label; every failed expensive episode has 8 keyframes | Re-run missing keys; never fill gaps by hand |
| G5 compile sanity | Each split has both labels for `regressed`; train has at least 400 pairs; OOD has at least 60 pairs; no prompt is longer than 3,000 characters; the golden test passes | Report the counts; do not change splits after training data has been read |

---

## 10. Troubleshooting

- `MUJOCO_GL` errors (`Could not load EGL` / `osmesa` not found): install `libosmesa6-dev` and set both `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa` before importing anything that imports mujoco.
- pygame display errors in PushT: `SDL_VIDEODRIVER=dummy`.
- Policy loads but success is near 0 for the base policy everywhere: the observation keys or normalization differ from `lerobot-eval`. Re-diff `envs.py` against the installed `lerobot_eval.py` step by step. Do not continue until `run_plain_episodes` matches `lerobot-eval` on the same seeds within plus or minus 0.15 success over 20 episodes.
- GCS 403: the service account lacks `objectAdmin` on the bucket, or the GitHub secret holds a different key.
- Spot VMs never start: check `PREEMPTIBLE_CPUS` quota in the region; try another zone in the same region; fall back to GitHub only and re-run sizing.
- GitHub jobs queue forever: the free plan allows 20 concurrent jobs per account across repos. Do not dispatch more than 20 shards at once.
- A key's Parquet is half-written: cannot happen with the temp-file plus `if_generation_match=0` upload. If you see a corrupt file, delete it and re-run that key.

---

## 11. Final message checklist (what Codex reports at the end)

- Versions: lerobot, torch, policy shas, Python.
- Sizing chosen and why; total episodes per env and tier; total CPU hours; GCP spend so far from the billing console.
- Gate results G1 to G5 with numbers.
- Paths: `gs://$BRIER_BUCKET/release/v1/` contents and the HF dataset URL.
- Every deviation from this guide and the reason.
