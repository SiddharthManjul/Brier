# Brier: Build Plan (Robot CI first, Software CI second)

Oct 29, 2026 · @Brooklyn

## Summary

Brier ships by Nov 4 as open-source Robot CI. When a robot policy changes, Brier predicts which evaluation scenarios will regress, runs only the evaluations it needs and guarantees a regression-detection rate. Software CI is the second adapter, built only if Robot CI passes its Nov 2 gate.

- **Pitch:** Brier is a world model plus a calibrated decision model for expensive engineering loops. It predicts what an evaluation will say before you pay for it.
- **Meta models:** Muse Glimmer (fine-tuned as the decision model), Muse Spark (reasoning, escalation, failure explanation), SAM 3.1 (perception on failure frames).
- **Track:** Push the Frontier.
- **Headline number to prove:** regressions caught per expensive-evaluation hour versus running everything, at a stated detection guarantee.
- **Deadline:** Nov 4, 9:00 PM ET, which is Nov 5, 7:30 AM IST (US daylight saving ends Nov 1). Internal submit target: Nov 5, 5:00 AM IST.

### What ships (definition of done)

1. `brier` engine: ledger, state compiler, cascade, conformal policy, CLI.
2. Robot CI adapter on LeRobot PushT and ALOHA, in simulation.
3. Fine-tuned Glimmer decision model with a calibration report.
4. Failure explainer: SAM 3.1 + Muse Spark.
5. A GitHub Action that gates policy PRs in a public demo repo.
6. The Brier platform: a Next.js app on Vercel showing every evaluated change, its failures and the model results.
7. README and a 3-minute video.

### Non-goals (never build these during the hackathon)

- GitHub App, OAuth, user accounts, billing, multi-tenancy
- Platform pages or features beyond the four listed in Platform (no auth, settings or admin)
- Real hardware, ZK proofs, chip CI
- Kubernetes, Terraform, Docker images beyond what Modal or Actions need
- Any feature not listed in `SCOPE.md`

## Budget and compute

The plan spends about $70 of the $480 available and leaves every credit pool with a reserve. Estimates are mine; measure on Day 1 and update this table.

| Pool | Used for | Planned | Ceiling |
| --- | --- | --- | --- |
| Meta API $150 | Muse Spark Contributor tier ($0.10 / $0.20 per M tokens) for explanations and escalations; SAM 3.1 on failure keyframes | \~$20 | $50 |
| Modal $30 | Glimmer zero-shot eval, QLoRA fine-tune on A100-80GB, eval batch serving, live demo endpoint (L4, scale to zero) | \~$22 | $30 |
| GCP $300 | Spot CPU rollouts, GCS bucket (shared data store, public images for the platform); GPU only if quota is approved | \~$30 | $120 |
| Vercel Hobby + Neon free tier | Platform hosting and Postgres | $0 | $0 |
| GitHub Actions | Rollout matrix jobs and the demo CI, free on public repos | $0 | $0 |
| Kaggle / Lightning free tiers | 3B student distillation, spare experiments | $0 | $0 |

### Rules

- Use Muse Spark **Contributor** variants only. The data is public simulation output, so letting Meta train on it costs nothing.
- Set spend alerts today: Meta billing page at $40, Modal workspace budget at $25, GCP budget alert at $100.
- GCP free trial blocks GPUs and caps concurrent vCPUs. Upgrade to a paid billing account today (remaining credits carry over) and file a GPU quota request for 1x A100 and 2x L4 in one region. Plan as if the quota is denied.
- Every GPU job writes checkpoints to GCS or a Modal Volume every 200 steps. No job runs without a resume path.

### If more credits arrive

| Credit | Upgrade it unlocks |
| --- | --- |
| AMD $100 (MI300X 192GB) | BF16 LoRA fine-tune instead of 4-bit QLoRA, plus ablation runs |
| Baseten (Muse Spark Open) | Native video input for the explainer instead of keyframes |
| Fireworks | A second, managed serving path for the fine-tuned Glimmer |
| GCP GPU quota | Ablation fine-tunes and 3B distillation off Modal |

### Fallbacks for the fine-tune, in order

1. Modal A100-80GB QLoRA (default).
2. GCP spot A100 if quota is approved.
3. Thunder Compute A100 80GB at about $0.78/hr, roughly $3 out of pocket.
4. Frozen-Glimmer decision head (no fine-tune), always trained in parallel as a safety net.

## System architecture

Brier is one engine with domain adapters. A change flows along the top row, back along the middle row, and every run, decision and label lands in the ledger that retrains the models.

&#91;embedded content: Brier architecture: 10 components, one feedback loop\]

Only pairs above the conformal threshold reach the expensive tier. Muse Spark can add evaluations but never remove one.

### The world model, stated precisely

| Element | Robot CI | Software CI |
| --- | --- | --- |
| State s | Change card, scenario, cheap-tier evidence, history | PR, CI job, repo history |
| Action a | skip, cheap, expensive | skip, lint, run job |
| Transition p(s' given s, a) | Predicted eval outcome per scenario | Predicted job outcome |
| Decision | Conformal policy over actions | Same |

The world model is action-conditioned: it predicts what running an evaluation would show. Exploration runs are what let it learn the effect of skipping.

### Adapter trait (Rust)

```rust
pub trait Adapter {
    type Change;
    type Scenario;
    fn card(&self, change: &Self::Change) -> ChangeCard;
    fn scenarios(&self) -> Vec<Self::Scenario>;
    fn compile_state(&self, c: &Self::Change, s: &Self::Scenario, ev: &Evidence) -> String; // the only prompt builder
    fn actions(&self) -> Vec<Action>;        // each with a relative cost
    fn questions(&self) -> Vec<Question>;    // noul | choice | score
    fn label(&self, runs: &[Run]) -> Labels; // the only label definition
}
```

## Tech stack and repo layout

Rust owns the engine and every text the models see. Python owns simulation and training. TypeScript owns the platform. Anything else needs a written reason in `SCOPE.md`.

| Layer | Choice | Why |
| --- | --- | --- |
| Engine, CLI, state compiler, policy, ledger | Rust: `clap`, `serde`, `rusqlite`, `reqwest`, `anyhow` | One binary that runs inside customer CI; your strongest language |
| Simulation | Python 3.11: `lerobot`, `gymnasium`, `gym-pusht`, `gym-aloha` | Pretrained policies and sim envs ready on the HF hub |
| T1 model | `lightgbm` + `scikit-learn` isotonic | Strong cheap baseline, CPU only |
| T2 training | `torch`, `transformers`, `peft`, `bitsandbytes` | Standard QLoRA path for a new architecture |
| T2 serving | `vllm` with LoRA and prefix caching, on Modal | Logprobs at the answer position, scale to zero |
| Reasoning and perception | Meta Model API (OpenAI-compatible client) | Muse Spark Contributor, SAM 3.1 |
| Platform | Next.js + TypeScript + shadcn/ui + Drizzle on Vercel | Easiest hosting; one place for runs, failures and model results |
| Storage | SQLite ledger per CI run (engine); Postgres on Neon (platform); Parquet episodes in GCS | Engine stays deterministic; platform gets a real database |
| CI and demo | GitHub Actions, Check Run summary via `GITHUB_TOKEN` | No GitHub App needed |

### Repo layout

```
brier/
  AGENTS.md            # Codex rules (section: Working with Codex)
  SCOPE.md             # frozen feature list; anything not here is out
  README.md
  Makefile             # make check, make test, make replay
  crates/brier-core/   # Rust: ledger, adapter trait, state compiler, cascade, policy, CLI, report
  py/brier_sim/        # envs, policies, change library, scenarios, rollout runner
  py/brier_ml/         # datasets, T1, prompt format loader, finetune, calibrate, eval
  py/brier_explain/    # keyframes, SAM 3.1 client, Muse Spark client
  serve/modal_app.py   # vLLM + LoRA endpoint: POST /decide
  apps/web/            # Next.js + TS + shadcn + Drizzle: 4 pages, POST /api/ingest
  .github/workflows/   # check.yml, rollouts.yml (matrix), robot-ci.yml (PR gate)
  data/                # gitignored; mirrors gs://brier-data
```

### Two rules that prevent the worst bugs

1. **One state compiler.** `brier compile` (Rust) produces the exact prompt text for training JSONL and for serving. Python never formats prompts itself. A mismatch between training and serving text silently destroys calibration.
2. **One label definition.** Labels are computed once by `brier label` from raw episode records. Training, eval and the platform all read that output.

## Robot CI data

Every training label comes from simulation episodes we run ourselves, so the ground truth is exact and the regression causes are known. The unit of data is a **(change, scenario) pair** with a full expensive-tier result.

### Environments and policies

| Env | Policy (HF hub) | Notes |
| --- | --- | --- |
| `gym_pusht/PushT-v0` | `lerobot/diffusion_pusht` | If CPU episodes exceed 30 s, cut diffusion inference steps or switch to `lerobot/vqbet_pusht` |
| `gym_aloha/AlohaTransferCube-v0` | `lerobot/act_aloha_sim_transfer_cube_human` | ACT is fast on CPU; MuJoCo needs EGL or OSMesa headless rendering |
| `gym_aloha/AlohaInsertion-v0` | `lerobot/act_aloha_sim_insertion_human` | Stretch: only if Day 2 throughput allows |

Day 1 measurement decides everything below: seconds per episode per env on a GitHub Actions runner and on a GCP spot core.

### Change library (the "policy PRs")

A change wraps the base policy with a perturbation. Each change gets a **change card**: what a real engineer would write in a policy PR (family, parameters, a one-line description). About 35% of changes are benign so the model learns to say "safe".

| Family | Parameters | Real-world analogue |
| --- | --- | --- |
| `obs_noise` | sigma in {0.01, 0.03, 0.05, 0.1} | New sensor, new camera driver |
| `action_latency` | k in {1, 2, 4} steps | New controller, network hop |
| `action_scale` | {0.8, 0.9, 1.1, 1.2} | Retuned gains |
| `action_bias` | drift per step | Calibration offset |
| `camera_shift` | pixel translation, rotation | Re-mounted camera |
| `occlusion` | patch size, location | Arm or cable in view |
| `brightness` | gain, contrast | Lighting change |
| `frame_drop` | drop probability | Dropped frames under load |
| `chunk_len` | n\_action\_steps change | Inference schedule change |
| `weight_noise` | sigma on weights | Under-trained or regressed checkpoint |
| `benign` | refactor, metadata-only, sigma 0 | Changes that should pass |

Rules: one or two families per change; magnitudes sampled so roughly half of non-benign changes regress on some scenarios but not all. Scenario-specific failure is the whole point.

### Scenarios

A scenario is a fixed seed set plus environment parameters: initial object pose region (4 bins), lighting (2 levels), camera jitter (on/off) and, for ALOHA, physics jitter (friction, mass, 2 levels). Target 16 scenarios per env.

### Evaluation tiers

| Tier | Episodes | Horizon | Extra randomization | Relative cost |
| --- | --- | --- | --- | --- |
| `cheap` | 3 | 50% | none | 0.075 |
| `expensive` (hardware stand-in) | 20 | 100% | obs noise + physics jitter | 1.0 |
| `baseline` (once per scenario) | 50 | 100% | as expensive | one-off |

The README states plainly that the expensive tier stands in for hardware.

### Labels (computed once by `brier label`)

- `regressed` (noul): expensive success rate below baseline success rate minus 0.15.
- `failure_mode` (choice): none, stall/timeout, overshoot/misalignment, grasp or drop failure, collision, oscillation. Derived from terminal state and trajectory statistics.
- `success_bucket` (score 1 to 5): expensive success rate in fifths.
- `cause_family`: the injected family. Used only to score the explainer, never shown to it.

### Dataset size

Pairs = changes x scenarios. Episodes = pairs x 23, plus baselines. Target: 80 changes x 16 scenarios = 1,280 pairs per env, about 29k episodes per env. Minimum viable: 40 changes x 12 scenarios per env. Each pair yields two training examples (with and without cheap-tier evidence), so 2 envs give about 5k examples at the target size.

### Splits

- **Train / calibration / test by change** (60 / 20 / 20). A change never appears in two splits.
- **OOD split:** hold out the `occlusion` and `chunk_len` families entirely. Report them separately.
- **Time split** for the replay: changes ordered by id, as if they arrived as PRs.

### Rollout farm

- `rollouts.yml`: a GitHub Actions matrix over (env, change shard). Up to 20 concurrent jobs on the free plan. Each job uploads Parquet episode records to `gs://brier-data/episodes/`.
- `gcp_spot.sh`: the same runner on spot CPU VMs for bulk generation when Actions is too slow. Idempotent: a finished (change, scenario, tier) key is skipped.
- Episode record: `env, policy, change_id, scenario_id, tier, seed, success, steps, terminal_state, reward_sum, keyframes_uri, wall_seconds`. Keyframes (8 per episode) are saved only for failed expensive episodes.

## Models and training

Four tiers, cheapest first. A pair moves up a tier only when the tier below is not confident. Muse Spark can add evaluations but can never remove one, so the guarantee always rests on calibrated models.

| Tier | Model | Runs on | Decides |
| --- | --- | --- | --- |
| T0 | Rules | Rust, in CI | Benign changes with identical weights or config-only diffs: skip expensive |
| T1 | LightGBM + isotonic calibration | Python CPU, exported for the engine | `regressed` from numeric features |
| T2 | Glimmer 30B + LoRA, option-token scoring | vLLM on Modal | `regressed`, `failure_mode`, `success_bucket` |
| T3 | Muse Spark (Contributor) | Meta API | Reasoned review for the uncertain band; can only request more evals |

### T1 features

Change family one-hots and magnitudes, scenario parameters, cheap-tier success and reward stats, baseline success rate, scenario fragility (regression rate of this scenario across training changes) and nearest-neighbour outcomes of similar past changes. Fit isotonic calibration on the calibration split.

### T2 prompt format (produced only by `brier compile`)

```
<system> You are Brier, a calibrated decision model for robot policy CI. Answer with one option letter.
<state>
CHANGE: id=c042 family=action_latency k=2 desc="Move inference to edge box"
SCENARIO: env=aloha_transfer_cube pose_bin=3 lighting=low physics_jitter=on
BASELINE: success=0.86 (50 eps)
CHEAP_TIER: success=1/3 mean_reward=0.41 (3 eps, 50% horizon)
HISTORY: scenario_fragility=0.31; similar_changes(k=5): 3 regressed, 2 passed
</state>
<question> Will the expensive evaluation regress by more than 15 points?
<options> A) yes  B) no
<answer>
```

Options are shuffled during training so the model cannot learn a position bias. Target length: at most 1,500 tokens.

### Training objective

The model's probability for each option comes from a softmax over the option-letter logits at the answer position. The loss is cross-entropy plus a Brier term (both are proper scoring rules). Score questions use the ranked probability score instead of the Brier term.

```latex
\mathcal{L} = -\log p_\theta(y^*) + \lambda \sum_{k} \left(p_\theta(k) - \mathbb{1}[k = y^*]\right)^2, \quad \lambda = 1
```

### Fine-tune configuration (Modal A100-80GB)

| Setting | Value |
| --- | --- |
| Base | Glimmer text decoder only (skip the 2B vision encoder) |
| Quantization | 4-bit NF4, bf16 compute (BF16 if AMD credits arrive) |
| LoRA | rank 16, alpha 32, dropout 0.05, attention + MLP projections |
| Optimizer | AdamW, lr 1e-4, cosine, 3% warmup |
| Batch | effective 32 via gradient accumulation |
| Sequence | 1,536 tokens, gradient checkpointing on |
| Epochs | 1 (second epoch only if calibration-split NLL still falls) |
| Checkpoints | every 200 steps to a Modal Volume |
| Smoke test | 50 steps; measure tokens/sec before the full run |

Custom training loop in plain PyTorch + PEFT. No Trainer subclasses, no Unsloth.

### Calibration after training

Fit one temperature per question type on the calibration split. Report ECE, Brier and NLL before and after. Never fit anything on the test split.

### Serving

vLLM with the LoRA adapter, prefix caching on, `max_tokens=1`, `logprobs=20`. Option probabilities = renormalized probabilities of the option letters. One request carries all questions for a pair. Endpoint: `POST /decide` returning `{question_id: {option: prob}}` plus model version.

### Frozen-head fallback (always trained)

Extract Glimmer's last-token hidden state for each compiled state once, then train a small MLP head with the same loss on CPU. This is the DRS-paper recipe and the safety net if the fine-tune fails. It is also an ablation row.

### T3: Muse Spark escalation

Called only when T2's `regressed` probability falls in the uncertain band. Input: compiled state, T2 distribution, cheap-tier keyframes. Output JSON: `{recommend: run_expensive | run_more_cheap | no_change, rationale}`. The engine applies it only if it adds evaluations.

### Failure explainer: SAM 3.1 + Muse Spark

For each failed expensive episode: 8 keyframes, SAM 3.1 masks for the task objects and gripper, simple geometry from the masks (final object offset, gripper-object distance). Muse Spark receives frames, masks and telemetry but **not** the change card. Output JSON: `{failure_mode, cause_hypothesis, evidence_frames, suggested_check}`. Scored against `cause_family`. Sim gives privileged state, but SAM is used because real robots only have cameras; say this in the README.

### Stretch: 3B student distillation

Only if Day 6 has time. Teacher = fine-tuned T2 probabilities on all compiled states; student = Glimmer's 3B companion model, if it shares the tokenizer. Loss = KL to teacher + the proper-scoring loss on labels. Train on Kaggle or Modal. Report latency and ECE against the teacher.

## Policy engine, guarantee and ledger

The guarantee Brier states: across policy changes, the expected fraction of regressed scenarios that Brier skips is at most alpha (default 0.05). Conformal risk control picks the threshold that makes this hold.

### Decision flow per change

1. **T0** clears benign changes. Done.
2. **Stage A:** run the cheap tier on every scenario (7.5% of an expensive run each).
3. **Score:** compile each pair with its cheap-tier evidence; T1 scores all pairs; T2 scores pairs where T1 is between its confident bounds.
4. **Stage B:** run the expensive tier on every pair with p\_fail at or above the threshold lambda.
5. **Escalate:** pairs within 0.1 of lambda go to T3, which may add expensive runs.
6. **Explore:** with probability epsilon = 0.05, run expensive on a pair the policy skipped. Log the propensity.
7. **Verdict:** `block` if any expensive run confirms a regression, otherwise `pass`. The check summary prints the guarantee and the evals saved.

### Threshold by conformal risk control

Per calibration change i, the loss L\_i(lambda) is the fraction of its regressed scenarios with p\_fail below lambda (0 if it has none). It is bounded by B = 1 and shrinks as lambda falls. Choose:

```latex
\hat{\lambda} = \sup\left\{\lambda : \frac{n}{n+1}\hat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha\right\}, \quad \hat{R}_n(\lambda) = \frac{1}{n}\sum_{i=1}^{n} L_i(\lambda)
```

This gives E\[L(lambda-hat)\] at most alpha for a new exchangeable change. Validate it: over 20 random calibration/test splits, the realized miss rate must stay at or below alpha in at least 18. Recompute lambda-hat whenever 10 new labelled changes land in the ledger.

### Exploration and unbiased monitoring

Exploration runs make the miss rate observable after deployment: inverse-propensity weighting of explored pairs estimates the true miss rate of skipped pairs. In the offline replay all labels are known, so the replay simulates exploration and checks the IPS estimate against the truth.

### Ledger schema (SQLite, append-only)

```sql
CREATE TABLE changes   (change_id TEXT PRIMARY KEY, env TEXT, card_json TEXT, created_at TEXT);
CREATE TABLE scenarios (scenario_id TEXT PRIMARY KEY, env TEXT, params_json TEXT, baseline_success REAL);
CREATE TABLE runs      (run_id TEXT PRIMARY KEY, change_id TEXT, scenario_id TEXT, tier TEXT,
                        episodes INT, successes INT, mean_reward REAL, uri TEXT, wall_seconds REAL);
CREATE TABLE decisions (decision_id TEXT PRIMARY KEY, change_id TEXT, scenario_id TEXT, tier_used TEXT,
                        model_version TEXT, probs_json TEXT, lambda REAL, action TEXT,
                        propensity REAL, reason TEXT, created_at TEXT);
CREATE TABLE labels    (change_id TEXT, scenario_id TEXT, regressed INT, failure_mode TEXT,
                        success_bucket INT, cause_family TEXT, PRIMARY KEY (change_id, scenario_id));
CREATE TABLE explanations (run_id TEXT PRIMARY KEY, model TEXT, json TEXT, cost_usd REAL);
```

No updates, no deletes. A correction is a new row with a later timestamp.

## Product surface

The product is a GitHub Action, a check summary and a four-page platform. That is all a judge needs to see, and all Codex should build.

### The policy PR flow

A demo repo `brier-demo-robot` holds `policies/<env>.yaml`: base checkpoint plus the change parameters. Opening a PR that edits this file triggers `robot-ci.yml`:

1. `brier card` builds the change card from the YAML diff.
2. Matrix job: cheap tier on all scenarios.
3. `brier decide` calls `/decide` and writes decisions to the ledger.
4. Matrix job: expensive tier on the selected pairs only.
5. `brier explain` runs on failed expensive episodes.
6. `brier report` posts the Check Run summary and uploads `ledger.sqlite` and `report.json` as artifacts, then pushes the run to the platform and links the check to its run page.

### Three scripted demo PRs

| PR | Change | Expected outcome |
| --- | --- | --- |
| #1 "Bump logging, no policy change" | benign | T0 pass in under a minute, zero expensive runs |
| #2 "Move inference to edge box" | `action_latency` k=2 | Regresses on 3 to 5 scenarios; Brier runs a few expensive evals and blocks |
| #3 "Re-mount wrist camera" | `camera_shift` | Regresses on low-light scenarios; explainer names the visual cause from SAM masks |

Pick the exact parameters from the dataset so each PR behaves as described.

### Check summary (what the PR shows)

```markdown
## Brier: BLOCK (2 regressions confirmed)
Guarantee: expected missed regressions <= 5% (lambda = 0.18, calibrated on 64 changes)
Expensive evals: 6 of 16 scenarios (63% saved vs run-all)

| Scenario | p(regress) | Action | Result |
| aloha/pose3/low-light | 0.91 | expensive | REGRESSED 0.86 -> 0.40 |
| aloha/pose1/normal | 0.04 | skipped | n/a |

### Why scenario aloha/pose3 failed
Gripper closes ~2 steps late; cube slips at frame 5 (SAM mask overlay). Suggested check: latency compensation.
```

## Platform

The Brier platform is a Next.js app on Vercel that shows every evaluated change, why it passed or was blocked and how well calibrated the models are. For the hackathon it is public and read-only; the engine pushes results to it through one ingest endpoint. It replaces the static dashboard.

### Stack

| Layer | Choice |
| --- | --- |
| Framework | Next.js App Router + TypeScript, Node runtime only |
| UI | shadcn/ui + Tailwind; charts from shadcn's chart component (Recharts) |
| Database | Postgres on Neon via the Vercel Marketplace integration, free tier |
| ORM and migrations | Drizzle |
| Images | Keyframes and SAM overlays as public objects under `gs://brier-data/public/` |
| Hosting | Vercel for the hackathon; Cloudflare (OpenNext) after it |
| Auth | None. Pages are public; ingest takes a bearer token from `BRIER_INGEST_TOKEN` |

Vercel hosts the hackathon build; the app moves to Cloudflare (OpenNext) afterwards. To make that move a config change, the app uses no Vercel-specific SDKs and talks to Postgres through Drizzle with the Neon serverless driver (\`drizzle-orm/neon-http\`), which runs on both.

### Data flow

The engine stays local-first: each CI run keeps its own SQLite ledger, which keeps replays and training deterministic. `brier report --push` sends one run bundle to `POST /api/ingest`, and the platform writes it to Postgres. Data flows one way only; the platform never writes back to the engine.

### Postgres schema (Drizzle, `apps/web/db/schema.ts`)

| Table | Columns |
| --- | --- |
| `changes` | id, env, title, card (jsonb), pr\_url, created\_at |
| `evaluations` | id, change\_id, verdict, alpha, lambda, evals\_run, evals\_total, hours\_saved\_pct, model\_version, created\_at |
| `pairs` | id, evaluation\_id, scenario\_id, p\_regress, action, result, success\_before, success\_after |
| `failures` | id, pair\_id, keyframe\_urls (jsonb), overlay\_urls (jsonb), explanation (jsonb) |
| `reports` | id, suite (robot or software), created\_at, body (jsonb: ablation rows, reliability bins, curves) |

Ingest is idempotent on evaluation id: a repeated push replaces that evaluation's rows in one transaction.

### Pages, and no others

1. `/`: pitch, headline numbers from the latest report, demo video, GitHub link.
2. `/runs`: table of evaluations (change, env, verdict, evals run of total, hours saved, date) with one env filter.
3. `/runs/[id]`: the guarantee line, the scenario table (p(regress), action, result) and the failure gallery with keyframes, SAM overlays and the explanation. The PR check summary links here.
4. `/models`: ablation table, reliability diagram, missed regressions versus eval-hours. Software CI appears as a tab on this page in phase 2, not as a new page.

The only API route is `POST /api/ingest`. Pages read Postgres in server components; there is no read API and no client-side data fetching library.

### Rules for Codex in `apps/web`

- shadcn/ui components only. No other UI kit, icon pack beyond lucide, or animation library.
- Server components by default; client components only where a chart needs them.
- One `db/schema.ts`, one `lib/db.ts`. Queries live in the page that uses them. No repository or service layers.
- No auth, settings, admin, search, pagination beyond 50 rows, theme toggle (system theme only), i18n or analytics.
- `pnpm lint`, `pnpm typecheck` and `pnpm build` join `make check`.

### After the hackathon, not now

GitHub OAuth (Auth.js), orgs with installation-scoped data, the GitHub App and usage billing per evaluated change. They are listed here so nobody starts them this week.

## Evaluation

The submission stands on one table and two plots: the ablation table, the reliability diagram and missed regressions versus eval hours. Everything is computed by `brier replay` on the test split and written to `report.json`.

### Metrics

| Metric | Definition | Where it shows |
| --- | --- | --- |
| ECE | Expected calibration error, 15 equal-mass bins, `regressed` question | Ablation table, reliability plot |
| Brier score, NLL | Proper scoring rules on `regressed` | Ablation table |
| AUROC | Ranking quality on `regressed` | Ablation table |
| Eval-hours saved | 1 minus expensive-eval hours used / run-all hours, at alpha = 0.05 | Ablation table, headline |
| Change-level recall | Share of regressed changes with at least one regression caught | Ablation table |
| Guarantee validity | Splits (of 20) where realized miss rate is at most alpha | Ablation table |
| OOD ECE | ECE on held-out families | Ablation table |
| Explainer accuracy | `cause_family` top-1 and top-2 from frames only | Explainer table |
| Latency | p50 / p95 per decision call | Ablation table |
| Cost | $ per change evaluated, all API and GPU costs | Ablation table |

### Ablation table (fill from `report.json`)

| Strategy | ECE | AUROC | Eval-hours saved | Change recall | Guarantee holds | OOD ECE | p95 latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Run all expensive | n/a | n/a | 0% | 100% | yes | n/a | n/a |
| Random at equal budget |  |  |  |  |  |  |  |
| Cheap-tier heuristic |  |  |  |  |  |  |  |
| T1 LightGBM |  |  |  |  |  |  |  |
| Glimmer base, zero-shot logprobs |  |  |  |  |  |  |  |
| Glimmer frozen head |  |  |  |  |  |  |  |
| Glimmer fine-tuned |  |  |  |  |  |  |  |
| Fine-tuned + temperature |  |  |  |  |  |  |  |
| Full cascade (T0 to T3) |  |  |  |  |  |  |  |
| 3B student (stretch) |  |  |  |  |  |  |  |

The cheap-tier heuristic runs expensive wherever cheap success falls below baseline minus 0.15. It is the honest baseline: if Brier does not beat it, the models add nothing.

### Acceptance criteria (checked on Nov 2 evening)

These are targets, not predictions. Missing one is a signal to cut scope, not to keep tuning.

- [ ] Fine-tuned + temperature ECE at most 0.05 on the test split
- [ ] Fine-tuned beats the cheap-tier heuristic on eval-hours saved at alpha = 0.05
- [ ] Guarantee holds in at least 18 of 20 splits
- [ ] Demo PRs #1 to #3 behave as scripted, end to end, in under 15 minutes each
- [ ] Explainer top-2 accuracy above 60% on test failures

## Software CI adapter (phase 2)

Software CI proves the engine is general: the same ledger, state compiler, policy and decision model, pointed at pull requests instead of policy changes. Build it only if every Robot CI acceptance criterion passes on Nov 2 evening. Budget: one day.

### Mapping onto the adapter trait

| Trait item | Robot CI | Software CI |
| --- | --- | --- |
| Change | Policy change card | PR: files changed, lines, author history, title |
| Scenario | Sim scenario | CI job (one matrix job = one scenario) |
| Cheap action | Cheap tier | Lint and import check |
| Expensive action | Expensive tier | Run that CI job |
| `regressed` | Success drop of 15+ points | Job failed on the PR head |
| Extra question | `failure_mode` | `reverted_30d`: PR reverted within 30 days |

### Data

1. Pick 3 public Python repos with GitHub Actions history, matrix jobs and over 1,000 merged PRs (choose on Nov 3 morning; check that job logs are still retained).
2. Mine with `gh api`: PRs, changed files, workflow runs and job conclusions on the head commit.
3. Revert labels: commits whose message starts with "Revert" and references a PR number.
4. Repo world model, minimal: file to job co-failure counts from history, plus a tree-sitter import graph for Python. Nothing more.

### What the demo shows

A replay over the last 300 PRs of one repo: CI job-minutes saved at alpha = 0.05 and the reliability diagram, on the platform's /models page under a Software tab. No live GitHub App and no live gating of someone else's repo.

### Cut line

If the replay is not producing numbers by Nov 3, 8:00 PM IST, drop software CI to a roadmap slide and spend the time on the video.

## Schedule

Seven days, each ending at a gate. A missed gate triggers the matching fallback in Risks the same night, never a later catch-up. All times are IST.

&#91;embedded content: Build schedule · 7 days, 7 gates (IST)\]

The Nov 2 gate is the only scope decision: software CI gets built only if every Robot CI acceptance criterion passes.

### Day 1, Thu Oct 29

- [ ] Accounts: upgrade GCP billing, file the GPU quota request, set budget alerts on all three pools, apply for AMD credits, create a Meta API key using Contributor variants, create the Modal workspace
- [ ] Create the public repo `brier` with AGENTS.md and SCOPE.md (T01)
- [ ] T02 ledger, T03 sim runner, T04 change library, T05 scenarios
- [ ] T06 timing on a GitHub Actions runner and one GCP spot VM; set the dataset size from the result
- [ ] T07 `rollouts.yml`, first matrix run

* **Gate:** three regression families run end to end on Actions and land in GCS

### Day 2, Fri Oct 30

- [ ] T08 GCP spot runner; bulk generation, PushT first, then ALOHA
- [ ] T09 baselines (50 episodes per scenario)
- [ ] T10 `brier label`, T11 `brier compile`, T12 T1, T13 eval harness on the partial data
- [ ] Ask #questions for the Stage 2 judging rubric
- [ ] Fireworks session on training Muse open models, 9:30 PM IST

* **Gate:** half the dataset labelled and T1 evaluated on it; check the "labels too easy" and "labels too noisy" risks

### Day 3, Sat Oct 31

- [ ] Dataset complete; freeze splits and publish the dataset card
- [ ] T14 base Glimmer endpoint on Modal; fill the zero-shot ablation row
- [ ] T15 QLoRA smoke test (50 steps, measure tokens/sec), then the full run
- [ ] T16 frozen head in parallel
- [ ] While the fine-tune runs: T30 platform scaffold deployed on Vercel, T31 ingest

* **Gate:** fine-tune running by 6 PM; if AMD credits are visible, the BF16 run goes there instead

### Day 4, Sun Nov 1

- [ ] T17 temperature scaling, T18 conformal policy and `brier decide`, T19 replay, T20 LoRA serving
- [ ] Ablation table v1

* **Gate:** guarantee holds in 18 of 20 splits; fine-tuned model beats the cheap-tier heuristic, or switch to the fallback story

### Day 5, Mon Nov 2

- [ ] T21 keyframes and SAM 3.1, T22 explainer and escalation
- [ ] T23 `robot-ci.yml` with the check summary, T24 demo repo PRs #1 to #3
- [ ] T32 `/runs` and `/runs/[id]`; the PR check links to the run page
- [ ] Evening: walk the acceptance checklist in Evaluation

* **Gate:** go or no-go on software CI

### Day 6, Tue Nov 3

- [ ] Go: T26 and T27, with the 8 PM cut line. No-go: T28 3B distillation
- [ ] T25 `/models` page and T33 landing page
- [ ] Last full day for submission questions; Discord office hours run 8:30 to 9:30 PM IST

* **Gate:** cut line at 8 PM

### Day 7, Wed Nov 4

- [ ] Code freeze at 2 PM; bug fixes only after that
- [ ] Record demo PRs, edit the video, T29 README
- [ ] Submit by Nov 5, 5:00 AM IST; confirm the repo is public and the video link opens logged out

* **Gate:** submitted

## Working with Codex

Codex builds only what a task card names, inside the files the card lists, in diffs you can review in five minutes. Three files enforce this: `AGENTS.md` (rules Codex reads on every run), `SCOPE.md` (the frozen feature list, edited only by you) and one task card per session.

### Your review loop for every task

1. One task per Codex session, fresh context. Paste the task card.
2. When it finishes, run `git diff --stat`. Reject the diff if it touches a file not on the card or exceeds the card's line budget.
3. Run `make check` and the card's acceptance commands yourself. Do not trust the summary.
4. Search the diff for unrequested extras: new flags, config keys, CLI subcommands, classes with one implementation, "TODO: future" code. Tell Codex to delete them, not to justify them.
5. Commit as `T07: brier compile`. The commit log doubles as the judges' proof of what was built during the hackathon.
6. If Codex fails the same card twice, split the card in half. Do not add instructions to a failing card.

### `SCOPE.md` (frozen)

```markdown
# SCOPE (frozen; only Brooklyn edits this file)

CLI commands, and no others: brier ledger init | label | compile | card | decide | replay | explain | report [--push]
HTTP endpoints, and no others: POST /decide (serve/modal_app.py), POST /api/ingest (apps/web)
Workflows, and no others: check.yml, rollouts.yml, robot-ci.yml
Platform: apps/web, Next.js App Router + TypeScript + shadcn/ui + Drizzle + Postgres (Neon), hosted on Vercel
Platform pages, and no others: /, /runs, /runs/[id], /models
Envs: PushT, AlohaTransferCube (AlohaInsertion only if T06 timing allows)
Models: T0 rules, T1 LightGBM + isotonic, T2 Glimmer + LoRA, T3 Muse Spark (Contributor), SAM 3.1 for failure frames
Phase 2 (only after the Nov 2 gate): software CI adapter + replay on public repos, shown as a tab on /models

Out of scope: auth, accounts, settings or admin pages, GitHub App, billing, a second UI library, Vercel-specific SDKs,
Edge runtime, databases beyond SQLite (engine) and Postgres (platform), Docker beyond Modal images, Kubernetes,
Terraform, real hardware, ZK proofs, chip CI
```

### `AGENTS.md`

```markdown
# AGENTS.md: rules for every Codex task in this repo

## Mission
Implement exactly the current task card. Nothing more. SCOPE.md is the outer boundary; the task card is the inner one.

## Hard rules
1. Touch only the files listed under "Files" in the task card. If another file must change, stop and explain why in your final message. Do not change it.
2. Add no dependency the card does not list. Never add a framework.
3. Add no feature, flag, config option, CLI subcommand, endpoint, page, environment variable or abstraction that the card does not name. No plugin systems, registries, base classes with one implementation or "future-proofing".
4. No refactors, renames or formatting changes outside the task.
5. Smallest diff that passes the acceptance commands. Stay under the card's line budget; if you cannot, stop and say so.
6. Each new public function gets one focused test that the acceptance commands run.
7. No mock modes, demo modes, silent fallbacks or retries the card does not ask for. Fail loudly with a clear error.
8. Prompt text is produced only by `brier compile` (Rust). Python never formats prompts.
9. Labels are computed only by `brier label`.
10. Secrets and paths come from env vars: META_API_KEY, BRIER_DATA (gs:// or local), MODAL_*, DATABASE_URL, BRIER_INGEST_TOKEN. Never hardcode them.
11. No new top-level files or directories.
12. Docstrings on public functions only. Do not edit README.md unless the card says so.
13. When a choice is unclear, take the simpler option and list the assumption in your final message.

## Definition of done
- The card's acceptance commands pass.
- `make check` passes: cargo fmt --check, cargo clippy -- -D warnings, cargo test, ruff check, pytest -q, and in apps/web: pnpm lint, pnpm typecheck, pnpm build.
- Final message lists: files changed, commands run and their results, assumptions, anything skipped.

## Conventions
- Rust: anyhow for errors, serde for JSON, clap for CLI, blocking reqwest. No async unless the card says so.
- Python 3.11: type hints, plain functions over classes, pathlib, logging (no print in library code).
- apps/web: Next.js App Router, TypeScript strict, server components by default, client components only for charts. shadcn/ui components only (lucide icons). Drizzle for every query; one db/schema.ts and one lib/db.ts; queries live in the page that uses them. Node runtime only, no Vercel-specific SDKs. No auth, settings, admin, search, theme toggle, i18n or analytics.
- IDs: change `c001`, scenario `<env>-s01`, run and decision ids are UUIDv4.
- Data files: Parquet for episodes, JSONL for compiled states, SQLite for the engine ledger, Postgres for the platform.
```

### Task card template

```
TASK T11: brier compile
Goal: Emit one JSONL line per (change, scenario, question) with the exact prompt text and label.
Files (only these): crates/brier-core/src/compile.rs, crates/brier-core/src/main.rs, crates/brier-core/tests/compile.rs
Inputs: ledger.sqlite (changes, scenarios, runs, labels)
Output: {"id", "split", "question", "options", "prompt", "label"} per line; prompt format exactly as in the plan
Out of scope: tokenization, truncation strategies, any Python
Dependencies allowed: none new
Acceptance: cargo test compile; brier compile --ledger fixtures/mini.sqlite --out /tmp/c.jsonl && head -1 /tmp/c.jsonl
Line budget: 250
```

### Task list

| ID | Task | Day | Depends on |
| --- | --- | --- | --- |
| T01 | Repo skeleton, Makefile, `check.yml`, AGENTS.md, SCOPE.md | 1 |  |
| T02 | Ledger schema + `brier ledger init` + insert helpers | 1 | T01 |
| T03 | Sim runner: env + policy + N episodes to Parquet | 1 | T01 |
| T04 | Change library: 11 families as policy wrappers | 1 | T03 |
| T05 | Scenario definitions for PushT and ALOHA | 1 | T03 |
| T06 | Timing benchmark: seconds per episode per env | 1 | T04, T05 |
| T07 | `rollouts.yml` matrix + GCS upload, idempotent keys | 1 | T06 |
| T08 | GCP spot runner script | 2 | T07 |
| T09 | Baselines: 50 episodes per scenario | 2 | T07 |
| T10 | `brier label` | 2 | T02, T09 |
| T11 | `brier compile` | 2 | T10 |
| T12 | T1 LightGBM + isotonic | 2 | T10 |
| T13 | Eval harness: metrics to `report.json` | 2 | T12 |
| T14 | Modal vLLM endpoint, base Glimmer, zero-shot option scoring | 3 | T11 |
| T15 | QLoRA training script with option-token proper-scoring loss | 3 | T11 |
| T16 | Frozen-head embeddings + MLP head | 3 | T14 |
| T30 | Platform scaffold: Next.js + shadcn + Drizzle schema + Neon, deployed on Vercel | 3 | T01 |
| T31 | `POST /api/ingest` + `brier report --push` | 3 | T30, T13 |
| T17 | Temperature scaling per question type | 4 | T15 |
| T18 | Conformal risk control + `brier decide` | 4 | T13 |
| T19 | `brier replay`: 20 splits, simulated exploration, IPS check | 4 | T18 |
| T20 | Serve fine-tuned LoRA at `POST /decide` | 4 | T15 |
| T21 | Keyframes + SAM 3.1 client + mask geometry | 5 | T09 |
| T22 | Muse Spark explainer + escalation client | 5 | T21 |
| T23 | `brier card`, `brier report`, `robot-ci.yml` check summary | 5 | T18, T20 |
| T24 | Demo repo with PRs #1 to #3 | 5 | T23 |
| T32 | `/runs` and `/runs/[id]` with the failure gallery | 5 | T31, T22 |
| T25 | `/models`: ablation table, reliability diagram, missed-vs-hours chart | 6 | T31, T19 |
| T33 | Landing page `/` | 6 | T32 |
| T26 | Software CI miner (gated) | 6 | Nov 2 gate |
| T27 | Software CI adapter + replay (gated) | 6 | T26 |
| T28 | 3B student distillation (stretch) | 6 | T17 |
| T29 | README with built-during-hackathon log | 7 | all |

## Submission

Submit by Nov 5, 5:00 AM IST, 2.5 hours before the hard deadline. The video and README must make it obvious what was built during the hackathon, or the project risks disqualification.

### README structure

1. One-line pitch and the headline number.
2. 30-second architecture: world model, decision model, reasoning, the guarantee.
3. Meta models used and where: Glimmer (fine-tuned decision model), Muse Spark (escalation, explanation), SAM 3.1 (failure perception).
4. Results: ablation table, reliability diagram, missed-regressions plot.
5. Honest limits: the expensive tier stands in for hardware; scenarios are simulated; the guarantee assumes exchangeable changes.
6. **Built during the hackathon:** first commit hash and date, the commit list T01 to T29, and the statement that no code predates Oct 27.
7. Reproduce: `make replay` on the published dataset; links to the LoRA adapter and dataset.
8. Prior work: TypeSafe Jev and RLCD, RLCR (Brier-score rewards), Meta's Diff Risk Score and Predictive Test Selection, conformal risk control, LeRobot.

### 3-minute video script

| Time | Shot | Line |
| --- | --- | --- |
| 0:00 to 0:20 | Robot sim failing | Every policy change needs evals across dozens of scenarios. Hardware time is the bottleneck. |
| 0:20 to 0:50 | Architecture diagram | A world model predicts each eval's outcome; a calibrated Glimmer decides which to run; Muse Spark reasons when unsure. |
| 0:50 to 1:50 | Live PR #1, then PR #2 | Benign PR passes in seconds. Latency PR: Brier runs 6 of 16 evals, catches the regression, blocks, explains it with SAM masks. |
| 1:50 to 2:30 | Platform /models | Reliability diagram, eval-hours saved at a 5% miss guarantee, ablation against the cheap heuristic. |
| 2:30 to 3:00 | Software CI tab or roadmap slide | Same engine on code. Next: real robot fleets, then software and chip CI. |

Record the PR demos in advance and cut them tight. Narrate over the recording; never wait on a live run in the video.

### Judging alignment

- **Stage 1 viability:** fits Push the Frontier (control and understanding of a physical system) and uses three Meta models, one of them fine-tuned.
- **Stage 2 scoring:** the rubric table did not render in the guide. Ask in #questions on Oct 30 and adjust this section.
- **Community vote:** post the video and platform link on X and LinkedIn on Nov 9 after the closing ceremony.

## Risks and fallbacks

The two risks most likely to sink the week are slow simulation and a fine-tune that does not beat the cheap heuristic. Both have a decision date.

| Risk | Early signal | Fallback | Decide by |
| --- | --- | --- | --- |
| Episodes too slow on CPU | T06 shows over 30 s per PushT episode | Fewer diffusion steps or VQ-BeT; drop AlohaInsertion; minimum dataset size | Oct 29 night |
| MuJoCo rendering fails headless | ALOHA env crashes on Actions | Switch `MUJOCO_GL` between egl and osmesa; run ALOHA on GCP only | Oct 30 noon |
| Labels too easy (change family gives away the answer) | T1 AUROC above 0.97 | Narrow magnitude ranges so outcomes depend on scenario; add benign look-alikes | Oct 30 night |
| Labels too noisy | Baseline success rate swings over 0.1 between reruns | Raise expensive episodes from 20 to 30 for borderline pairs only | Oct 30 night |
| Glimmer fails to load in transformers or PEFT | T14 or T15 smoke test errors | Frozen-head path with vLLM embeddings; report fine-tune as next step | Oct 31 noon |
| Modal credit runs out mid-train | Spend alert at $25 | Resume from the last checkpoint on Thunder Compute (about $3) | Oct 31 |
| Platform eats the week | A web task over budget or a page outside SCOPE | Ship `/runs/[id]` and `/models` only; the README becomes the landing page | Nov 2 |
| Fine-tune does not beat the cheap heuristic | Ablation v1 on Nov 1 | Lead with calibration and the guarantee; full cascade still beats run-all | Nov 1 night |
| Guarantee fails in replay | Fewer than 18 of 20 splits hold | Raise calibration split size; use alpha 0.1 and say so | Nov 1 night |
| SAM 3.1 or Muse Spark API errors or rate limits | First 20 explainer calls | Explain from frames + sim telemetry without masks | Nov 2 |
| Codex scope creep | Diff over budget or files off-card | Reject, split the card, re-run | Every task |
| Demo PR runs take too long | Over 15 min end to end | Pre-run PRs and show recordings; keep the live run for Q&A | Nov 3 |
| Submission platform issues | Upload fails | Submit by 5:00 AM IST with 2.5 hours buffer; video unlisted on YouTube | Nov 5 |
