# Brier

Brier predicts which robot evaluation scenarios will regress after a policy change, then selects expensive simulation runs under a calibrated risk policy.

**Status, November 2, 2026:** the private dataset passed all five dataset gates and independent release verification: 5,152 production Parquet keys, 60,480 episodes, 2,560 authoritative labeled pairs and 15,360 Rust-compiled records. Glimmer completed the authorized single epoch (249 optimizer steps), final scoring and temperature calibration. GPU spend was **$2.181175 total**, including **$1.015840 for full training**. The local trained adapter is `data/runs/glimmer-ft-v1/adapter`; measured metrics are `data/runs/glimmer-ft-v1/metrics.json`. These private files are gitignored. No second epoch, public model publication or deployment has run.

**Deployment is on hold.** The prepared full-cascade replay has test ECE **0.128743**, above the unchanged **0.05** target. Its conservative risk threshold is zero: it selects all 368 test pairs and saves **0%** of evaluation hours. Eighteen of twenty diagnostic splits meet the target. Training completion establishes an artifact, not the product acceptance gate or useful savings. Real Modal serving, Meta provider calls, hosted database integration and end-to-end PR demos still need credential-backed tests.

Repository: [SiddharthManjul/Brier](https://github.com/SiddharthManjul/Brier). The [build plan](<Brier Build Plan (Robot CI first, Software CI second).md>), [scope](SCOPE.md), and [task rules](AGENTS.md) define the boundaries.

## How it works

1. A policy change and a scenario become a structured change card. PushT and AlohaTransferCube are the supported simulation environments.
2. Three cheap episodes provide evidence and save one actual final image each. The expensive tier uses twenty episodes; the stored reference baseline uses fifty per scenario.
3. The decision cascade combines identity rules, LightGBM with isotonic calibration, and option probabilities from a pinned language model. The three questions are regression, failure mode, and success bucket. Rust `brier compile` produces every prompt; Rust `brier label` computes the authoritative labels.
4. Temperature scaling fits one positive temperature per question. A separate change-level conformal risk calibration selects the threshold. Scenarios at or above that threshold run; five-percent exploration also samples below it with logged propensities.
5. The ledger records decisions and measured runs. `brier report` produces a Markdown verdict and, when complete, a platform ingest bundle. The four-page web app shows runs, actual failure assets, and measured model reports.

The world-model role is predicting what an evaluation would show from a change, scenario, and available evidence. It does not generate replacement simulation trajectories. The plan assigns Glimmer to decision scoring, Muse Spark Contributor to escalation and explanation, and SAM 3.1 to failure-frame perception. Meta's [Glimmer model guide](https://dev.meta.ai/docs/muse-glimmer/get-the-model) identifies the published `meta-models/Muse-Glimmer-30B` resource: a multimodal model with a nested text decoder. The trained adapter uses immutable revision `cec94fcfd6512f2cfe708138c32533b346ea4f69`; `brier-model.json`, `run_meta.json`, the independent dataset-readiness receipt and the completed spend ledger bind the measured run.

SAM segmentation, Muse Spark explanation/escalation, and the Rust `brier explain` path pass local integration checks; real provider calls remain pending. The clients use the documented `sam-3.1` and `muse-spark-1.3-contributor` identities and send the project's `META_API_KEY` as the provider Bearer credential. SAM decoding uses the official `meta-sam-parser==0.1.0` dependency. These APIs expose version-named identities here; no immutable SAM/Muse revision is claimed. Contracts come from Meta's [SAM overview](https://dev.meta.ai/docs/sam/overview), [structured-output documentation](https://dev.meta.ai/docs/structured-output), and [image-understanding documentation](https://dev.meta.ai/docs/image-understanding).

Rust compiles all explanation and escalation text. Python first requests the three Rust-named SAM concepts, unions matched instances, saves actual overlays, and returns measured geometry. Rust then compiles the Muse prompt from observed frames, geometry, scenario parameters, and allowlisted telemetry; the explainer never receives the change card or hidden cause-family label. A missing required mask fails clearly. Muse receives interleaved raw/overlay images and returns structured evidence-frame indices and an explanation.

T3 is called only for T2 regression probabilities in `[0.1, 0.9]`. Both `run_expensive` and `run_more_cheap` recommendations add a full expensive evaluation, preserving the original three-episode cheap feature vector used in calibration. `no_change` preserves the existing decision; T3 cannot remove selected evaluations. API costs remain `null` unless measured. T0 can separately prove an unchanged policy identity without making a statistical claim. The Robot CI workflow now includes explanation, evidence upload, and platform ingest; an operational gate still requires deployment configuration and real end-to-end evidence.

## What the guarantee means

For a change, loss is the fraction of its regressed scenarios that the threshold skips; a change with no regressions has zero loss. With a scorer fixed independently of risk calibration, and exchangeable calibration and future changes, the policy bounds **expected change-level missed fraction** at `alpha`. It does not guarantee every change's realized loss or validity under arbitrary distribution shift.

Training changes, probability-calibration changes, risk-calibration changes, and held-out test/OOD changes must stay separate. The frozen `calibration` split must itself be divided by change: isotonic/temperature fitting uses one pool, conformal risk calibration uses another. Reusing a probability-fitting change for risk calibration invalidates the required independence. The main test/OOD table uses a threshold fitted without those labels. The separate twenty-split diagnostic repartitions the held-out calibration/test pool; probability-fitting changes remain excluded.

Guide 2 reserves nineteen calibration changes by a fixed hash of their IDs before model selection or temperature fitting. At least twenty calibration changes are required; the remaining changes form the fitting pool. Checkpoints and model/calibration metadata record both lists. Frozen-head early stopping, fine-tune evaluation, and temperature fitting exclude the reserved labels, while scoring still produces their predictions. Prepare T1 with the complete frozen dataset and `train_t1(dataset, output, calibration_change_ids=fit)`, where `fit` is the seven-change probability-fitting pool; score the complete original dataset. Its artifact records the actual isotonic fitting IDs. Include model-selection and probability-fitting IDs in replay provenance. Nineteen is the minimum for the default risk bound, not evidence of useful savings.

Before live selection, `decide` refreshes the supplied risk-calibration artifact when at least ten new eligible changes have accumulated. Eligibility requires the frozen `calibration` split, no decisions for that change, authoritative labels for every environment scenario, and exactly one complete three-episode cheap and twenty-episode expensive result per scenario. Training, probability-fitting, existing risk-pool, current-live, test, and OOD changes are excluded. Every qualifying change is rescored with the same fixed T1/T2 cascade, model version, and temperatures; T3 is not used for these risk scores. After validation, the artifact is replaced atomically, with a source-change check. A successful independent refresh remains valid if later live scoring fails. Independently collected complete calibration data must still be supplied; selectively observed live labels are never automatically recycled into the pool. Exchangeability remains an assumption.

The finite-sample bound includes `1 / (n + 1)`. At `alpha = 0.05`, at least nineteen independent risk-calibration changes are necessary, even with zero observed calibration loss. Nineteen is a minimum, not a promise of useful savings. Replay's twenty seeded split checks are an empirical diagnostic, distinct from the statistical assumption.

Replay can compute inverse-propensity diagnostics because its offline ledger contains the true regression denominator for every scenario. That denominator is not identifiable from selectively observed live runs alone. Simulation is a stand-in for expensive hardware evaluation; no real-hardware performance is established.

## Local checks

Use Python 3.11, Rust with `rustfmt` and `clippy`, Node.js 24, and pnpm 10.30.3. Python dependencies are declared in `pyproject.toml`; Rust and web dependencies have lockfiles.

```sh
python3.11 -m venv .venv
.venv/bin/python3.11 -m pip install -e '.[dev]'
pnpm --dir apps/web install --frozen-lockfile
make check
```

`make check` runs Rust formatting, Clippy with warnings denied, Rust tests, Ruff, pytest, and the web lint, typecheck, native Node tests, and production build. It uses `.venv/bin/python3.11` when present. It does not launch rollouts, GPU training, deployment, or database migrations. The CI installation sequence is in [check.yml](.github/workflows/check.yml).

Guide 2 has a separate dependency manifest in [py/pyproject.toml](py/pyproject.toml), mirrored in [modal_train.py](py/brier_ml/modal_train.py). Keep its environment separate from the root LeRobot environment. Its CPU acceptance command is `PYTHONPATH=py python -m pytest -q py/brier_ml/tests`, using that training environment. The guide tests are outside the root pytest collection. The training image pins Torch 2.15.0, Transformers 5.22.1 and PEFT 0.21.2; the completed Guide 2 GPU run used this separate environment. Serving uses a separate vLLM image whose remote execution remains untested.

## Reproduce a measured replay

The verified source release remains private at `gs://brier_bucket/release/v1`; its measured dataset card is `data/rollout-v1/DATASET_CARD.md`. The prepared local decision bundle is `data/decision-v1-audited`, containing checksummed copies of the ledger, compiled states and frozen scenario catalog, fitted T1, the exact trained adapter identity, three temperatures, the independent seven-fit/nineteen-risk partition, risk calibration, both frozen policy baselines and the measured replay report. Nothing is publicly published. Set `BRIER_DATA` to this existing **local** bundle directory, or a separately prepared directory containing:

| File | Required contents |
| --- | --- |
| `ledger.sqlite` | Authoritative labels, scenario baselines, exactly one three-episode cheap result per retained pair, and complete measured twenty-episode expensive results for the offline counterfactual. |
| `compiled.jsonl` | Rust-compiled records with `id`, `split`, `question`, `options`, `prompt`, and `label`; splits are frozen by change. |
| `replay-scores.json` | Independently fitted strategy predictions with their training and probability-calibration provenance. |

The scores file has one top-level `strategies` object. Each strategy has `model_version`, `training_change_ids`, `probability_calibration_change_ids`, and `probabilities`. The two provenance fields are arrays of change IDs. `probabilities` maps retained regression/cheap compiled IDs, such as `c001/pusht-s01/regressed/cheap`, to finite numbers in `[0, 1]`. Scores are required for all retained risk-calibration, test, and OOD pairs. Optional `latencies_seconds` and `costs_usd` maps must cover every evaluation pair if supplied; absent measurements remain absent. Unknown fields are rejected.

For a new input directory whose `report.json` does not already exist:

```sh
make replay
```

This is exactly:

```sh
cargo run --locked --bin brier -- replay \
  --ledger "$BRIER_DATA/ledger.sqlite" \
  --compiled "$BRIER_DATA/compiled.jsonl" \
  --scores "$BRIER_DATA/replay-scores.json" \
  --out "$BRIER_DATA/report.json"
```

Defaults are `--alpha 0.05 --seed 0`. The prepared `data/decision-v1-audited` bundle already contains `report.json`; use the separate `recheck-report.json` example below to reproduce it without overwriting an artifact. Replay reads actual labels and durations, adds run-all, cheap-heuristic, and random-at-equal-expected-budget baselines, and reports only supplied scored strategies. Primary ablation metrics, costs, reliability bins, and threshold curves use the test cohort; OOD calibration is reported separately. The JSON includes measured hours, change recall, ECE, Brier score, NLL, AUROC when defined, and twenty seeded split results. Persisted held-out explanations can also be scored for top-one/top-two injected-family agreement without exposing the hidden family to the provider. ECE uses fifteen quantile bins with tied scores kept together. API/GPU cost and latency appear only when provided as measurements.

For artifact preparation and live decisions, inspect the implemented interfaces:

```sh
cargo run --locked --bin brier -- label --help
cargo run --locked --bin brier -- compile --help
cargo run --locked --bin brier -- card --help
cargo run --locked --bin brier -- decide --help
cargo run --locked --bin brier -- explain --help
cargo run --locked --bin brier -- report --help
PYTHONPATH=py .venv/bin/python -m brier_ml.prepare --help
```

`decide` requires `--ledger`, `--compiled`, `--calibration`, and `--change`, with optional `--alpha`. Its risk-calibration artifact binds the actual scorer versions, temperatures, T1 artifact, and separate probability/risk change IDs. A live change must be unlabelled and outside those fitting pools. Scoring needs the configured `MODAL_DECIDE_URL`; no model endpoint is embedded in the repository.

`explain --ledger ... --run ...` takes the aggregate expensive-run UUID recorded in the ledger. It processes every failed individual episode in that twenty-episode Parquet group, using each episode's eight saved keyframes, and stores the aggregate explanation in the ledger. Local images and overlays remain under `BRIER_DATA`; publishing HTTPS image URLs is a separate deployment step.

`report` requires `--ledger`, `--change`, and `--out`. The output file is Markdown; stdout is structured JSON containing `markdown`, `bundle`, and `verdict`. Optional `--metrics` attaches an existing measured replay report; `--push` takes the actual HTTPS `/api/ingest` URL and requires `BRIER_INGEST_TOKEN`. Repeated reports reuse decision-derived IDs. A regression confirmed by a selected run blocks; incomplete selected results remain pending; both return a nonzero CLI exit after writing the summary. Skipped scenarios do not expose offline labels as live observations. `hoursSavedPct` is `null` unless measured full counterfactual durations exist; evaluation-count reduction is reported separately.

Robot CI requires repository variable `BRIER_INGEST_URL` to contain the actual deployed HTTPS `/api/ingest` URL and secret `BRIER_INGEST_TOKEN` to authorize the push. This URL is deployment configuration, not a supplied public service. The workflow uploads actual evidence before `report --push` and archives its JSON stdout, ledger, and assets. With a GCS `BRIER_DATA` prefix, the report maps already uploaded relative `public/keyframes` and `public/overlays` paths to canonical GCS HTTPS URLs. Public read access for the intended evidence prefix must be configured through GCS IAM during deployment; the workflow does not make the bucket public. Local artifact publication remains a deployment task; no workflow has been run as part of this local implementation.

## Trained model and serving contract

The completed Guide 2 run is `glimmer-ft-v1`: BF16, microbatch eight, effective batch thirty-two, one epoch and 249 steps. Its pinned base is `meta-models/Muse-Glimmer-30B@cec94fcfd6512f2cfe708138c32533b346ea4f69`. The model version additionally binds a SHA256 of the adapter configuration and weights; the same identity is in the risk-calibration artifact and must be returned by `/decide`. Temperature scaling happens once in Rust after the endpoint returns raw option probabilities.

[serve/modal_app.py](serve/modal_app.py) defines only `POST /decide`. The named Modal secret `brier-model` supplies `MODAL_MODEL_ID` and `MODAL_MODEL_REVISION`. The existing private `brier-models` volume contains `runs/glimmer-ft-v1`; a future serving launch must use `MODAL_VOLUME_NAME=brier-models` and `MODAL_CHECKPOINT_DIR=/checkpoints/runs/glimmer-ft-v1`. Checkpoint configuration is supplied when defining the Modal app so the volume is mounted. These values describe the completed private run; no new volume or checkpoint is created by import.

The service verifies the pinned model, rank, dataset binding and completed one-epoch metadata before constructing vLLM. Guide 2 requires matching `run_meta.json` and disjoint probability/risk pools; older `finetune.py` artifacts use their explicit `smoke=false` dataset-bound manifest format. A smoke checkpoint cannot stand in for the trained adapter. Rust sends exactly three compiled questions, without labels; the endpoint returns the original IDs and the complete normalized option distributions. It never rewrites Rust prompts or applies a second temperature transform.

The Python 3.11 serving image pins CUDA 13.0.3, vLLM 0.28.0 and FastAPI 0.162.0, using native text-only Glimmer and BF16 on an A100-80GB. The [Meta runtime guide](https://dev.meta.ai/docs/muse-glimmer/deploy) and [vLLM recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B) describe this runtime. The [pinned vLLM model source](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/muse_glimmer.py) declares LoRA support and maps the attention output-gate projection separately from MLP gates. Actual GPU adapter loading and memory/latency acceptance still require a separately authorized serving smoke test; training on Transformers does not verify vLLM execution. Missing answer-letter log probabilities fail clearly. Prompts may use at most 1,536 tokens, with one additional context position for generation.

No training command is required to connect Muse Spark or SAM: both clients use their hosted APIs. Real access and response contracts remain to be tested. Additional Glimmer training, student distillation and a second epoch are outside this integration task.

## Credentials and manual integration checks

Keep secrets in the environment or the service's secret manager. Never commit credentials, copy `.env` files into artifacts, or paste secret values into issue comments. Existing deployment remains on hold; the following configuration is for the next credential-backed test session, not authorization to deploy or run paid inference.

| Value | Needed for |
| --- | --- |
| `DATABASE_URL` | A Neon Postgres connection string for the four-page app and ingest route. Use a dedicated development database for the first integration check. |
| `BRIER_INGEST_TOKEN` | A generated shared bearer token, identical in the app and the CLI/Robot CI sender. |
| `META_API_KEY` | Meta API access entitled to Muse Spark Contributor and SAM 3.1. No fine-tuning credential is needed. |
| Modal authentication (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`, or existing local Modal login) | Access to the existing private model volume and any later authorized serving test. |
| `MODAL_MODEL_ID`, `MODAL_MODEL_REVISION` | The pinned base identity above, supplied to the named `brier-model` secret when serving is approved. |
| `MODAL_VOLUME_NAME`, `MODAL_CHECKPOINT_DIR` | The private volume and run directory described above. |
| `MODAL_DECIDE_URL` | The actual HTTPS `/decide` URL after serving is separately approved and verified. No URL is invented here. |
| `BRIER_DATA` | Local prepared bundle path for local tests; a separate private decision-bundle GCS prefix for future Robot CI. Do not overwrite the immutable dataset release. |
| GitHub variables `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_SERVICE_ACCOUNT` | A federated identity authorized for the intended private decision-bundle/evidence prefixes. |
| GitHub variable `BRIER_INGEST_URL` | The eventual actual HTTPS `/api/ingest` address; not required for offline checks. |

Local checks without provider calls or deployments:

```sh
make check
BRIER_DATA="$PWD/data/decision-v1-audited" cargo run --locked --bin brier -- replay \
  --ledger "$PWD/data/decision-v1-audited/ledger.sqlite" \
  --compiled "$PWD/data/decision-v1-audited/compiled.jsonl" \
  --scores "$PWD/data/decision-v1-audited/replay-scores.json" \
  --out "$PWD/data/decision-v1-audited/recheck-report.json"
pnpm --dir apps/web dev
```

Replay creates a new output and fails if it already exists. Preparing a new bundle uses the documented `brier_ml.prepare` interface; it is a CPU-only artifact operation and preserves the source release. Missing database configuration should render a setup state; it is not a reason to invent dashboard data. Once the development database credential is supplied, apply the existing migration with `pnpm --dir apps/web db:migrate` before exercising real ingest. Migration changes the configured database and has not been run by this task.

To reproduce the verified bundle into a fresh output directory, build the Rust binary and supply both immutable receipts. The training receipt binds the adapter, completed epoch, scores and temperatures; the readiness receipt binds the frozen dataset. The command fails if its output already exists:

```sh
cargo build --locked --bin brier
PYTHONPATH=py .venv/bin/python -m brier_ml.prepare \
  --dataset data/rollout-v1 --runs data/runs \
  --readiness data/rollout-v1/recovery/training-readiness-d2976bf834144f1bcc70b9761fa584af7ef0894e89b282a78cf8dd0a5037a9ed.json \
  --training-receipt data/rollout-v1/recovery/training-completion-0b77c8f224e6ada19c8e5b3c371b9dd1e3b73b7c7564ff119ffce1d23f36ae3b.json \
  --output data/decision-v1-reproduced --brier target/debug/brier
```

After separately approving provider/GPU testing, verify `/decide` against held-out Rust-compiled questions and its exact bundle model version, then run one genuine `explain` request against saved failure frames. Configure `DATABASE_URL` and `BRIER_INGEST_TOKEN` in the app environment, and send an actual `brier report --push` bundle to its `/api/ingest` route. Inspect all four pages against the ingested rows. These checks need real service access; passing local fixtures is not reported as passing them.

The prepared Robot CI workflow fetches the decision bundle, verifies every named file checksum including the frozen scenario IDs and seed catalog, checks the model and separate fit/risk provenance, and selects the matching PushT or ALOHA frozen baseline before any rollout. Its live report includes the measured replay metrics. The checked-in policy template deliberately has `revision: null`: replace it with the actual deployed immutable policy snapshot only when running real Robot CI is approved. Do not dispatch the workflow merely to preview the frontend.

## Implementation audit, November 2

The four-page application and Robot CI source are implemented, with local contract tests. Credential-backed acceptance is still pending. This is not a claim that the entire submission is finished.

Audit checks passed: `make check` (126 Rust, 492 root Python and 15 web tests, plus lint, typecheck and production build); 224 separate Guide 1 tests; 310 separate Guide 2 CPU tests. Rust reproduced all 15,360 frozen training records byte for byte. A temporary-fixture cross-stage probe covered all sixteen scenarios in each environment through decisions, reports and web ingest validation; its scoring endpoint was local and its episodes deterministic fixtures, so it is not evidence of live provider behavior. The fresh private `data/decision-v1-audited` bundle verifies all nine file hashes, including the frozen scenario catalog; the prior bundle and released dataset are preserved.

| Area | Current evidence | Remaining acceptance |
| --- | --- | --- |
| Dataset, Rust labels/prompts, trained model | Independently verified private release; one epoch, 249 steps; exact adapter and seven-fit/nineteen-risk partition | No new training or data changes are required for integration testing. |
| Four web pages and ingest | Responsive scoped design, atomic ingest contract and measured-report validation | Apply the existing migration to a development Neon database; ingest actual rows and inspect the populated pages. |
| Native Robot CI and failure evidence | Frozen scenario IDs/seeds, native evaluation semantics, staged imports, raw frame/mask contracts and PR metadata | Real vLLM LoRA load, Muse/SAM requests and a complete policy PR run have not been exercised. |
| Model acceptance | Full cascade: test ECE 0.128743, 0% hours saved, 18/20 diagnostic splits. Fine-tuned-only: ECE 0.150758, 28.66% hours saved, 15/20 splits. Cheap heuristic: 1.30% hours saved. | ECE target fails. The cascade does not beat the heuristic on savings. The alternate row is not a replacement selected from held-out results. Explainer top-two accuracy is unmeasured. |
| Demo and submission | Source, private artifacts and setup instructions are prepared | Three actual timed PR demos, public evidence permissions, deployed services and the recorded three-minute video remain unfinished. Deployment and publication stay on hold. |

Future live evaluations use the same native Guide 1 rollout path and pinned policies as collection. Guide-native policy parameters are `s` for action scale, `b` for action bias, `shift`/`angle` for camera shift and `fraction` for occlusion; the other family parameter names are unchanged. The workflow does not silently reinterpret legacy `scale`, `drift`, `dx`/`dy` or rectangle parameters as these trained semantics. The frozen release and existing trained artifacts remain unchanged.

The PDF's sign-in, onboarding, Scenarios and Settings workflows remain deferred under the user's four-page scope decision. A missing demo recording is not replaced with a fabricated video or sample evaluation.

## Platform and results

The Node-runtime app in `apps/web` has `/`, `/runs`, `/runs/[id]`, `/models`, and `POST /api/ingest`. It uses shadcn/ui and Drizzle with Neon HTTP for every database query. `DATABASE_URL` is loaded lazily; there is no build-time connection. The ingest route uses `BRIER_INGEST_TOKEN`, validates relationships and HTTPS asset URLs, and atomically replaces one evaluation's rows. Database migrations are checked in but have not been applied to a hosted database.

The frontend follows the supplied design within the frozen four-page scope; sign-in, onboarding, Scenarios and Settings remain outside this build. Missing measurements display as pending. The `/models` table and charts consume actual ingested Robot reports. Private local artifacts are ready for integration; public image access, dataset/adapter publication, hosted deployment and a demo video remain on hold.

Software CI tasks T26/T27 remain gated on the Nov 2 Robot acceptance checklist: test ECE at most 0.05; better evaluation-hour savings than the cheap heuristic at `alpha = 0.05`; at least eighteen of twenty replay splits within target; all three end-to-end PR demos under fifteen minutes each; and explainer top-two accuracy above sixty percent. The measured ECE and cascade-savings gates fail; live-demo and explainer gates remain unverified. No Software adapter or platform tab is presented as available.

## Built during the hackathon

The first recorded project commit is [`916b0f4`](https://github.com/SiddharthManjul/Brier/commit/916b0f4), dated **October 29, 2026**. Available repository history records no Brier project commit before October 27. The Robot CI and platform implementation is recorded in one-file commits; the private Guide 1 dataset and authorized Guide 2 one-epoch training are complete.

Selected initial implementation commits by task (later implementation, follow-up fixes, and tests are in the full log):

| Task | Commit |
| --- | --- |
| T01–T02, scaffold and ledger | `916b0f4` |
| T03, simulation runner | `2a412f7` |
| T04, change families | `bc82d4c` |
| T05, scenarios | `cee3a2e` |
| T06, timing benchmark | `78dd0c7` |
| T07, resumable rollout shards | `a7574e9` |
| T08, spot runner | `9bfd6e2` |
| T09, baselines | `37e4df4` |
| T10, authoritative labels | `e827eee` |
| T11, prompt compiler | `8e07b5f` |
| T12, calibrated LightGBM | `68798d7` |
| T13, measured evaluation report | `0db2101` |

Use `git log --reverse --oneline` for the actual complete commit list. Real rollout collection and the authorized one-epoch GPU training are complete. Real provider API validation, serving smoke tests, hosted migrations/deployment, public demo PRs and video recording remain separate work; source integration does not complete those tasks.

The build plan credits LeRobot, conformal risk control, Meta's Diff Risk Score and Predictive Test Selection, TypeSafe Jev/RLCD, and RLCR's Brier-score rewards as prior work. No comparative performance claim is made.
