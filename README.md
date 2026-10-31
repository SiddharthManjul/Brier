# Brier

Brier predicts which robot evaluation scenarios will regress after a policy change, then selects expensive simulation runs under a calibrated risk policy.

**Status:** the ungated Robot CI and platform code is implemented locally. Local checks pass: 114 Rust tests, 443 Python tests, 11 web tests, formatting, lint, type checking, and the production build. Guide 1's rollout, aggregation, label and prompt code is implemented; real GitHub timing is in progress, and collection and dataset quality gates remain pending. Guide 2 training code is in progress. There is no measured performance headline, published evaluation dataset, released adapter, deployed service, or recorded end-to-end demo yet. Passing local checks does not establish that a trained model saves evaluation hours.

Repository: [SiddharthManjul/Brier](https://github.com/SiddharthManjul/Brier). The [build plan](<Brier Build Plan (Robot CI first, Software CI second).md>), [scope](SCOPE.md), and [task rules](AGENTS.md) define the boundaries.

## How it works

1. A policy change and a scenario become a structured change card. PushT and AlohaTransferCube are the supported simulation environments.
2. Three cheap episodes provide evidence and save one actual final image each. The expensive tier uses twenty episodes; the stored reference baseline uses fifty per scenario.
3. The decision cascade combines identity rules, LightGBM with isotonic calibration, and option probabilities from a pinned language model. The three questions are regression, failure mode, and success bucket. Rust `brier compile` produces every prompt; Rust `brier label` computes the authoritative labels.
4. Temperature scaling fits one positive temperature per question. A separate change-level conformal risk calibration selects the threshold. Scenarios at or above that threshold run; five-percent exploration also samples below it with logged propensities.
5. The ledger records decisions and measured runs. `brier report` produces a Markdown verdict and, when complete, a platform ingest bundle. The four-page web app shows runs, actual failure assets, and measured model reports.

The world-model role is predicting what an evaluation would show from a change, scenario, and available evidence. It does not generate replacement simulation trajectories. The plan assigns Glimmer to decision scoring, Muse Spark Contributor to escalation and explanation, and SAM 3.1 to failure-frame perception. Meta's [Glimmer model guide](https://dev.meta.ai/docs/muse-glimmer/get-the-model) identifies the published `meta-models/Muse-Glimmer-30B` resource: a multimodal model with a nested text decoder. `MODAL_MODEL_ID` and an immutable `MODAL_MODEL_REVISION` remain required; the code does not automatically choose this resource or a revision. No trained Brier adapter is claimed here.

SAM segmentation, Muse Spark explanation/escalation, and the Rust `brier explain` path pass local integration checks; real provider calls remain pending. The clients use the documented `sam-3.1` and `muse-spark-1.3-contributor` identities and send the project's `META_API_KEY` as the provider Bearer credential. SAM decoding uses the official `meta-sam-parser==0.1.0` dependency. These APIs expose version-named identities here; no immutable SAM/Muse revision is claimed. Contracts come from Meta's [SAM overview](https://dev.meta.ai/docs/sam/overview), [structured-output documentation](https://dev.meta.ai/docs/structured-output), and [image-understanding documentation](https://dev.meta.ai/docs/image-understanding).

Rust compiles all explanation and escalation text. Python first requests the three Rust-named SAM concepts, unions matched instances, saves actual overlays, and returns measured geometry. Rust then compiles the Muse prompt from observed frames, geometry, scenario parameters, and allowlisted telemetry; the explainer never receives the change card or hidden cause-family label. A missing required mask fails clearly. Muse receives interleaved raw/overlay images and returns structured evidence-frame indices and an explanation.

T3 is called only for T2 regression probabilities in `[0.1, 0.9]`. Both `run_expensive` and `run_more_cheap` recommendations add a full expensive evaluation, preserving the original three-episode cheap feature vector used in calibration. `no_change` preserves the existing decision; T3 cannot remove selected evaluations. API costs remain `null` unless measured. T0 can separately prove an unchanged policy identity without making a statistical claim. The Robot CI workflow now includes explanation, evidence upload, and platform ingest; an operational gate still requires deployment configuration and real end-to-end evidence.

## What the guarantee means

For a change, loss is the fraction of its regressed scenarios that the threshold skips; a change with no regressions has zero loss. With a scorer fixed independently of risk calibration, and exchangeable calibration and future changes, the policy bounds **expected change-level missed fraction** at `alpha`. It does not guarantee every change's realized loss or validity under arbitrary distribution shift.

Training changes, probability-calibration changes, risk-calibration changes, and held-out test/OOD changes must stay separate. The frozen `calibration` split must itself be divided by change: isotonic/temperature fitting uses one pool, conformal risk calibration uses another. Reusing a probability-fitting change for risk calibration invalidates the required independence. The main test/OOD table uses a threshold fitted without those labels. The separate twenty-split diagnostic repartitions the held-out calibration/test pool; probability-fitting changes remain excluded.

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

## Reproduce a measured replay

A complete replay dataset is not published yet. Set `BRIER_DATA` to an existing **local** directory containing:

| File | Required contents |
| --- | --- |
| `ledger.sqlite` | Authoritative labels, scenario baselines, exactly one three-episode cheap result per retained pair, and complete measured twenty-episode expensive results for the offline counterfactual. |
| `compiled.jsonl` | Rust-compiled records with `id`, `split`, `question`, `options`, `prompt`, and `label`; splits are frozen by change. |
| `replay-scores.json` | Independently fitted strategy predictions with their training and probability-calibration provenance. |

The scores file has one top-level `strategies` object. Each strategy has `model_version`, `training_change_ids`, `probability_calibration_change_ids`, and `probabilities`. The two provenance fields are arrays of change IDs. `probabilities` maps retained regression/cheap compiled IDs, such as `c001/pusht-s01/regressed/cheap`, to finite numbers in `[0, 1]`. Scores are required for all retained risk-calibration, test, and OOD pairs. Optional `latencies_seconds` and `costs_usd` maps must cover every evaluation pair if supplied; absent measurements remain absent. Unknown fields are rejected.

With those real artifacts already prepared:

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

Defaults are `--alpha 0.05 --seed 0`. The output must not already exist. Replay reads actual labels and durations, adds run-all, cheap-heuristic, and random-at-equal-expected-budget baselines, and reports only supplied scored strategies. Primary ablation metrics, costs, reliability bins, and threshold curves use the test cohort; OOD calibration is reported separately. The JSON includes measured hours, change recall, ECE, Brier score, NLL, AUROC when defined, and twenty seeded split results. Persisted held-out explanations can also be scored for top-one/top-two injected-family agreement without exposing the hidden family to the provider. ECE uses fifteen quantile bins with tied scores kept together. API/GPU cost and latency appear only when provided as measurements.

For artifact preparation and live decisions, inspect the implemented interfaces:

```sh
cargo run --locked --bin brier -- label --help
cargo run --locked --bin brier -- compile --help
cargo run --locked --bin brier -- card --help
cargo run --locked --bin brier -- decide --help
cargo run --locked --bin brier -- explain --help
cargo run --locked --bin brier -- report --help
```

`decide` requires `--ledger`, `--compiled`, `--calibration`, and `--change`, with optional `--alpha`. Its risk-calibration artifact binds the actual scorer versions, temperatures, T1 artifact, and separate probability/risk change IDs. A live change must be unlabelled and outside those fitting pools. Scoring needs the configured `MODAL_DECIDE_URL`; no model endpoint is embedded in the repository.

`explain --ledger ... --run ...` takes the aggregate expensive-run UUID recorded in the ledger. It processes every failed individual episode in that twenty-episode Parquet group, using each episode's eight saved keyframes, and stores the aggregate explanation in the ledger. Local images and overlays remain under `BRIER_DATA`; publishing HTTPS image URLs is a separate deployment step.

`report` requires `--ledger`, `--change`, and `--out`. The output file is Markdown; stdout is structured JSON containing `markdown`, `bundle`, and `verdict`. Optional `--metrics` attaches an existing measured replay report; `--push` takes the actual HTTPS `/api/ingest` URL and requires `BRIER_INGEST_TOKEN`. Repeated reports reuse decision-derived IDs. A regression confirmed by a selected run blocks; incomplete selected results remain pending; both return a nonzero CLI exit after writing the summary. Skipped scenarios do not expose offline labels as live observations. `hoursSavedPct` is `null` unless measured full counterfactual durations exist; evaluation-count reduction is reported separately.

Robot CI requires repository variable `BRIER_INGEST_URL` to contain the actual deployed HTTPS `/api/ingest` URL and secret `BRIER_INGEST_TOKEN` to authorize the push. This URL is deployment configuration, not a supplied public service. The workflow uploads actual evidence before `report --push` and archives its JSON stdout, ledger, and assets. With a GCS `BRIER_DATA` prefix, the report maps already uploaded relative `public/keyframes` and `public/overlays` paths to canonical GCS HTTPS URLs. Public read access for the intended evidence prefix must be configured through GCS IAM during deployment; the workflow does not make the bucket public. Local artifact publication remains a deployment task; no workflow has been run as part of this local implementation.

## Model jobs: configuration examples only

**Do not execute these examples until real GPU training is authorized and model/data configuration is verified.** They allocate remote compute; they have not been executed as part of this documentation task.

Training requires `MODAL_MODEL_ID`, a forty-hex immutable `MODAL_MODEL_REVISION`, `MODAL_VOLUME_NAME`, `MODAL_CHECKPOINT_DIR`, and `BRIER_DATA`. Data and checkpoint directories must be under the Modal Volume mount `/checkpoints`, and compiled data must already be present there. A `gs://` path must first be transferred; the trainer does not perform that transfer. The following assumes those environment variables are configured for the intended remote volume:

```sh
# Configuration example only; launches an A100-80GB job when executed.
modal run py/brier_ml/finetune.py --compiled "$BRIER_DATA/compiled.jsonl" --smoke
```

Smoke training is fifty optimizer updates. Full training uses the same command without `--smoke`, with a separate checkpoint directory; smoke and full configurations cannot resume each other's checkpoints. Training and frozen-embedding extraction include the native `MuseGlimmerForConditionalGeneration` path and its nested text configuration. The training image pins `transformers==5.15.0` and `peft==0.19.1`, matching [Transformers’ declared PEFT minimum](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/dependency_versions_table.py#L38); Glimmer QLoRA targets only text-decoder projection modules. The implementation uses rank-sixteen QLoRA, cross entropy plus Brier or ordinal ranked-probability loss, periodic checkpoints, and validation of the dataset/checkpoint configuration on resume. Compatibility and throughput still require an actual run with the configured checkpoint.

Student distillation additionally requires `MODAL_STUDENT_MODEL_ID` and `MODAL_STUDENT_MODEL_REVISION`, a compiled-file-bound teacher artifact, and exactly compatible teacher/student tokenizers. The teacher artifact must be inside `BRIER_DATA`; targets come from training rows only. No published 3B Glimmer companion has been identified in the verified model guide. The implemented distillation path is conditional on finding a compatible student; it does not establish that a 3B checkpoint or distilled artifact exists. A model's nominal size does not establish tokenizer compatibility.

```sh
# Configuration example only; also launches remote GPU compute when executed.
modal run py/brier_ml/distill.py \
  --compiled "$BRIER_DATA/compiled.jsonl" \
  --teacher "$BRIER_DATA/teacher.json" --smoke
```

[serve/modal_app.py](serve/modal_app.py) defines the Modal vLLM `POST /decide` service. It expects the named `brier-model` secret to contain the pinned base-model configuration. LoRA serving additionally requires the configured checkpoint directory and volume; the adapter must match that base revision. The Python 3.11 image uses CUDA 13.0.3 and vLLM 0.28.0, with the native text-only Glimmer path and BF16 on an A100-80GB. This replaces the earlier L4/bitsandbytes assumption: the published 30B model needs about 72 GB for BF16 serving, and the selected vLLM runtime has no bitsandbytes loader. These requirements follow the [Meta runtime guide](https://dev.meta.ai/docs/muse-glimmer/deploy) and [vLLM recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B). Exact option-token log probabilities are requested within a maximum of twenty; the current API requires the requested count to match the supplied option-token IDs. The image has not been built or deployed, and GPU memory, native adapter loading, dependency compatibility, and latency still require an authorized smoke run. An actual endpoint URL and released adapter remain pending. Frozen-head training is a separate implemented path, not evidence that either model has been trained.

## Platform and results

The Node-runtime app in `apps/web` has `/`, `/runs`, `/runs/[id]`, `/models`, and `POST /api/ingest`. It uses shadcn/ui and Drizzle with Neon HTTP for every database query. `DATABASE_URL` is loaded lazily; there is no build-time connection. The ingest route uses `BRIER_INGEST_TOKEN`, validates relationships and HTTPS asset URLs, and atomically replaces one evaluation's rows. Database migrations are checked in but have not been applied to a hosted database.

For a configured local development environment, use `pnpm --dir apps/web dev`. Missing measurements display as pending. The `/models` table and charts consume actual Robot reports: no ablation numbers, reliability plot, evaluation savings, model weights, public failure-image URLs, deployment URL, or demo-video link are invented here. Dataset, adapter, and video publication remain pending.

Software CI tasks T26/T27 remain gated on the Nov 2 Robot acceptance checklist: test ECE at most 0.05; better evaluation-hour savings than the cheap heuristic at `alpha = 0.05`; at least eighteen of twenty replay splits within target; all three end-to-end PR demos under fifteen minutes each; and explainer top-two accuracy above sixty percent. These are unverified targets. No Software adapter or platform tab is presented as available.

## Built during the hackathon

The first recorded project commit is [`916b0f4`](https://github.com/SiddharthManjul/Brier/commit/916b0f4), dated **October 29, 2026**. Available repository history records no Brier project commit before October 27. The ungated Robot CI and platform implementation is now recorded in one-file commits in the local history; Guide 1 rollout-data work is in progress.

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

Use `git log --reverse --oneline` for the actual complete commit list. Real rollout collection, GPU experiments, real provider API validation, hosted migrations/deployment, public demo PRs, and video recording remain separate work; local implementation does not complete those tasks.

The build plan credits LeRobot, conformal risk control, Meta's Diff Risk Score and Predictive Test Selection, TypeSafe Jev/RLCD, and RLCR's Brier-score rewards as prior work. No comparative performance claim is made.
