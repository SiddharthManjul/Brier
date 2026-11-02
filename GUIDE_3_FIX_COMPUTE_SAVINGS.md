# Guide 3: Make Brier save real compute, with the guarantee intact

Owner: Brooklyn. Executor: Codex. Source of truth for what this guide does not restate: the Brier build plan, `AGENTS.md`, `SCOPE.md`, Guide 1 and Guide 2. Where this guide is more specific, it wins, and Brooklyn authorizes every file, flag and dependency it names.

## The goal

A calibrated decision model is only worth anything if it cuts the expensive evaluations a team pays for while stating, with a guarantee, what that cut can cost them. Today the cascade runs every scenario, so the savings are 0%. This guide's target is the largest savings the data supports at alpha = 0.05, with the guarantee still valid. A result Meta's engineers will not see through has to look like their own Predictive Test Selection result: a big cut in test cost at a fixed, stated rate of faulty changes caught (they report a factor of two lower infrastructure cost with over 99.9% of faulty changes still reported).

## Why it runs everything today

Brier picks its skip threshold with conformal risk control over calibration **changes**:

```
lambda_hat = sup { lambda : n/(n+1) * R_hat(lambda) + B/(n+1) <= alpha },   B = 1
```

The `B/(n+1)` term is a fixed finite-sample penalty. Room left for misses at each n:

| Calibration changes n | Max R_hat at alpha 0.05 | at alpha 0.10 |
| --- | --- | --- |
| 16 | none: lambda_hat = 0, run everything | 0.044 |
| 19 | 0.000 | 0.053 |
| 26 | 0.014 | 0.065 |
| 40 | 0.026 | 0.078 |
| 64 | 0.035 | 0.086 |
| 100 | 0.041 | 0.091 |
| 150 | 0.043 | 0.093 |

The plan's 60/20/20 split leaves about 13 calibration changes per env. That is the "none" row, so running everything is the mathematically correct output. On top of that, the plan's loss counts every missed regressed **scenario**, which is much stricter than what a CI gate needs. Both are design problems, fixed below. Code bugs that cause the same symptom are ruled out first (S1, S2).

## The five levers (all of them, not a menu)

| Lever | What changes | Why it is valid | Where savings come from |
| --- | --- | --- | --- |
| L1 Change-level guarantee | The primary guarantee becomes "expected fraction of regressing changes that pass Brier undetected <= alpha". The scenario-level miss rate is still measured and reported | This is the verdict the gate actually makes, and the metric Meta's PTS reports ("faulty changes reported"). Conformal risk control works for any bounded loss that is monotone in lambda; the change-level miss loss is both | A regressing change only needs one of its regressed scenarios selected, so lambda can be far higher |
| L2 Sequential expensive runs with early stop | Selected scenarios run in descending `p_regress`; the first confirmed regression fixes the verdict at BLOCK and the remaining expensive runs are cancelled. "Run full suite" stays available on demand | After a confirmed regression the change-level loss is 0 whatever else runs, so stopping cannot create a miss | Regressing changes usually cost 1 or 2 expensive runs instead of all selected ones |
| L3 More calibration changes | Production lambda calibrates on all held-out changes; extra changes are rolled out only if that pool is under 50 (S3 table) | Same generator distribution, never seen in training, never chosen by looking at test results | n moves from the "none" row to about 100+ |
| L4 Adaptive cheap tier | Score every scenario first with the no-evidence probabilities (`noev`, already trained). Run the cheap tier only where `noev` is not confident | Both variants are trained and calibrated; the confidence cut-offs are fixed constants, and the guarantee is computed on the policy exactly as executed, cheap-tier decisions included | The cheap tier's fixed 7.5% x 16 = 1.2 expensive-equivalents per change shrinks |
| L5 Escalation only below lambda, capped | Muse Spark only sees skipped pairs just below lambda and adds at most 2 runs per change | It can only add runs, so the guarantee is untouched | Stops T3 from re-adding what the policy skipped |

T0 already clears benign changes (about 35% of the library) with zero cheap and zero expensive runs. Make sure the replay counts that, because it is the cheapest win and it is real.

## The plan

| Step | What | Time box (IST) |
| --- | --- | --- |
| S1 | Diagnose: reasons on every decision, independent Python reimplementation, cross-check with Rust | Nov 2, 2 h |
| S2 | Fix any bug found | Nov 2, 1 to 2 h |
| S3 | Only if S1 shows pooled held-out n under 50: roll out extra held-out changes in the background while S4 is coded | start right after S1, done by Nov 3 12:00 |
| S4 | Implement L1, L2, L4, L5 and the policy-as-executed replay | Nov 2 to Nov 3 morning |
| S5 | Score new changes, recompute lambda, replay, report | Nov 3 by 16:00 |
| S6 | Re-verify demo PRs, update platform numbers, hand over for credentials | Nov 3 by 20:00 |

Hard rules:

- Nothing is chosen by looking at test results. Every threshold comes from calibration data or from the replay's random splits.
- Never compute conformal losses per scenario pair to inflate n (scenarios inside a change are not exchangeable).
- New changes come from the same generator distribution as the original library (35% benign, same family mix and magnitudes). Never add benign-only changes to grow n.
- The guarantee printed anywhere names its level: "change-level: expected regressing changes missed <= 5%". The scenario-level miss rate is printed next to it. Never present the change-level number as a scenario-level one.
- No new CLI subcommands. New flags only where a card names them.

---

## S1 Diagnose

### Card F1: decision reasons in the ledger

```
TASK F1: decision reasons
Goal: Every decision row records why its action was chosen.
Files (only these): the Rust file that implements brier decide's policy (T18, for example crates/brier-core/src/policy.rs), its test file
Dependencies allowed: none new
Line budget: 80
```

Spec: set `decisions.reason` (the column already exists) to exactly one of:

- `t0_benign` (rules cleared it)
- `policy_above_lambda` (`p_regress >= lambda`)
- `policy_below_lambda` (skipped)
- `t3_added` (Muse Spark asked for an expensive run)
- `t3_no_change`
- `exploration` (epsilon draw)
- `t3_capped` (escalation over the per-change cap)
- `early_stop_not_needed` (selected, not run because the verdict was already BLOCK)
- `cheap_skipped_confident_low` (adaptive cheap tier skipped the pair)
- `fallback_error` (any path where a model call failed and the code ran expensive to be safe)

`fallback_error` must also log the error. If such a fallback path exists today and was not requested by a card, it violates `AGENTS.md` rule 7; keep it only as a logged, counted reason and report how often it fired.

Acceptance: `cargo test` for the policy file, plus a test where each reason is produced once.

### Card F2: independent policy diagnostics (Python)

```
TASK F2: policy_diag
Goal: Recompute lambda_hat, the risk curve and savings independently from the probability files, and compare with the Rust policy.
Files (only these): py/brier_ml/policy_diag.py, py/brier_ml/tests/test_policy_diag.py
Dependencies allowed: numpy
Line budget: 220
```

Spec. Inputs: the compiled JSONL (for labels, splits and change ids), the probability files the policy uses (`probs_temp/<split>.jsonl` from Guide 2, question `regressed`, variant `ev`), and optionally `ledger.sqlite` (for the Rust decisions and reasons).

Outputs, printed and written to `data/diag/policy_diag.json`:

1. Which probabilities the Rust policy actually reads: model name, variant, temperature applied or not. Compare 20 pairs' `p_regress` from the ledger's `decisions.probs_json` with the probability file. Any mismatch over 1e-6 is a finding.
2. `p_yes` sanity: on calibration, mean `p_yes` for regressed pairs vs not regressed pairs, and AUROC. If the regressed mean is lower, the letter mapping is flipped.
3. n per env and pooled, alpha in {0.05, 0.10, 0.20}, and for each: `lambda_hat` by the formula above on a grid of every distinct calibration `p` value plus 0 and 1 (exact, no coarse grid), `R_hat(lambda)` at 41 points from 0 to 1, and the predicted fraction of expensive runs on calibration.
4. The same `lambda_hat` read from the Rust implementation (run `brier decide` or the function the replay calls on the calibration set) and the absolute difference. Over 1e-9 is a finding.
5. The calibration changes with the largest `L_i` at a moderate lambda (0.1): change id, family, the regressed scenarios and their `p_regress`. These are the model's confident misses.
6. From the ledger, if replay or demo decisions exist: counts per `reason`, per env.

Acceptance: `pytest -q py/brier_ml/tests/test_policy_diag.py` with a synthetic case where the answer is known: n = 30 changes, probabilities that perfectly separate, so `lambda_hat` at alpha 0.05 equals the smallest regressed `p` and the expensive fraction equals the regressed fraction. A second synthetic case with n = 15 must give `lambda_hat = 0` at alpha 0.05.

### Runbook S1

```bash
python -m brier_ml.policy_diag --compiled data/compiled.jsonl --probs data/runs/glimmer-ft-v1/probs_temp --ledger data/ledger.sqlite --out data/diag
```

Classify the root cause with this table and write the result at the top of the final message:

| Finding | Root cause | Go to |
| --- | --- | --- |
| Rust and Python `lambda_hat` differ, or the comparison or search direction is wrong | bug | S2 |
| The policy reads zero-shot, `noev` or untempered probabilities, or `p_no` | wiring bug | S2 |
| Most expensive runs have reason `t3_added` | escalation | S4 (card F7) |
| Any `fallback_error` | wiring or credential bug | S2 |
| n < 20 at the level used (per env or pooled) | sample size | S3 and S4 |
| n >= 26 pooled, but `R_hat` at small lambda already exceeds the allowed value | model misses on some scenarios | S4 (the change-level guarantee only needs one regressed scenario per change caught) |

More than one row can apply. Fix in table order.

---

## S2 Fix bugs

Use the plan's own task ids for commits (T18 policy, T19 replay, T22 escalation). Typical fixes, each one a small diff with a test that fails before and passes after:

- `lambda_hat` must be the **largest** lambda on the exact grid that satisfies the bound; if none does, it is 0 and the summary must say "calibration set too small for alpha".
- Expensive runs iff `p_regress >= lambda_hat`.
- `L_i` divides by the change's regressed scenarios only, and is 0 for changes with none. Benign and non-regressing changes still count in n.
- `p_regress` is the probability of the option whose text is `yes`, never a fixed letter (train options were shuffled; check that the policy never reads train-format rows).
- The policy reads fine-tuned, temperature-scaled, `ev` probabilities. `ev` is correct because Stage A runs the cheap tier on every scenario before scoring.
- A model-call failure fails the decision loudly. No silent "run expensive" fallback.

Re-run S1 after S2. S3 starts right after S1 if its table says so; S4 starts once no bug row applies.

---

## S3 More calibration changes (only as many as the data needs)

Nothing is retrained and no existing data is rebuilt. The model, temperatures, train split and every existing rollout stay as they are. This step only adds held-out changes, and only if the pooled held-out count is too small. Decide from S1's printed pooled held-out n (calibration plus test, non-OOD, both envs):

| Pooled held-out n from S1 | Extension |
| --- | --- |
| 50 or more | Skip S3 entirely |
| 30 to 49 | `--extend 20` per env |
| under 30 | `--extend 40` per env |

The numbers below use `--extend 60` as the example; use the value from this table.

### Card F3: extend the change library

```
TASK F3: change library extension
Goal: Generate additional changes with new ids from the same distribution, assigned to the held-out pool.
Files (only these): py/brier_sim/changes.py, its test; the Rust compile file (T11) and its test
Dependencies allowed: none new
Line budget: 80
```

Spec:

- `python -m brier_sim.changes --env <env> --extend 60 --seed 2027 --write` appends 60 changes per env with ids continuing after the last existing id. Same family mix, magnitude grid (`magnitudes_<env>.json`) and 35% benign share. Existing changes are never modified.
- `brier compile --heldout-from <first new id>` (single new flag) assigns every change with id at or above it to `calibration`, or to `ood` if its families include an OOD family. Train is untouched, so no retraining is needed.

### Runbook S3

1. Generate and upload the appended `changes_<env>.jsonl`.
2. Dispatch rollouts for the new changes only (baselines exist): Guide 1 runbook R7, ALOHA first because it is faster. Put every free GitHub job and the spot VMs on it.
3. As each env completes: Guide 1 R8 to R10 (aggregate, `brier label`, `brier compile --heldout-from ...`).
4. Score the new held-out examples with the fine-tuned adapter on Modal (Guide 2 `score_fn` restricted to the new ids; add an `--ids-from` file argument to `score.py` if it is missing). About $1. Apply the existing `temps.json`; do not refit temperatures.

Cut-off: whatever is fully labelled by Nov 3, 12:00 IST goes in. Partial changes (some scenarios missing) are excluded entirely.

Pooled held-out n after this step at full size: about 26 original calibration + 26 original test + about 96 new non-OOD changes, so roughly 150. With a smaller original library, proportionally less; the bound table at the top shows what each n buys.

---

## S4 Implement the levers

### Card F4: change-level risk and the policy as executed

```
TASK F4: change-level conformal risk
Goal: Calibrate lambda on the change-level miss loss of the policy exactly as it executes, and keep the scenario-level loss as a reported metric.
Files (only these): the Rust policy file (T18), the Rust replay file (T19), their tests
Dependencies allowed: none new
Line budget: 200
```

Spec:

- For a calibration change i and threshold lambda, simulate the policy as executed (T0, the adaptive cheap tier from F6, the threshold, L2's ordering; exploration and escalation excluded because they can only add runs and are not counted toward the guarantee). Define:
  - `L_change_i(lambda) = 1` if change i has at least one regressed scenario and **none** of its regressed scenarios would be run expensive; otherwise 0.
  - `L_scen_i(lambda)` = the plan's original loss (fraction of the change's regressed scenarios not run), reported only.
- Both losses are non-increasing as lambda falls. Assert this on the calibration data at runtime and fail loudly if it is violated (it would mean the simulated policy is not monotone in lambda, which would break the guarantee).
- `lambda_hat` = the largest lambda on the exact grid (every distinct calibration `p` plus 0 and 1) with `n/(n+1) * mean(L_change) + 1/(n+1) <= alpha`, pooled over envs and over all held-out changes (production). 0 if none.
- `brier decide --alpha <a> --guarantee change|scenario` (default `change`). `scenario` keeps the old behavior for comparison.
- Replay: 20 random 75/25 partitions of the held-out pool (seeds 0 to 19). Per alpha in {0.05, 0.10} and per guarantee level, report: eval-hours used and saved **counting cheap and expensive runs** (cheap = 0.075 of an expensive run, from the tier table), expensive runs per change (mean, by benign, regressing and non-regressing), realized change-level miss rate and scenario-level miss rate (mean and worst split), splits meeting alpha out of 20, change recall. OOD separately with the production lambda.
- Exploration (epsilon = 0.05) stays and is reported separately as overhead.

Acceptance: `cargo test`; a fixture where the change-level lambda is strictly higher than the scenario-level one; a fixture where early stop (F5) changes cost but not `L_change`.

### Card F5: sequential expensive runs with early stop

```
TASK F5: early stop
Goal: Run selected expensive scenarios in descending p_regress and stop at the first confirmed regression.
Files (only these): the Rust decide/report path (T18, T23), .github/workflows/robot-ci.yml, their tests
Dependencies allowed: none new
Line budget: 140
```

Spec:

- `brier decide` emits the selected scenarios as an ordered list (descending `p_regress`, ties by scenario id).
- `robot-ci.yml` runs them in waves of `W = 4` matrix jobs (enough parallelism to keep wall time low). After each wave, `brier report` checks for a confirmed regression (the plan's label rule on the run). If one is confirmed, the workflow cancels the remaining waves, posts BLOCK and lists the unrun selected scenarios as "not needed: verdict already BLOCK". Otherwise it runs the next wave.
- The check summary gets a "Run full suite" instruction: re-run the workflow with input `full=true`, which runs every selected scenario without early stop. Nothing else changes.
- Replay simulates the same waves so its cost numbers match the live gate.

Acceptance: workflow lint (`actionlint` if present, else `gh workflow view`), a replay fixture with a regression in wave 1 that costs exactly 4 expensive runs.

### Card F6: adaptive cheap tier

```
TASK F6: adaptive cheap tier
Goal: Use no-evidence probabilities to skip the cheap tier where the model is already confident.
Files (only these): the Rust policy file (T18), the replay file (T19), their tests
Dependencies allowed: none new
Line budget: 120
```

Spec:

- Stage A' (new, before Stage A): score every pair with the `noev` variant. A pair is "confident low" if `p_noev < lambda_low` and "confident high" if `p_noev >= lambda_high`. Confident-high pairs go straight to the expensive queue, confident-low pairs are skipped, and only the rest run the cheap tier and get `ev` scores as before.
- `lambda_low = 0.02` and `lambda_high = 0.90` are fixed constants, chosen now before any held-out result is seen, and never tuned. That keeps the guarantee exact: F4 computes `lambda_hat` on the policy as executed with these constants in place, so no threshold is selected twice on the same calibration data. If the replay later shows the cheap tier still dominates cost, report it; do not tune these on calibration or test.
- The serving endpoint already takes either variant (the prompt differs only in the CHEAP_TIER line), so no model change is needed.

Acceptance: replay fixture where a confident-low pair skips the cheap tier and the reported cheap cost drops.

### Card F7: escalation only below lambda, capped

```
TASK F7: escalation band and cap
Goal: Send to Muse Spark only skipped pairs just below lambda; cap additions per change.
Files (only these): the Rust escalation file (T22), its test
Dependencies allowed: none new
Line budget: 60
```

Spec: escalate only pairs with `lambda_hat - 0.10 <= p < lambda_hat`; at most 2 additions per change (`ESCALATION_CAP = 2`, a constant), the 2 highest `p` first; the rest are recorded as `t3_capped`. Escalation never removes a run.

Acceptance: test with 5 band pairs all answering `run_expensive`: exactly 2 added.

### Runbook S4

```bash
cargo build --release
brier replay --ledger data/ledger.sqlite --probs data/runs/glimmer-ft-v1/probs_temp --alphas 0.05,0.10 --guarantee change --out data/report_change.json
brier replay --ledger data/ledger.sqlite --probs data/runs/glimmer-ft-v1/probs_temp --alphas 0.05,0.10 --guarantee scenario --out data/report_scenario.json
python -m brier_ml.policy_diag --compiled data/compiled.jsonl --probs data/runs/glimmer-ft-v1/probs_temp --ledger data/ledger.sqlite --out data/diag --pooled-heldout --guarantee change
```

F2 gains `--pooled-heldout` and `--guarantee` in the same session as F4, and must reproduce Rust's `lambda_hat` to 1e-9 for both guarantee levels.

---

## S5 Recompute and report

After S3's new changes are scored:

1. Re-run the S4 runbook. Gate: Rust and Python agree; the change-level guarantee holds in at least 18 of 20 splits at alpha 0.05.
2. If it holds, the headline is the alpha 0.05 change-level result: "X% of evaluation hours saved; expected regressing changes missed at most 5% (realized: Y%); scenario-level misses Z%". X counts cheap, expensive and exploration runs.
3. If it does not hold in 18 of 20 splits at 0.05, stop and report the per-split numbers before writing anything public. That would mean the probabilities are miscalibrated on held-out changes or a policy component is not monotone, and it needs S1 again, not a different alpha.
4. Update `report.json`, `/models`, README results and the check summary from the replay output only. Nothing typed by hand.
5. README "How Brier saves compute" section, in this order: T0 clears benign changes for free; the adaptive cheap tier; the calibrated threshold; early stop after the first confirmed regression. One line each with its share of the savings from the replay's breakdown.
6. README "Honest limits" gains: the guarantee is change-level and assumes new changes are exchangeable with the calibration changes; the scenario-level miss rate is reported alongside.

---

## S6 Re-verify the demo, then hand over

| Demo PR | Expected |
| --- | --- |
| #1 "Bump logging, no policy change" | T0 pass, 0 cheap and 0 expensive runs |
| #2 "Move inference to edge box" | BLOCK after the first wave or two; the summary shows expensive runs used, runs skipped and runs not needed after the verdict |
| #3 "Re-mount wrist camera" | BLOCK on low-light scenarios; the explainer names the visual cause |

Then `brier report --push` and confirm `/runs`, `/runs/[id]` and `/models` show the new numbers.

### Handover (Brooklyn fills these in to test)

| Variable | Where | Used by |
| --- | --- | --- |
| `DATABASE_URL` | Vercel env and `apps/web/.env.local` | platform (Neon Postgres) |
| `BRIER_INGEST_TOKEN` | Vercel env, demo repo secret, local shell | `POST /api/ingest`, `brier report --push` |
| `META_API_KEY` | demo repo secret, local shell | Muse Spark, SAM 3.1 |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | demo repo secrets, local shell | the `/decide` endpoint |
| `BRIER_DECIDE_URL` | demo repo variable | `brier decide` |

Codex leaves them unset, lists them in its final message and writes the exact test commands for this repo (they depend on how T18, T23 and T24 were built): run the platform, run `brier decide` on demo PR #2's change card, re-run `robot-ci.yml` on PR #2, open the new run page.

---

## Final message checklist

- S1 root cause rows and the S2 fixes.
- n before and after S3; `lambda_hat` per alpha and guarantee level; Rust and Python agreement.
- Replay tables for both guarantee levels, the savings breakdown by lever, and the headline.
- Demo PR results and their costs.
- Extra spend: GitHub minutes (free), GCP spot, Modal.
- The handover table with nothing filled in.
