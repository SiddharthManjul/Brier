use crate::policy::{calibrate, LabeledChange, PairScore};
use crate::policy::{calibrate_executed, ExecutedChange, ExecutedPair, ExecutedPolicy, Guarantee};
use crate::replay::{execute_change, heldout_partitions};
use crate::replay::{replay, ReplayChange, ReplayPair};
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OpenFlags};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scores {
    strategies: BTreeMap<String, ModelScores>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelScores {
    model_version: String,
    training_change_ids: BTreeSet<String>,
    probability_calibration_change_ids: BTreeSet<String>,
    probabilities: BTreeMap<String, f64>,
    #[serde(default)]
    latencies_seconds: BTreeMap<String, f64>,
    #[serde(default)]
    costs_usd: BTreeMap<String, f64>,
}

struct Pair {
    id: String,
    change: String,
    scenario: String,
    split: String,
    regressed: bool,
    seconds: f64,
    heuristic: bool,
}

struct ExecutedMeasurement {
    change: ExecutedChange,
    split: String,
    benign: bool,
    seconds: BTreeMap<String, f64>,
}

/// Replays frozen fine-tuned ev/noev scores with explicit pooled held-out Guide 3 calibration.
pub fn write_guide3_report(
    ledger: &Path,
    compiled: &Path,
    probs: &Path,
    out: &Path,
    alphas: &[f64],
    guarantee: Guarantee,
    seed: u64,
) -> Result<()> {
    ensure!(!alphas.is_empty(), "at least one alpha required");
    let run_dir = probs
        .parent()
        .context("probability directory needs a run parent")?;
    let metadata: Value = serde_json::from_slice(&fs::read(run_dir.join("run_meta.json"))?)?;
    let metrics: Value = serde_json::from_slice(&fs::read(run_dir.join("metrics.json"))?)?;
    let temperatures: Value = serde_json::from_slice(&fs::read(run_dir.join("temps.json"))?)?;
    ensure!(
        metadata["calibration_partition"] == metrics["calibration_partition"],
        "probability fitting provenance mismatch"
    );
    let fitted: BTreeSet<_> = metadata["calibration_partition"]["fit_change_ids"]
        .as_array()
        .context("missing probability/model-fitting exclusion IDs")?
        .iter()
        .map(|id| id.as_str().context("invalid fitting ID").map(str::to_owned))
        .collect::<Result<_>>()?;
    ensure!(
        !fitted.is_empty(),
        "fitting exclusion IDs must not be empty"
    );
    let revision = metadata["revision"]
        .as_str()
        .context("missing model revision")?;
    let temperature = temperatures["regressed"]
        .as_f64()
        .filter(|p| p.is_finite() && *p > 0.0)
        .context("invalid regression temperature")?;
    let mut expected = BTreeMap::new();
    for line in fs::read_to_string(compiled)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        if row["question"] != "regressed" || row["split"] == "train" {
            continue;
        }
        let id = row["id"]
            .as_str()
            .context("missing compiled ID")?
            .to_owned();
        ensure!(
            id == format!(
                "{}:{}:regressed:{}",
                row["change_id"].as_str().context("missing change ID")?,
                row["scenario_id"].as_str().context("missing scenario ID")?,
                row["variant"].as_str().context("missing variant")?
            ) && matches!(row["variant"].as_str(), Some("ev" | "noev"))
                && matches!(row["split"].as_str(), Some("calibration" | "test" | "ood")),
            "invalid Guide 3 compiled identity"
        );
        ensure!(expected.insert(id, row).is_none(), "duplicate compiled ID");
    }
    for id in &fitted {
        ensure!(
            expected
                .values()
                .any(|row| row["change_id"] == *id && row["split"] == "calibration"),
            "fitting exclusion must name an original calibration change"
        );
    }
    let mut predictions = BTreeMap::new();
    let mut model = None;
    for split in ["calibration", "test", "ood"] {
        for line in fs::read_to_string(probs.join(format!("{split}.jsonl")))?.lines() {
            let row: Value = serde_json::from_str(line)?;
            if row["question"] != "regressed" {
                continue;
            }
            let id = row["id"].as_str().context("missing probability ID")?;
            let source = expected
                .get(id)
                .context("probability ID not present in compiled data")?;
            ensure!(
                row["split"] == split
                    && row["split"] == source["split"]
                    && row["variant"] == source["variant"]
                    && row["options"] == source["options"]
                    && row["label"] == source["label"]
                    && row["revision"] == revision
                    && row["temperature"].as_f64() == Some(temperature),
                "probability provenance/temperature mismatch for {id}"
            );
            let name = row["model"]
                .as_str()
                .filter(|s| !s.is_empty())
                .context("missing model identity")?;
            ensure!(
                row["model"] == metrics["model"],
                "score model is not the completed fine-tuned model"
            );
            if let Some(previous) = &model {
                ensure!(previous == name, "mixed probability model identities");
            } else {
                model = Some(name.to_owned());
            }
            let options = row["options"].as_array().context("missing options")?;
            ensure!(
                options.len() == 2
                    && options.iter().filter(|v| **v == "yes").count() == 1
                    && options.iter().filter(|v| **v == "no").count() == 1,
                "regression options must contain yes/no once"
            );
            let yes = options.iter().position(|v| *v == "yes").unwrap();
            let values = row["probs"]
                .as_object()
                .context("missing letter probabilities")?;
            ensure!(
                values.len() == 2
                    && values.values().all(|v| v
                        .as_f64()
                        .is_some_and(|p| p.is_finite() && (0.0..=1.0).contains(&p)))
                    && (values.values().map(|v| v.as_f64().unwrap()).sum::<f64>() - 1.0).abs()
                        < 1e-6,
                "invalid letter probabilities"
            );
            let p = values
                .get(if yes == 0 { "A" } else { "B" })
                .and_then(Value::as_f64)
                .context("missing yes probability")?;
            ensure!(
                predictions.insert(id.to_owned(), p).is_none(),
                "duplicate probability ID"
            );
        }
    }
    ensure!(
        expected.len() == predictions.len(),
        "incomplete ev/noev probabilities"
    );
    let connection = Connection::open_with_flags(ledger, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut changes: BTreeMap<String, ExecutedMeasurement> = BTreeMap::new();
    for (id, row) in &expected {
        let change = row["change_id"].as_str().unwrap();
        if row["variant"] != "ev" || fitted.contains(change) {
            continue;
        }
        let scenario = row["scenario_id"].as_str().unwrap();
        let split = row["split"].as_str().unwrap();
        let label: i64 = connection.query_row(
            "SELECT regressed FROM labels WHERE change_id=?1 AND scenario_id=?2",
            params![change, scenario],
            |r| r.get(0),
        )?;
        ensure!(matches!(label, 0 | 1), "invalid authoritative label");
        let label_index = match row["label"].as_str() {
            Some("A") => 0,
            Some("B") => 1,
            _ => anyhow::bail!("invalid compiled label"),
        };
        ensure!(
            (row["options"][label_index] == "yes") == (label == 1),
            "compiled/authoritative label disagreement"
        );
        let (count, episodes, seconds): (i64, i64, f64) = connection.query_row("SELECT count(*),sum(episodes),sum(wall_seconds) FROM runs WHERE change_id=?1 AND scenario_id=?2 AND tier='expensive'", params![change, scenario], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        ensure!(
            count == 1 && episodes == 20 && seconds.is_finite() && seconds > 0.0,
            "incomplete measured expensive result"
        );
        let card: String = connection.query_row(
            "SELECT card_json FROM changes WHERE change_id=?1",
            [change],
            |r| r.get(0),
        )?;
        let card: Value = serde_json::from_str(&card)?;
        let benign = card["families"]
            .as_array()
            .is_some_and(|f| f.len() == 1 && f[0]["family"] == "benign");
        let entry = changes
            .entry(change.to_owned())
            .or_insert_with(|| ExecutedMeasurement {
                change: ExecutedChange {
                    change_id: change.to_owned(),
                    pairs: Vec::new(),
                },
                split: split.into(),
                benign,
                seconds: BTreeMap::new(),
            });
        ensure!(entry.split == split, "change leaks across splits");
        let noev = id.strip_suffix(":ev").context("invalid ev ID")?.to_owned() + ":noev";
        let noev_source = expected
            .get(&noev)
            .context("missing no-evidence compiled pair")?;
        ensure!(
            noev_source["label"] == row["label"]
                && noev_source["options"] == row["options"]
                && noev_source["split"] == row["split"],
            "ev/noev label or split mismatch"
        );
        entry.change.pairs.push(ExecutedPair {
            scenario_id: scenario.into(),
            probability: predictions[id],
            noev_probability: *predictions
                .get(&noev)
                .context("missing no-evidence probability")?,
            regressed: label == 1,
        });
        entry.seconds.insert(scenario.into(), seconds);
    }
    ensure!(!changes.is_empty(), "no independent held-out changes");
    for change in changes.values() {
        let first = &change.change.pairs[0].scenario_id;
        let env: String = connection.query_row(
            "SELECT env FROM scenarios WHERE scenario_id=?1",
            [first],
            |r| r.get(0),
        )?;
        let expected: usize =
            connection.query_row("SELECT count(*) FROM scenarios WHERE env=?1", [&env], |r| {
                r.get(0)
            })?;
        ensure!(
            change.change.pairs.len() == expected,
            "partial held-out change {}",
            change.change.change_id
        );
        for pair in &change.change.pairs {
            let actual: String = connection.query_row(
                "SELECT env FROM scenarios WHERE scenario_id=?1",
                [&pair.scenario_id],
                |r| r.get(0),
            )?;
            ensure!(actual == env, "mixed scenario environments in change");
        }
    }
    let heldout: Vec<_> = changes.values().filter(|c| c.split != "ood").collect();
    let ids: Vec<_> = heldout.iter().map(|c| c.change.change_id.clone()).collect();
    let partitions = heldout_partitions(&ids)?;
    let mut reports = serde_json::Map::new();
    for &alpha in alphas {
        let policy = calibrate_executed(
            &heldout.iter().map(|c| c.change.clone()).collect::<Vec<_>>(),
            alpha,
            guarantee,
        )?;
        let mut splits = Vec::new();
        for (split_seed, (calibration, tests)) in partitions.iter().enumerate() {
            let split_policy = calibrate_executed(
                &calibration
                    .iter()
                    .map(|id| changes[id].change.clone())
                    .collect::<Vec<_>>(),
                alpha,
                guarantee,
            )?;
            let mut measured = executed_metrics(
                &tests.iter().map(|id| &changes[id]).collect::<Vec<_>>(),
                &split_policy,
                seed ^ split_seed as u64,
            )?;
            measured["seed"] = json!(split_seed);
            measured["policy"] = serde_json::to_value(split_policy)?;
            measured["calibration_change_ids"] = json!(calibration);
            measured["test_change_ids"] = json!(tests);
            splits.push(measured);
        }
        let field = if guarantee == Guarantee::Change {
            "unconditional_change_miss_rate"
        } else {
            "selection_mean_scenario_miss"
        };
        let within = splits
            .iter()
            .filter(|s| s[field].as_f64().unwrap() <= alpha)
            .count();
        let mut summary = serde_json::Map::new();
        for key in [
            "eval_hours",
            "eval_hours_saved",
            "expensive_runs_per_change",
            "unconditional_change_miss_rate",
            "conditional_regressing_change_miss_rate",
            "selection_mean_scenario_miss",
            "actual_mean_scenario_miss",
            "actual_scenario_miss_rate",
            "change_recall",
        ] {
            let values: Vec<_> = splits.iter().filter_map(|s| s[key].as_f64()).collect();
            let worst = if matches!(key, "eval_hours_saved" | "change_recall") {
                f64::min
            } else {
                f64::max
            };
            summary.insert(key.into(), json!({"mean":(!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64),"worst":values.into_iter().reduce(worst)}));
        }
        for category in ["benign", "regressing", "non_regressing"] {
            let values: Vec<_> = splits
                .iter()
                .filter_map(|s| s["expensive_runs_by_category"][category]["mean"].as_f64())
                .collect();
            summary.insert(format!("expensive_runs_per_{category}_change"),json!({"mean":(!values.is_empty()).then(||values.iter().sum::<f64>()/values.len() as f64),"worst":values.into_iter().reduce(f64::max)}));
        }
        let mut breakdown = serde_json::Map::new();
        for key in [
            "t0_saved_seconds",
            "adaptive_cheap_saved_seconds",
            "cheap_overhead_seconds",
            "threshold_saved_seconds",
            "early_stop_saved_seconds",
            "exploration_selected_overhead_seconds",
        ] {
            breakdown.insert(
                key.into(),
                json!(
                    splits
                        .iter()
                        .map(|s| s["breakdown"][key].as_f64().unwrap()
                            / (s["run_all_hours"].as_f64().unwrap() * 3600.0))
                        .sum::<f64>()
                        / splits.len() as f64
                ),
            );
        }
        summary.insert(
            "savings_breakdown_mean_baseline_fraction".into(),
            Value::Object(breakdown),
        );
        let ood: Vec<_> = changes.values().filter(|c| c.split == "ood").collect();
        reports.insert(alpha.to_string(), json!({"production_policy":policy,"splits":splits,"summary":summary,"splits_within_alpha":within,"required_splits_within_alpha":18,"replay_gate_passed":within>=18,"production_pool_descriptive":executed_metrics(&heldout,&policy,seed)?,"ood":if ood.is_empty(){Value::Null}else{executed_metrics(&ood,&policy,seed)?}}));
    }
    let mut source_hashes = BTreeMap::new();
    for path in [
        ledger.to_path_buf(),
        compiled.to_path_buf(),
        run_dir.join("run_meta.json"),
        run_dir.join("metrics.json"),
        run_dir.join("temps.json"),
        probs.join("calibration.jsonl"),
        probs.join("test.jsonl"),
        probs.join("ood.jsonl"),
    ] {
        source_hashes.insert(
            path.display().to_string(),
            format!("{:x}", Sha256::digest(fs::read(path)?)),
        );
    }
    let report = json!({"guide3":{"guarantee":guarantee,"guarantee_denominator":"all changes; conditional regressing-change misses are measured separately", "heldout_changes":heldout.len(),"heldout_change_ids":ids,"excluded_fit_change_ids":fitted,"model_version":model,"revision":revision,"temperature":temperature,"source_sha256":source_hashes,"t0_credit_verified":false,"t0_saved_seconds":0,"cheap_expensive_equivalent":0.075,"exploration_epsilon":0.05,"exploration_seed":seed,"partition_seeds":[0,19],"wave_size":4,"wave_mode":if guarantee==Guarantee::Change{"early_stop_after_confirmed_regression"}else{"full_selected_suite"},"alphas":reports}});
    serde_json::to_writer_pretty(
        File::create_new(out).context("create new Guide 3 report")?,
        &report,
    )?;
    Ok(())
}

fn executed_metrics(
    changes: &[&ExecutedMeasurement],
    policy: &ExecutedPolicy,
    seed: u64,
) -> Result<Value> {
    let (
        mut baseline,
        mut cheap,
        mut expensive,
        mut selected_cost,
        mut policy_cost,
        mut exploration_cost,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (
        mut positives,
        mut selected_missed,
        mut actual_missed,
        mut regressed_changes,
        mut caught,
        mut selected_count,
        mut run_count,
        mut cheap_count,
        mut exploration_count,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0);
    let (mut selection_mean, mut actual_mean) = (0.0, 0.0);
    let mut groups: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut executions = Vec::new();
    for (index, change) in changes.iter().enumerate() {
        let decisions = execute_change(&change.change, policy, seed ^ index as u64)?;
        let outcomes: BTreeMap<_, _> = change
            .change
            .pairs
            .iter()
            .map(|p| (p.scenario_id.as_str(), p.regressed))
            .collect();
        let (mut n, mut selected_loss, mut actual_loss, mut used) = (0, 0, 0, 0);
        for decision in &decisions {
            let seconds = change.seconds[&decision.scenario_id];
            let positive = outcomes[decision.scenario_id.as_str()];
            baseline += seconds;
            if decision.cheap_run {
                cheap += 0.075 * seconds;
                cheap_count += 1;
            }
            if decision.policy_selected {
                policy_cost += seconds;
            }
            if decision.selected {
                selected_cost += seconds;
                selected_count += 1;
            }
            if decision.ran {
                expensive += seconds;
                run_count += 1;
                used += 1;
            }
            if decision.exploration {
                exploration_cost += seconds;
                exploration_count += 1;
            }
            if positive {
                n += 1;
                selected_loss += usize::from(!decision.selected);
                actual_loss += usize::from(!decision.ran);
            }
        }
        positives += n;
        selected_missed += selected_loss;
        actual_missed += actual_loss;
        regressed_changes += usize::from(n > 0);
        caught += usize::from(n > actual_loss);
        selection_mean += selected_loss as f64 / n.max(1) as f64;
        actual_mean += actual_loss as f64 / n.max(1) as f64;
        let group = groups
            .entry(if change.benign {
                "benign"
            } else if n > 0 {
                "regressing"
            } else {
                "non_regressing"
            })
            .or_default();
        group.0 += 1;
        group.1 += used;
        executions.push(json!({"change_id":change.change.change_id,"decisions":decisions}));
    }
    let count = changes.len() as f64;
    ensure!(
        count > 0.0
            && baseline.is_finite()
            && cheap.is_finite()
            && expensive.is_finite()
            && baseline > 0.0,
        "invalid aggregate replay costs"
    );
    Ok(
        json!({"changes":changes.len(),"regressed_changes":regressed_changes,"caught_changes":caught,"unconditional_change_miss_rate":(regressed_changes-caught) as f64/count,"conditional_regressing_change_miss_rate":(regressed_changes>0).then(|| (regressed_changes-caught) as f64/regressed_changes as f64),"change_recall":(regressed_changes>0).then(|| caught as f64/regressed_changes as f64),"selection_mean_scenario_miss":selection_mean/count,"actual_mean_scenario_miss":actual_mean/count,"selection_scenario_miss_rate":(positives>0).then(|| selected_missed as f64/positives as f64),"actual_scenario_miss_rate":(positives>0).then(|| actual_missed as f64/positives as f64),"selected_expensive_runs":selected_count,"actual_expensive_runs":run_count,"expensive_runs_per_change":run_count as f64/count,"expensive_runs_by_category":groups.iter().map(|(k,(n,r))| (k,json!({"changes":n,"mean":*r as f64/ *n as f64}))).collect::<BTreeMap<_,_>>(),"cheap_runs":cheap_count,"exploration_draws":exploration_count,"eval_hours":(cheap+expensive)/3600.0,"run_all_hours":baseline/3600.0,"eval_hours_saved":1.0-(cheap+expensive)/baseline,"breakdown":{"t0_saved_seconds":0,"all_cheap_seconds":0.075*baseline,"adaptive_cheap_saved_seconds":0.075*baseline-cheap,"cheap_overhead_seconds":cheap,"threshold_saved_seconds":baseline-policy_cost,"early_stop_saved_seconds":selected_cost-expensive,"exploration_selected_overhead_seconds":exploration_cost},"executions":executions}),
    )
}

/// Builds measured ablations from ledger labels and independently fitted scores.
/// Compiled split membership is frozen; probability-fitting changes are excluded
/// from the risk-calibration pool. Missing measurements are never filled in.
pub fn write_report(
    ledger: &Path,
    compiled: &Path,
    scores: &Path,
    out: &Path,
    alpha: f64,
    seed: u64,
) -> Result<()> {
    let scores: Scores = serde_json::from_slice(&fs::read(scores)?)?;
    ensure!(
        !scores.strategies.is_empty(),
        "at least one scored strategy is required"
    );
    let mut examples = BTreeMap::new();
    let mut pairs_seen = BTreeSet::new();
    let mut splits = BTreeMap::new();
    for line in fs::read_to_string(compiled)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let id = row["id"].as_str().context("compiled id is missing")?;
        let guide = id.contains(':');
        let parts: Vec<_> = id.split(if guide { ':' } else { '/' }).collect();
        ensure!(parts.len() == 4, "invalid compiled id");
        if guide {
            ensure!(
                row["change_id"] == parts[0]
                    && row["scenario_id"] == parts[1]
                    && row["pair_id"] == format!("{}:{}", parts[0], parts[1])
                    && row["question"] == parts[2]
                    && row["variant"] == parts[3]
                    && matches!(parts[3], "ev" | "noev"),
                "invalid guide compiled identity"
            );
        }
        let split = row["split"].as_str().context("compiled split is missing")?;
        ensure!(
            matches!(split, "train" | "calibration" | "test" | "ood"),
            "invalid split"
        );
        if let Some(previous) = splits.insert(parts[0].to_owned(), split.to_owned()) {
            ensure!(previous == split, "change appears in multiple splits");
        }
        if parts[2] == "regressed" && parts[3] == if guide { "ev" } else { "cheap" } {
            ensure!(
                pairs_seen.insert((parts[0].to_owned(), parts[1].to_owned())),
                "duplicate compiled pair"
            );
            ensure!(
                examples
                    .insert(
                        id.to_owned(),
                        (parts[0].to_owned(), parts[1].to_owned(), split.to_owned())
                    )
                    .is_none(),
                "duplicate compiled pair"
            );
        }
    }
    ensure!(
        !examples.is_empty(),
        "compiled regression/cheap records are required"
    );
    let mut probability_fitted = BTreeSet::new();
    for (name, model) in &scores.strategies {
        ensure!(
            !name.trim().is_empty() && !model.model_version.trim().is_empty(),
            "strategy and model version must be nonempty"
        );
        for id in &model.training_change_ids {
            ensure!(
                splits.get(id).is_some_and(|split| split == "train"),
                "training provenance must name original train changes"
            );
        }
        for id in &model.probability_calibration_change_ids {
            ensure!(
                splits.get(id).is_some_and(|split| split == "calibration"),
                "probability calibration provenance must name calibration changes"
            );
            probability_fitted.insert(id.clone());
        }
        for (id, p) in &model.probabilities {
            ensure!(
                examples.contains_key(id) && p.is_finite() && (0.0..=1.0).contains(p),
                "invalid prediction for {id}"
            );
        }
    }
    let connection = Connection::open_with_flags(ledger, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut pairs = Vec::new();
    for (id, (change, scenario, split)) in examples {
        if split == "train" || probability_fitted.contains(&change) {
            continue;
        }
        let regressed: i64 = connection.query_row(
            "SELECT regressed FROM labels WHERE change_id=?1 AND scenario_id=?2",
            params![change, scenario],
            |row| row.get(0),
        )?;
        ensure!(
            matches!(regressed, 0 | 1),
            "invalid authoritative regression label"
        );
        let (seconds, episodes): (f64, i64) = connection.query_row(
            "SELECT sum(wall_seconds),sum(episodes) FROM runs WHERE change_id=?1 AND scenario_id=?2 AND tier='expensive'",
            params![change, scenario], |row| Ok((row.get(0)?, row.get(1)?)))?;
        ensure!(
            seconds.is_finite() && seconds > 0.0 && episodes == 20,
            "one complete expensive result is required per pair"
        );
        let mut query = connection.prepare(
            "SELECT r.episodes,r.successes,s.baseline_success FROM runs r JOIN scenarios s USING(scenario_id) WHERE r.change_id=?1 AND r.scenario_id=?2 AND r.tier='cheap'",
        )?;
        let cheap_runs = query
            .query_map(params![change, scenario], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        ensure!(
            cheap_runs.len() == 1 && cheap_runs[0].0 == 3,
            "exactly one three-episode cheap result is required per pair"
        );
        let (_, successes, baseline) = cheap_runs[0];
        ensure!(
            successes.is_finite()
                && (0.0..=3.0).contains(&successes)
                && (0.0..=1.0).contains(&baseline),
            "invalid cheap or baseline success"
        );
        let cheap = successes / 3.0;
        pairs.push(Pair {
            id,
            change,
            scenario,
            split,
            regressed: regressed == 1,
            seconds,
            heuristic: cheap < baseline - 0.15,
        });
    }
    let evaluation: Vec<_> = pairs.iter().filter(|pair| pair.split == "test").collect();
    ensure!(
        !evaluation.is_empty(),
        "held-out test pairs are required for the primary evaluation"
    );
    let all_hours = evaluation.iter().map(|pair| pair.seconds).sum::<f64>() / 3600.0;
    ensure!(
        all_hours.is_finite() && all_hours > 0.0,
        "aggregate evaluation time must be finite and positive"
    );
    let mut rows = BTreeMap::new();
    rows.insert(
        "Run all expensive".to_owned(),
        selection(&evaluation, &vec![true; evaluation.len()]),
    );
    rows.insert(
        "Cheap-tier heuristic".to_owned(),
        selection(
            &evaluation,
            &evaluation
                .iter()
                .map(|pair| pair.heuristic)
                .collect::<Vec<_>>(),
        ),
    );
    let mut reference_budget = None;
    for (name, model) in scores.strategies {
        ensure!(
            !rows.contains_key(&name),
            "scored strategy collides with a baseline"
        );
        let mut changes: BTreeMap<String, ReplayChange> = BTreeMap::new();
        let mut risk: BTreeMap<String, LabeledChange> = BTreeMap::new();
        for pair in &pairs {
            let p = *model
                .probabilities
                .get(&pair.id)
                .with_context(|| format!("missing prediction for {}", pair.id))?;
            if pair.split == "ood" {
                continue;
            }
            changes
                .entry(pair.change.clone())
                .or_insert_with(|| ReplayChange {
                    change_id: pair.change.clone(),
                    split: pair.split.clone(),
                    pairs: Vec::new(),
                })
                .pairs
                .push(ReplayPair {
                    scenario_id: pair.scenario.clone(),
                    probability: p,
                    regressed: pair.regressed,
                    wall_seconds: pair.seconds,
                });
            if pair.split == "calibration" {
                risk.entry(pair.change.clone())
                    .or_insert_with(|| LabeledChange {
                        change_id: pair.change.clone(),
                        pairs: Vec::new(),
                    })
                    .pairs
                    .push(PairScore {
                        scenario_id: pair.scenario.clone(),
                        probability: p,
                        regressed: pair.regressed,
                    });
            }
        }
        let policy = calibrate(&risk.into_values().collect::<Vec<_>>(), alpha)?;
        let replay = replay(&changes.into_values().collect::<Vec<_>>(), alpha, seed)?;
        let probabilities: Vec<_> = evaluation
            .iter()
            .map(|pair| model.probabilities[&pair.id])
            .collect();
        let selected: Vec<_> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| p >= policy.lambda || unit(seed, i) < 0.05)
            .collect();
        let mut row = selection(&evaluation, &selected);
        let budget = row["eval_hours"].as_f64().unwrap();
        if reference_budget.is_none() || name.to_lowercase().contains("full cascade") {
            reference_budget = Some(budget);
        }
        let labels: Vec<_> = evaluation.iter().map(|pair| pair.regressed).collect();
        row.as_object_mut().unwrap().extend(
            binary_metrics(&labels, &probabilities)
                .as_object()
                .unwrap()
                .clone(),
        );
        for split in ["test", "ood"] {
            let cohort: Vec<_> = pairs.iter().filter(|pair| pair.split == split).collect();
            if !cohort.is_empty() {
                row[format!("{split}_metrics")] = binary_metrics(
                    &cohort.iter().map(|pair| pair.regressed).collect::<Vec<_>>(),
                    &cohort
                        .iter()
                        .map(|pair| model.probabilities[&pair.id])
                        .collect::<Vec<_>>(),
                );
            }
        }
        let test_pairs: Vec<_> = evaluation
            .iter()
            .copied()
            .filter(|pair| pair.split == "test")
            .collect();
        let test_probabilities: Vec<_> = test_pairs
            .iter()
            .map(|pair| model.probabilities[&pair.id])
            .collect();
        let mut thresholds = test_probabilities.clone();
        thresholds.sort_by(f64::total_cmp);
        thresholds.dedup();
        if let Some(last) = thresholds.last() {
            thresholds.push(f64::from_bits(last.to_bits() + 1));
        }
        row["threshold_curve_split"] = json!("test");
        row["threshold_curve"] = Value::Array(
            thresholds
                .into_iter()
                .map(|threshold| {
                    let mut point = selection(
                        &test_pairs,
                        &test_probabilities
                            .iter()
                            .map(|&p| p >= threshold)
                            .collect::<Vec<_>>(),
                    );
                    point["threshold"] = json!(threshold);
                    point
                })
                .collect(),
        );
        for (measurements, kind) in [
            (&model.latencies_seconds, "latency"),
            (&model.costs_usd, "cost"),
        ] {
            if measurements.is_empty() {
                continue;
            }
            let mut values = Vec::new();
            for pair in &evaluation {
                let value = *measurements
                    .get(&pair.id)
                    .context("partial measurement vector")?;
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "measurements must be finite and nonnegative"
                );
                values.push(value);
            }
            if kind == "latency" {
                values.sort_by(f64::total_cmp);
                row["p50_latency_seconds"] = json!(quantile(&values, 0.5));
                row["p95_latency_seconds"] = json!(quantile(&values, 0.95));
            } else {
                let total_cost = values.iter().sum::<f64>();
                ensure!(total_cost.is_finite(), "aggregate measured cost overflowed");
                row["total_cost_usd"] = json!(total_cost);
                row["cost_usd_per_change"] = json!(
                    total_cost
                        / evaluation
                            .iter()
                            .map(|pair| &pair.change)
                            .collect::<BTreeSet<_>>()
                            .len() as f64
                );
            }
        }
        row["model_version"] = json!(model.model_version);
        row["guarantee_splits"] = json!(replay.empirical_splits_within_alpha);
        row["lambda"] = json!(policy.lambda);
        row["replay"] = serde_json::to_value(replay)?;
        rows.insert(name, row);
    }
    let probability = reference_budget.unwrap() / all_hours;
    rows.insert(
        "Random at equal expected budget".into(),
        selection(
            &evaluation,
            &evaluation
                .iter()
                .enumerate()
                .map(|(i, _)| unit(seed ^ 0xa076_1d64_78bd_642f, i) < probability)
                .collect::<Vec<_>>(),
        ),
    );
    let mut report = json!({"evaluation_splits": evaluation.iter().map(|pair| &pair.split).collect::<BTreeSet<_>>(), "alpha": alpha, "seed": seed, "strategies": rows});
    if let Some(metrics) = crate::explain::score_explanations(ledger, compiled)? {
        report["explainer_metrics"] = metrics;
    }
    let file = File::create_new(out).context("create new replay report")?;
    serde_json::to_writer_pretty(file, &report)?;
    Ok(())
}

fn selection(pairs: &[&Pair], selected: &[bool]) -> Value {
    let mut changes: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let (mut baseline, mut cost, mut regressions, mut missed) = (0.0, 0.0, 0, 0);
    for (pair, &run) in pairs.iter().zip(selected) {
        baseline += pair.seconds;
        if run {
            cost += pair.seconds;
        }
        let entry = changes.entry(&pair.change).or_default();
        if pair.regressed {
            regressions += 1;
            entry.0 += 1;
            if !run {
                missed += 1;
                entry.1 += 1;
            }
        }
    }
    let regressed_changes = changes.values().filter(|(n, _)| *n > 0).count();
    let caught_changes = changes.values().filter(|(n, missed)| n > missed).count();
    json!({"pairs":pairs.len(),"selected_pairs":selected.iter().filter(|&&run| run).count(),
        "regressed_pairs":regressions,"missed_regressions":missed,
        "missed_regression_fraction":(regressions>0).then(|| missed as f64/regressions as f64),
        "mean_change_loss":changes.values().map(|(n,m)| *m as f64/(*n).max(1) as f64).sum::<f64>()/changes.len() as f64,
        "regressed_changes":regressed_changes,"caught_changes":caught_changes,
        "change_recall":(regressed_changes>0).then(|| caught_changes as f64/regressed_changes as f64),
        "eval_hours":cost/3600.0,"run_all_hours":baseline/3600.0,"eval_hours_saved":1.0-cost/baseline})
}

fn binary_metrics(labels: &[bool], probabilities: &[f64]) -> Value {
    let count = labels.len() as f64;
    let mut sorted = probabilities.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut edges: Vec<_> = (0..=15)
        .map(|i| quantile(&sorted, i as f64 / 15.0))
        .collect();
    edges.dedup();
    let internal = if edges.len() > 2 {
        &edges[1..edges.len() - 1]
    } else {
        &[]
    };
    let mut bins: BTreeMap<usize, (usize, f64, usize)> = BTreeMap::new();
    let (mut brier, mut nll) = (0.0, 0.0);
    for (&label, &p) in labels.iter().zip(probabilities) {
        let bin = bins
            .entry(internal.partition_point(|edge| *edge < p))
            .or_default();
        bin.0 += 1;
        bin.1 += p;
        bin.2 += usize::from(label);
        let truth = if label { 1.0 } else { 0.0 };
        brier += (p - truth).powi(2);
        let clipped = p.clamp(1e-15, 1.0 - 1e-15);
        nll -= if label {
            clipped.ln()
        } else {
            (-clipped).ln_1p()
        };
    }
    let ece: f64 = bins
        .values()
        .map(|(_, p, y)| (*p - *y as f64).abs() / count)
        .sum();
    let positive = labels.iter().filter(|&&label| label).count();
    let negative = labels.len() - positive;
    let auroc = if positive > 0 && negative > 0 {
        let mut order: Vec<_> = labels
            .iter()
            .copied()
            .zip(probabilities.iter().copied())
            .collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (mut i, mut negatives, mut wins) = (0, 0, 0.0);
        while i < order.len() {
            let end = i + order[i..]
                .iter()
                .take_while(|row| row.1 == order[i].1)
                .count();
            let positives = order[i..end].iter().filter(|row| row.0).count();
            let tied_negative = end - i - positives;
            wins += positives as f64 * (negatives as f64 + 0.5 * tied_negative as f64);
            negatives += tied_negative;
            i = end;
        }
        Some(wins / (positive * negative) as f64)
    } else {
        None
    };
    json!({"ece":ece,"brier":brier/count,"nll":nll/count,"auroc":auroc,
        "reliability_bins":bins.values().map(|(n,p,y)| json!({"count":n,"mean_probability":p/ *n as f64,"observed_regression":*y as f64/ *n as f64})).collect::<Vec<_>>()})
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

fn unit(seed: u64, index: usize) -> f64 {
    let mut value = seed.wrapping_add((index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
}

#[cfg(test)]
mod guide3_cost_tests {
    use super::*;

    #[test]
    fn four_job_wave_cost_keeps_change_detection_and_separates_actual_scenario_misses() {
        let change = ExecutedMeasurement {
            change: ExecutedChange {
                change_id: "c100".into(),
                pairs: (0..12)
                    .map(|i| ExecutedPair {
                        scenario_id: format!("s{i:02}"),
                        probability: 0.8,
                        noev_probability: 0.95,
                        regressed: true,
                    })
                    .collect(),
            },
            split: "test".into(),
            benign: false,
            seconds: (0..12).map(|i| (format!("s{i:02}"), 1.0)).collect(),
        };
        let policy = ExecutedPolicy {
            alpha: 0.05,
            lambda: 0.8,
            calibration_changes: 30,
            guarantee: Guarantee::Change,
            certified: true,
            run_all: false,
            risk_bound: 1.0 / 31.0,
            guarantee_metric: "unconditional_change_miss".into(),
        };
        let result = executed_metrics(&[&change], &policy, 0).unwrap();
        assert_eq!(result["actual_expensive_runs"], 4);
        assert_eq!(result["cheap_runs"], 0);
        assert_eq!(result["eval_hours"], 4.0 / 3600.0);
        assert_eq!(result["unconditional_change_miss_rate"], 0.0);
        assert_eq!(result["selection_scenario_miss_rate"], 0.0);
        assert_eq!(result["actual_scenario_miss_rate"], 8.0 / 12.0);
        assert_eq!(result["breakdown"]["early_stop_saved_seconds"], 8.0);
        let mut scenario = policy;
        scenario.guarantee = Guarantee::Scenario;
        assert_eq!(
            executed_metrics(&[&change], &scenario, 0).unwrap()["actual_expensive_runs"],
            12
        );
    }
}
