use crate::policy::{calibrate, LabeledChange, PairScore};
use crate::replay::{replay, ReplayChange, ReplayPair};
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OpenFlags};
use serde::Deserialize;
use serde_json::{json, Value};
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
