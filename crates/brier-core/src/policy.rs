use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const DEFAULT_ALPHA: f64 = 0.05;
pub const LAMBDA_LOW: f64 = 0.02;
pub const LAMBDA_HIGH: f64 = 0.90;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Guarantee {
    Change,
    Scenario,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutedPair {
    pub scenario_id: String,
    pub probability: f64,
    pub noev_probability: f64,
    pub regressed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutedChange {
    pub change_id: String,
    pub pairs: Vec<ExecutedPair>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutedPolicy {
    pub alpha: f64,
    pub lambda: f64,
    pub calibration_changes: usize,
    pub guarantee: Guarantee,
    pub certified: bool,
    pub run_all: bool,
    pub risk_bound: f64,
    pub guarantee_metric: String,
}

/// Applies the fixed adaptive mask, or bypasses it for an uncertified run-all policy.
pub fn selected(pair: &ExecutedPair, policy: &ExecutedPolicy) -> bool {
    policy.run_all || adaptive_selected(pair, policy.lambda)
}

/// Measures unconditional change misses or equally weighted per-change scenario misses.
pub fn executed_loss(change: &ExecutedChange, lambda: f64, level: Guarantee) -> Result<f64> {
    validate_executed(change)?;
    ensure!(
        lambda.is_finite() && (0.0..=1.0).contains(&lambda),
        "invalid lambda"
    );
    Ok(executed_change_loss(change, lambda, level))
}

/// Calibrates the exact executed adaptive policy; uncertifiable data yields explicit run-all.
pub fn calibrate_executed(
    changes: &[ExecutedChange],
    alpha: f64,
    guarantee: Guarantee,
) -> Result<ExecutedPolicy> {
    ensure!(
        alpha.is_finite() && (0.0..=1.0).contains(&alpha),
        "invalid alpha"
    );
    ensure!(!changes.is_empty(), "calibration changes must not be empty");
    let mut ids = HashSet::new();
    let mut grid = vec![0.0, 1.0];
    for change in changes {
        validate_executed(change)?;
        ensure!(
            ids.insert(&change.change_id),
            "duplicate calibration change"
        );
        grid.extend(change.pairs.iter().map(|pair| pair.probability));
    }
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    for change in changes {
        for level in [Guarantee::Change, Guarantee::Scenario] {
            ensure!(
                grid.windows(2)
                    .all(|w| executed_change_loss(change, w[0], level)
                        <= executed_change_loss(change, w[1], level)),
                "executed loss is not monotone"
            );
        }
    }
    let bound = |lambda| {
        (1.0 + changes
            .iter()
            .map(|change| executed_change_loss(change, lambda, guarantee))
            .sum::<f64>())
            / (changes.len() + 1) as f64
    };
    let lambda = grid
        .into_iter()
        .rev()
        .find(|&lambda| bound(lambda) <= alpha);
    Ok(ExecutedPolicy {
        alpha,
        lambda: lambda.unwrap_or(0.0),
        calibration_changes: changes.len(),
        guarantee,
        certified: lambda.is_some(),
        run_all: lambda.is_none(),
        risk_bound: bound(lambda.unwrap_or(0.0)),
        guarantee_metric: match guarantee {
            Guarantee::Change => "unconditional_change_miss",
            Guarantee::Scenario => "mean_scenario_miss",
        }
        .into(),
    })
}

fn adaptive_selected(pair: &ExecutedPair, lambda: f64) -> bool {
    pair.noev_probability >= LAMBDA_HIGH
        || (pair.noev_probability >= LAMBDA_LOW && pair.probability >= lambda)
}

fn executed_change_loss(change: &ExecutedChange, lambda: f64, level: Guarantee) -> f64 {
    let positive = change.pairs.iter().filter(|pair| pair.regressed).count();
    let missed = change
        .pairs
        .iter()
        .filter(|pair| pair.regressed && !adaptive_selected(pair, lambda))
        .count();
    match level {
        Guarantee::Change => {
            if positive > 0 && positive == missed {
                1.0
            } else {
                0.0
            }
        }
        Guarantee::Scenario => missed as f64 / positive.max(1) as f64,
    }
}

fn validate_executed(change: &ExecutedChange) -> Result<()> {
    validate_change(&LabeledChange {
        change_id: change.change_id.clone(),
        pairs: change
            .pairs
            .iter()
            .map(|pair| PairScore {
                scenario_id: pair.scenario_id.clone(),
                probability: pair.probability,
                regressed: pair.regressed,
            })
            .collect(),
    })?;
    ensure!(
        change
            .pairs
            .iter()
            .all(|pair| pair.noev_probability.is_finite()
                && (0.0..=1.0).contains(&pair.noev_probability)),
        "invalid no-evidence probability"
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairScore {
    pub scenario_id: String,
    pub probability: f64,
    pub regressed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabeledChange {
    pub change_id: String,
    pub pairs: Vec<PairScore>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub alpha: f64,
    pub lambda: f64,
    pub calibration_changes: usize,
}

/// Computes the fraction of regressed scenarios skipped at `lambda`.
/// Equality retains a scenario; changes without regressions have zero loss.
pub fn missed_fraction(change: &LabeledChange, lambda: f64) -> Result<f64> {
    ensure!(
        lambda.is_finite() && (0.0..=1.0).contains(&lambda),
        "lambda must be finite and between zero and one"
    );
    validate_change(change)?;
    Ok(change_loss(change, lambda))
}

/// Selects the largest threshold satisfying the change-level CRC bound.
/// With fixed model scores and exchangeable calibration and future changes,
/// CRC bounds expected missed fraction, not every change's realized loss.
/// The caller must supply calibration changes only, never test changes.
pub fn calibrate(changes: &[LabeledChange], alpha: f64) -> Result<Policy> {
    ensure!(
        alpha.is_finite() && (0.0..=1.0).contains(&alpha),
        "alpha must be finite and between zero and one"
    );
    ensure!(!changes.is_empty(), "calibration changes must not be empty");

    let mut change_ids = HashSet::new();
    let mut candidates = vec![0.0, 1.0];
    for change in changes {
        validate_change(change)?;
        ensure!(
            change_ids.insert(change.change_id.as_str()),
            "duplicate calibration change: {}",
            change.change_id
        );
        candidates.extend(change.pairs.iter().map(|pair| pair.probability));
    }

    let denominator = changes.len() as f64 + 1.0;
    let minimum_bound = 1.0 / denominator;
    ensure!(
        minimum_bound <= alpha,
        "cannot certify alpha {} with {} calibration changes: minimum bound is {}",
        alpha,
        changes.len(),
        minimum_bound
    );

    candidates.sort_by(f64::total_cmp);
    candidates.dedup_by(|left, right| left == right);
    let lambda = candidates
        .into_iter()
        .rev()
        .find(|&threshold| {
            let total_loss: f64 = changes
                .iter()
                .map(|change| change_loss(change, threshold))
                .sum();
            (total_loss + 1.0) / denominator <= alpha
        })
        .context("no calibration threshold satisfies the finite-sample bound")?;

    Ok(Policy {
        alpha,
        lambda,
        calibration_changes: changes.len(),
    })
}

fn validate_change(change: &LabeledChange) -> Result<()> {
    ensure!(
        !change.change_id.trim().is_empty(),
        "change id must not be empty"
    );
    ensure!(
        !change.pairs.is_empty(),
        "change {} has no scenario pairs",
        change.change_id
    );

    let mut scenario_ids = HashSet::new();
    for pair in &change.pairs {
        ensure!(
            !pair.scenario_id.trim().is_empty(),
            "scenario id must not be empty for change {}",
            change.change_id
        );
        ensure!(
            scenario_ids.insert(pair.scenario_id.as_str()),
            "duplicate scenario {} for change {}",
            pair.scenario_id,
            change.change_id
        );
        ensure!(
            pair.probability.is_finite() && (0.0..=1.0).contains(&pair.probability),
            "probability must be finite and between zero and one for {} / {}",
            change.change_id,
            pair.scenario_id
        );
    }
    Ok(())
}

fn change_loss(change: &LabeledChange, lambda: f64) -> f64 {
    let regressed = change.pairs.iter().filter(|pair| pair.regressed).count();
    if regressed == 0 {
        return 0.0;
    }
    let missed = change
        .pairs
        .iter()
        .filter(|pair| pair.regressed && pair.probability < lambda)
        .count();
    missed as f64 / regressed as f64
}
