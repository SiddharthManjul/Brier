use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const DEFAULT_ALPHA: f64 = 0.05;

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
