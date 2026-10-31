use crate::policy::{calibrate, missed_fraction, LabeledChange, PairScore, Policy};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayPair {
    pub scenario_id: String,
    pub probability: f64,
    pub regressed: bool,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayChange {
    pub change_id: String,
    pub split: String,
    pub pairs: Vec<ReplayPair>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomSplit {
    pub calibration_change_ids: Vec<String>,
    pub test_change_ids: Vec<String>,
    pub lambda: f64,
    pub mean_change_loss: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub scenario_id: String,
    pub probability: f64,
    pub policy_selected: bool,
    pub selected: bool,
    pub propensity: f64,
    pub observed_regressed: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChangeReplay {
    pub change_id: String,
    pub decisions: Vec<Decision>,
    pub offline_true_regressions: usize,
    pub policy_loss: f64,
    pub realized_loss: f64,
    pub ips_policy_loss: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub policy: Policy,
    pub initial_calibration_change_ids: Vec<String>,
    pub changes: Vec<ChangeReplay>,
    pub test_baseline_wall_seconds: f64,
    pub test_selected_wall_seconds: f64,
    pub test_selected_hours: f64,
    pub test_savings_fraction: f64,
    pub pair_regression_recall: Option<f64>,
    pub change_regression_recall: Option<f64>,
    pub mean_change_loss: f64,
    pub policy_mean_change_loss: f64,
    pub ips_policy_mean_change_loss: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub seed: u64,
    pub alpha: f64,
    pub splits: Vec<RandomSplit>,
    pub empirical_splits_within_alpha: usize,
    pub timeline: Timeline,
}

/// Replays fixed scores on calibration/test changes with twenty seeded splits.
/// Scores must come from a scorer fixed independently of this held-out pool.
/// The timeline calibrates on an ID-ordered prefix and keeps that policy fixed.
/// IPS uses offline true regression denominators, unavailable in a live ledger.
/// Costs cover expensive test-period runs, including five-percent exploration.
pub fn replay(changes: &[ReplayChange], alpha: f64, seed: u64) -> Result<ReplayReport> {
    let mut changes = changes.to_vec();
    changes.sort_by(|a, b| a.change_id.cmp(&b.change_id));
    let mut ids = HashSet::new();
    let mut calibration_count = 0;
    for change in &mut changes {
        ensure!(
            matches!(change.split.as_str(), "calibration" | "test"),
            "replay accepts calibration/test changes only"
        );
        calibration_count += usize::from(change.split == "calibration");
        ensure!(
            ids.insert(change.change_id.clone()),
            "duplicate replay change"
        );
        change
            .pairs
            .sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
        missed_fraction(&labeled(change), 0.0)?;
        for pair in &change.pairs {
            ensure!(
                pair.wall_seconds.is_finite() && pair.wall_seconds > 0.0,
                "wall_seconds must be finite and positive"
            );
        }
    }
    ensure!(
        calibration_count > 0 && calibration_count < changes.len(),
        "replay requires calibration and test changes"
    );
    let labels: Vec<_> = changes.iter().map(labeled).collect();
    let policy = calibrate(&labels[..calibration_count], alpha)?;
    let mut rng = Rng(seed);
    let mut splits = Vec::with_capacity(20);
    for _ in 0..20 {
        let mut indices: Vec<_> = (0..changes.len()).collect();
        for end in (1..indices.len()).rev() {
            let selected = rng.index(end + 1);
            indices.swap(end, selected);
        }
        let calibration: Vec<_> = indices[..calibration_count]
            .iter()
            .map(|&i| labels[i].clone())
            .collect();
        let split_policy = calibrate(&calibration, alpha)?;
        let loss: f64 = indices[calibration_count..]
            .iter()
            .map(|&i| missed_fraction(&labels[i], split_policy.lambda))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum();
        let change_ids = |indices: &[usize]| {
            indices
                .iter()
                .map(|&i| changes[i].change_id.clone())
                .collect()
        };
        splits.push(RandomSplit {
            calibration_change_ids: change_ids(&indices[..calibration_count]),
            test_change_ids: change_ids(&indices[calibration_count..]),
            lambda: split_policy.lambda,
            mean_change_loss: loss / (changes.len() - calibration_count) as f64,
        });
    }
    let initial_ids = changes[..calibration_count]
        .iter()
        .map(|change| change.change_id.clone())
        .collect();
    Ok(ReplayReport {
        seed,
        alpha,
        empirical_splits_within_alpha: splits
            .iter()
            .filter(|split| split.mean_change_loss <= alpha)
            .count(),
        splits,
        timeline: timeline(
            &changes[calibration_count..],
            policy,
            initial_ids,
            seed ^ 0xd1b5_4a32_d192_ed03,
        )?,
    })
}

fn labeled(change: &ReplayChange) -> LabeledChange {
    LabeledChange {
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
    }
}

fn timeline(
    tests: &[ReplayChange],
    policy: Policy,
    initial_ids: Vec<String>,
    seed: u64,
) -> Result<Timeline> {
    let mut rng = Rng(seed);
    let mut changes = Vec::new();
    let (mut baseline, mut cost) = (0.0, 0.0);
    let (mut regressions, mut detected, mut regressed_changes, mut detected_changes) = (0, 0, 0, 0);
    let (mut policy_total, mut realized_total, mut ips_total) = (0.0, 0.0, 0.0);
    for change in tests {
        let true_regressions = change.pairs.iter().filter(|pair| pair.regressed).count();
        let (mut policy_misses, mut realized_misses, mut ips_misses) = (0, 0, 0.0);
        let mut decisions = Vec::new();
        let mut caught = false;
        for pair in &change.pairs {
            let policy_selected = pair.probability >= policy.lambda;
            let propensity = if policy_selected { 1.0 } else { 0.05 };
            let selected = policy_selected || rng.unit() < 0.05;
            baseline += pair.wall_seconds;
            if selected {
                cost += pair.wall_seconds;
                detected += usize::from(pair.regressed);
                caught |= pair.regressed;
            }
            if pair.regressed && !policy_selected {
                policy_misses += 1;
                if selected {
                    ips_misses += 1.0 / propensity;
                }
            }
            realized_misses += usize::from(pair.regressed && !selected);
            decisions.push(Decision {
                scenario_id: pair.scenario_id.clone(),
                probability: pair.probability,
                policy_selected,
                selected,
                propensity,
                observed_regressed: selected.then_some(pair.regressed),
            });
        }
        let denominator = true_regressions.max(1) as f64;
        let policy_loss = policy_misses as f64 / denominator;
        let realized_loss = realized_misses as f64 / denominator;
        let ips_policy_loss = ips_misses / denominator;
        regressions += true_regressions;
        regressed_changes += usize::from(true_regressions > 0);
        detected_changes += usize::from(caught);
        policy_total += policy_loss;
        realized_total += realized_loss;
        ips_total += ips_policy_loss;
        changes.push(ChangeReplay {
            change_id: change.change_id.clone(),
            decisions,
            offline_true_regressions: true_regressions,
            policy_loss,
            realized_loss,
            ips_policy_loss,
        });
    }
    ensure!(
        baseline.is_finite() && cost.is_finite(),
        "aggregate replay cost overflowed"
    );
    let count = tests.len() as f64;
    Ok(Timeline {
        policy,
        initial_calibration_change_ids: initial_ids,
        changes,
        test_baseline_wall_seconds: baseline,
        test_selected_wall_seconds: cost,
        test_selected_hours: cost / 3600.0,
        test_savings_fraction: 1.0 - cost / baseline,
        pair_regression_recall: (regressions > 0).then(|| detected as f64 / regressions as f64),
        change_regression_recall: (regressed_changes > 0)
            .then(|| detected_changes as f64 / regressed_changes as f64),
        mean_change_loss: realized_total / count,
        policy_mean_change_loss: policy_total / count,
        ips_policy_mean_change_loss: ips_total / count,
    })
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
    fn index(&mut self, upper: usize) -> usize {
        let bound = upper as u64;
        let limit = u64::MAX - u64::MAX % bound;
        loop {
            let value = self.next();
            if value < limit {
                return (value % bound) as usize;
            }
        }
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
}
