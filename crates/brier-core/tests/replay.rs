use brier_core::policy::DEFAULT_ALPHA;
use brier_core::replay::{replay, ReplayChange, ReplayPair};
use std::collections::HashSet;

fn fixture(tests: usize, probability: f64) -> Vec<ReplayChange> {
    (1..=19 + tests)
        .map(|i| ReplayChange {
            change_id: format!("c{i:03}"),
            split: if i <= 19 { "calibration" } else { "test" }.into(),
            pairs: vec![ReplayPair {
                scenario_id: "pusht-s01".into(),
                probability: if i <= 19 { 0.8 } else { probability },
                regressed: true,
                wall_seconds: 2.0,
            }],
        })
        .collect()
}

#[test]
fn replay_is_seeded_and_independent_of_input_order() {
    let mut changes = fixture(20, 0.1);
    let first = replay(&changes, DEFAULT_ALPHA, 42).unwrap();
    assert_eq!(first, replay(&changes, DEFAULT_ALPHA, 42).unwrap());
    changes.reverse();
    assert_eq!(first, replay(&changes, DEFAULT_ALPHA, 42).unwrap());
    assert_ne!(first, replay(&changes, DEFAULT_ALPHA, 43).unwrap());
}

#[test]
fn twenty_splits_keep_changes_disjoint_and_whole() {
    let report = replay(&fixture(20, 0.8), DEFAULT_ALPHA, 42).unwrap();
    assert_eq!(report.splits.len(), 20);
    assert_eq!(report.empirical_splits_within_alpha, 20);
    for split in report.splits {
        let calibration: HashSet<_> = split.calibration_change_ids.iter().collect();
        let test: HashSet<_> = split.test_change_ids.iter().collect();
        assert_eq!(calibration.len(), 19);
        assert_eq!(test.len(), 20);
        assert!(calibration.is_disjoint(&test));
        assert_eq!(calibration.union(&test).count(), 39);
        assert_eq!(split.lambda, 0.8);
        assert_eq!(split.mean_change_loss, 0.0);
    }
}

#[test]
fn timeline_uses_past_prefix_and_retains_threshold_equality() {
    let timeline = replay(&fixture(20, 0.8), DEFAULT_ALPHA, 3)
        .unwrap()
        .timeline;
    assert_eq!(timeline.initial_calibration_change_ids[0], "c001");
    assert_eq!(timeline.initial_calibration_change_ids[18], "c019");
    assert_eq!(timeline.changes[0].change_id, "c020");
    for change in timeline.changes {
        for decision in change.decisions {
            assert!(decision.policy_selected && decision.selected);
            assert_eq!(decision.propensity, 1.0);
            assert_eq!(decision.observed_regressed, Some(true));
        }
    }
}

#[test]
fn exploration_logs_observations_propensities_and_measured_costs() {
    let timeline = replay(&fixture(40, 0.1), DEFAULT_ALPHA, 13)
        .unwrap()
        .timeline;
    let mut selected = 0;
    for change in &timeline.changes {
        for decision in &change.decisions {
            assert!(!decision.policy_selected);
            assert_eq!(decision.propensity, 0.05);
            assert_eq!(
                decision.observed_regressed,
                decision.selected.then_some(true)
            );
            selected += usize::from(decision.selected);
        }
    }
    let fraction = selected as f64 / 40.0;
    assert_eq!(timeline.test_baseline_wall_seconds, 80.0);
    assert_eq!(timeline.test_selected_wall_seconds, selected as f64 * 2.0);
    assert_eq!(timeline.test_savings_fraction, 1.0 - fraction);
    assert_eq!(timeline.pair_regression_recall, Some(fraction));
    assert_eq!(timeline.change_regression_recall, Some(fraction));
    assert_eq!(timeline.mean_change_loss, 1.0 - fraction);
    assert_eq!(timeline.ips_policy_mean_change_loss, selected as f64 / 2.0);
}

#[test]
fn ips_recovers_offline_policy_loss_across_seeds() {
    let changes = fixture(40, 0.1);
    let total: f64 = (0..400)
        .map(|seed| {
            let timeline = replay(&changes, DEFAULT_ALPHA, seed).unwrap().timeline;
            assert_eq!(timeline.policy_mean_change_loss, 1.0);
            timeline.ips_policy_mean_change_loss
        })
        .sum();
    assert!((total / 400.0 - 1.0).abs() < 0.15);
}

#[test]
fn change_recall_and_loss_do_not_pool_unbalanced_scenarios() {
    let mut changes = fixture(2, 0.1);
    changes[20].pairs = (1..=9)
        .map(|i| ReplayPair {
            scenario_id: format!("pusht-s{i:02}"),
            probability: 0.9,
            regressed: true,
            wall_seconds: 2.0,
        })
        .collect();
    let timeline = replay(&changes, DEFAULT_ALPHA, 8).unwrap().timeline;
    assert_eq!(timeline.policy_mean_change_loss, 0.5);
    assert_eq!(
        timeline.mean_change_loss,
        timeline.changes[0].realized_loss / 2.0
    );
    let first_caught = timeline.changes[0].decisions[0].selected;
    assert_eq!(
        timeline.change_regression_recall,
        Some(if first_caught { 1.0 } else { 0.5 })
    );
    assert_eq!(
        timeline.pair_regression_recall,
        Some(if first_caught { 1.0 } else { 0.9 })
    );
}

#[test]
fn no_regressions_have_zero_loss_and_undefined_recall() {
    let mut changes = fixture(20, 0.1);
    for change in &mut changes {
        change.pairs[0].regressed = false;
    }
    let timeline = replay(&changes, DEFAULT_ALPHA, 1).unwrap().timeline;
    assert_eq!(timeline.mean_change_loss, 0.0);
    assert_eq!(timeline.policy_mean_change_loss, 0.0);
    assert_eq!(timeline.ips_policy_mean_change_loss, 0.0);
    assert_eq!(timeline.pair_regression_recall, None);
    assert_eq!(timeline.change_regression_recall, None);
}

#[test]
fn rejects_training_ood_duplicates_and_invalid_costs() {
    for split in ["train", "ood", ""] {
        let mut changes = fixture(20, 0.8);
        changes[0].split = split.into();
        assert!(replay(&changes, DEFAULT_ALPHA, 1).is_err());
    }
    let mut changes = fixture(20, 0.8);
    changes[20].change_id = changes[0].change_id.clone();
    assert!(replay(&changes, DEFAULT_ALPHA, 1).is_err());
    for cost in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        let mut changes = fixture(20, 0.8);
        changes[20].pairs[0].wall_seconds = cost;
        assert!(replay(&changes, DEFAULT_ALPHA, 1).is_err());
    }
    let changes = fixture(20, 0.8);
    assert!(replay(&changes[1..], DEFAULT_ALPHA, 1).is_err());
    assert!(replay(&changes[..19], DEFAULT_ALPHA, 1).is_err());
    assert!(replay(&[], DEFAULT_ALPHA, 1).is_err());
}
