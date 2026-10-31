use brier_core::policy::{calibrate, missed_fraction, LabeledChange, PairScore, DEFAULT_ALPHA};

fn change(id: &str, outcomes: &[(f64, bool)]) -> LabeledChange {
    LabeledChange {
        change_id: id.to_owned(),
        pairs: outcomes
            .iter()
            .enumerate()
            .map(|(index, &(probability, regressed))| PairScore {
                scenario_id: format!("pusht-s{index:02}"),
                probability,
                regressed,
            })
            .collect(),
    }
}

fn calibration() -> Vec<LabeledChange> {
    (1..=19)
        .map(|index| change(&format!("c{index:03}"), &[(0.4, true)]))
        .collect()
}

#[test]
fn loss_counts_only_regressions_and_retains_threshold_equality() {
    let input = change("c001", &[(0.2, true), (0.8, true), (0.1, false)]);
    assert_eq!(missed_fraction(&input, 0.8).unwrap(), 0.5);
}

#[test]
fn calibration_uses_equal_change_weights_for_unbalanced_scenario_counts() {
    let mut changes: Vec<_> = (1..=17)
        .map(|index| change(&format!("c{index:03}"), &[(0.01, false)]))
        .collect();
    changes.push(change("c018", &[(0.2, true)]));
    changes.push(change("c019", &[(0.8, true); 9]));
    let policy = calibrate(&changes, 0.1).unwrap();
    assert_eq!(policy.lambda, 0.8);
    assert_eq!(policy.calibration_changes, 19);
    assert_eq!(policy.alpha, 0.1);
}

#[test]
fn calibration_preserves_scores_equal_to_the_selected_threshold() {
    let changes = calibration();
    let policy = calibrate(&changes, DEFAULT_ALPHA).unwrap();
    assert_eq!(policy.lambda, 0.4);
    assert_eq!(missed_fraction(&changes[0], policy.lambda).unwrap(), 0.0);
}

#[test]
fn changes_without_regressions_count_toward_the_finite_sample_bound() {
    let changes: Vec<_> = (1..=19)
        .map(|index| change(&format!("c{index:03}"), &[(0.01, false)]))
        .collect();
    let policy = calibrate(&changes, DEFAULT_ALPHA).unwrap();
    assert_eq!(policy.lambda, 1.0);
    assert_eq!(policy.calibration_changes, 19);
    assert_eq!(missed_fraction(&changes[0], 1.0).unwrap(), 0.0);
}

#[test]
fn probability_one_is_retained_at_the_maximum_threshold() {
    let changes: Vec<_> = (1..=19)
        .map(|index| change(&format!("c{index:03}"), &[(1.0, true)]))
        .collect();
    assert_eq!(calibrate(&changes, DEFAULT_ALPHA).unwrap().lambda, 1.0);
}

#[test]
fn insufficient_calibration_size_cannot_claim_certification() {
    let changes = calibration();
    let error = calibrate(&changes[..18], DEFAULT_ALPHA).unwrap_err();
    assert!(error.to_string().contains("cannot certify"));
    assert!(calibrate(&changes, 0.0).is_err());
    assert!(calibrate(&[], DEFAULT_ALPHA).is_err());
}

#[test]
fn rejects_nonfinite_and_out_of_range_values() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
        let mut changes = calibration();
        changes[0].pairs[0].probability = invalid;
        assert!(calibrate(&changes, DEFAULT_ALPHA).is_err());
        assert!(missed_fraction(&changes[0], 0.5).is_err());
        assert!(calibrate(&calibration(), invalid).is_err());
        assert!(missed_fraction(&calibration()[0], invalid).is_err());
    }
}

#[test]
fn rejects_duplicate_changes_and_duplicate_scenarios() {
    let mut changes = calibration();
    changes[1].change_id = changes[0].change_id.clone();
    assert!(calibrate(&changes, DEFAULT_ALPHA).is_err());

    let mut changes = calibration();
    let duplicate = changes[0].pairs[0].clone();
    changes[0].pairs.push(duplicate);
    assert!(calibrate(&changes, DEFAULT_ALPHA).is_err());
}

#[test]
fn rejects_empty_identifiers_and_empty_scenario_groups() {
    let mut changes = calibration();
    changes[0].change_id = " ".to_owned();
    assert!(calibrate(&changes, DEFAULT_ALPHA).is_err());

    let mut changes = calibration();
    changes[0].pairs[0].scenario_id.clear();
    assert!(calibrate(&changes, DEFAULT_ALPHA).is_err());

    let mut changes = calibration();
    changes[0].pairs.clear();
    assert!(calibrate(&changes, DEFAULT_ALPHA).is_err());
}
