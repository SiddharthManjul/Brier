use brier_core::card::card;
use serde_json::json;
use std::{fs, path::PathBuf};

struct Fixture {
    directory: PathBuf,
    before: PathBuf,
    after: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!("brier-card-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let before = directory.join("before.yaml");
        let after = directory.join("after.yaml");
        let fixture = Self {
            directory,
            before,
            after,
        };
        fixture.write("description before", "benign\n    params: {}", "info");
        fixture
    }

    fn write(&self, description: &str, family: &str, logging: &str) {
        let source = format!("env: gym_pusht/PushT-v0\npolicy:\n  id: lerobot/diffusion_pusht\n  revision: {}\ndescription: {description}\nfamilies:\n  - family: {family}\nlogging:\n  level: {logging}\n", "a".repeat(40));
        fs::write(&self.before, &source).unwrap();
        fs::write(&self.after, &source).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

#[test]
fn metadata_only_changes_prove_identical_pinned_behavior_and_preserve_the_card_contract() {
    let f = Fixture::new();
    let before = fs::read_to_string(&f.before).unwrap();
    fs::write(
        &f.after,
        before
            .replace("description before", "logging only change")
            .replace("level: info", "level: debug"),
    )
    .unwrap();
    let report = card(&f.before, &f.after, "c123").unwrap();
    assert!(report.t0_identical);
    assert_eq!(report.identity.before, report.identity.after);
    assert_eq!(report.card.as_object().unwrap().len(), 2);
    assert_eq!(report.card["description"], "logging only change");
    assert_eq!(
        report.card["families"],
        json!([{"family":"benign","params":{}}])
    );
    assert_eq!(report.policy.revision, "a".repeat(40));
}

#[test]
fn a_benign_name_cannot_clear_a_changed_checkpoint_revision() {
    let f = Fixture::new();
    fs::write(
        &f.after,
        fs::read_to_string(&f.after)
            .unwrap()
            .replace(&"a".repeat(40), &"b".repeat(40)),
    )
    .unwrap();
    let report = card(&f.before, &f.after, "c124").unwrap();
    assert!(!report.t0_identical);
    assert_ne!(report.identity.before.policy, report.identity.after.policy);
}

#[test]
fn positive_stochastic_perturbations_never_claim_identical_weights_or_behavior() {
    let f = Fixture::new();
    for family in [
        "obs_noise\n    params: {sigma: 0.01}",
        "weight_noise\n    params: {sigma: 0.001}",
        "frame_drop\n    params: {p: 0.2}",
    ] {
        f.write("same configured perturbation", family, "info");
        let report = card(&f.before, &f.after, "c125").unwrap();
        assert!(!report.t0_identical);
        assert_eq!(report.identity.before, report.identity.after);
        assert!(!report.identity.before.effective_families.is_empty());
    }
}

#[test]
fn zero_noise_parameters_are_proven_noops_without_rewriting_the_change_card() {
    let f = Fixture::new();
    let source = fs::read_to_string(&f.after).unwrap().replace("family: benign\n    params: {}", "family: obs_noise\n    params: {sigma: 0.0}\n  - family: weight_noise\n    params: {sigma: 0.0}");
    fs::write(&f.after, source).unwrap();
    let report = card(&f.before, &f.after, "c126").unwrap();
    assert!(report.t0_identical);
    assert!(report.identity.after.effective_families.is_empty());
    assert_eq!(report.card["families"].as_array().unwrap().len(), 2);
}

#[test]
fn evaluation_affecting_family_composition_order_is_preserved() {
    let f = Fixture::new();
    let source = fs::read_to_string(&f.before).unwrap().replace("family: benign\n    params: {}", "family: action_scale\n    params: {scale: 0.9}\n  - family: action_latency\n    params: {k: 2}");
    fs::write(&f.before, &source).unwrap();
    let reversed = source.replace("family: action_scale\n    params: {scale: 0.9}\n  - family: action_latency\n    params: {k: 2}", "family: action_latency\n    params: {k: 2}\n  - family: action_scale\n    params: {scale: 0.9}");
    fs::write(&f.after, reversed).unwrap();
    let report = card(&f.before, &f.after, "c127").unwrap();
    assert!(!report.t0_identical);
    assert_ne!(report.identity.before, report.identity.after);
    assert_eq!(report.card["families"][0]["family"], "action_latency");
}

#[test]
fn malformed_snapshots_parameters_and_identifiers_fail_clearly() {
    let f = Fixture::new();
    let source = fs::read_to_string(&f.before).unwrap();
    for invalid in [
        source.replace(&"a".repeat(40), "main"),
        source.replace("params: {}", "params: {unrecognized: 1}"),
        source.clone() + "inference_config: {}\n",
        source.replace(
            "family: benign\n    params: {}",
            "family: action_latency\n    params: {k: 0}",
        ),
        source.replace(
            "family: benign\n    params: {}",
            "family: action_latency\n    params: {k: 1.5}",
        ),
        "env: [\n".to_string(),
    ] {
        fs::write(&f.after, invalid).unwrap();
        assert!(card(&f.before, &f.after, "c128").is_err());
    }
    fs::write(&f.after, source).unwrap();
    for invalid in ["c000", "c12", "123", "c9999999999999999999999999"] {
        assert!(card(&f.before, &f.after, invalid).is_err());
    }
}

#[test]
fn card_cli_emits_the_identity_proof_consumed_by_robot_ci() {
    let f = Fixture::new();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_brier"))
        .arg("card")
        .arg("--before")
        .arg(&f.before)
        .arg("--after")
        .arg(&f.after)
        .args(["--change", "c129"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["change_id"], "c129");
    assert_eq!(report["t0_identical"], true);
    assert_eq!(report["identity"]["before"], report["identity"]["after"]);
}
