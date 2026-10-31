use brier_core::{ledger, replay_report::write_report};
use rusqlite::params;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

fn fixture() -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let directory = std::env::temp_dir().join(format!("brier-replay-report-{}", Uuid::new_v4()));
    fs::create_dir(&directory).unwrap();
    let ledger_path = directory.join("ledger.sqlite");
    let compiled = directory.join("compiled.jsonl");
    let predictions = directory.join("scores.json");
    let connection = ledger::init(&ledger_path).unwrap();
    connection
        .execute(
            "INSERT INTO scenarios (scenario_id,baseline_success) VALUES ('pusht-s01',1.0)",
            [],
        )
        .unwrap();
    let mut lines = String::new();
    let mut scores = serde_json::Map::new();
    let mut latency = serde_json::Map::new();
    let mut costs = serde_json::Map::new();
    for i in 1..=40 {
        let change = format!("c{i:03}");
        let split = if i <= 19 {
            "calibration"
        } else if i == 40 {
            "ood"
        } else {
            "test"
        };
        let id = format!("{change}/pusht-s01/regressed/cheap");
        lines.push_str(&format!(
            "{}\n",
            json!({"id":id,"split":split,"question":"regressed"})
        ));
        scores.insert(id.clone(), json!(if i <= 19 { 0.8 } else { 0.2 }));
        latency.insert(id.clone(), json!(0.01));
        costs.insert(id, json!(0.001));
        connection
            .execute(
                "INSERT INTO labels (change_id,scenario_id,regressed) VALUES (?1,'pusht-s01',?2)",
                params![change, i != 40],
            )
            .unwrap();
        for (tier, episodes, successes) in [("cheap", 3, 1), ("expensive", 20, 2)] {
            connection.execute("INSERT INTO runs (run_id,change_id,scenario_id,tier,episodes,successes,wall_seconds) VALUES (?1,?2,'pusht-s01',?3,?4,?5,3600.0)",params![Uuid::new_v4().to_string(),change,tier,episodes,successes]).unwrap();
        }
    }
    fs::write(&compiled, lines).unwrap();
    fs::write(
        &predictions,
        serde_json::to_vec(&json!({"strategies":{"Full cascade":{
            "model_version":"fixed-test-model", "training_change_ids":[],
            "probability_calibration_change_ids":[], "probabilities":scores,
            "latencies_seconds":latency, "costs_usd":costs,
        }}}))
        .unwrap(),
    )
    .unwrap();
    (directory, ledger_path, compiled, predictions)
}

#[test]
fn report_uses_ledger_truth_preserves_heldout_metrics_and_is_seeded() {
    let (directory, ledger, compiled, predictions) = fixture();
    let output = directory.join("report.json");
    write_report(&ledger, &compiled, &predictions, &output, 0.05, 42).unwrap();
    let report: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["evaluation_splits"], json!(["test"]));
    let rows = &report["strategies"];
    assert_eq!(rows["Run all expensive"]["run_all_hours"], 20.0);
    assert_eq!(rows["Run all expensive"]["change_recall"], 1.0);
    assert_eq!(rows["Run all expensive"]["eval_hours_saved"], 0.0);
    assert!(rows["Run all expensive"].get("ece").is_none());
    let full = &rows["Full cascade"];
    assert!((full["test_metrics"]["brier"].as_f64().unwrap() - 0.64).abs() < 1e-12);
    assert!((full["ood_metrics"]["brier"].as_f64().unwrap() - 0.04).abs() < 1e-12);
    assert_eq!(full["test_metrics"]["auroc"], Value::Null);
    assert_eq!(full["p95_latency_seconds"], 0.01);
    assert_eq!(full["replay"]["splits"].as_array().unwrap().len(), 20);
    assert_eq!(full["lambda"], 0.8);
    assert_eq!(full["threshold_curve_split"], "test");
    assert_eq!(full["threshold_curve"][0]["run_all_hours"], 20.0);
    assert_eq!(full["threshold_curve"][0]["pairs"], 20);
    let second = directory.join("second.json");
    write_report(&ledger, &compiled, &predictions, &second, 0.05, 42).unwrap();
    assert_eq!(fs::read(output).unwrap(), fs::read(second).unwrap());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn report_rejects_test_fitting_missing_predictions_and_incomplete_measurements() {
    let (directory, ledger, compiled, predictions) = fixture();
    let original: Value = serde_json::from_slice(&fs::read(&predictions).unwrap()).unwrap();
    for mutation in ["leakage", "prediction", "latency"] {
        let mut input = original.clone();
        let strategy = &mut input["strategies"]["Full cascade"];
        match mutation {
            "leakage" => strategy["probability_calibration_change_ids"] = json!(["c020"]),
            "prediction" => {
                strategy["probabilities"]
                    .as_object_mut()
                    .unwrap()
                    .remove("c020/pusht-s01/regressed/cheap");
            }
            _ => {
                strategy["latencies_seconds"]
                    .as_object_mut()
                    .unwrap()
                    .remove("c020/pusht-s01/regressed/cheap");
            }
        }
        fs::write(&predictions, serde_json::to_vec(&input).unwrap()).unwrap();
        assert!(write_report(
            &ledger,
            &compiled,
            &predictions,
            &directory.join("report.json"),
            0.05,
            42
        )
        .is_err());
        assert!(!directory.join("report.json").exists());
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn report_excludes_probability_fitting_changes_from_risk_calibration() {
    let (directory, ledger, compiled, predictions) = fixture();
    let mut input: Value = serde_json::from_slice(&fs::read(&predictions).unwrap()).unwrap();
    input["strategies"]["Full cascade"]["probability_calibration_change_ids"] = json!(["c001"]);
    fs::write(&predictions, serde_json::to_vec(&input).unwrap()).unwrap();
    let error = write_report(
        &ledger,
        &compiled,
        &predictions,
        &directory.join("report.json"),
        0.05,
        42,
    )
    .unwrap_err();
    assert!(error.to_string().contains("cannot certify"));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn tied_binary_scores_have_half_auroc_and_one_calibration_bin() {
    let (directory, ledger, compiled, predictions) = fixture();
    let connection = rusqlite::Connection::open(&ledger).unwrap();
    connection
        .execute_batch(
            "DROP TRIGGER labels_no_UPDATE; UPDATE labels SET regressed=0 WHERE change_id='c020';",
        )
        .unwrap();
    let output = directory.join("report.json");
    write_report(&ledger, &compiled, &predictions, &output, 0.05, 42).unwrap();
    let report: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    let row = &report["strategies"]["Full cascade"];
    assert_eq!(row["auroc"], 0.5);
    assert!((row["brier"].as_f64().unwrap() - (19.0 * 0.64 + 0.04) / 20.0).abs() < 1e-12);
    assert_eq!(row["reliability_bins"].as_array().unwrap().len(), 1);
    assert_eq!(row["reliability_bins"][0]["count"], 20);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn report_rejects_duplicate_partial_and_invalid_cheap_results_without_changing_scores() {
    for mutation in [
        "INSERT INTO runs VALUES ('duplicate','c020','pusht-s01','cheap',3,3,NULL,NULL,1.0)",
        "UPDATE runs SET episodes=2 WHERE change_id='c020' AND tier='cheap'",
        "DELETE FROM runs WHERE change_id='c020' AND tier='cheap'",
        "UPDATE runs SET successes=-1 WHERE change_id='c020' AND tier='cheap'",
        "UPDATE runs SET successes=4 WHERE change_id='c020' AND tier='cheap'",
        "UPDATE runs SET successes=1e999 WHERE change_id='c020' AND tier='cheap'",
        "UPDATE scenarios SET baseline_success=1.1 WHERE scenario_id='pusht-s01'",
    ] {
        let (directory, ledger, compiled, predictions) = fixture();
        let original_compiled = fs::read(&compiled).unwrap();
        let original_predictions = fs::read(&predictions).unwrap();
        let connection = rusqlite::Connection::open(&ledger).unwrap();
        connection
            .execute_batch("DROP TRIGGER runs_no_UPDATE; DROP TRIGGER runs_no_DELETE; DROP TRIGGER scenarios_no_UPDATE;")
            .unwrap();
        connection.execute_batch(mutation).unwrap();
        let output = directory.join("report.json");
        assert!(write_report(&ledger, &compiled, &predictions, &output, 0.05, 42).is_err());
        assert!(!output.exists());
        assert_eq!(fs::read(compiled).unwrap(), original_compiled);
        assert_eq!(fs::read(predictions).unwrap(), original_predictions);
        fs::remove_dir_all(directory).unwrap();
    }
}
