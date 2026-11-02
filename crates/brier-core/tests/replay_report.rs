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
fn report_reads_guide_ids_without_changing_pair_weights_or_scores() {
    let (directory, ledger, compiled, predictions) = fixture();
    let original = directory.join("legacy.json");
    write_report(&ledger, &compiled, &predictions, &original, 0.05, 42).unwrap();
    let source = fs::read_to_string(&compiled).unwrap();
    let mut rows = Vec::new();
    for line in source.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let parts: Vec<_> = row["id"].as_str().unwrap().split('/').collect();
        rows.push(json!({"id":format!("{}:{}:regressed:ev",parts[0],parts[1]),
            "pair_id":format!("{}:{}",parts[0],parts[1]),"change_id":parts[0],"scenario_id":parts[1],
            "env":"pusht","variant":"ev","question":"regressed","options":["yes","no"],
            "prompt":"fixture<answer>","label":"A","split":row["split"]}));
    }
    fs::write(
        &compiled,
        rows.iter()
            .map(|row| format!("{row}\n"))
            .collect::<String>(),
    )
    .unwrap();
    let input = fs::read_to_string(&predictions)
        .unwrap()
        .replace("/pusht-s01/regressed/cheap", ":pusht-s01:regressed:ev");
    fs::write(&predictions, input).unwrap();
    let output = directory.join("guide.json");
    write_report(&ledger, &compiled, &predictions, &output, 0.05, 42).unwrap();
    assert_eq!(fs::read(original).unwrap(), fs::read(output).unwrap());
    rows[0]["change_id"] = json!("c999");
    fs::write(
        &compiled,
        rows.iter()
            .map(|row| format!("{row}\n"))
            .collect::<String>(),
    )
    .unwrap();
    assert!(write_report(
        &ledger,
        &compiled,
        &predictions,
        &directory.join("bad.json"),
        0.05,
        42
    )
    .is_err());
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

fn guide3_fixture() -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let (directory, ledger_path, compiled, scores) = fixture();
    let connection = rusqlite::Connection::open(&ledger_path).unwrap();
    connection
        .execute_batch("DROP TRIGGER scenarios_no_UPDATE; UPDATE scenarios SET env='pusht';")
        .unwrap();
    let source: Value = serde_json::from_slice(&fs::read(scores).unwrap()).unwrap();
    let mut compiled_rows = String::new();
    let run = directory.join("run");
    let probs = run.join("probs_temp");
    fs::create_dir_all(&probs).unwrap();
    let mut probability_rows = std::collections::BTreeMap::<&str, String>::new();
    for i in 1..=40 {
        let change = format!("c{i:03}");
        let split = if i <= 19 {
            "calibration"
        } else if i == 40 {
            "ood"
        } else {
            "test"
        };
        connection
            .execute(
                "INSERT INTO changes(change_id,env,card_json) VALUES (?1,'pusht',?2)",
                params![
                    change,
                    json!({"families":[{"family":"action_latency","params":{"k":1}}]}).to_string()
                ],
            )
            .unwrap();
        for variant in ["ev", "noev"] {
            let id = format!("{change}:pusht-s01:regressed:{variant}");
            let row = json!({"id":id,"pair_id":format!("{change}:pusht-s01"),"change_id":change,"scenario_id":"pusht-s01","split":split,"question":"regressed","variant":variant,"options":["yes","no"],"label":if i==40{"B"}else{"A"}});
            compiled_rows.push_str(&format!("{row}\n"));
            let p = if variant == "noev" {
                0.5
            } else {
                source["strategies"]["Full cascade"]["probabilities"]
                    [format!("{change}/pusht-s01/regressed/cheap")]
                .as_f64()
                .unwrap()
            };
            let prediction = json!({"id":id,"split":split,"question":"regressed","variant":variant,"options":["yes","no"],"label":row["label"],"probs":{"A":p,"B":1.0-p},"model":"FT","revision":"pin","temperature":1.0});
            probability_rows
                .entry(split)
                .or_default()
                .push_str(&format!("{prediction}\n"));
        }
    }
    fs::write(&compiled, compiled_rows).unwrap();
    for (split, rows) in probability_rows {
        fs::write(probs.join(format!("{split}.jsonl")), rows).unwrap();
    }
    let partition = json!({"fit_change_ids":["c001"],"risk_change_ids":["c002"]});
    fs::write(
        run.join("run_meta.json"),
        json!({"revision":"pin","calibration_partition":partition}).to_string(),
    )
    .unwrap();
    fs::write(
        run.join("metrics.json"),
        json!({"model":"FT","calibration_partition":partition}).to_string(),
    )
    .unwrap();
    fs::write(run.join("temps.json"), json!({"regressed":1.0}).to_string()).unwrap();
    (directory, ledger_path, compiled, probs)
}

#[test]
fn guide3_report_reads_exact_tempered_variants_and_excludes_fitting_and_ood_from_pool() {
    use brier_core::{policy::Guarantee, replay_report::write_guide3_report};
    let (directory, ledger, compiled, probs) = guide3_fixture();
    let original = fs::read(&ledger).unwrap();
    let output = directory.join("guide3.json");
    write_guide3_report(
        &ledger,
        &compiled,
        &probs,
        &output,
        &[0.05, 0.1],
        Guarantee::Change,
        42,
    )
    .unwrap();
    let report: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    let guide = &report["guide3"];
    assert_eq!(guide["heldout_changes"], 38);
    assert_eq!(guide["excluded_fit_change_ids"], json!(["c001"]));
    assert_eq!(guide["t0_credit_verified"], false);
    assert_eq!(guide["alphas"]["0.05"]["production_policy"]["lambda"], 0.2);
    assert_eq!(
        guide["alphas"]["0.05"]["splits"].as_array().unwrap().len(),
        20
    );
    let production = &guide["alphas"]["0.05"]["production_pool_descriptive"];
    assert_eq!(production["actual_expensive_runs"], 38);
    assert_eq!(production["cheap_runs"], 38);
    assert!((production["eval_hours_saved"].as_f64().unwrap() + 0.075).abs() < 1e-12);
    assert_eq!(production["unconditional_change_miss_rate"], 0.0);
    assert_eq!(guide["alphas"]["0.05"]["ood"]["changes"], 1);
    assert_eq!(fs::read(ledger).unwrap(), original);
    assert_eq!(guide["source_sha256"].as_object().unwrap().len(), 8);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn guide3_report_rejects_missing_noev_scores_and_changed_temperature_before_output() {
    use brier_core::{policy::Guarantee, replay_report::write_guide3_report};
    for mutation in ["missing", "temperature", "model"] {
        let (directory, ledger, compiled, probs) = guide3_fixture();
        let path = probs.join("test.jsonl");
        let mut rows: Vec<Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        match mutation {
            "missing" => {
                rows.pop();
            }
            "temperature" => rows[0]["temperature"] = json!(2.0),
            _ => rows[0]["model"] = json!("zero-shot"),
        }
        fs::write(
            &path,
            rows.iter().map(|r| format!("{r}\n")).collect::<String>(),
        )
        .unwrap();
        let output = directory.join("bad.json");
        assert!(write_guide3_report(
            &ledger,
            &compiled,
            &probs,
            &output,
            &[0.05],
            Guarantee::Change,
            42
        )
        .is_err());
        assert!(!output.exists());
        fs::remove_dir_all(directory).unwrap();
    }
}
