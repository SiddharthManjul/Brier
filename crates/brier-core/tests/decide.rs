use brier_core::{cascade::FEATURE_NAMES, decide::decide};
use parquet::{
    data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use rusqlite::Connection;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

static MODAL_ENV: Mutex<()> = Mutex::new(());

struct Fixture {
    directory: PathBuf,
    ledger: PathBuf,
    compiled: PathBuf,
    calibration: PathBuf,
}

impl Fixture {
    fn new(t1: f64, risk: f64) -> Self {
        let directory = std::env::temp_dir().join(format!("brier-decide-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let ledger = directory.join("ledger.sqlite");
        let compiled = directory.join("compiled.jsonl");
        let calibration = directory.join("risk.json");
        let connection = brier_core::ledger::init(&ledger).unwrap();
        connection.execute_batch("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<10)
            INSERT INTO changes SELECT printf('c%03d',x),'gym_pusht/PushT-v0',json_object('description','policy update','families',json_array(json_object('family','action_latency','params',json_object('k',x%4+1)))),'now' FROM n;
            INSERT INTO scenarios VALUES('pusht-s01','gym_pusht/PushT-v0','{\"pose_bin\":0,\"lighting\":\"normal\",\"camera_jitter\":false}',0.8);
            INSERT INTO labels SELECT change_id,'pusht-s01',CAST(substr(change_id,2) AS INTEGER)%2,'stall/timeout',3,'action_latency' FROM changes;
            INSERT INTO runs SELECT change_id||'-cheap',change_id,scenario_id,'cheap',3,2,0.41,'raw',3.0 FROM labels;").unwrap();
        brier_core::compile::compile(&ledger, &compiled).unwrap();
        connection.execute_batch("WITH RECURSIVE n(x) AS (SELECT 100 UNION ALL SELECT x+1 FROM n WHERE x<118)
            INSERT INTO changes SELECT printf('c%03d',x),'gym_pusht/PushT-v0','{\"description\":\"risk pool\",\"families\":[{\"family\":\"action_latency\",\"params\":{\"k\":2}}]}','later' FROM n;
            INSERT INTO labels SELECT change_id,'pusht-s01',1,'stall/timeout',1,'action_latency' FROM changes WHERE change_id BETWEEN 'c100' AND 'c118';
            INSERT INTO changes VALUES('c999','gym_pusht/PushT-v0','{\"description\":\"new policy\",\"families\":[{\"family\":\"action_latency\",\"params\":{\"k\":2}}]}','latest');
            INSERT INTO runs VALUES('live-cheap','c999','pusht-s01','cheap',3,2,0.41,'raw',3.0);").unwrap();
        let t1_artifact = json!({"format_version":1, "model_version": format!("t1@{}", "a".repeat(64)),
            "training_change_ids": (1..=6).map(|n| format!("c{n:03}")).collect::<Vec<_>>(),
            "calibration_change_ids": ["c007", "c008"], "feature_names": FEATURE_NAMES.as_slice(),
            "booster": {"objective": "binary sigmoid:1", "tree_info": [{"tree_structure": {"leaf_value": 0.0}}]},
            "isotonic": {"x": [0.5], "y": [t1]}});
        let artifact = json!({"format_version":1, "model_version": format!("example/model@{}", "b".repeat(40)),
            "t1": t1_artifact, "temperatures": {"regressed":1.0,"failure_mode":1.0,"success_bucket":1.0},
            "probability_calibration_change_ids": ["c007","c008"],
            "changes": (100..=118).map(|n| json!({"change_id":format!("c{n:03}"), "pairs":[{
                "scenario_id":"pusht-s01", "probability":risk, "regressed":true}]})).collect::<Vec<_>>()});
        fs::write(&calibration, serde_json::to_vec(&artifact).unwrap()).unwrap();
        Self {
            directory,
            ledger,
            compiled,
            calibration,
        }
    }

    fn artifact(&self) -> Value {
        serde_json::from_slice(&fs::read(&self.calibration).unwrap()).unwrap()
    }

    fn save(&self, artifact: &Value) {
        fs::write(&self.calibration, serde_json::to_vec(artifact).unwrap()).unwrap();
    }

    fn decision_count(&self) -> i64 {
        Connection::open(&self.ledger)
            .unwrap()
            .query_row("SELECT count(*) FROM decisions", [], |row| row.get(0))
            .unwrap()
    }

    fn add_complete_change(&self, number: usize, split: &str) {
        let change = format!("c{number:03}");
        let connection = Connection::open(&self.ledger).unwrap();
        connection.execute("INSERT INTO changes VALUES(?1,'gym_pusht/PushT-v0','{\"description\":\"independent calibration collection\",\"families\":[{\"family\":\"action_latency\",\"params\":{\"k\":2}}]}','later')", [&change]).unwrap();
        connection
            .execute(
                "INSERT INTO labels VALUES(?1,'pusht-s01',1,'stall/timeout',3,'HIDDEN_CAUSE')",
                [&change],
            )
            .unwrap();
        for (tier, count) in [("cheap", 3), ("expensive", 20)] {
            connection
                .execute(
                    "INSERT INTO runs VALUES(?1,?2,'pusht-s01',?3,?4,2,0.41,'raw',3.0)",
                    rusqlite::params![uuid::Uuid::new_v4().to_string(), change, tier, count],
                )
                .unwrap();
        }
        let source = fs::read_to_string(&self.compiled).unwrap();
        let templates: Vec<Value> = source
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|row| row["id"].as_str().unwrap().starts_with("c007/"))
            .collect();
        assert_eq!(templates.len(), 6);
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&self.compiled)
            .unwrap();
        for mut row in templates {
            row["id"] =
                json!(row["id"]
                    .as_str()
                    .unwrap()
                    .replacen("c007/", &format!("{change}/"), 1));
            row["split"] = json!(split);
            writeln!(file, "{}", row).unwrap();
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

fn server(
    version: String,
    missing_option: bool,
    probability: f64,
) -> (String, thread::JoinHandle<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/decide", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let handle = thread::spawn(move || {
        let started = Instant::now();
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        started.elapsed() < Duration::from_secs(5),
                        "test server received no request"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let (offset, length) = loop {
            let mut buffer = [0_u8; 8192];
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(offset) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..offset]);
                assert!(headers.starts_with("POST /decide HTTP/1.1"));
                let length: usize = headers
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .unwrap()
                    .1
                    .trim()
                    .parse()
                    .unwrap();
                break (offset + 4, length);
            }
        };
        while bytes.len() < offset + length {
            let mut buffer = [0_u8; 8192];
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
        }
        let request: Value = serde_json::from_slice(&bytes[offset..offset + length]).unwrap();
        let mut answers = serde_json::Map::new();
        for question in request["questions"].as_array().unwrap() {
            let id = question["id"].as_str().unwrap();
            let probabilities = if id.contains("/regressed/") {
                if missing_option {
                    json!({"A":0.8})
                } else {
                    json!({"A":probability,"B":1.0-probability})
                }
            } else {
                let options = question["options"].as_object().unwrap();
                Value::Object(
                    options
                        .keys()
                        .map(|key| (key.clone(), json!(1.0 / options.len() as f64)))
                        .collect(),
                )
            };
            answers.insert(id.to_string(), probabilities);
        }
        let response =
            serde_json::to_string(&json!({"model_version":version,"questions":answers})).unwrap();
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).unwrap();
        request
    });
    (url, handle)
}

fn cheap_parquet(path: &Path, directory: &Path) -> Vec<PathBuf> {
    let png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\0\x01\0\0\0\x01\x08\x02\0\0\0\x90wS\xde\0\0\0\x0cIDATx\x9cc\xf8\xcf\xc0\0\0\x03\x01\x01\0\xc9\xfe\x92\xef\0\0\0\0IEND\xaeB`\x82";
    let frames: Vec<_> = (0..3)
        .map(|seed| {
            let relative = format!(
                "public/keyframes/pusht/c999/pusht-s01/{seed}/{}",
                uuid::Uuid::new_v4()
            );
            fs::create_dir_all(directory.join(&relative)).unwrap();
            fs::write(directory.join(&relative).join("frame-0010.png"), png).unwrap();
            relative
        })
        .collect();
    let schema = Arc::new(
        parse_message_type(
            "message episodes {
        REQUIRED BYTE_ARRAY env (UTF8); REQUIRED BYTE_ARRAY policy (UTF8);
        REQUIRED BYTE_ARRAY change_id (UTF8); REQUIRED BYTE_ARRAY scenario_id (UTF8);
        REQUIRED BYTE_ARRAY tier (UTF8); REQUIRED INT64 seed; REQUIRED BOOLEAN success;
        REQUIRED INT64 steps; REQUIRED BYTE_ARRAY terminal_state (UTF8); REQUIRED DOUBLE reward_sum;
        OPTIONAL BYTE_ARRAY keyframes_uri (UTF8); REQUIRED DOUBLE wall_seconds; }",
        )
        .unwrap(),
    );
    let mut writer = SerializedFileWriter::new(
        File::create(path).unwrap(),
        schema,
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    let terminal = json!({"trajectory":{"collision_steps":0,"action_direction_changes":0,
        "action_transitions":9,"object_max_height":0.0,"object_final_height":0.0,
        "goal_distance_min":1.0,"goal_distance_final":2.0},
        "cause_family":"HIDDEN_LABEL","card":"HIDDEN_CARD"})
    .to_string();
    let mut group = writer.next_row_group().unwrap();
    for index in 0..12 {
        let mut column = group.next_column().unwrap().unwrap();
        match index {
            5 | 7 => {
                let values = if index == 5 { [0, 1, 2] } else { [10; 3] };
                column
                    .typed::<Int64Type>()
                    .write_batch(&values, None, None)
                    .unwrap();
            }
            6 => {
                column
                    .typed::<BoolType>()
                    .write_batch(&[true, true, false], None, None)
                    .unwrap();
            }
            9 | 11 => {
                let value = if index == 9 { 0.41 } else { 1.0 };
                column
                    .typed::<DoubleType>()
                    .write_batch(&[value; 3], None, None)
                    .unwrap();
            }
            10 => {
                let values: Vec<_> = frames
                    .iter()
                    .map(|value| ByteArray::from(value.as_str()))
                    .collect();
                column
                    .typed::<ByteArrayType>()
                    .write_batch(&values, Some(&[1; 3]), None)
                    .unwrap();
            }
            _ => {
                let value = match index {
                    0 => "gym_pusht/PushT-v0",
                    1 => "lerobot/diffusion_pusht",
                    2 => "c999",
                    3 => "pusht-s01",
                    4 => "cheap",
                    8 => &terminal,
                    _ => unreachable!(),
                };
                column
                    .typed::<ByteArrayType>()
                    .write_batch(&vec![ByteArray::from(value); 3], None, None)
                    .unwrap();
            }
        }
        column.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
    frames
        .iter()
        .map(|relative| fs::canonicalize(directory.join(relative).join("frame-0010.png")).unwrap())
        .collect()
}

#[test]
fn uncertain_t2_reads_actual_cheap_evidence_and_t3_only_adds_evaluations() {
    use std::os::unix::fs::PermissionsExt;

    let python = Command::new("python3")
        .args(["-c", "import sys; print(sys.executable)"])
        .output()
        .unwrap();
    assert!(python.status.success());
    let python = String::from_utf8(python.stdout).unwrap();
    for (recommend, risk, reason) in [
        ("run_expensive", 0.8, "t3_addition"),
        ("no_change", 0.4, "threshold"),
    ] {
        let fixture = Fixture::new(0.5, risk);
        let raw = fixture.directory.join("cheap.parquet");
        let frames = cheap_parquet(&raw, &fixture.directory);
        let connection = Connection::open(&fixture.ledger).unwrap();
        connection
            .execute_batch("DROP TRIGGER runs_no_UPDATE")
            .unwrap();
        connection
            .execute(
                "UPDATE runs SET uri=?1 WHERE run_id='live-cheap'",
                [raw.to_str().unwrap()],
            )
            .unwrap();
        let bin = fixture.directory.join("bin");
        fs::create_dir(&bin).unwrap();
        fs::write(fixture.directory.join("recommend"), recommend).unwrap();
        let bridge = format!(
            "#!{}\n{}",
            python.trim(),
            r#"
import json, os, sys
from pathlib import Path
assert sys.argv[1:] == ['-m', 'brier_explain.client']
root = Path(os.environ['BRIER_DATA'])
payload = json.load(sys.stdin)
assert payload['operation'] == 'escalate'
assert len(payload['frames']) == 3
for frame in payload['frames']:
    assert Path(frame['path']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
with (root / 't3-request.json').open('x') as stream: json.dump(payload, stream)
json.dump({'model':'muse-spark-1.3-contributor','recommend':(root/'recommend').read_text(),
           'rationale':'Three recorded final frames justify this additive review.','cost_usd':None}, sys.stdout)
"#
        );
        fs::write(bin.join("python3"), bridge).unwrap();
        fs::set_permissions(bin.join("python3"), fs::Permissions::from_mode(0o755)).unwrap();
        let artifact = fixture.artifact();
        let (url, handle) = server(
            artifact["model_version"].as_str().unwrap().to_string(),
            false,
            0.5,
        );
        let result = Command::new(env!("CARGO_BIN_EXE_brier"))
            .arg("decide")
            .arg("--ledger")
            .arg(&fixture.ledger)
            .arg("--compiled")
            .arg(&fixture.compiled)
            .arg("--calibration")
            .arg(&fixture.calibration)
            .args(["--change", "c999"])
            .env("MODAL_DECIDE_URL", url)
            .env("BRIER_DATA", &fixture.directory)
            .env("PATH", &bin)
            .output()
            .unwrap();
        let t2 = handle.join().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: Value = serde_json::from_slice(&result.stdout).unwrap();
        let decision = &report["decisions"][0];
        assert_eq!(decision["tier_used"], "T2+T3");
        assert_eq!(decision["action"], "expensive");
        assert_eq!(decision["reason"], reason);
        assert_eq!(decision["propensity"], 1.0);
        assert_eq!(decision["lambda"], risk);
        assert_eq!(decision["probabilities"]["regressed"]["yes"], 0.5);
        assert_eq!(
            decision["probabilities"]["escalation"]["recommend"],
            recommend
        );
        assert_eq!(
            decision["probabilities"]["escalation"]["model"],
            "muse-spark-1.3-contributor"
        );
        let request: Value =
            serde_json::from_slice(&fs::read(fixture.directory.join("t3-request.json")).unwrap())
                .unwrap();
        assert_eq!(request.as_object().unwrap().len(), 3);
        let prompt = request["prompt"].as_str().unwrap();
        assert!(prompt.contains(t2["questions"][0]["prompt"].as_str().unwrap()));
        assert!(prompt.contains("CHEAP_OBSERVATIONS:") && prompt.contains("\"reward_sum\":0.41"));
        assert!(!prompt.contains("HIDDEN_LABEL") && !prompt.contains("HIDDEN_CARD"));
        for (index, path) in frames.iter().enumerate() {
            assert_eq!(request["frames"][index], json!({"index":index,"path":path}));
        }
        let stored: String = connection
            .query_row(
                "SELECT probs_json FROM decisions WHERE change_id='c999'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&stored).unwrap(),
            decision["probabilities"]
        );
        assert_eq!(fixture.decision_count(), 1);
        let labels: i64 = connection
            .query_row(
                "SELECT count(*) FROM labels WHERE change_id='c999'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(labels, 0);
    }
}

#[test]
fn live_t1_decisions_are_logged_once_without_computing_labels() {
    let fixture = Fixture::new(0.95, 0.95);
    let report = decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05,
    )
    .unwrap();
    let decision = &report.decisions[0];
    assert_eq!(decision.tier_used, "T1");
    assert_eq!(decision.action, "expensive");
    assert_eq!(decision.propensity, 1.0);
    assert_eq!(decision.reason, "threshold");
    assert_eq!(decision.probabilities["regressed"]["yes"], 0.95);
    assert!(uuid::Uuid::parse_str(&decision.decision_id).is_ok());
    assert_eq!(fixture.decision_count(), 1);
    let connection = Connection::open(&fixture.ledger).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM labels WHERE change_id='c999'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert!(decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05
    )
    .is_err());
    assert_eq!(fixture.decision_count(), 1);
}

#[test]
fn independent_calibration_refresh_waits_for_ten_and_persists_all_candidates() {
    for count in [9, 10, 12] {
        let fixture = Fixture::new(0.95, 0.95);
        for number in 200..200 + count {
            fixture.add_complete_change(number, "calibration");
        }
        let original = fs::read(&fixture.calibration).unwrap();
        let report = decide(
            &fixture.ledger,
            &fixture.compiled,
            &fixture.calibration,
            "c999",
            0.05,
        )
        .unwrap();
        let expected = if count < 10 { 19 } else { 19 + count };
        assert_eq!(report.policy.calibration_changes, expected);
        let artifact = fixture.artifact();
        assert_eq!(artifact["changes"].as_array().unwrap().len(), expected);
        if count < 10 {
            assert_eq!(fs::read(&fixture.calibration).unwrap(), original);
        } else {
            let added = &artifact["changes"].as_array().unwrap()[19..];
            for (offset, change) in added.iter().enumerate() {
                assert_eq!(change["change_id"], format!("c{:03}", 200 + offset));
                assert_eq!(change["pairs"][0]["probability"], 0.95);
                assert_eq!(change["pairs"][0]["regressed"], true);
            }
            assert_eq!(
                report.decisions[0].probabilities["policy"]["calibration_changes"],
                expected
            );
            let compiled =
                brier_core::compile::compile_live(&fixture.ledger, "c200", &fixture.compiled)
                    .unwrap();
            assert!(compiled
                .iter()
                .flat_map(|pair| &pair.questions)
                .all(|question| question.get("label").is_none()
                    && !question["prompt"]
                        .as_str()
                        .unwrap()
                        .contains("HIDDEN_CAUSE")));
        }
        assert_eq!(fixture.decision_count(), 1);
    }
}

#[test]
fn selective_heldout_fitted_incomplete_and_duplicate_results_do_not_trigger_refresh() {
    let fixture = Fixture::new(0.95, 0.95);
    for number in 200..209 {
        fixture.add_complete_change(number, "calibration");
    }
    for (number, split) in [
        (290, "calibration"),
        (291, "test"),
        (292, "ood"),
        (293, "calibration"),
        (294, "calibration"),
        (295, "calibration"),
    ] {
        fixture.add_complete_change(number, split);
    }
    let connection = Connection::open(&fixture.ledger).unwrap();
    connection.execute_batch("INSERT INTO decisions VALUES('selective','c290','pusht-s01','T1','fixed','{}',0.5,'expensive',1.0,'selected','later');
        DROP TRIGGER runs_no_UPDATE; UPDATE runs SET episodes=19 WHERE change_id='c293' AND tier='expensive';
        INSERT INTO runs SELECT 'duplicate-cheap',change_id,scenario_id,tier,episodes,successes,mean_reward,uri,wall_seconds FROM runs WHERE change_id='c294' AND tier='cheap';
        INSERT INTO runs SELECT 'fit-expensive',change_id,scenario_id,'expensive',20,2,0.4,'raw',3.0 FROM runs WHERE change_id='c007';
        DROP TRIGGER labels_no_DELETE; DELETE FROM labels WHERE change_id='c295';").unwrap();
    let original = fs::read(&fixture.calibration).unwrap();
    let report = decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05,
    )
    .unwrap();
    assert_eq!(report.policy.calibration_changes, 19);
    assert_eq!(fs::read(&fixture.calibration).unwrap(), original);
    assert_eq!(fixture.decision_count(), 2);
}

#[test]
fn failed_refresh_scoring_leaves_original_artifact_and_no_live_decisions() {
    let fixture = Fixture::new(0.95, 0.95);
    for number in 200..210 {
        fixture.add_complete_change(number, "calibration");
    }
    let mut artifact = fixture.artifact();
    artifact["t1"]["booster"]["objective"] = json!("regression");
    fixture.save(&artifact);
    let original = fs::read(&fixture.calibration).unwrap();
    assert!(decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05
    )
    .is_err());
    assert_eq!(fs::read(&fixture.calibration).unwrap(), original);
    assert_eq!(fixture.decision_count(), 0);
    assert!(!fs::read_dir(&fixture.directory).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with(".brier-risk-")));
}

#[test]
fn probability_fitting_and_existing_risk_changes_are_not_counted_as_new() {
    for existing_pool in [false, true] {
        let fixture = Fixture::new(0.95, 0.95);
        for number in 200..210 {
            fixture.add_complete_change(number, "calibration");
        }
        let mut artifact = fixture.artifact();
        if existing_pool {
            artifact["changes"][0]["change_id"] = json!("c200");
        } else {
            artifact["probability_calibration_change_ids"]
                .as_array_mut()
                .unwrap()
                .push(json!("c200"));
        }
        fixture.save(&artifact);
        let original = fs::read(&fixture.calibration).unwrap();
        let report = decide(
            &fixture.ledger,
            &fixture.compiled,
            &fixture.calibration,
            "c999",
            0.05,
        )
        .unwrap();
        assert_eq!(report.policy.calibration_changes, 19);
        assert_eq!(fs::read(&fixture.calibration).unwrap(), original);
        assert_eq!(fixture.decision_count(), 1);
    }
}

#[test]
fn successful_independent_refresh_survives_a_later_live_evidence_failure() {
    let fixture = Fixture::new(0.95, 0.95);
    for number in 200..210 {
        fixture.add_complete_change(number, "calibration");
    }
    Connection::open(&fixture.ledger)
        .unwrap()
        .execute_batch(
            "DROP TRIGGER runs_no_UPDATE; UPDATE runs SET episodes=2 WHERE change_id='c999';",
        )
        .unwrap();
    let error = decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05,
    )
    .unwrap_err();
    assert!(error.to_string().contains("three-episode"));
    assert_eq!(fixture.artifact()["changes"].as_array().unwrap().len(), 29);
    assert_eq!(fixture.decision_count(), 0);
}

#[test]
fn live_training_changes_are_rejected_even_when_their_runtime_label_is_absent() {
    let fixture = Fixture::new(0.95, 0.95);
    Connection::open(&fixture.ledger)
        .unwrap()
        .execute_batch("DROP TRIGGER labels_no_DELETE; DELETE FROM labels WHERE change_id='c001';")
        .unwrap();
    let error = decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c001",
        0.05,
    )
    .unwrap_err();
    assert!(error.to_string().contains("model training change"));
    assert_eq!(fixture.decision_count(), 0);
}

#[test]
fn uncertain_pairs_send_rust_questions_and_scale_semantic_probabilities() {
    let _guard = MODAL_ENV.lock().unwrap();
    let fixture = Fixture::new(0.5, 0.8);
    let mut artifact = fixture.artifact();
    artifact["temperatures"]["regressed"] = json!(2.0);
    artifact["model_version"] = json!(format!(
        "example/model@{}+lora@{}",
        "b".repeat(40),
        "c".repeat(64)
    ));
    fixture.save(&artifact);
    let (url, handle) = server(
        artifact["model_version"].as_str().unwrap().to_string(),
        false,
        0.99,
    );
    let previous = std::env::var_os("MODAL_DECIDE_URL");
    std::env::set_var("MODAL_DECIDE_URL", url);
    let result = decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05,
    );
    if let Some(previous) = previous {
        std::env::set_var("MODAL_DECIDE_URL", previous);
    } else {
        std::env::remove_var("MODAL_DECIDE_URL");
    }
    let request = handle.join().unwrap();
    let report = result.unwrap();
    assert_eq!(report.decisions[0].tier_used, "T2");
    assert!(
        (report.decisions[0].probabilities["regressed"]["yes"]
            .as_f64()
            .unwrap()
            - 0.99_f64.sqrt() / (0.99_f64.sqrt() + 0.01_f64.sqrt()))
        .abs()
            < 1e-12
    );
    assert_eq!(report.decisions[0].model_version, artifact["model_version"]);
    assert_eq!(request["questions"].as_array().unwrap().len(), 3);
    for question in request["questions"].as_array().unwrap() {
        assert_eq!(question.as_object().unwrap().len(), 3);
        assert!(question["prompt"]
            .as_str()
            .unwrap()
            .contains("CHEAP_TIER: success=2/3"));
        assert!(question.get("label").is_none());
    }
}

#[test]
fn wrong_model_versions_and_incomplete_distributions_leave_no_decisions() {
    let _guard = MODAL_ENV.lock().unwrap();
    for missing_option in [false, true] {
        let fixture = Fixture::new(0.5, 0.8);
        let artifact = fixture.artifact();
        let version = if missing_option {
            artifact["model_version"].as_str().unwrap().to_string()
        } else {
            "wrong-model".into()
        };
        let (url, handle) = server(version, missing_option, 0.99);
        let previous = std::env::var_os("MODAL_DECIDE_URL");
        std::env::set_var("MODAL_DECIDE_URL", url);
        let result = decide(
            &fixture.ledger,
            &fixture.compiled,
            &fixture.calibration,
            "c999",
            0.05,
        );
        if let Some(previous) = previous {
            std::env::set_var("MODAL_DECIDE_URL", previous);
        } else {
            std::env::remove_var("MODAL_DECIDE_URL");
        }
        handle.join().unwrap();
        assert!(result.is_err());
        assert_eq!(fixture.decision_count(), 0);
    }
}

#[test]
fn risk_provenance_labels_and_finite_sample_requirements_are_enforced() {
    for mutation in ["overlap", "label", "small", "legacy", "live_fit"] {
        let fixture = Fixture::new(0.95, 0.95);
        let mut artifact = fixture.artifact();
        match mutation {
            "overlap" => artifact["changes"][0]["change_id"] = json!("c007"),
            "label" => artifact["changes"][0]["pairs"][0]["regressed"] = json!(false),
            "small" => {
                artifact["changes"].as_array_mut().unwrap().pop();
            }
            "live_fit" => artifact["probability_calibration_change_ids"]
                .as_array_mut()
                .unwrap()
                .push(json!("c999")),
            _ => {
                artifact["t1"]
                    .as_object_mut()
                    .unwrap()
                    .remove("training_change_ids");
            }
        }
        fixture.save(&artifact);
        assert!(decide(
            &fixture.ledger,
            &fixture.compiled,
            &fixture.calibration,
            "c999",
            0.05
        )
        .is_err());
        assert_eq!(fixture.decision_count(), 0);
    }
}

#[test]
fn incomplete_cheap_evidence_fails_and_benign_names_cannot_bypass_scoring() {
    let fixture = Fixture::new(0.95, 0.95);
    let connection = Connection::open(&fixture.ledger).unwrap();
    connection
        .execute_batch(
            "DROP TRIGGER runs_no_UPDATE; UPDATE runs SET episodes=2 WHERE change_id='c999';",
        )
        .unwrap();
    let error = decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05,
    )
    .unwrap_err();
    assert!(error.to_string().contains("three-episode"));
    assert_eq!(fixture.decision_count(), 0);
    connection.execute_batch("UPDATE runs SET episodes=3 WHERE change_id='c999'; DROP TRIGGER changes_no_UPDATE;
        UPDATE changes SET card_json='{\"description\":\"metadata only claimed\",\"families\":[{\"family\":\"benign\",\"params\":{}}]}' WHERE change_id='c999';").unwrap();
    let report = decide(
        &fixture.ledger,
        &fixture.compiled,
        &fixture.calibration,
        "c999",
        0.05,
    )
    .unwrap();
    assert_eq!(report.decisions[0].tier_used, "T1");
    assert_eq!(report.decisions[0].action, "expensive");
}

#[test]
fn decide_cli_emits_the_logged_report() {
    let fixture = Fixture::new(0.95, 0.95);
    let result = Command::new(env!("CARGO_BIN_EXE_brier"))
        .arg("decide")
        .arg("--ledger")
        .arg(&fixture.ledger)
        .arg("--compiled")
        .arg(&fixture.compiled)
        .arg("--calibration")
        .arg(&fixture.calibration)
        .args(["--change", "c999"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["change_id"], "c999");
    assert_eq!(report["policy"]["calibration_changes"], 19);
    assert_eq!(fixture.decision_count(), 1);
}
