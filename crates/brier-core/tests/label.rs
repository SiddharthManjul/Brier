use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use parquet::{
    basic::Compression,
    data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use rusqlite::Connection;
use serde_json::{json, Value};
use uuid::Uuid;

struct Fixture {
    directory: PathBuf,
    ledger: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

fn write_episodes(path: &Path, tier: &str, count: usize, successes: usize, telemetry: bool) {
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
    let properties = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build(),
    );
    let mut writer =
        SerializedFileWriter::new(File::create(path).unwrap(), schema, properties).unwrap();
    let terminal = if telemetry {
        json!({"trajectory": {"collision_steps": 0, "action_direction_changes": 0, "action_transitions": 10, "object_max_height": 0.0, "object_final_height": 0.0, "goal_distance_min": 10.0, "goal_distance_final": 10.0}}).to_string()
    } else {
        "{}".into()
    };
    let mut group = writer.next_row_group().unwrap();
    for column in 0..12 {
        let mut col = group.next_column().unwrap().unwrap();
        match column {
            5 | 7 => {
                let values: Vec<i64> = (0..count)
                    .map(|n| if column == 5 { n as i64 } else { 10 })
                    .collect();
                col.typed::<Int64Type>()
                    .write_batch(&values, None, None)
                    .unwrap();
            }
            6 => {
                col.typed::<BoolType>()
                    .write_batch(
                        &(0..count).map(|n| n < successes).collect::<Vec<_>>(),
                        None,
                        None,
                    )
                    .unwrap();
            }
            9 | 11 => {
                col.typed::<DoubleType>()
                    .write_batch(&vec![1.0; count], None, None)
                    .unwrap();
            }
            10 => {
                col.typed::<ByteArrayType>()
                    .write_batch(&[], Some(&vec![0; count]), None)
                    .unwrap();
            }
            _ => {
                let value = match column {
                    0 => "gym_pusht/PushT-v0",
                    1 => "lerobot/diffusion_pusht",
                    2 => {
                        if tier == "baseline" {
                            "c001"
                        } else {
                            "c002"
                        }
                    }
                    3 => "pusht-s01",
                    4 => tier,
                    8 => &terminal,
                    _ => unreachable!(),
                };
                col.typed::<ByteArrayType>()
                    .write_batch(&vec![ByteArray::from(value); count], None, None)
                    .unwrap();
            }
        }
        col.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
}

fn fixture(successes: usize, telemetry: bool, expensive_count: usize) -> Fixture {
    let directory = std::env::temp_dir().join(format!("brier-label-{}", Uuid::new_v4()));
    fs::create_dir_all(directory.join("episodes")).unwrap();
    let ledger = directory.join("ledger.sqlite");
    brier_core::ledger::init(&ledger).unwrap();
    fs::write(directory.join("catalog.json"), json!({"changes": [
      {"change_id":"c001","env":"gym_pusht/PushT-v0","card":{"description":"unchanged","families":[{"family":"benign","params":{}}]}},
      {"change_id":"c002","env":"gym_pusht/PushT-v0","card":{"description":"gain change","families":[{"family":"action_scale","params":{"scale":0.9}}]}}
    ],"scenarios":[{"scenario_id":"pusht-s01","env":"gym_pusht/PushT-v0","params":{"pose_bin":0,"lighting":"normal","camera_jitter":false},"seeds":(0..50).collect::<Vec<_>>()}]}).to_string()).unwrap();
    for (tier, count, wins) in [
        ("baseline", 50, 40),
        ("cheap", 3, 2),
        ("expensive", expensive_count, successes),
    ] {
        write_episodes(
            &directory.join("episodes").join(format!("{tier}.parquet")),
            tier,
            count,
            wins,
            telemetry,
        );
    }
    Fixture { directory, ledger }
}

fn command(f: &Fixture) -> std::process::Output {
    command_at(f, &f.directory.join("episodes"))
}

fn command_at(f: &Fixture, episodes: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_brier"))
        .arg("label")
        .arg("--ledger")
        .arg(&f.ledger)
        .arg("--episodes")
        .arg(episodes)
        .arg("--catalog")
        .arg(f.directory.join("catalog.json"))
        .output()
        .unwrap()
}

fn stage(f: &Fixture, name: &str, tier: &str) -> PathBuf {
    let path = f.directory.join(name);
    fs::create_dir(&path).unwrap();
    fs::copy(
        f.directory.join("episodes").join(format!("{tier}.parquet")),
        path.join(format!("{tier}.parquet")),
    )
    .unwrap();
    path
}

fn counts(f: &Fixture) -> (i64, i64) {
    Connection::open(&f.ledger)
        .unwrap()
        .query_row(
            "SELECT (SELECT count(*) FROM runs),(SELECT count(*) FROM labels)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap()
}

#[test]
fn label_imports_snappy_records_and_uses_strict_regression_boundary() {
    for (wins, regressed, bucket) in [
        (13, 0, 4),
        (12, 1, 3),
        (4, 1, 1),
        (5, 1, 2),
        (20, 0, 5),
        (0, 1, 1),
    ] {
        let f = fixture(wins, true, 20);
        let output = command(&f);
        assert!(output.status.success(), "{:?}", output);
        let connection = Connection::open(&f.ledger).unwrap();
        let row: (i64, String, i64, String) = connection
            .query_row(
                "SELECT regressed,failure_mode,success_bucket,cause_family FROM labels",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            row,
            (
                regressed,
                if wins == 20 { "none" } else { "stall/timeout" }.into(),
                bucket,
                "action_scale".into()
            )
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM runs", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            3
        );
    }
}

#[test]
fn label_rejects_missing_failure_telemetry_without_changing_ledger() {
    let f = fixture(12, false, 20);
    let output = command(&f);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires trajectory"));
    assert_eq!(
        Connection::open(&f.ledger)
            .unwrap()
            .query_row("SELECT count(*) FROM changes", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn label_rejects_partial_and_duplicate_groups_atomically() {
    let f = fixture(12, true, 19);
    assert!(!command(&f).status.success());
    write_episodes(
        &f.directory.join("episodes/expensive.parquet"),
        "expensive",
        20,
        12,
        true,
    );
    fs::copy(
        f.directory.join("episodes/cheap.parquet"),
        f.directory.join("episodes/duplicate.parquet"),
    )
    .unwrap();
    let output = command(&f);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("duplicate episode seed"));
    fs::remove_file(f.directory.join("episodes/duplicate.parquet")).unwrap();
    assert!(command(&f).status.success());
    let repeated = command(&f);
    assert!(!repeated.status.success());
    assert!(String::from_utf8_lossy(&repeated.stderr).contains("already imported"));
    assert_eq!(
        Connection::open(&f.ledger)
            .unwrap()
            .query_row("SELECT count(*) FROM labels", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn staged_baseline_cheap_and_expensive_imports_create_one_authoritative_label() {
    let f = fixture(12, true, 20);
    let baseline = stage(&f, "baseline-stage", "baseline");
    let cheap = stage(&f, "cheap-stage", "cheap");
    let expensive = stage(&f, "expensive-stage", "expensive");
    let first = command_at(&f, &baseline);
    assert!(first.status.success(), "{:?}", first);
    assert_eq!(counts(&f), (1, 0));
    let repeated = command_at(&f, &baseline);
    assert!(!repeated.status.success());
    assert!(String::from_utf8_lossy(&repeated.stderr).contains("already imported"));
    fs::write(
        f.directory.join("episodes/expensive.parquet"),
        b"unavailable ground truth",
    )
    .unwrap();
    let stage_a = command_at(&f, &cheap);
    assert!(stage_a.status.success(), "{:?}", stage_a);
    assert_eq!(counts(&f), (2, 0));
    assert!(!command_at(&f, &cheap).status.success());
    assert_eq!(counts(&f), (2, 0));
    let stage_b = command_at(&f, &expensive);
    assert!(stage_b.status.success(), "{:?}", stage_b);
    assert_eq!(counts(&f), (3, 1));
    let connection = Connection::open(&f.ledger).unwrap();
    let label: (String, i64, String, i64) = connection
        .query_row(
            "SELECT change_id,regressed,failure_mode,success_bucket FROM labels",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(label, ("c002".into(), 1, "stall/timeout".into(), 3));
    assert!(!command_at(&f, &expensive).status.success());
    assert_eq!(counts(&f), (3, 1));
}

#[test]
fn partial_stages_and_expensive_before_cheap_leave_prior_state_unchanged() {
    let f = fixture(12, true, 20);
    let baseline = stage(&f, "baseline-stage", "baseline");
    let cheap = stage(&f, "cheap-stage", "cheap");
    let expensive = stage(&f, "expensive-stage", "expensive");
    let missing_baseline = command_at(&f, &cheap);
    assert!(!missing_baseline.status.success());
    assert!(String::from_utf8_lossy(&missing_baseline.stderr).contains("50-episode baseline"));
    assert_eq!(counts(&f), (0, 0));
    assert!(command_at(&f, &baseline).status.success());
    let missing_cheap = command_at(&f, &expensive);
    assert!(!missing_cheap.status.success());
    assert!(String::from_utf8_lossy(&missing_cheap.stderr).contains("three-episode cheap run"));
    assert_eq!(counts(&f), (1, 0));
    write_episodes(&cheap.join("cheap.parquet"), "cheap", 2, 1, false);
    assert!(!command_at(&f, &cheap).status.success());
    assert_eq!(counts(&f), (1, 0));
    write_episodes(&cheap.join("cheap.parquet"), "cheap", 3, 2, false);
    assert!(command_at(&f, &cheap).status.success());
    assert_eq!(counts(&f), (2, 0));
    write_episodes(
        &expensive.join("expensive.parquet"),
        "expensive",
        19,
        12,
        true,
    );
    assert!(!command_at(&f, &expensive).status.success());
    assert_eq!(counts(&f), (2, 0));
    write_episodes(
        &expensive.join("expensive.parquet"),
        "expensive",
        20,
        12,
        true,
    );
    assert!(command_at(&f, &expensive).status.success());
    assert_eq!(counts(&f), (3, 1));
}

#[test]
fn staged_import_rejects_catalog_drift_against_stored_baseline_scenario() {
    let f = fixture(12, true, 20);
    let baseline = stage(&f, "baseline-stage", "baseline");
    let cheap = stage(&f, "cheap-stage", "cheap");
    assert!(command_at(&f, &baseline).status.success());
    let path = f.directory.join("catalog.json");
    let mut catalog: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    catalog["scenarios"][0]["params"]["lighting"] = json!("low");
    fs::write(&path, serde_json::to_vec(&catalog).unwrap()).unwrap();
    let result = command_at(&f, &cheap);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("catalog changes existing scenario"));
    assert_eq!(counts(&f), (1, 0));
}

fn guide_fixture(env: &str, baseline_wins: usize, wins: usize) -> Fixture {
    let directory = std::env::temp_dir().join(format!("brier-guide-label-{}", Uuid::new_v4()));
    fs::create_dir_all(&directory).unwrap();
    let ledger = directory.join("ledger.sqlite");
    brier_core::ledger::init(&ledger).unwrap();
    let change = if env == "pusht" { "c002" } else { "c081" };
    let scenario = format!("{env}-s01");
    let seeds: Vec<_> = (0..50).map(|index| 11 + index * 13).collect();
    let card = json!({"change_id":change,"env":env,"desc":"Move inference to edge box",
        "families":[{"family":"action_latency","params":{"k":2}}],"seed":2026});
    let definition = json!({"scenario_id":scenario,"env":env,"params":{"pose_bin":1,"lighting":"normal","camera_jitter":false},
        "seeds":seeds,"baseline_success":baseline_wins as f64 / 50.0});
    fs::write(directory.join("changes.jsonl"), format!("{card}\n")).unwrap();
    fs::write(directory.join("scenarios.jsonl"), format!("{definition}\n")).unwrap();
    let mut rows = Vec::new();
    for (tier, count, successes) in [
        ("baseline", 50, baseline_wins),
        ("cheap", 3, 2),
        ("expensive", 20, wins),
    ] {
        for (index, seed) in seeds.iter().enumerate().take(count) {
            let success = index < successes;
            let peak = if success {
                if env == "pusht" {
                    1.0
                } else {
                    4.0
                }
            } else {
                0.0
            };
            rows.push(json!({"env":env,"policy_repo":if env == "pusht" {"lerobot/diffusion_pusht"} else {"lerobot/act_aloha_sim_transfer_cube_human"},
                "policy_revision":"84a7c23178445c6bbf7e1a884ff497017910f653","change_id":if tier == "baseline" {"base"} else {change},
                "scenario_id":scenario,"tier":tier,"episode_idx":index,"seed":seed,"success":success,"steps":100,"max_steps":300,
                "reward_sum":10.1,"max_reward":peak,"final_reward":peak,"coverage_final":if env == "pusht" {json!(0.1)} else {Value::Null},
                "coverage_max":if env == "pusht" {json!(0.1)} else {Value::Null},"block_disp_px":if env == "pusht" {json!(10.0)} else {Value::Null},
                "touched_right":if env == "aloha" {json!(false)} else {Value::Null},"touched_left":if env == "aloha" {json!(false)} else {Value::Null},
                "lifted":if env == "aloha" {json!(false)} else {Value::Null},"table_contact_steps":if env == "aloha" {json!(0)} else {Value::Null},
                "action_flip_rate":0.0,"terminal_state_json":"{}","keyframes_uri":Value::Null,"wall_seconds":0.1,
                "lerobot_version":"0.3.3","code_commit":"0123456789012345678901234567890123456789","host":"unit-test"}));
        }
    }
    write_jsonl(&directory.join("episodes.jsonl"), &rows);
    Fixture { directory, ledger }
}

fn write_jsonl(path: &Path, rows: &[Value]) {
    fs::write(
        path,
        rows.iter()
            .map(|row| format!("{row}\n"))
            .collect::<String>(),
    )
    .unwrap();
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn guide_command(f: &Fixture) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_brier"))
        .args(["label", "--ledger"])
        .arg(&f.ledger)
        .arg("--changes")
        .arg(f.directory.join("changes.jsonl"))
        .arg("--scenarios")
        .arg(f.directory.join("scenarios.jsonl"))
        .arg("--episodes")
        .arg(f.directory.join("episodes.jsonl"))
        .env("BRIER_DATA", f.directory.join("data"))
        .output()
        .unwrap()
}

fn guide_label(f: &Fixture) -> (i64, String, i64, String) {
    Connection::open(&f.ledger)
        .unwrap()
        .query_row(
            "SELECT regressed,failure_mode,success_bucket,cause_family FROM labels",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap()
}

#[test]
fn label_guide_rules_match_every_measured_branch_and_boundary_fixture() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../fixtures/label/guide_modes.json")).unwrap();
    for case in cases {
        let f = guide_fixture(
            case["env"].as_str().unwrap(),
            50,
            case["wins"].as_u64().unwrap() as usize,
        );
        let mut rows = read_jsonl(&f.directory.join("episodes.jsonl"));
        for row in rows
            .iter_mut()
            .filter(|row| row["tier"] == "expensive" && row["success"] == false)
        {
            for (key, value) in case["metrics"].as_object().unwrap() {
                row[key] = value.clone();
            }
        }
        write_jsonl(&f.directory.join("episodes.jsonl"), &rows);
        let output = guide_command(&f);
        assert!(output.status.success(), "{}: {:?}", case["name"], output);
        assert_eq!(
            guide_label(&f).1,
            case["expected"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn label_guide_buckets_use_half_open_fifths_and_regression_is_strict() {
    for wins in 0..=20 {
        let f = guide_fixture("pusht", 40, wins);
        let output = guide_command(&f);
        assert!(output.status.success(), "{:?}", output);
        let (regressed, mode, bucket, cause) = guide_label(&f);
        assert_eq!(regressed, i64::from(wins < 13));
        assert_eq!(mode, if wins < 13 { "stall/timeout" } else { "none" });
        assert_eq!(bucket, (wins as i64 / 4 + 1).min(5));
        assert_eq!(cause, "action_latency");
    }
}

#[test]
fn label_guide_ties_follow_rule_order_and_composite_cause_uses_first_family() {
    let f = guide_fixture("aloha", 50, 0);
    let mut cards = read_jsonl(&f.directory.join("changes.jsonl"));
    cards[0]["families"] = json!([{"family":"weight_noise","params":{"sigma":0.03}}, {"family":"action_latency","params":{"k":2}}]);
    write_jsonl(&f.directory.join("changes.jsonl"), &cards);
    let mut rows = read_jsonl(&f.directory.join("episodes.jsonl"));
    for row in rows.iter_mut().filter(|row| row["tier"] == "expensive") {
        if row["episode_idx"].as_u64().unwrap() < 10 {
            row["action_flip_rate"] = json!(0.36);
        } else {
            row["table_contact_steps"] = json!(20);
        }
    }
    write_jsonl(&f.directory.join("episodes.jsonl"), &rows);
    let output = guide_command(&f);
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(
        guide_label(&f),
        (1, "oscillation".into(), 1, "weight_noise".into())
    );
}

#[test]
fn label_guide_rerun_is_byte_identical_with_stable_v5_runs_and_exact_card_metadata() {
    let f = guide_fixture("pusht", 40, 12);
    let first = guide_command(&f);
    assert!(first.status.success(), "{:?}", first);
    let before = fs::read(&f.ledger).unwrap();
    let second = guide_command(&f);
    assert!(second.status.success(), "{:?}", second);
    assert_eq!(first.stdout, second.stdout);
    assert_eq!(before, fs::read(&f.ledger).unwrap());
    let connection = Connection::open(&f.ledger).unwrap();
    let (env, card): (String, String) = connection
        .query_row("SELECT env,card_json FROM changes", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(env, "gym_pusht/PushT-v0");
    assert_eq!(
        serde_json::from_str::<Value>(&card).unwrap(),
        read_jsonl(&f.directory.join("changes.jsonl"))[0]
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM changes WHERE change_id='base'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    let mut query = connection
        .prepare("SELECT run_id,change_id,scenario_id,tier,episodes,uri FROM runs ORDER BY tier")
        .unwrap();
    for row in query
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, String>(5)?,
            ))
        })
        .unwrap()
    {
        let (id, change, scenario, tier, count, uri) = row.unwrap();
        let key = format!("{change}:{scenario}:{tier}");
        assert_eq!(
            id,
            Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_bytes()).to_string()
        );
        assert_eq!(Uuid::parse_str(&id).unwrap().get_version_num(), 5);
        assert_eq!(
            count,
            match tier.as_str() {
                "baseline" => 50,
                "cheap" => 3,
                _ => 20,
            }
        );
        assert!(uri.ends_with(&format!(
            "/episodes/env=pusht/tier={tier}/change={change}/{scenario}.parquet"
        )));
    }
    let summary: Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(summary["pairs"], 1);
    assert_eq!(summary["regressed_rate"], 1.0);
    assert_eq!(summary["per_family"]["action_latency"]["pairs"], 1);
    assert_eq!(summary["per_scenario"]["pusht-s01"]["regressed_rate"], 1.0);
    assert_eq!(summary["per_env"]["pusht"]["regressed_rate"], 1.0);
    assert_eq!(summary["failure_modes"]["stall/timeout"], 1);
    assert!(summary["benign_false_regression_rate"].is_null());
}

#[test]
fn label_guide_benign_summary_reports_measured_false_regression() {
    let f = guide_fixture("pusht", 50, 0);
    let mut cards = read_jsonl(&f.directory.join("changes.jsonl"));
    cards[0]["families"] = json!([{"family":"benign","params":{}}]);
    write_jsonl(&f.directory.join("changes.jsonl"), &cards);
    let output = guide_command(&f);
    assert!(output.status.success(), "{:?}", output);
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["benign_false_regression_rate"], 1.0);
}

#[test]
fn label_guide_refuses_partial_duplicate_unpaired_and_invalid_measurements_atomically() {
    for fault in [
        "partial",
        "duplicate",
        "wrong-index",
        "wrong-seed",
        "wrong-env",
        "pilot",
        "base-card",
        "null-metric",
        "negative-time",
        "missing-cheap",
    ] {
        let f = guide_fixture("pusht", 40, 12);
        let mut rows = read_jsonl(&f.directory.join("episodes.jsonl"));
        let last = rows.len() - 1;
        match fault {
            "partial" => {
                rows.pop();
            }
            "duplicate" => rows.push(rows[last].clone()),
            "wrong-index" => rows[last]["episode_idx"] = json!(19_999),
            "wrong-seed" => rows[last]["seed"] = json!(123_456),
            "wrong-env" => rows[last]["env"] = json!("aloha"),
            "pilot" => rows[last]["tier"] = json!("pilot"),
            "base-card" => rows[0]["change_id"] = json!("c002"),
            "null-metric" => rows[last]["coverage_max"] = Value::Null,
            "negative-time" => rows[last]["wall_seconds"] = json!(-0.1),
            _ => rows.retain(|row| row["tier"] != "cheap"),
        }
        write_jsonl(&f.directory.join("episodes.jsonl"), &rows);
        let before = fs::read(&f.ledger).unwrap();
        assert!(!guide_command(&f).status.success(), "{fault}");
        assert_eq!(before, fs::read(&f.ledger).unwrap(), "{fault}");
    }
}

#[test]
fn label_guide_conflicting_metadata_rolls_back_new_rows_and_preserves_append_only_triggers() {
    let f = guide_fixture("pusht", 40, 12);
    assert!(guide_command(&f).status.success());
    let before = fs::read(&f.ledger).unwrap();
    let mut cards = read_jsonl(&f.directory.join("changes.jsonl"));
    cards[0]["desc"] = json!("Changed frozen description");
    let mut new_card = cards[0].clone();
    new_card["change_id"] = json!("c001");
    cards.push(new_card);
    write_jsonl(&f.directory.join("changes.jsonl"), &cards);
    let output = guide_command(&f);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("conflicting existing changes row"));
    assert_eq!(before, fs::read(&f.ledger).unwrap());
    let connection = Connection::open(&f.ledger).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='trigger'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        18
    );
    assert!(connection
        .execute("UPDATE changes SET env='bad'", [])
        .is_err());
    assert!(connection.execute("DELETE FROM labels", []).is_err());
}

#[test]
fn label_guide_compares_all_run_statistics_and_uri_without_replacing_rows() {
    for (column, value) in [
        ("run_id", "'changed-id'"),
        ("episodes", "19"),
        ("successes", "0"),
        ("mean_reward", "123.0"),
        ("uri", "'wrong-uri'"),
        ("wall_seconds", "99.0"),
    ] {
        let f = guide_fixture("pusht", 40, 12);
        assert!(guide_command(&f).status.success());
        let connection = Connection::open(&f.ledger).unwrap();
        let trigger: String = connection
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='runs_no_UPDATE'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        connection
            .execute_batch("DROP TRIGGER runs_no_UPDATE")
            .unwrap();
        connection
            .execute_batch(&format!(
                "UPDATE runs SET {column}={value} WHERE tier='expensive'"
            ))
            .unwrap();
        connection.execute_batch(&trigger).unwrap();
        drop(connection);
        let before = fs::read(&f.ledger).unwrap();
        let output = guide_command(&f);
        assert!(!output.status.success(), "{column}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("conflicting existing runs row"),
            "{:?}",
            output
        );
        assert_eq!(before, fs::read(&f.ledger).unwrap());
    }
}

#[test]
fn label_guide_cli_requires_exactly_one_complete_input_form() {
    let f = guide_fixture("pusht", 40, 12);
    for flags in [
        vec![],
        vec!["--changes", "changes.jsonl"],
        vec!["--scenarios", "scenarios.jsonl"],
        vec![
            "--catalog",
            "catalog.json",
            "--changes",
            "changes.jsonl",
            "--scenarios",
            "scenarios.jsonl",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_brier"))
            .args(["label", "--ledger"])
            .arg(&f.ledger)
            .arg("--episodes")
            .arg(f.directory.join("episodes.jsonl"))
            .args(flags)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("error:"));
    }
}

#[test]
fn label_guide_base_keys_stay_distinct_across_environments_without_synthetic_changes() {
    let f = guide_fixture("pusht", 40, 12);
    let other = guide_fixture("aloha", 50, 20);
    for name in ["changes", "scenarios", "episodes"] {
        let path = f.directory.join(format!("{name}.jsonl"));
        let mut rows = read_jsonl(&path);
        rows.extend(read_jsonl(&other.directory.join(format!("{name}.jsonl"))));
        write_jsonl(&path, &rows);
    }
    let output = guide_command(&f);
    assert!(output.status.success(), "{:?}", output);
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["pairs"], 2);
    assert_eq!(summary["per_env"]["pusht"]["regressed_rate"], 1.0);
    assert_eq!(summary["per_env"]["aloha"]["regressed_rate"], 0.0);
    let connection = Connection::open(&f.ledger).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(DISTINCT run_id) FROM runs WHERE change_id='base'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM changes", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert!(guide_command(&f).status.success());
}

#[test]
fn label_guide_rejects_manifest_drift_and_invalid_cards_before_mutation() {
    for fault in [
        "baseline-rate",
        "duplicate-scenario",
        "duplicate-change",
        "negative-change-seed",
        "unknown-family",
        "multiline-desc",
    ] {
        let f = guide_fixture("pusht", 40, 12);
        let mut scenarios = read_jsonl(&f.directory.join("scenarios.jsonl"));
        let mut cards = read_jsonl(&f.directory.join("changes.jsonl"));
        match fault {
            "baseline-rate" => scenarios[0]["baseline_success"] = json!(0.6),
            "duplicate-scenario" => scenarios.push(scenarios[0].clone()),
            "duplicate-change" => cards.push(cards[0].clone()),
            "negative-change-seed" => cards[0]["seed"] = json!(-1),
            "unknown-family" => cards[0]["families"][0]["family"] = json!("unknown"),
            _ => cards[0]["desc"] = json!("First line\nSecond line"),
        }
        write_jsonl(&f.directory.join("changes.jsonl"), &cards);
        write_jsonl(&f.directory.join("scenarios.jsonl"), &scenarios);
        let before = fs::read(&f.ledger).unwrap();
        assert!(!guide_command(&f).status.success(), "{fault}");
        assert_eq!(before, fs::read(&f.ledger).unwrap());
    }
}
