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
use serde_json::json;
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
    Command::new(env!("CARGO_BIN_EXE_brier"))
        .arg("label")
        .arg("--ledger")
        .arg(&f.ledger)
        .arg("--episodes")
        .arg(f.directory.join("episodes"))
        .arg("--catalog")
        .arg(f.directory.join("catalog.json"))
        .output()
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
