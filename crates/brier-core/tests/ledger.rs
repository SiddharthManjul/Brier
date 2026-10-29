use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use brier_core::ledger::*;
use rusqlite::Connection;

const RUN_ID: &str = "01234567-89ab-4cde-8f01-23456789abcd";

fn memory_ledger() -> Connection {
    init(Path::new(":memory:")).unwrap()
}

fn assert_stored(connection: &Connection, table: &str, columns: &str, expected: &str) {
    let actual: String = connection
        .query_row(
            &format!("SELECT json_array({columns}) FROM {table}"),
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn init_creates_exact_planned_tables_and_columns() {
    let connection = memory_ledger();
    for (table, columns) in [
        ("changes", "change_id,env,card_json,created_at"),
        ("scenarios", "scenario_id,env,params_json,baseline_success"),
        ("runs", "run_id,change_id,scenario_id,tier,episodes,successes,mean_reward,uri,wall_seconds"),
        ("decisions", "decision_id,change_id,scenario_id,tier_used,model_version,probs_json,lambda,action,propensity,reason,created_at"),
        ("labels", "change_id,scenario_id,regressed,failure_mode,success_bucket,cause_family"),
        ("explanations", "run_id,model,json,cost_usd"),
    ] {
        let actual: String = connection.query_row(
            "SELECT group_concat(name, ',') FROM pragma_table_info(?1)", [table], |row| row.get(0),
        ).unwrap();
        assert_eq!(actual, columns);
    }
    let count: i64 = connection
        .query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 6);
}

#[test]
fn insert_change_preserves_quoted_json_and_rejects_duplicate_id() {
    let connection = memory_ledger();
    let row = Change {
        change_id: "c001",
        env: "pusht",
        card_json: r#"{"desc":"O'Brien"}"#,
        created_at: "2026-10-29T00:00:00Z",
    };
    insert_change(&connection, &row).unwrap();
    assert!(insert_change(&connection, &row).is_err());
    assert_stored(
        &connection,
        "changes",
        "change_id,env,card_json,created_at",
        r#"["c001","pusht","{\"desc\":\"O'Brien\"}","2026-10-29T00:00:00Z"]"#,
    );
}

#[test]
fn insert_scenario_preserves_fields_and_rejects_duplicate_id() {
    let connection = memory_ledger();
    let row = Scenario {
        scenario_id: "pusht-s01",
        env: "pusht",
        params_json: "{}",
        baseline_success: 0.85,
    };
    insert_scenario(&connection, &row).unwrap();
    assert!(insert_scenario(&connection, &row).is_err());
    assert_stored(
        &connection,
        "scenarios",
        "scenario_id,env,params_json,baseline_success",
        r#"["pusht-s01","pusht","{}",0.85]"#,
    );
}

#[test]
fn insert_run_preserves_fields_and_rejects_duplicate_id() {
    let connection = memory_ledger();
    let row = Run {
        run_id: RUN_ID,
        change_id: "c001",
        scenario_id: "pusht-s01",
        tier: "expensive",
        episodes: 20,
        successes: 9,
        mean_reward: 0.4,
        uri: "gs://test/episodes.parquet",
        wall_seconds: 120.5,
    };
    insert_run(&connection, &row).unwrap();
    assert!(insert_run(&connection, &row).is_err());
    assert_stored(
        &connection,
        "runs",
        "run_id,change_id,scenario_id,tier,episodes,successes,mean_reward,uri,wall_seconds",
        r#"["01234567-89ab-4cde-8f01-23456789abcd","c001","pusht-s01","expensive",20,9,0.4,"gs://test/episodes.parquet",120.5]"#,
    );
}

#[test]
fn insert_decision_preserves_fields_and_rejects_duplicate_id() {
    let connection = memory_ledger();
    let row = Decision {
        decision_id: RUN_ID,
        change_id: "c001",
        scenario_id: "pusht-s01",
        tier_used: "T1",
        model_version: "v1",
        probs_json: r#"{"yes":0.8}"#,
        lambda: 0.18,
        action: "expensive",
        propensity: 1.0,
        reason: "above threshold",
        created_at: "2026-10-29T00:00:00Z",
    };
    insert_decision(&connection, &row).unwrap();
    assert!(insert_decision(&connection, &row).is_err());
    assert_stored(&connection, "decisions", "decision_id,change_id,scenario_id,tier_used,model_version,probs_json,lambda,action,propensity,reason,created_at", r#"["01234567-89ab-4cde-8f01-23456789abcd","c001","pusht-s01","T1","v1","{\"yes\":0.8}",0.18,"expensive",1.0,"above threshold","2026-10-29T00:00:00Z"]"#);
}

#[test]
fn insert_label_preserves_fields_and_rejects_duplicate_pair() {
    let connection = memory_ledger();
    let row = Label {
        change_id: "c001",
        scenario_id: "pusht-s01",
        regressed: true,
        failure_mode: "stall/timeout",
        success_bucket: 2,
        cause_family: "action_latency",
    };
    insert_label(&connection, &row).unwrap();
    assert!(insert_label(&connection, &row).is_err());
    assert_stored(
        &connection,
        "labels",
        "change_id,scenario_id,regressed,failure_mode,success_bucket,cause_family",
        r#"["c001","pusht-s01",1,"stall/timeout",2,"action_latency"]"#,
    );
}

#[test]
fn insert_explanation_preserves_fields_and_rejects_duplicate_run() {
    let connection = memory_ledger();
    let row = Explanation {
        run_id: RUN_ID,
        model: "spark",
        json: r#"{"failure_mode":"stall"}"#,
        cost_usd: 0.01,
    };
    insert_explanation(&connection, &row).unwrap();
    assert!(insert_explanation(&connection, &row).is_err());
    assert_stored(
        &connection,
        "explanations",
        "run_id,model,json,cost_usd",
        r#"["01234567-89ab-4cde-8f01-23456789abcd","spark","{\"failure_mode\":\"stall\"}",0.01]"#,
    );
}

#[test]
fn append_only_guards_survive_reopening() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "brier-ledger-{}-{nonce}.sqlite",
        std::process::id()
    ));
    drop(init(&path).unwrap());
    let connection = Connection::open(&path).unwrap();
    for (table, keys, values) in [
        ("changes", "change_id", "'c001'"),
        ("scenarios", "scenario_id", "'pusht-s01'"),
        ("runs", "run_id", "'01234567-89ab-4cde-8f01-23456789abcd'"),
        (
            "decisions",
            "decision_id",
            "'01234567-89ab-4cde-8f01-23456789abcd'",
        ),
        ("labels", "change_id,scenario_id", "'c001','pusht-s01'"),
        (
            "explanations",
            "run_id",
            "'01234567-89ab-4cde-8f01-23456789abcd'",
        ),
    ] {
        let insert = format!("INSERT INTO {table} ({keys}) VALUES ({values})");
        connection.execute(&insert, []).unwrap();
        let key = keys.split(',').next().unwrap();
        for sql in [
            format!("UPDATE {table} SET {key} = {key}"),
            format!("DELETE FROM {table}"),
            insert.replace("INSERT INTO", "INSERT OR REPLACE INTO"),
        ] {
            let error = connection.execute(&sql, []).unwrap_err().to_string();
            assert!(error.contains("append-only"), "{sql}: {error}");
        }
    }
    let error = connection
        .execute(
            "INSERT OR REPLACE INTO changes (rowid, change_id) VALUES (1, 'c002')",
            [],
        )
        .unwrap_err();
    assert!(error.to_string().contains("append-only"));
    drop(connection);
    std::fs::remove_file(path).unwrap();
}
