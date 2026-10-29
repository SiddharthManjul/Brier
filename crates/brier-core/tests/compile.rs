use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    process::Command,
};

#[test]
fn compile_preserves_options_change_splits_and_train_only_history() {
    let directory = std::env::temp_dir().join(format!("brier-compile-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&directory).unwrap();
    let ledger = directory.join("ledger.sqlite");
    let connection = brier_core::ledger::init(&ledger).unwrap();
    connection.execute_batch("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<11), e(offset,env) AS (VALUES(0,'gym_pusht/PushT-v0'),(80,'gym_aloha/AlohaTransferCube-v0')) INSERT INTO changes SELECT printf('c%03d',x+offset),env,json_object('description','policy update','families',json_array(json_object('family',CASE WHEN x=11 THEN 'occlusion' ELSE 'action_latency' END,'params',json(CASE WHEN x=11 THEN json_object('x',0,'y',0,'width',10,'height',10) ELSE json_object('k',x%4+1) END)))),'now' FROM n,e;
      INSERT INTO scenarios VALUES('pusht-s01','gym_pusht/PushT-v0','{\"pose_bin\":0,\"lighting\":\"normal\",\"camera_jitter\":false}',0.8),('aloha_transfer_cube-s01','gym_aloha/AlohaTransferCube-v0','{\"pose_bin\":0,\"lighting\":\"low\",\"camera_jitter\":true,\"physics_jitter\":true}',0.8);
      INSERT INTO labels SELECT change_id,scenario_id,CAST(substr(change_id,2) AS INTEGER)%2,'stall/timeout',3,'action_latency' FROM changes JOIN scenarios USING(env);
      INSERT INTO runs SELECT change_id||'-cheap',change_id,scenario_id,'cheap',3,2,0.41,'raw',3.0 FROM labels;").unwrap();
    let run = |name: &str| {
        Command::new(env!("CARGO_BIN_EXE_brier"))
            .arg("compile")
            .arg("--ledger")
            .arg(&ledger)
            .arg("--out")
            .arg(directory.join(name))
            .output()
            .unwrap()
    };
    let first = run("first.jsonl");
    assert!(first.status.success(), "{:?}", first);
    let read = |name: &str| {
        fs::read_to_string(directory.join(name))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>()
    };
    let rows = read("first.jsonl");
    assert_eq!(rows.len(), 132);
    let mut changes = BTreeMap::<(String, String), BTreeSet<String>>::new();
    for row in &rows {
        let parts: Vec<_> = row["id"].as_str().unwrap().split('/').collect();
        assert_eq!(parts.len(), 4);
        changes
            .entry((parts[1].into(), row["split"].as_str().unwrap().into()))
            .or_default()
            .insert(parts[0].into());
        assert!(row["options"].get(row["label"].as_str().unwrap()).is_some());
        let prompt = row["prompt"].as_str().unwrap();
        assert!(prompt.ends_with("<answer>"));
        assert_eq!(prompt.contains("CHEAP_TIER:"), parts[3] == "cheap");
        if parts[0] == "c009" || parts[0] == "c089" {
            assert!(prompt
                .contains("scenario_fragility=0.50; similar_changes(k=5): 2 regressed, 3 passed"));
        }
    }
    for scenario in ["pusht-s01", "aloha_transfer_cube-s01"] {
        for (split, count) in [("train", 6), ("calibration", 2), ("test", 2), ("ood", 1)] {
            assert_eq!(changes[&(scenario.into(), split.into())].len(), count);
        }
    }
    assert!(run("second.jsonl").status.success());
    assert_eq!(rows, read("second.jsonl"));
    assert!(!run("first.jsonl").status.success());
    assert_eq!(rows, read("first.jsonl"));
    connection.execute_batch("DROP TRIGGER labels_no_UPDATE; UPDATE labels SET regressed=1-regressed WHERE CAST(substr(change_id,2) AS INTEGER)%80>=7;").unwrap();
    assert!(run("changed.jsonl").status.success());
    for (before, after) in rows.iter().zip(read("changed.jsonl")) {
        assert_eq!(before["prompt"], after["prompt"]);
    }
    connection.execute_batch("INSERT INTO runs SELECT 'duplicate',change_id,scenario_id,'cheap',3,2,0.4,'raw',1.0 FROM labels LIMIT 1;").unwrap();
    assert!(!run("invalid.jsonl").status.success());
    assert!(!directory.join("invalid.jsonl").exists());
    drop(connection);
    fs::remove_dir_all(directory).unwrap();
}
