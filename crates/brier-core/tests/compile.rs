use serde_json::{json, Value};
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

#[test]
fn live_compilation_needs_no_target_labels_and_freezes_training_history() {
    let directory = std::env::temp_dir().join(format!("brier-live-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&directory).unwrap();
    let ledger = directory.join("ledger.sqlite");
    let compiled = directory.join("compiled.jsonl");
    let connection = brier_core::ledger::init(&ledger).unwrap();
    connection.execute_batch("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<10)
        INSERT INTO changes SELECT printf('c%03d',x),'gym_pusht/PushT-v0',json_object('description','policy update','families',json_array(json_object('family','action_latency','params',json_object('k',x%4+1)))),'now' FROM n;
        INSERT INTO scenarios VALUES('pusht-s01','gym_pusht/PushT-v0','{\"pose_bin\":0,\"lighting\":\"normal\",\"camera_jitter\":false}',0.8);
        INSERT INTO labels SELECT change_id,'pusht-s01',CAST(substr(change_id,2) AS INTEGER)%2,'stall/timeout',3,'action_latency' FROM changes;
        INSERT INTO runs SELECT change_id||'-cheap',change_id,scenario_id,'cheap',3,2,0.41,'raw',3.0 FROM labels;").unwrap();
    brier_core::compile::compile(&ledger, &compiled).unwrap();
    let original: Vec<Value> = fs::read_to_string(&compiled)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let existing = brier_core::compile::compile_live(&ledger, "c009", &compiled).unwrap();
    for question in &existing[0].questions {
        let expected = original
            .iter()
            .find(|row| row["id"] == question["id"])
            .unwrap();
        assert_eq!(question["prompt"], expected["prompt"]);
        assert_eq!(question["options"], expected["options"]);
        assert!(question.get("label").is_none());
    }
    connection.execute_batch("INSERT INTO changes VALUES('c999','gym_pusht/PushT-v0','{\"description\":\"policy update\",\"families\":[{\"family\":\"action_latency\",\"params\":{\"k\":2}}]}','later');
        INSERT INTO runs VALUES('new-cheap','c999','pusht-s01','cheap',3,2,0.41,'raw',3.0);").unwrap();
    let before = brier_core::compile::compile_live(&ledger, "c999", &compiled).unwrap();
    assert_eq!(before[0].features[35], 0.5);
    assert_eq!(before[0].features[36], 0.4);
    connection.execute_batch("WITH RECURSIVE n(x) AS (SELECT 800 UNION ALL SELECT x+1 FROM n WHERE x<819)
        INSERT INTO changes SELECT printf('c%03d',x),'gym_pusht/PushT-v0','{\"description\":\"later history\",\"families\":[{\"family\":\"action_latency\",\"params\":{\"k\":2}}]}','later' FROM n;
        INSERT INTO labels SELECT change_id,'pusht-s01',1,'stall/timeout',1,'action_latency' FROM changes WHERE change_id BETWEEN 'c800' AND 'c819';").unwrap();
    let after = brier_core::compile::compile_live(&ledger, "c999", &compiled).unwrap();
    assert_eq!(before[0].features, after[0].features);
    assert_eq!(before[0].questions, after[0].questions);
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
    drop(connection);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn sam_prompts_name_only_observable_task_objects() {
    let pusht = brier_core::compile::sam_prompts("gym_pusht/PushT-v0").unwrap();
    let aloha = brier_core::compile::sam_prompts("gym_aloha/AlohaTransferCube-v0").unwrap();
    assert_eq!(pusht.as_object().unwrap().len(), 3);
    assert_eq!(aloha["object"], "the red cube");
    assert_eq!(pusht["object"], "the T-shaped block");
    assert!(brier_core::compile::sam_prompts("unknown").is_err());
}

fn telemetry() -> Value {
    json!({"collision_steps":0,"action_direction_changes":0,"action_transitions":10,
        "object_max_height":0.0,"object_final_height":0.0,"goal_distance_min":10.0,
        "goal_distance_final":11.0,"steps":10,"reward_sum":0.5})
}

#[test]
fn explanation_prompt_accepts_observations_and_rejects_hidden_fields() {
    let scenario = json!({"pose_bin":0,"lighting":"normal","camera_jitter":false});
    let geometry = json!({"final_object_offset":[1.0,2.0],"gripper_object_distance":3.0});
    let render = |scenario: &Value, telemetry: &Value, geometry: &Value| {
        brier_core::compile::explanation_prompt("gym_pusht/PushT-v0", scenario, telemetry, geometry)
    };
    let prompt = render(&scenario, &telemetry(), &geometry).unwrap();
    assert!(prompt.contains("raw0, overlay0"));
    assert!(prompt.contains("final_object_offset"));
    let mut leaked = telemetry();
    leaked["cause_family"] = json!("secret injection");
    assert!(render(&scenario, &leaked, &geometry).is_err());
    let mut leaked = scenario.clone();
    leaked["card"] = json!({"family":"hidden"});
    assert!(render(&leaked, &telemetry(), &geometry).is_err());
    assert!(render(
        &scenario,
        &telemetry(),
        &json!({"final_object_offset":[1.0],"gripper_object_distance":3.0})
    )
    .is_err());
}

#[test]
fn escalation_prompt_preserves_fixed_scores_and_requires_three_observations() {
    let state = json!({"change_id":"c001","scenario_id":"pusht-s01","features":[],
        "questions":[{"id":"c001/pusht-s01/regressed/cheap","prompt":"compiled state<answer>","options":{"A":"yes","B":"no"}}]});
    let scores = json!({"regressed":{"yes":0.4,"no":0.6}});
    let observations = json!({"observations":[telemetry(),telemetry(),telemetry()],"frames":[]});
    let prompt =
        brier_core::compile::escalation_prompt(&state, &scores, 0.5, &observations).unwrap();
    assert!(prompt.contains("compiled state<answer>"));
    assert!(prompt.contains("cannot cancel evaluations"));
    assert!(prompt.contains("CALIBRATED_THRESHOLD: 0.5"));
    assert!(brier_core::compile::escalation_prompt(
        &state,
        &scores,
        0.5,
        &json!({"observations":[telemetry()]})
    )
    .is_err());
    assert!(brier_core::compile::escalation_prompt(
        &state,
        &json!({"regressed":{"yes":0.4,"no":0.7}}),
        0.5,
        &observations
    )
    .is_err());
}
