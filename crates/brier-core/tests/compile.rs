use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command};

struct Fixture {
    directory: PathBuf,
    ledger: PathBuf,
    connection: Connection,
}
impl Fixture {
    fn new() -> Self {
        let directory =
            std::env::temp_dir().join(format!("brier-compile-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let ledger = directory.join("ledger.sqlite");
        let connection = brier_core::ledger::init(&ledger).unwrap();
        Self {
            directory,
            ledger,
            connection,
        }
    }
    fn add(&self, change: &str, env: &str, families: Value, regressed: bool) {
        let (full, scenario, scenario_params) = if env == "pusht" {
            (
                "gym_pusht/PushT-v0",
                "pusht-s01",
                json!({"pose_bin":1,"lighting":"normal","camera_jitter":false}),
            )
        } else {
            (
                "gym_aloha/AlohaTransferCube-v0",
                "aloha-s07",
                json!({"pose_bin":3,"lighting":"low","physics_jitter":true,"camera_jitter":false}),
            )
        };
        let card = json!({"change_id":change,"env":env,"families":families,"desc":"Move inference to edge box","seed":2026});
        self.connection
            .execute(
                "INSERT INTO changes VALUES(?1,?2,?3,'now')",
                params![change, full, card.to_string()],
            )
            .unwrap();
        self.connection
            .execute(
                "INSERT INTO scenarios SELECT ?1,?2,?3,0.86 WHERE NOT EXISTS (SELECT 1 FROM scenarios WHERE scenario_id=?1)",
                params![scenario, full, scenario_params.to_string()],
            )
            .unwrap();
        self.connection
            .execute(
                "INSERT INTO labels VALUES(?1,?2,?3,'stall/timeout',3,'action_latency')",
                params![change, scenario, regressed],
            )
            .unwrap();
        self.connection
            .execute(
                "INSERT INTO runs VALUES(?1,?2,?3,'cheap',3,1,0.41,'raw',1.0)",
                params![format!("{change}-cheap"), change, scenario],
            )
            .unwrap();
    }
    fn compile(&self, name: &str) -> (std::process::Output, Vec<Value>) {
        let path = self.directory.join(name);
        let result = Command::new(env!("CARGO_BIN_EXE_brier"))
            .args(["compile", "--ledger"])
            .arg(&self.ledger)
            .arg("--out")
            .arg(&path)
            .output()
            .unwrap();
        let rows = if result.status.success() {
            fs::read_to_string(path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        } else {
            vec![]
        };
        (result, rows)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}
fn latency(k: i64) -> Value {
    json!([{"family":"action_latency","params":{"k":k}}])
}
fn fixture() -> Fixture {
    let f = Fixture::new();
    for (id, regressed) in [
        ("c001", false),
        ("c002", true),
        ("c003", false),
        ("c004", true),
        ("c009", true),
    ] {
        f.add(id, "pusht", latency(2), regressed);
    }
    f.add("c011","pusht",json!([{"family":"action_latency","params":{"k":2}},{"family":"occlusion","params":{"fraction":0.1}}]),true);
    f.add(
        "c013",
        "pusht",
        json!([{"family":"chunk_len","params":{"n_action_steps":12}}]),
        false,
    );
    for (id, regressed) in [
        ("c081", true),
        ("c084", false),
        ("c085", true),
        ("c087", false),
    ] {
        f.add(id, "aloha", latency(2), regressed);
    }
    f.add("c082","aloha",json!([{"family":"action_latency","params":{"k":2}},{"family":"obs_noise","params":{"sigma":0.03}}]),true);
    f
}

#[test]
fn guide_metadata_sha_splits_shuffling_labels_and_counts_are_stable() {
    let f = fixture();
    let (first, rows) = f.compile("first.jsonl");
    assert!(first.status.success(), "{:?}", first);
    assert_eq!(rows.len(), 72);
    let expected = BTreeMap::from([
        ("c001", "calibration"),
        ("c002", "train"),
        ("c003", "train"),
        ("c004", "test"),
        ("c009", "train"),
        ("c011", "ood"),
        ("c013", "ood"),
        ("c081", "train"),
        ("c082", "calibration"),
        ("c084", "train"),
        ("c085", "train"),
        ("c087", "train"),
    ]);
    let canonical = |question: &str| -> Value {
        match question {
            "regressed" => json!(["yes", "no"]),
            "failure_mode" => json!([
                "none",
                "stall or timeout",
                "overshoot or misalignment",
                "grasp or drop",
                "collision",
                "oscillation"
            ]),
            _ => json!(["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"]),
        }
    };
    let mut shuffled = false;
    for row in &rows {
        assert_eq!(row.as_object().unwrap().len(), 11);
        let change = row["change_id"].as_str().unwrap();
        let scenario = row["scenario_id"].as_str().unwrap();
        let question = row["question"].as_str().unwrap();
        let variant = row["variant"].as_str().unwrap();
        assert_eq!(row["split"], expected[change]);
        assert_eq!(row["pair_id"], format!("{change}:{scenario}"));
        assert_eq!(
            row["id"],
            format!("{change}:{scenario}:{question}:{variant}")
        );
        assert_eq!(
            row["env"],
            if scenario.starts_with("aloha-") {
                "aloha"
            } else {
                "pusht"
            }
        );
        let index = row["label"].as_str().unwrap().as_bytes()[0] - b'A';
        let label: bool = f
            .connection
            .query_row(
                "SELECT regressed FROM labels WHERE change_id=?1",
                [change],
                |r| r.get(0),
            )
            .unwrap();
        let target = match question {
            "regressed" => {
                if label {
                    "yes"
                } else {
                    "no"
                }
            }
            "failure_mode" => "stall or timeout",
            _ => "40-59%",
        };
        assert_eq!(row["options"][index as usize], target);
        if row["split"] == "train" {
            shuffled |= row["options"] != canonical(question);
        } else {
            assert_eq!(row["options"], canonical(question));
        }
        assert!(row["prompt"].as_str().unwrap().ends_with("<answer>"));
        assert_eq!(
            row["prompt"]
                .as_str()
                .unwrap()
                .contains("CHEAP_TIER: not run"),
            variant == "noev"
        );
    }
    assert!(shuffled);
    let stdout = String::from_utf8(first.stdout).unwrap();
    assert!(
        stdout.contains("split=train")
            && stdout.contains("question=failure_mode")
            && stdout.contains("variant=noev")
    );
    assert_eq!(rows, f.compile("second.jsonl").1);
    assert!(!f.compile("first.jsonl").0.status.success());
    f.add("c099", "pusht", latency(2), true);
    let (_, extended) = f.compile("extended.jsonl");
    for row in &rows {
        assert_eq!(
            row["split"],
            extended
                .iter()
                .find(|other| other["id"] == row["id"])
                .unwrap()["split"]
        );
    }
}

#[test]
fn all_questions_and_variants_match_literal_prompt_goldens() {
    let f = fixture();
    let (result, rows) = f.compile("golden.jsonl");
    assert!(result.status.success(), "{:?}", result);
    const PREFIX: &str = r#"<system> You are Brier, a calibrated decision model for robot policy CI. Answer with one option letter.
<state>
CHANGE: id=c082 family=action_latency k=2 + obs_noise sigma=0.03 desc="Move inference to edge box"
SCENARIO: env=aloha_transfer_cube pose_bin=3 lighting=low physics_jitter=on camera_jitter=off
BASELINE: success=0.86 (50 eps)
CHEAP_TIER: success=1/3 mean_reward=0.41 (3 eps, 50% horizon)
HISTORY: scenario_fragility=0.50; similar_changes(k=4): 2 regressed, 2 passed
</state>
"#;
    for (question, ending) in [("regressed", "<question> Will the expensive evaluation regress by more than 15 points?\n<options> A) yes  B) no\n<answer>"),
        ("failure_mode", "<question> If the expensive evaluation regresses, what is the main failure mode?\n<options> A) none  B) stall or timeout  C) overshoot or misalignment  D) grasp or drop  E) collision  F) oscillation\n<answer>"),
        ("success_bucket", "<question> What will the expensive success rate be?\n<options> A) 0-19%  B) 20-39%  C) 40-59%  D) 60-79%  E) 80-100%\n<answer>")] {
        for variant in ["ev","noev"] {
            let expected=format!("{PREFIX}{ending}");
            let expected=if variant=="noev" {expected.replace("CHEAP_TIER: success=1/3 mean_reward=0.41 (3 eps, 50% horizon)","CHEAP_TIER: not run")} else {expected};
            let row=rows.iter().find(|row|row["id"]==format!("c082:aloha-s07:{question}:{variant}")).unwrap();
            assert_eq!(row["prompt"].as_str().unwrap().as_bytes(),expected.as_bytes());
        }
    }
    let pusht = rows
        .iter()
        .find(|row| row["id"] == "c001:pusht-s01:regressed:ev")
        .unwrap();
    assert!(pusht["prompt"]
        .as_str()
        .unwrap()
        .contains("SCENARIO: env=pusht pose_bin=1 lighting=normal camera_jitter=off"));
}

#[test]
fn heldout_labels_and_magnitudes_never_enter_history_and_train_self_is_excluded() {
    let f = fixture();
    let (_, before) = f.compile("before.jsonl");
    f.connection.execute_batch("DROP TRIGGER labels_no_UPDATE; UPDATE labels SET regressed=1-regressed WHERE change_id IN ('c001','c004','c011','c013','c082')").unwrap();
    let (_, after) = f.compile("after.jsonl");
    for (before, after) in before.iter().zip(&after) {
        assert_eq!(before["prompt"], after["prompt"]);
    }
    let own = after
        .iter()
        .find(|r| r["id"] == "c002:pusht-s01:regressed:ev")
        .unwrap()["prompt"]
        .clone();
    f.connection
        .execute(
            "UPDATE labels SET regressed=1-regressed WHERE change_id='c002'",
            [],
        )
        .unwrap();
    let (_, changed) = f.compile("changed.jsonl");
    assert_eq!(
        own,
        changed
            .iter()
            .find(|r| r["id"] == "c002:pusht-s01:regressed:ev")
            .unwrap()["prompt"]
    );
    f.connection.execute_batch("DROP TRIGGER changes_no_UPDATE; UPDATE changes SET card_json=json_set(card_json,'$.families[0].params.k',1000000) WHERE change_id IN ('c001','c004','c082')").unwrap();
    let (_, magnitude) = f.compile("magnitude.jsonl");
    for row in &changed {
        if row["split"] == "train" {
            assert_eq!(
                row["prompt"],
                magnitude.iter().find(|r| r["id"] == row["id"]).unwrap()["prompt"]
            );
        }
    }
}

#[test]
fn history_prefers_family_jaccard_then_magnitude_then_change_id() {
    let f = Fixture::new();
    f.add("c001", "pusht", latency(2), true);
    for (id, k, label) in [
        ("c002", 100, true),
        ("c003", 2, false),
        ("c009", 3, false),
        ("c011", 4, false),
        ("c013", 5, false),
        ("c014", 6, false),
    ] {
        f.add(id, "pusht", latency(k), label);
    }
    f.add(
        "c015",
        "pusht",
        json!([{"family":"benign","params":{}}]),
        true,
    );
    let (result, rows) = f.compile("ranked.jsonl");
    assert!(result.status.success(), "{:?}", result);
    let row = rows
        .iter()
        .find(|r| r["id"] == "c001:pusht-s01:regressed:ev")
        .unwrap();
    assert!(row["prompt"]
        .as_str()
        .unwrap()
        .contains("similar_changes(k=5): 0 regressed, 5 passed"));
    // Equal magnitudes at the fifth-place cutoff are resolved by the lower change ID.
    f.connection.execute_batch("DROP TRIGGER changes_no_UPDATE; UPDATE changes SET card_json=json_set(card_json,'$.families[0].params.k',6) WHERE change_id='c002'").unwrap();
    let (_, tied) = f.compile("tied.jsonl");
    let row = tied
        .iter()
        .find(|r| r["id"] == "c001:pusht-s01:regressed:ev")
        .unwrap();
    assert!(row["prompt"]
        .as_str()
        .unwrap()
        .contains("similar_changes(k=5): 1 regressed, 4 passed"));
}

#[test]
fn live_compilation_preserves_transport_and_uses_frozen_training_history() {
    let f = fixture();
    let (_, original) = f.compile("compiled.jsonl");
    let compiled = f.directory.join("compiled.jsonl");
    let existing = brier_core::compile::compile_live(&f.ledger, "c082", &compiled).unwrap();
    assert_eq!(existing[0].features.last().unwrap(), &0.5);
    for question in &existing[0].questions {
        let kind = question["id"].as_str().unwrap().split('/').nth(2).unwrap();
        let expected = original
            .iter()
            .find(|row| row["id"] == format!("c082:aloha-s07:{kind}:ev"))
            .unwrap();
        assert_eq!(question["prompt"], expected["prompt"]);
        assert!(question["options"].is_object() && question.get("label").is_none());
    }
    f.connection.execute("INSERT INTO changes VALUES('c999','gym_aloha/AlohaTransferCube-v0',?1,'later')",[json!({"change_id":"c999","env":"aloha","families":latency(2),"desc":"New policy","seed":2026}).to_string()]).unwrap();
    f.connection
        .execute_batch(
            "INSERT INTO runs VALUES('live','c999','aloha-s07','cheap',3,1,0.41,'raw',1.0)",
        )
        .unwrap();
    let before = brier_core::compile::compile_live(&f.ledger, "c999", &compiled).unwrap();
    f.add("c900", "aloha", latency(100000), false);
    let after = brier_core::compile::compile_live(&f.ledger, "c999", &compiled).unwrap();
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(after).unwrap()
    );
    f.connection.execute_batch("DROP TRIGGER labels_no_UPDATE; UPDATE labels SET regressed=1-regressed WHERE change_id='c081'").unwrap();
    assert!(brier_core::compile::compile_live(&f.ledger, "c999", &compiled).is_err());
}

#[test]
fn empty_history_duplicate_runs_and_bad_snapshots_fail_clearly() {
    let empty = Fixture::new();
    empty.add("c001", "pusht", latency(2), true);
    empty.add("c004", "pusht", latency(2), false);
    let (result, _) = empty.compile("empty.jsonl");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("at least one other training change"));
    assert!(!empty.directory.join("empty.jsonl").exists());
    let f = fixture();
    let (_, rows) = f.compile("compiled.jsonl");
    for (key, value) in [
        ("env", json!("pusht")),
        ("pair_id", json!("wrong")),
        ("variant", json!("bad")),
        ("options", json!(["yes", "yes"])),
        ("label", json!("Z")),
    ] {
        let mut rows = rows.clone();
        let row = rows.iter_mut().find(|r| r["change_id"] == "c081").unwrap();
        row[key] = value;
        let path = f.directory.join("invalid.jsonl");
        fs::write(
            &path,
            rows.iter()
                .map(|row| format!("{row}\n"))
                .collect::<String>(),
        )
        .unwrap();
        assert!(
            brier_core::compile::compile_live(&f.ledger, "c082", &path).is_err(),
            "{key}"
        );
    }
    f.connection.execute_batch("INSERT INTO runs SELECT 'duplicate',change_id,scenario_id,'cheap',3,1,0.41,'raw',1.0 FROM labels LIMIT 1").unwrap();
    assert!(!f.compile("duplicate.jsonl").0.status.success());
    assert!(!f.directory.join("duplicate.jsonl").exists());
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
    assert!(render(
        &json!({"pose_bin":4,"lighting":"normal","camera_jitter":false}),
        &telemetry(),
        &geometry
    )
    .is_ok());
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

#[test]
fn sam_prompts_keep_observed_concepts_for_both_supported_tasks() {
    let pusht = brier_core::compile::sam_prompts("gym_pusht/PushT-v0").unwrap();
    let aloha = brier_core::compile::sam_prompts("gym_aloha/AlohaTransferCube-v0").unwrap();
    assert_eq!(pusht["object"], "the T-shaped block");
    assert_eq!(aloha["object"], "the red cube");
    assert!(brier_core::compile::sam_prompts("unsupported").is_err());
}

#[test]
fn legacy_snapshots_and_existing_training_pairs_keep_live_prompt_parity() {
    let f = fixture();
    let (_, rows) = f.compile("guide.jsonl");
    let old = rows.iter().filter(|row|row["question"]=="regressed" && row["variant"]=="ev").map(|row| {
        let options=row["options"].as_array().unwrap().iter().enumerate().map(|(i,value)|(((b'A'+i as u8) as char).to_string(),value.clone())).collect::<BTreeMap<_,_>>();
        json!({"id":format!("{}/{}/regressed/cheap",row["change_id"].as_str().unwrap(),row["scenario_id"].as_str().unwrap()),
            "split":row["split"],"question":"regressed","options":options,"prompt":row["prompt"],"label":row["label"]})
    }).collect::<Vec<_>>();
    let legacy = f.directory.join("legacy.jsonl");
    fs::write(
        &legacy,
        old.iter().map(|row| format!("{row}\n")).collect::<String>(),
    )
    .unwrap();
    for change in ["c082", "c002"] {
        let from_guide =
            brier_core::compile::compile_live(&f.ledger, change, &f.directory.join("guide.jsonl"))
                .unwrap();
        let from_legacy = brier_core::compile::compile_live(&f.ledger, change, &legacy).unwrap();
        assert_eq!(
            serde_json::to_value(&from_guide).unwrap(),
            serde_json::to_value(&from_legacy).unwrap()
        );
        for question in &from_guide[0].questions {
            let kind = question["id"].as_str().unwrap().split('/').nth(2).unwrap();
            let original = rows
                .iter()
                .find(|row| {
                    row["change_id"] == change && row["question"] == kind && row["variant"] == "ev"
                })
                .unwrap();
            assert_eq!(question["prompt"], original["prompt"]);
        }
    }
}

#[test]
fn legacy_aloha_ledger_ids_compile_and_load_without_renaming() {
    let f = Fixture::new();
    for (change, label) in [
        ("c081", true),
        ("c082", false),
        ("c084", false),
        ("c085", true),
        ("c087", false),
    ] {
        f.add(change, "aloha", latency(2), label);
    }
    f.connection.execute_batch("DROP TRIGGER scenarios_no_UPDATE; DROP TRIGGER labels_no_UPDATE; DROP TRIGGER runs_no_UPDATE; DROP TRIGGER changes_no_UPDATE;
        UPDATE scenarios SET scenario_id='aloha_transfer_cube-s01',params_json=json_set(params_json,'$.pose_bin',0);
        UPDATE labels SET scenario_id='aloha_transfer_cube-s01'; UPDATE runs SET scenario_id='aloha_transfer_cube-s01';
        UPDATE changes SET card_json=json_object('description',json_extract(card_json,'$.desc'),'families',json_extract(card_json,'$.families'));").unwrap();
    let (result, rows) = f.compile("legacy-aloha.jsonl");
    assert!(result.status.success(), "{:?}", result);
    assert!(rows
        .iter()
        .all(|row| row["scenario_id"] == "aloha_transfer_cube-s01" && row["env"] == "aloha"));
    let pairs = brier_core::compile::compile_live(
        &f.ledger,
        "c082",
        &f.directory.join("legacy-aloha.jsonl"),
    )
    .unwrap();
    assert_eq!(pairs[0].scenario_id, "aloha_transfer_cube-s01");
    for question in &pairs[0].questions {
        let kind = question["id"].as_str().unwrap().split('/').nth(2).unwrap();
        let expected = rows
            .iter()
            .find(|row| row["id"] == format!("c082:aloha_transfer_cube-s01:{kind}:ev"))
            .unwrap();
        assert_eq!(question["prompt"], expected["prompt"]);
    }
}
