use parquet::{
    data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use rusqlite::Connection;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};
use uuid::Uuid;

const PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\xf8\xff\xff\x3f\x00\x05\xfe\x02\xfe\x0d\xef\x46\xb8\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

struct Fixture {
    directory: PathBuf,
    ledger: PathBuf,
    run: String,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

fn parquet(path: &Path, directory: &Path, successes: usize) {
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
    let terminal = json!({"trajectory":{"collision_steps":0,"action_direction_changes":0,"action_transitions":10,"object_max_height":0.0,"object_final_height":0.0,"goal_distance_min":10.0,"goal_distance_final":11.0},"cause_family":"HIDDEN_INJECTION","card":"HIDDEN_CARD"}).to_string();
    let frame_directory = directory.join("public/keyframes/failed");
    fs::create_dir_all(&frame_directory).unwrap();
    for index in 0..8 {
        fs::write(frame_directory.join(format!("frame-{index:04}.png")), PNG).unwrap();
    }
    let mut group = writer.next_row_group().unwrap();
    for index in 0..12 {
        let mut column = group.next_column().unwrap().unwrap();
        match index {
            5 | 7 => {
                column
                    .typed::<Int64Type>()
                    .write_batch(
                        &(0..20)
                            .map(|seed| if index == 5 { seed } else { 10 })
                            .collect::<Vec<_>>(),
                        None,
                        None,
                    )
                    .unwrap();
            }
            6 => {
                column
                    .typed::<BoolType>()
                    .write_batch(
                        &(0..20).map(|seed| seed < successes).collect::<Vec<_>>(),
                        None,
                        None,
                    )
                    .unwrap();
            }
            9 | 11 => {
                column
                    .typed::<DoubleType>()
                    .write_batch(&[1.0; 20], None, None)
                    .unwrap();
            }
            10 => {
                column
                    .typed::<ByteArrayType>()
                    .write_batch(
                        &vec![ByteArray::from("public/keyframes/failed"); 20],
                        Some(&[1; 20]),
                        None,
                    )
                    .unwrap();
            }
            _ => {
                let value = match index {
                    0 => "gym_pusht/PushT-v0",
                    1 => "lerobot/diffusion_pusht",
                    2 => "c002",
                    3 => "pusht-s01",
                    4 => "expensive",
                    8 => &terminal,
                    _ => unreachable!(),
                };
                column
                    .typed::<ByteArrayType>()
                    .write_batch(&vec![ByteArray::from(value); 20], None, None)
                    .unwrap();
            }
        }
        column.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
}

fn fixture(mode: &str) -> Fixture {
    let directory = std::env::temp_dir().join(format!("brier-explain-{}", Uuid::new_v4()));
    fs::create_dir_all(directory.join("bridge/brier_explain")).unwrap();
    fs::write(directory.join("bridge/brier_explain/__init__.py"), "").unwrap();
    fs::write(directory.join("mode"), mode).unwrap();
    fs::write(directory.join("bridge/brier_explain/client.py"),r#"
import json, os, shutil, sys
from pathlib import Path
p=json.load(sys.stdin)
root=Path(os.environ['BRIER_DATA'])
mode=(root/'mode').read_text()
assert 'HIDDEN_INJECTION' not in json.dumps(p) and 'HIDDEN_CARD' not in json.dumps(p)
with (root/'calls').open('a') as stream: stream.write(p['operation']+'\n')
if mode=='bridge_failure': raise SystemExit('fixture provider rejected request')
rows=[]
for source in p['episodes']:
    if p['operation']=='segment':
        assert set(source)=={'id','frames','sam_prompts'}
        frames=[]
        for frame in source['frames']:
            overlay=root/('overlay-'+str(frame['index'])+'.png')
            shutil.copyfile(frame['path'],overlay)
            frames.append({**frame,'overlay_path':str(overlay)})
        geometry={'final_object_offset':[1.0,2.0],'gripper_object_distance':3.0}
        if mode=='geometry': geometry['final_object_offset']=[1.0]
        if mode=='frame': frames[0]['path']=frames[1]['path']
        if mode=='outside': frames[0]['overlay_path']='/etc/passwd'
        rows.append({'id':source['id'],'frames':frames,'geometry':geometry,'model':'sam-3.1'})
    else:
        assert 'SAM_GEOMETRY' in source['prompt'] and 'raw0, overlay0' in source['prompt']
        explanation={'failure_mode':'stall/timeout','cause_hypothesis':'action_latency | brightness\nThe object remains away from the target.','evidence_frames':[0,7],'suggested_check':'Inspect the final object position.'}
        if mode=='evidence': explanation['evidence_frames']=[8]
        if mode=='fields': explanation['cause_family']='hidden'
        if mode=='cause_unknown': explanation['cause_hypothesis']='unknown\nObserved rationale.'
        if mode=='cause_duplicate': explanation['cause_hypothesis']='brightness | brightness\nObserved rationale.'
        if mode=='cause_no_rationale': explanation['cause_hypothesis']='brightness'
        rows.append({'id':source['id'],'model':'fixture-muse','explanation':explanation,'cost_usd':None})
        if mode=='missing_cost': del rows[-1]['cost_usd']
        if mode=='negative_cost': rows[-1]['cost_usd']=-1
    if mode=='identity': rows[-1]['id']='incorrect'
json.dump({'episodes':rows},sys.stdout)
"#).unwrap();
    let ledger = directory.join("ledger.sqlite");
    let connection = brier_core::ledger::init(&ledger).unwrap();
    connection
        .execute(
            "INSERT INTO scenarios VALUES('pusht-s01','gym_pusht/PushT-v0',?1,0.8)",
            [json!({"pose_bin":0,"lighting":"normal","camera_jitter":false}).to_string()],
        )
        .unwrap();
    let run = Uuid::new_v4().to_string();
    parquet(&directory.join("raw.parquet"), &directory, 19);
    connection
        .execute(
            "INSERT INTO runs VALUES(?1,'c002','pusht-s01','expensive',20,19,1.0,?2,20.0)",
            [&run, directory.join("raw.parquet").to_str().unwrap()],
        )
        .unwrap();
    Fixture {
        directory,
        ledger,
        run,
    }
}

fn command(f: &Fixture) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_brier"))
        .args(["explain", "--ledger"])
        .arg(&f.ledger)
        .args(["--run", &f.run])
        .current_dir(&f.directory)
        .env("BRIER_DATA", &f.directory)
        .env("PYTHONPATH", f.directory.join("bridge"))
        .output()
        .unwrap()
}

#[test]
fn explain_persists_real_frame_references_and_nullable_cost_once() {
    let f = fixture("valid");
    let result = command(&f);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["model"], "fixture-muse");
    assert!(report["cost_usd"].is_null());
    assert_eq!(report["episodes"].as_array().unwrap().len(), 1);
    let episode = &report["episodes"][0];
    assert_eq!(episode["seed"], 19);
    assert_eq!(
        Uuid::parse_str(episode["id"].as_str().unwrap())
            .unwrap()
            .get_version_num(),
        4
    );
    assert_eq!(episode["local_keyframes"].as_array().unwrap().len(), 8);
    assert_eq!(episode["local_overlays"].as_array().unwrap().len(), 8);
    assert_eq!(episode["keyframe_urls"], json!([]));
    let connection = Connection::open(&f.ledger).unwrap();
    let (stored, cost): (String, Option<f64>) = connection
        .query_row(
            "SELECT json,cost_usd FROM explanations WHERE run_id=?1",
            [&f.run],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(serde_json::from_str::<Value>(&stored).unwrap(), report);
    assert_eq!(cost, None);
    assert!(!command(&f).status.success());
    assert_eq!(
        fs::read_to_string(f.directory.join("calls")).unwrap(),
        "segment\nexplain\n"
    );
}

#[test]
fn invalid_provider_evidence_never_appends_an_explanation() {
    for mode in [
        "bridge_failure",
        "identity",
        "geometry",
        "frame",
        "outside",
        "evidence",
        "fields",
        "missing_cost",
        "negative_cost",
        "cause_unknown",
        "cause_duplicate",
        "cause_no_rationale",
    ] {
        let f = fixture(mode);
        let result = command(&f);
        assert!(!result.status.success(), "accepted {mode}");
        let count: i64 = Connection::open(&f.ledger)
            .unwrap()
            .query_row("SELECT count(*) FROM explanations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0, "persisted {mode}");
    }
}

#[test]
fn missing_real_keyframes_fail_before_model_work() {
    let f = fixture("valid");
    fs::remove_file(f.directory.join("public/keyframes/failed/frame-0007.png")).unwrap();
    let result = command(&f);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("exactly 8 actual keyframes"));
    assert!(!f.directory.join("calls").exists());
}

#[test]
fn raw_group_mismatch_fails_before_model_work() {
    let f = fixture("valid");
    let connection = Connection::open(&f.ledger).unwrap();
    connection
        .execute_batch("DROP TRIGGER runs_no_UPDATE; UPDATE runs SET successes=18;")
        .unwrap();
    let result = command(&f);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("differs from stored aggregate"));
    assert!(!f.directory.join("calls").exists());
}

#[test]
fn score_explanations_separates_heldout_splits_and_never_fills_missing_metrics_with_zero() {
    let f = fixture("valid");
    let compiled = f.directory.join("compiled.jsonl");
    fs::write(&compiled,"{\"id\":\"c002/pusht-s01/regressed/cheap\",\"split\":\"test\",\"question\":\"regressed\"}\n{\"id\":\"c003/pusht-s01/regressed/cheap\",\"split\":\"ood\",\"question\":\"regressed\"}\n{\"id\":\"c004/pusht-s01/regressed/cheap\",\"split\":\"train\",\"question\":\"regressed\"}\n").unwrap();
    assert!(
        brier_core::explain::score_explanations(&f.ledger, &compiled)
            .unwrap()
            .is_none()
    );
    let result = command(&f);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(brier_core::explain::score_explanations(&f.ledger, &compiled).is_err());
    let connection = Connection::open(&f.ledger).unwrap();
    connection.execute_batch("INSERT INTO labels VALUES('c002','pusht-s01',1,'stall/timeout',1,'brightness+obs_noise');").unwrap();
    let metrics = brier_core::explain::score_explanations(&f.ledger, &compiled)
        .unwrap()
        .unwrap();
    assert_eq!(metrics["test"]["episodes"], 1);
    assert_eq!(metrics["test"]["changes"], 1);
    assert_eq!(metrics["test"]["top1_injected_family_agreement"], 0.0);
    assert_eq!(metrics["test"]["top2_injected_family_agreement"], 1.0);
    assert!(metrics["ood"]["top1_injected_family_agreement"].is_null());
    let second_run = Uuid::new_v4().to_string();
    connection.execute("INSERT INTO runs VALUES(?1,'c003','pusht-s01','expensive',20,19,1.0,'raw.parquet',20.0)",[&second_run]).unwrap();
    connection.execute_batch("INSERT INTO labels VALUES('c003','pusht-s01',1,'stall/timeout',1,'action_latency');
        INSERT INTO runs VALUES('training','c004','pusht-s01','expensive',20,19,1.0,'raw.parquet',20.0);
        INSERT INTO explanations VALUES('training','fixture-muse','{\"must_not_score\":true}',NULL);").unwrap();
    let mut second = report;
    second["run_id"] = json!(second_run);
    second["episodes"][0]["id"] = json!(Uuid::new_v4().to_string());
    connection
        .execute(
            "INSERT INTO explanations VALUES(?1,'fixture-muse',?2,NULL)",
            [&second_run, &second.to_string()],
        )
        .unwrap();
    let metrics = brier_core::explain::score_explanations(&f.ledger, &compiled)
        .unwrap()
        .unwrap();
    assert_eq!(metrics["test"]["top1_injected_family_agreement"], 0.0);
    assert_eq!(metrics["ood"]["top1_injected_family_agreement"], 1.0);
    assert_eq!(metrics["ood"]["top2_injected_family_agreement"], 1.0);
    assert!(metrics["definition"]
        .as_str()
        .unwrap()
        .contains("not causal proof"));
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
    assert_eq!(
        brier_core::explain::score_explanations(&f.ledger, &compiled)
            .unwrap()
            .unwrap(),
        metrics
    );
    rows[0]["change_id"] = json!("c999");
    fs::write(
        &compiled,
        rows.iter()
            .map(|row| format!("{row}\n"))
            .collect::<String>(),
    )
    .unwrap();
    assert!(brier_core::explain::score_explanations(&f.ledger, &compiled).is_err());
}
