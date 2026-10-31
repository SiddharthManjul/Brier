use crate::compile;
use anyhow::{ensure, Context, Result};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, RowAccessor},
};
use rusqlite::{params, Connection, OpenFlags};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use uuid::Uuid;

const METRICS: [&str; 7] = [
    "collision_steps",
    "action_direction_changes",
    "action_transitions",
    "object_max_height",
    "object_final_height",
    "goal_distance_min",
    "goal_distance_final",
];
const MODES: [&str; 6] = [
    "none",
    "collision",
    "oscillation",
    "grasp or drop failure",
    "overshoot/misalignment",
    "stall/timeout",
];

#[derive(Debug, Serialize)]
pub struct ExplainReport {
    pub run_id: String,
    pub model: String,
    pub cost_usd: Option<f64>,
    pub episodes: Vec<Value>,
}

struct Episode {
    seed: i64,
    telemetry: Value,
    frames: Value,
}
struct Group {
    env: String,
    scenario: Value,
    episodes: Vec<Episode>,
}

fn data_root() -> Result<PathBuf> {
    let value =
        std::env::var("BRIER_DATA").context("BRIER_DATA must name an existing local directory")?;
    ensure!(
        !value.contains("://"),
        "explanation requires local BRIER_DATA artifacts"
    );
    let root = Path::new(&value)
        .canonicalize()
        .context("resolve BRIER_DATA")?;
    ensure!(
        root.is_dir(),
        "BRIER_DATA must be an existing local directory"
    );
    Ok(root)
}

fn artifact(root: &Path, value: &str) -> Result<PathBuf> {
    ensure!(
        !value.is_empty() && !value.contains("://"),
        "artifact must reference a local path"
    );
    let supplied = Path::new(value);
    let path = if supplied.is_absolute() {
        supplied.to_path_buf()
    } else {
        root.join(supplied)
    };
    let path = path.canonicalize().context("resolve recorded artifact")?;
    ensure!(
        path.starts_with(root),
        "artifact must remain inside BRIER_DATA"
    );
    Ok(path)
}

fn png(root: &Path, value: &str) -> Result<PathBuf> {
    ensure!(
        Path::new(value).is_absolute(),
        "frame paths must be absolute"
    );
    let path = artifact(root, value)?;
    ensure!(path.is_file(), "frame must be an existing PNG file");
    let mut signature = [0; 8];
    File::open(&path)?.read_exact(&mut signature)?;
    ensure!(
        &signature == b"\x89PNG\r\n\x1a\n",
        "frame must contain actual PNG data"
    );
    Ok(path)
}

fn frames(root: &Path, uri: &str, count: usize) -> Result<Value> {
    let directory = artifact(root, uri)?;
    ensure!(
        directory.is_dir(),
        "keyframes_uri must reference a frame directory"
    );
    let mut paths = fs::read_dir(directory)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    ensure!(
        paths.len() == count,
        "episode requires exactly {count} actual keyframes"
    );
    let mut unique = BTreeSet::new();
    paths
        .into_iter()
        .enumerate()
        .map(|(index, path)| {
            let path = png(root, path.to_str().context("non-UTF8 frame path")?)?;
            ensure!(unique.insert(path.clone()), "duplicate episode frame path");
            Ok(json!({"index":index,"path":path}))
        })
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}

fn group(connection: &Connection, root: &Path, run: &str, tier: &str) -> Result<Group> {
    let (change, scenario_id, stored_tier, expected, successes, mean_reward, uri, seconds, env, scenario): (String,String,String,i64,i64,f64,String,f64,String,String) = connection.query_row(
        "SELECT r.change_id,r.scenario_id,r.tier,r.episodes,r.successes,r.mean_reward,r.uri,r.wall_seconds,s.env,s.params_json FROM runs r JOIN scenarios s USING(scenario_id) WHERE r.run_id=?1", [run], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?,row.get(8)?,row.get(9)?)))
        .context("run and scenario must exist in the ledger")?;
    let count = if tier == "cheap" { 3 } else { 20 };
    ensure!(
        stored_tier == tier && expected == count && (0..=expected).contains(&successes),
        "run must be a complete {tier} group"
    );
    ensure!(
        tier == "cheap" || successes < expected,
        "expensive run has no failed episodes"
    );
    ensure!(
        mean_reward.is_finite() && seconds.is_finite() && seconds >= 0.0,
        "invalid stored run measurements"
    );
    compile::sam_prompts(&env)?;
    let scenario: Value = serde_json::from_str(&scenario)?;
    let uris: Vec<String> = if uri.starts_with('[') {
        serde_json::from_str(&uri).context("invalid run artifact paths")?
    } else {
        vec![uri]
    };
    ensure!(!uris.is_empty(), "run has no episode artifacts");
    let mut sources = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    let (mut found_successes, mut reward, mut wall) = (0, 0.0, 0.0);
    let mut episodes = Vec::new();
    for uri in uris {
        let path = artifact(root, &uri)?;
        ensure!(
            sources.insert(path.clone()),
            "duplicate run episode artifact"
        );
        let reader = SerializedFileReader::new(File::open(path)?)?;
        for row in reader.get_row_iter(None)? {
            let row = row?;
            ensure!(
                row.get_column_iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
                    == [
                        "env",
                        "policy",
                        "change_id",
                        "scenario_id",
                        "tier",
                        "seed",
                        "success",
                        "steps",
                        "terminal_state",
                        "reward_sum",
                        "keyframes_uri",
                        "wall_seconds"
                    ],
                "unexpected episode schema"
            );
            if row.get_string(2)? != &change
                || row.get_string(3)? != &scenario_id
                || row.get_string(4)? != tier
            {
                continue;
            }
            let policy = if env.contains("PushT") {
                "lerobot/diffusion_pusht"
            } else {
                "lerobot/act_aloha_sim_transfer_cube_human"
            };
            ensure!(
                row.get_string(0)? == &env && row.get_string(1)? == policy,
                "episode identity differs from stored run"
            );
            let seed = row.get_long(5)?;
            ensure!(
                seed >= 0 && seeds.insert(seed),
                "duplicate or invalid episode seed"
            );
            let success = row.get_bool(6)?;
            let steps = row.get_long(7)?;
            let episode_reward = row.get_double(9)?;
            let elapsed = row.get_double(11)?;
            ensure!(
                steps > 0 && episode_reward.is_finite() && elapsed.is_finite() && elapsed >= 0.0,
                "invalid episode measurements"
            );
            found_successes += i64::from(success);
            reward += episode_reward;
            wall += elapsed;
            if tier == "expensive" && success {
                continue;
            }
            let terminal: Value = serde_json::from_str(row.get_string(8)?)?;
            let mut telemetry = json!({"steps":steps,"reward_sum":episode_reward});
            for key in METRICS {
                let value = terminal["trajectory"][key]
                    .as_f64()
                    .with_context(|| format!("episode requires observed trajectory.{key}"))?;
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "invalid observed trajectory.{key}"
                );
                telemetry[key] = json!(value);
            }
            ensure!(
                telemetry["action_direction_changes"].as_f64()
                    <= telemetry["action_transitions"].as_f64()
                    && telemetry["goal_distance_min"].as_f64()
                        <= telemetry["goal_distance_final"].as_f64(),
                "inconsistent observed trajectory"
            );
            let uri = row
                .get_column_iter()
                .nth(10)
                .and_then(|(_, field)| match field {
                    Field::Str(value) => Some(value.as_str()),
                    _ => None,
                })
                .context("episode has no actual recorded keyframes")?;
            episodes.push(Episode {
                seed,
                telemetry,
                frames: frames(root, uri, if tier == "cheap" { 1 } else { 8 })?,
            });
        }
    }
    ensure!(
        seeds.len() == expected as usize
            && found_successes == successes
            && (reward / expected as f64 - mean_reward).abs() <= 1e-8 * (1.0 + mean_reward.abs())
            && (wall - seconds).abs() <= 1e-8 * (1.0 + seconds),
        "raw episode group differs from stored aggregate"
    );
    episodes.sort_by_key(|episode| episode.seed);
    Ok(Group {
        env,
        scenario,
        episodes,
    })
}

pub(crate) fn cheap_evidence(ledger: &Path, change: &str, scenario: &str) -> Result<Value> {
    let root = data_root()?;
    let connection = Connection::open_with_flags(ledger, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut query = connection.prepare(
        "SELECT run_id FROM runs WHERE change_id=?1 AND scenario_id=?2 AND tier='cheap'",
    )?;
    let runs = query
        .query_map([change, scenario], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    ensure!(
        runs.len() == 1,
        "cheap evidence requires exactly one three-episode run"
    );
    let group = group(&connection, &root, &runs[0], "cheap")?;
    let observations: Vec<_> = group
        .episodes
        .iter()
        .map(|episode| episode.telemetry.clone())
        .collect();
    let frames: Vec<_> = group
        .episodes
        .into_iter()
        .enumerate()
        .map(|(index, episode)| json!({"index":index,"path":episode.frames[0]["path"]}))
        .collect();
    ensure!(
        frames
            .iter()
            .map(|frame| frame["path"].as_str())
            .collect::<BTreeSet<_>>()
            .len()
            == 3,
        "cheap evidence requires three distinct final frames"
    );
    Ok(json!({"observations":observations,"frames":frames}))
}

pub(crate) fn bridge(payload: &Value) -> Result<Value> {
    let mut child = Command::new("python3")
        .args(["-m", "brier_explain.client"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("start brier_explain.client using the active Python environment")?;
    serde_json::to_writer(
        child.stdin.as_mut().context("open explanation stdin")?,
        payload,
    )?;
    child
        .stdin
        .take()
        .context("close explanation stdin")?
        .flush()?;
    let output = child.wait_with_output()?;
    ensure!(
        output.status.success(),
        "explanation bridge failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).context("explanation bridge returned invalid JSON")
}

fn returned_episodes<'a>(response: &'a Value, expected: &[Value]) -> Result<Vec<&'a Value>> {
    let episodes = response["episodes"]
        .as_array()
        .context("bridge result requires episodes")?;
    ensure!(
        episodes.len() == expected.len()
            && episodes
                .iter()
                .zip(expected)
                .all(|(actual, source)| actual["id"] == source["id"]),
        "bridge episode identities differ from requested episodes"
    );
    Ok(episodes.iter().collect())
}

fn cause_predictions(value: &str) -> Result<Vec<&str>> {
    let (ranked, rationale) = value.split_once('\n').context(
        "cause_hypothesis requires ranked family names, newline, and observed-evidence rationale",
    )?;
    let families: Vec<_> = ranked.split(" | ").collect();
    ensure!(
        (1..=2).contains(&families.len())
            && !rationale.trim().is_empty()
            && families
                .iter()
                .all(|family| compile::CAUSE_FAMILIES.contains(family))
            && families.iter().collect::<BTreeSet<_>>().len() == families.len(),
        "cause_hypothesis requires one or two distinct canonical family names and a rationale"
    );
    Ok(families)
}

/// Scores persisted held-out hypotheses against injected labels without exposing labels to a provider.
pub fn score_explanations(ledger: &Path, compiled: &Path) -> Result<Option<Value>> {
    let mut splits = BTreeMap::new();
    for line in BufReader::new(File::open(compiled)?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let id = row["id"]
            .as_str()
            .context("compiled scoring row requires id")?;
        let guide = id.contains(':');
        let parts: Vec<_> = id.split(if guide { ':' } else { '/' }).collect();
        ensure!(
            parts.len() == 4 && parts.iter().all(|s| !s.is_empty()),
            "invalid compiled scoring id"
        );
        let change = parts[0];
        if guide {
            ensure!(
                row["change_id"] == change
                    && row["scenario_id"] == parts[1]
                    && row["pair_id"] == format!("{}:{}", change, parts[1])
                    && row["question"] == parts[2]
                    && row["variant"] == parts[3]
                    && matches!(parts[3], "ev" | "noev"),
                "invalid guide compiled scoring identity"
            );
        }
        let split = row["split"]
            .as_str()
            .context("compiled scoring row requires split")?;
        ensure!(
            matches!(split, "train" | "calibration" | "test" | "ood"),
            "invalid compiled scoring split"
        );
        if let Some(previous) = splits.insert(change.to_string(), split.to_string()) {
            ensure!(
                previous == split,
                "scoring change appears in multiple frozen splits"
            );
        }
    }
    ensure!(!splits.is_empty(), "compiled scoring pool is empty");
    let connection = Connection::open_with_flags(ledger, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut query = connection.prepare("SELECT r.run_id,r.change_id,r.scenario_id,r.tier,r.episodes,r.successes,e.json,l.cause_family FROM explanations e JOIN runs r USING(run_id) LEFT JOIN labels l ON l.change_id=r.change_id AND l.scenario_id=r.scenario_id ORDER BY r.run_id")?;
    let rows = query.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, i64>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, Option<String>>(7)?,
        ))
    })?;
    let mut counts = [[0usize; 3]; 2];
    let mut changes = [BTreeSet::new(), BTreeSet::new()];
    let mut episode_ids = BTreeSet::new();
    for row in rows {
        let (run, change, _, tier, episodes, successes, stored, truth) = row?;
        let index = match splits.get(&change).map(String::as_str) {
            Some("test") => 0,
            Some("ood") => 1,
            _ => continue,
        };
        ensure!(
            tier == "expensive" && episodes == 20 && (0..episodes).contains(&successes),
            "scored explanation must reference a failed complete expensive run"
        );
        let truth = truth
            .context("held-out explanation is missing its authoritative cause_family label")?;
        let actual: Vec<_> = truth.split('+').collect();
        ensure!(
            (1..=2).contains(&actual.len())
                && actual
                    .iter()
                    .all(|family| compile::CAUSE_FAMILIES.contains(family))
                && actual.iter().collect::<BTreeSet<_>>().len() == actual.len(),
            "invalid authoritative injected cause_family"
        );
        let stored: Value = serde_json::from_str(&stored)?;
        let predictions = stored["episodes"]
            .as_array()
            .context("held-out explanation requires per-episode predictions")?;
        ensure!(
            predictions.len() == (episodes - successes) as usize,
            "held-out explanation count differs from failed episode count"
        );
        for prediction in predictions {
            let id = prediction["id"]
                .as_str()
                .context("scored episode is missing UUID")?;
            let uuid = Uuid::parse_str(id)?;
            ensure!(
                uuid.get_version_num() == 4
                    && uuid.to_string() == id
                    && episode_ids.insert(id.to_string()),
                "invalid or duplicate scored episode UUID in run {run}"
            );
            let ranked = cause_predictions(
                prediction["cause_hypothesis"]
                    .as_str()
                    .context("missing scored cause_hypothesis")?,
            )?;
            counts[index][0] += 1;
            counts[index][1] += usize::from(actual.contains(&ranked[0]));
            counts[index][2] += usize::from(ranked.iter().any(|family| actual.contains(family)));
        }
        changes[index].insert(change);
    }
    if counts.iter().all(|count| count[0] == 0) {
        return Ok(None);
    }
    let mut metrics = json!({"unit":"explained failed expensive episode", "definition":"Top-k injected-family agreement: at least one of the first k predicted families matches any injected component in brier label cause_family. This measures observational agreement, not causal proof."});
    for (index, split) in ["test", "ood"].iter().enumerate() {
        let count = counts[index];
        let rate = |hits: usize| {
            if count[0] == 0 {
                None
            } else {
                Some(hits as f64 / count[0] as f64)
            }
        };
        metrics[*split] = json!({"episodes":count[0],"changes":changes[index].len(),"top1_injected_family_agreement":rate(count[1]),"top2_injected_family_agreement":rate(count[2])});
    }
    Ok(Some(metrics))
}

/// Explains every recorded failed episode of one expensive run and atomically appends its evidence.
pub fn explain(ledger: &Path, run: &str) -> Result<ExplainReport> {
    let id = Uuid::parse_str(run).context("run must be a UUIDv4")?;
    ensure!(
        id.get_version_num() == 4 && id.to_string() == run,
        "run must be a canonical UUIDv4"
    );
    let root = data_root()?;
    let connection = Connection::open_with_flags(ledger, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    ensure!(
        !connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM explanations WHERE run_id=?1)",
            [run],
            |row| row.get::<_, bool>(0)
        )?,
        "run already has an append-only explanation"
    );
    let group = group(&connection, &root, run, "expensive")?;
    let concepts = compile::sam_prompts(&group.env)?;
    let requests: Vec<_> = group.episodes.iter().map(|episode| json!({"id":Uuid::new_v4().to_string(),"frames":episode.frames,"sam_prompts":concepts})).collect();
    let segmented = bridge(&json!({"operation":"segment","episodes":requests}))?;
    let segmented = returned_episodes(&segmented, &requests)?;
    let mut explain_requests = Vec::new();
    for ((segmented, request), episode) in segmented.iter().zip(&requests).zip(&group.episodes) {
        ensure!(
            segmented["model"] == "sam-3.1",
            "unexpected SAM model identity"
        );
        let frames = segmented["frames"]
            .as_array()
            .context("SAM result requires frames")?;
        ensure!(frames.len() == 8, "SAM result requires eight frames");
        let mut unique = BTreeSet::new();
        for (index, frame) in frames.iter().enumerate() {
            ensure!(
                frame["index"] == index
                    && frame["path"] == request["frames"][index]["path"]
                    && frame.as_object().is_some_and(|object| object.len() == 3),
                "SAM changed original frame identity"
            );
            for key in ["path", "overlay_path"] {
                let path = png(
                    &root,
                    frame[key].as_str().context("SAM frame path missing")?,
                )?;
                ensure!(
                    unique.insert(path),
                    "SAM raw and overlay paths must be distinct"
                );
            }
        }
        let prompt = compile::explanation_prompt(
            &group.env,
            &group.scenario,
            &episode.telemetry,
            &segmented["geometry"],
        )?;
        explain_requests.push(json!({"id":request["id"],"prompt":prompt,"frames":frames}));
    }
    let response = bridge(&json!({"operation":"explain","episodes":explain_requests}))?;
    let answers = returned_episodes(&response, &requests)?;
    let mut report = ExplainReport {
        run_id: run.into(),
        model: String::new(),
        cost_usd: Some(0.0),
        episodes: Vec::new(),
    };
    for (((answer, segmented), request), episode) in answers
        .iter()
        .zip(segmented)
        .zip(&requests)
        .zip(&group.episodes)
    {
        let model = answer["model"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .context("missing Muse model identity")?;
        ensure!(
            report.model.is_empty() || report.model == model,
            "explanation model identity changed within a run"
        );
        report.model = model.into();
        let explanation = &answer["explanation"];
        ensure!(
            explanation
                .as_object()
                .is_some_and(|object| object.len() == 4)
                && explanation["failure_mode"]
                    .as_str()
                    .is_some_and(|value| MODES.contains(&value)),
            "invalid explanation fields or failure mode"
        );
        for key in ["cause_hypothesis", "suggested_check"] {
            ensure!(
                explanation[key]
                    .as_str()
                    .is_some_and(|value| !value.trim().is_empty()),
                "explanation requires nonempty {key}"
            );
        }
        cause_predictions(explanation["cause_hypothesis"].as_str().unwrap())?;
        let evidence = explanation["evidence_frames"]
            .as_array()
            .context("missing explanation evidence_frames")?;
        ensure!(
            !evidence.is_empty()
                && evidence
                    .iter()
                    .all(|value| value.as_u64().is_some_and(|index| index < 8))
                && evidence
                    .iter()
                    .map(Value::to_string)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == evidence.len(),
            "explanation references invalid or duplicate evidence frames"
        );
        let cost = match answer
            .get("cost_usd")
            .context("missing explanation cost provenance")?
        {
            Value::Null => None,
            value => {
                let cost = value.as_f64().context("invalid explanation cost")?;
                ensure!(cost.is_finite() && cost >= 0.0, "invalid explanation cost");
                Some(cost)
            }
        };
        report.cost_usd = report.cost_usd.zip(cost).map(|(total, cost)| total + cost);
        let local = |key: &str| -> Result<Vec<String>> {
            segmented["frames"]
                .as_array()
                .unwrap()
                .iter()
                .map(|frame| {
                    let path = artifact(
                        &root,
                        frame[key]
                            .as_str()
                            .context("missing persisted frame path")?,
                    )?;
                    Ok(path
                        .strip_prefix(&root)?
                        .to_str()
                        .context("non-UTF8 persisted path")?
                        .to_string())
                })
                .collect()
        };
        let mut stored = explanation.clone();
        stored["id"] = request["id"].clone();
        stored["seed"] = json!(episode.seed);
        stored["model"] = json!(model);
        stored["geometry"] = segmented["geometry"].clone();
        stored["keyframe_urls"] = json!([]);
        stored["overlay_urls"] = json!([]);
        stored["local_keyframes"] = json!(local("path")?);
        stored["local_overlays"] = json!(local("overlay_path")?);
        stored["cost_usd"] = json!(cost);
        report.episodes.push(stored);
    }
    ensure!(
        report.cost_usd.is_none_or(f64::is_finite),
        "explanation total cost overflow"
    );
    connection
        .execute(
            "INSERT INTO explanations(run_id,model,json,cost_usd) VALUES(?1,?2,?3,?4)",
            params![
                run,
                report.model,
                serde_json::to_string(&report)?,
                report.cost_usd
            ],
        )
        .context("append completed explanation")?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::{
        data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type},
        file::{properties::WriterProperties, writer::SerializedFileWriter},
        schema::parser::parse_message_type,
    };
    use std::sync::Arc;

    #[test]
    fn cheap_evidence_requires_three_actual_final_frames_and_filters_hidden_metadata() {
        let directory =
            std::env::temp_dir().join(format!("brier-cheap-evidence-{}", Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let ledger = directory.join("ledger.sqlite");
        let connection = crate::ledger::init(&ledger).unwrap();
        connection.execute_batch("INSERT INTO scenarios VALUES('pusht-s01','gym_pusht/PushT-v0','{\"pose_bin\":0,\"lighting\":\"normal\",\"camera_jitter\":false}',0.8);
            INSERT INTO runs VALUES('cheap','c001','pusht-s01','cheap',3,2,1.0,'cheap.parquet',3.0);").unwrap();
        let schema = Arc::new(parse_message_type("message episodes {
            REQUIRED BYTE_ARRAY env (UTF8); REQUIRED BYTE_ARRAY policy (UTF8);
            REQUIRED BYTE_ARRAY change_id (UTF8); REQUIRED BYTE_ARRAY scenario_id (UTF8);
            REQUIRED BYTE_ARRAY tier (UTF8); REQUIRED INT64 seed; REQUIRED BOOLEAN success;
            REQUIRED INT64 steps; REQUIRED BYTE_ARRAY terminal_state (UTF8); REQUIRED DOUBLE reward_sum;
            OPTIONAL BYTE_ARRAY keyframes_uri (UTF8); REQUIRED DOUBLE wall_seconds; }").unwrap());
        let mut writer = SerializedFileWriter::new(
            File::create(directory.join("cheap.parquet")).unwrap(),
            schema,
            Arc::new(WriterProperties::builder().build()),
        )
        .unwrap();
        let mut group = writer.next_row_group().unwrap();
        let terminal = json!({"trajectory":{"collision_steps":0,"action_direction_changes":0,"action_transitions":10,"object_max_height":0.0,"object_final_height":0.0,"goal_distance_min":10.0,"goal_distance_final":11.0},"cause_family":"must not escape"}).to_string();
        for seed in 0..3 {
            fs::create_dir(directory.join(seed.to_string())).unwrap();
            fs::write(
                directory.join(seed.to_string()).join("final.png"),
                b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\xf8\xff\xff\x3f\x00\x05\xfe\x02\xfe\x0d\xef\x46\xb8\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82",
            )
            .unwrap();
        }
        for index in 0..12 {
            let mut column = group.next_column().unwrap().unwrap();
            match index {
                5 | 7 => {
                    column
                        .typed::<Int64Type>()
                        .write_batch(
                            &if index == 5 { [0, 1, 2] } else { [10, 10, 10] },
                            None,
                            None,
                        )
                        .unwrap();
                }
                6 => {
                    column
                        .typed::<BoolType>()
                        .write_batch(&[true, true, false], None, None)
                        .unwrap();
                }
                9 | 11 => {
                    column
                        .typed::<DoubleType>()
                        .write_batch(&[1.0; 3], None, None)
                        .unwrap();
                }
                10 => {
                    column
                        .typed::<ByteArrayType>()
                        .write_batch(
                            &[
                                ByteArray::from("0"),
                                ByteArray::from("1"),
                                ByteArray::from("2"),
                            ],
                            Some(&[1, 1, 1]),
                            None,
                        )
                        .unwrap();
                }
                _ => {
                    let value = match index {
                        0 => "gym_pusht/PushT-v0",
                        1 => "lerobot/diffusion_pusht",
                        2 => "c001",
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
        let previous = std::env::var_os("BRIER_DATA");
        std::env::set_var("BRIER_DATA", &directory);
        let evidence = cheap_evidence(&ledger, "c001", "pusht-s01").unwrap();
        assert_eq!(evidence["observations"].as_array().unwrap().len(), 3);
        assert_eq!(evidence["frames"].as_array().unwrap().len(), 3);
        assert_eq!(evidence["frames"][2]["index"], 2);
        assert!(!evidence.to_string().contains("must not escape"));
        fs::remove_file(directory.join("2/final.png")).unwrap();
        assert!(cheap_evidence(&ledger, "c001", "pusht-s01").is_err());
        if let Some(value) = previous {
            std::env::set_var("BRIER_DATA", value);
        } else {
            std::env::remove_var("BRIER_DATA");
        }
        fs::remove_dir_all(directory).unwrap();
    }
}
