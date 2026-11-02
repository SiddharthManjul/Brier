use crate::{
    cascade, compile, explain, ledger,
    policy::{
        calibrate, calibrate_executed, selected, ExecutedChange, ExecutedPair, Guarantee,
        LabeledChange, PairScore, Policy, LAMBDA_HIGH, LAMBDA_LOW,
    },
};
use anyhow::{ensure, Context, Result};
use reqwest::blocking::Client;
use rusqlite::{Connection, OpenFlags, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::Write,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RiskCalibration {
    pub format_version: u8,
    pub model_version: String,
    pub t1: Value,
    pub temperatures: BTreeMap<String, f64>,
    pub probability_calibration_change_ids: Vec<String>,
    pub changes: Vec<LabeledChange>,
    #[serde(default)]
    pub executed_changes: Vec<ExecutedChange>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Decision {
    pub decision_id: String,
    pub change_id: String,
    pub scenario_id: String,
    pub tier_used: String,
    pub model_version: String,
    pub probabilities: Value,
    pub lambda: f64,
    pub action: String,
    pub propensity: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DecideReport {
    pub change_id: String,
    pub policy: Policy,
    pub decisions: Vec<Decision>,
    pub cheap_required: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    model_version: String,
    questions: BTreeMap<String, BTreeMap<String, f64>>,
}

fn ids(value: &Value, field: &str) -> Result<BTreeSet<String>> {
    let values = value[field]
        .as_array()
        .with_context(|| format!("T1 artifact requires {field}"))?;
    let mut output = BTreeSet::new();
    for value in values {
        let id = value
            .as_str()
            .filter(|id| !id.trim().is_empty())
            .context("invalid T1 provenance id")?;
        ensure!(output.insert(id.to_string()), "duplicate T1 provenance id");
    }
    ensure!(!output.is_empty(), "T1 provenance must not be empty");
    Ok(output)
}

fn hex_revision(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn pinned_version(value: &str) -> bool {
    let (base, adapter) = match value.split_once("+lora@") {
        Some((base, adapter)) => (base, Some(adapter)),
        None => (value, None),
    };
    base.rsplit_once('@').is_some_and(|(model, revision)| {
        !model.is_empty()
            && !model.contains(char::is_whitespace)
            && !model.contains('@')
            && hex_revision(revision, 40)
    }) && adapter.is_none_or(|revision| hex_revision(revision, 64))
}

fn validate(
    calibration: &RiskCalibration,
    frozen: &compile::Snapshot,
    connection: &Connection,
) -> Result<()> {
    ensure!(
        matches!(calibration.format_version, 1 | 2),
        "unsupported risk calibration format"
    );
    ensure!(
        pinned_version(&calibration.model_version),
        "risk model_version requires base@40-hex with an optional +lora@64-hex digest"
    );
    ensure!(
        calibration.t1["model_version"]
            .as_str()
            .and_then(|value| value.strip_prefix("t1@"))
            .is_some_and(|value| hex_revision(value, 64)),
        "T1 artifact requires model_version=t1@64-hex-digest"
    );
    let training = ids(&calibration.t1, "training_change_ids")?;
    let fitted = ids(&calibration.t1, "calibration_change_ids")?;
    let expected_training: BTreeSet<_> = frozen
        .splits
        .iter()
        .filter(|(_, split)| *split == "train")
        .map(|(id, _)| id.clone())
        .collect();
    ensure!(
        training == expected_training,
        "T1 training provenance disagrees with frozen compiled train IDs"
    );
    let mut probability_fit = BTreeSet::new();
    for id in &calibration.probability_calibration_change_ids {
        ensure!(
            !id.trim().is_empty() && probability_fit.insert(id.clone()),
            "invalid or duplicate probability calibration id"
        );
    }
    ensure!(
        fitted.is_subset(&probability_fit) && training.is_disjoint(&probability_fit),
        "probability calibration provenance must include T1 isotonic IDs and exclude training"
    );
    for id in &probability_fit {
        ensure!(
            !frozen
                .splits
                .get(id)
                .is_some_and(|split| matches!(split.as_str(), "train" | "test" | "ood")),
            "probability calibration overlaps frozen train/test/ood changes"
        );
    }
    ensure!(
        calibration.temperatures.len() == 3
            && ["regressed", "failure_mode", "success_bucket"]
                .iter()
                .all(|question| calibration
                    .temperatures
                    .get(*question)
                    .is_some_and(|value| value.is_finite() && (0.05..=20.0).contains(value))),
        "three fitted question temperatures in [0.05,20] are required"
    );
    if calibration.format_version == 2 {
        let expected = frozen
            .splits
            .iter()
            .filter(|(id, split)| {
                matches!(split.as_str(), "calibration" | "test") && !probability_fit.contains(*id)
            })
            .map(|(id, _)| id.clone())
            .collect::<BTreeSet<_>>();
        ensure!(
            calibration
                .changes
                .iter()
                .map(|change| change.change_id.clone())
                .collect::<BTreeSet<_>>()
                == expected,
            "executed risk pool must contain every frozen held-out change and exclude fitting/OOD"
        );
    }
    for change in &calibration.changes {
        ensure!(
            !training.contains(&change.change_id)
                && !probability_fit.contains(&change.change_id)
                && !frozen
                    .splits
                    .get(&change.change_id)
                    .is_some_and(|split| matches!(split.as_str(), "train" | "ood")
                        || (calibration.format_version == 1 && split == "test")),
            "risk calibration overlaps fitting or unsupported held-out splits"
        );
        let env: String = connection
            .query_row(
                "SELECT env FROM changes WHERE change_id=?1",
                [&change.change_id],
                |row| row.get(0),
            )
            .context("risk change missing from ledger")?;
        let mut query = connection.prepare("SELECT scenario_id FROM scenarios WHERE env=?1")?;
        let expected = query
            .query_map([env], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<BTreeSet<_>>>()?;
        let actual: BTreeSet<_> = change
            .pairs
            .iter()
            .map(|pair| pair.scenario_id.clone())
            .collect();
        ensure!(
            !expected.is_empty() && actual == expected && actual.len() == change.pairs.len(),
            "risk calibration needs every scenario exactly once per change"
        );
        for pair in &change.pairs {
            let label: i64 = connection
                .query_row(
                    "SELECT regressed FROM labels WHERE change_id=?1 AND scenario_id=?2",
                    [&change.change_id, &pair.scenario_id],
                    |row| row.get(0),
                )
                .context("risk pair requires a brier label result")?;
            ensure!(
                (label == 0 || label == 1) && (label == 1) == pair.regressed,
                "risk calibration label disagrees with ledger"
            );
        }
    }
    Ok(())
}

fn score_t2(pair: &compile::CompiledPair, calibration: &RiskCalibration) -> Result<Value> {
    let url = env::var("MODAL_DECIDE_URL")
        .context("MODAL_DECIDE_URL is required for uncertain T1 pairs")?;
    ensure!(!url.trim().is_empty(), "MODAL_DECIDE_URL must not be empty");
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let response = client
        .post(url)
        .json(&json!({"questions": pair.questions}))
        .send()
        .context("call Modal POST /decide")?;
    ensure!(
        response.status().is_success(),
        "Modal /decide returned {}",
        response.status()
    );
    let response: Response = response.json().context("invalid Modal /decide response")?;
    ensure!(
        response.model_version == calibration.model_version,
        "Modal model_version differs from risk calibration"
    );
    ensure!(
        response.questions.len() == pair.questions.len(),
        "Modal returned unexpected question IDs"
    );
    let mut output = serde_json::Map::new();
    for question in &pair.questions {
        let id = question["id"]
            .as_str()
            .context("missing compiled question id")?;
        let kind = id
            .split('/')
            .nth(2)
            .context("invalid compiled question id")?;
        let probabilities = response
            .questions
            .get(id)
            .context("Modal omitted a compiled question")?;
        let options = question["options"]
            .as_object()
            .context("missing compiled options")?;
        ensure!(
            probabilities.len() == options.len()
                && probabilities.keys().all(|key| options.contains_key(key)),
            "Modal probability keys differ from compiled options"
        );
        ensure!(
            probabilities
                .values()
                .all(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
                && (probabilities.values().sum::<f64>() - 1.0).abs() <= 1.1e-7,
            "Modal option probabilities must be positive, finite and normalized"
        );
        let maximum = probabilities
            .values()
            .map(|value| value.ln())
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: BTreeMap<_, _> = probabilities
            .iter()
            .map(|(letter, probability)| {
                (
                    letter,
                    ((probability.ln() - maximum) / calibration.temperatures[kind]).exp(),
                )
            })
            .collect();
        let total: f64 = weights.values().sum();
        let mut semantic = serde_json::Map::new();
        for (letter, weight) in weights {
            let meaning = options[letter]
                .as_str()
                .context("invalid compiled option meaning")?;
            semantic.insert(meaning.to_string(), json!(weight / total));
        }
        output.insert(kind.to_string(), Value::Object(semantic));
    }
    Ok(Value::Object(output))
}

fn independently_complete(connection: &Connection, change: &str) -> Result<bool> {
    let decisions: i64 = connection.query_row(
        "SELECT count(*) FROM decisions WHERE change_id=?1",
        [change],
        |row| row.get(0),
    )?;
    if decisions != 0 {
        return Ok(false);
    }
    let mut query = connection.prepare("SELECT s.scenario_id,l.regressed FROM changes c JOIN scenarios s ON s.env=c.env LEFT JOIN labels l ON l.change_id=c.change_id AND l.scenario_id=s.scenario_id WHERE c.change_id=?1 ORDER BY s.scenario_id")?;
    let scenarios = query
        .query_map([change], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if scenarios.is_empty() {
        return Ok(false);
    }
    for (scenario, label) in scenarios {
        if !matches!(label, Some(0 | 1)) {
            return Ok(false);
        }
        let mut query = connection.prepare("SELECT tier,episodes,successes,mean_reward,wall_seconds FROM runs WHERE change_id=?1 AND scenario_id=?2 AND tier IN ('cheap','expensive')")?;
        let runs = query
            .query_map([change, &scenario], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, f64>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (tier, episodes) in [("cheap", 3), ("expensive", 20)] {
            let group: Vec<_> = runs.iter().filter(|run| run.0 == tier).collect();
            if group.len() != 1
                || group[0].1 != episodes
                || !(0..=episodes).contains(&group[0].2)
                || !group[0].3.is_finite()
                || !group[0].4.is_finite()
                || group[0].4 < 0.0
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn refresh(
    paths: (&Path, &Path, &Path),
    source: &[u8],
    calibration: &mut RiskCalibration,
    frozen: &compile::Snapshot,
    connection: &mut Connection,
    live: &str,
    alpha: f64,
) -> Result<()> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let training = ids(&calibration.t1, "training_change_ids")?;
    let fitted = ids(&calibration.t1, "calibration_change_ids")?;
    let mut candidates = Vec::new();
    for (change, split) in &frozen.splits {
        if split == "calibration"
            && change != live
            && !training.contains(change)
            && !fitted.contains(change)
            && !calibration
                .probability_calibration_change_ids
                .contains(change)
            && !calibration
                .changes
                .iter()
                .any(|item| &item.change_id == change)
            && independently_complete(&transaction, change)?
        {
            candidates.push(change.clone());
        }
    }
    if candidates.len() < 10 {
        transaction.commit()?;
        return Ok(());
    }
    let mut updated = calibration.clone();
    for change in candidates {
        let mut pairs = Vec::new();
        for pair in compile::compile_live(paths.0, &change, paths.1)? {
            let t1 = cascade::predict_t1(&calibration.t1, &pair.features)?;
            let probability = if (0.1..=0.9).contains(&t1) {
                score_t2(&pair, calibration)?["regressed"]["yes"]
                    .as_f64()
                    .context("missing refreshed semantic regression probability")?
            } else {
                t1
            };
            let label: i64 = transaction.query_row(
                "SELECT regressed FROM labels WHERE change_id=?1 AND scenario_id=?2",
                [&change, &pair.scenario_id],
                |row| row.get(0),
            )?;
            pairs.push(PairScore {
                scenario_id: pair.scenario_id,
                probability,
                regressed: label == 1,
            });
        }
        updated.changes.push(LabeledChange {
            change_id: change,
            pairs,
        });
    }
    validate(&updated, frozen, &transaction)?;
    calibrate(&updated.changes, alpha)?;
    publish_refresh(paths.2, source, &updated)?;
    transaction.commit()?;
    *calibration = updated;
    Ok(())
}

fn publish_refresh(path: &Path, source: &[u8], updated: &RiskCalibration) -> Result<()> {
    let path = path
        .canonicalize()
        .context("resolve calibration refresh destination")?;
    let temporary = path.with_file_name(format!(".brier-risk-{}.tmp", Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut file =
            File::create_new(&temporary).context("create calibration refresh temporary file")?;
        serde_json::to_writer_pretty(&mut file, updated)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        ensure!(
            fs::read(&path)? == source,
            "calibration artifact changed during refresh"
        );
        fs::rename(&temporary, &path).context("publish refreshed calibration artifact")?;
        Ok(())
    })();
    if temporary.exists() {
        fs::remove_file(&temporary).context("remove calibration refresh temporary file")?;
    }
    result
}

fn action(probability: f64, lambda: f64, draw: f64) -> (&'static str, f64, &'static str) {
    if probability >= lambda {
        ("expensive", 1.0, "policy_above_lambda")
    } else if draw < 0.05 {
        ("expensive", 0.05, "exploration")
    } else {
        ("skip", 0.05, "policy_below_lambda")
    }
}

fn add_review(
    action: &str,
    propensity: f64,
    _reason: &str,
    review: &Value,
) -> Result<(String, f64, String)> {
    let fields = review
        .as_object()
        .context("T3 response must be an object")?;
    ensure!(
        fields.len() == 4
            && ["model", "recommend", "rationale", "cost_usd"]
                .iter()
                .all(|field| fields.contains_key(*field))
            && review["model"] == "muse-spark-1.3-contributor"
            && review["rationale"]
                .as_str()
                .is_some_and(|text| !text.trim().is_empty())
            && (review["cost_usd"].is_null()
                || review["cost_usd"]
                    .as_f64()
                    .is_some_and(|cost| cost.is_finite() && cost >= 0.0)),
        "invalid Contributor T3 response"
    );
    match review["recommend"].as_str() {
        Some("run_expensive" | "run_more_cheap") => {
            // Additional evidence uses the full tier: the calibrated cheap feature
            // vector remains the original three episodes.
            Ok(("expensive".into(), 1.0, "t3_added".into()))
        }
        Some("no_change") => Ok((action.into(), propensity, "t3_no_change".into())),
        _ => anyhow::bail!("unknown T3 recommendation"),
    }
}

const ESCALATION_CAP: usize = 2;

fn escalate(
    decisions: &mut [Decision],
    lambda: f64,
    mut review: impl FnMut(&Decision) -> Result<Value>,
) -> Result<()> {
    let mut additions = 0;
    for decision in decisions {
        let p = decision.probabilities["regressed"]["yes"]
            .as_f64()
            .context("invalid escalation probability")?;
        if decision.action != "skip"
            || decision.tier_used == "T2-noev"
            || p < lambda - 0.10
            || p >= lambda
        {
            continue;
        }
        if additions == ESCALATION_CAP {
            decision.reason = "t3_capped".into();
            continue;
        }
        let response = review(decision)?;
        (decision.action, decision.propensity, decision.reason) = add_review(
            &decision.action,
            decision.propensity,
            &decision.reason,
            &response,
        )?;
        additions += usize::from(decision.action == "expensive");
        decision.probabilities["escalation"] = response;
        decision.tier_used.push_str("+T3");
    }
    Ok(())
}

/// Scores every live scenario, applies CRC and exploration, and atomically logs decisions.
/// Stage A must already contain exactly three cheap episodes per scenario. T1's
/// uncertain band is [0.1,0.9]. No family name alone enables a T0 shortcut.
/// At least ten complete, independent frozen-calibration changes trigger a validated
/// artifact refresh before live scoring; that refresh survives later live errors.
pub fn decide(
    ledger_path: &Path,
    compiled: &Path,
    calibration: &Path,
    change_id: &str,
    alpha: f64,
) -> Result<DecideReport> {
    decide_guarantee(
        ledger_path,
        compiled,
        calibration,
        change_id,
        alpha,
        Guarantee::Change,
    )
}

/// Executes the requested guarantee, returning a nonpersisted cheap plan before evidence exists.
pub fn decide_guarantee(
    ledger_path: &Path,
    compiled: &Path,
    calibration: &Path,
    change_id: &str,
    alpha: f64,
    guarantee: Guarantee,
) -> Result<DecideReport> {
    ensure!(
        change_id.len() >= 4
            && change_id.starts_with('c')
            && change_id[1..].bytes().all(|b| b.is_ascii_digit())
            && change_id[1..].bytes().any(|b| b != b'0'),
        "change must start at c001 with at least three digits"
    );
    let calibration_path = calibration;
    let source = fs::read(calibration_path)?;
    let mut calibration: RiskCalibration = serde_json::from_slice(&source)?;
    let frozen = compile::snapshot(compiled)?;
    let mut connection =
        Connection::open_with_flags(ledger_path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    validate(&calibration, &frozen, &connection)?;
    ensure!(
        frozen
            .splits
            .get(change_id)
            .is_none_or(|split| split != "train"),
        "live change must not be a model training change"
    );
    ensure!(
        !calibration
            .changes
            .iter()
            .any(|change| change.change_id == change_id),
        "live change must not be a risk calibration change"
    );
    ensure!(
        !calibration
            .probability_calibration_change_ids
            .iter()
            .any(|id| id == change_id),
        "live change must not be a probability calibration change"
    );
    let labels: i64 = connection.query_row(
        "SELECT count(*) FROM labels WHERE change_id=?1",
        [change_id],
        |row| row.get(0),
    )?;
    ensure!(labels == 0, "decide requires an unlabelled live change");
    let existing: i64 = connection.query_row(
        "SELECT count(*) FROM decisions WHERE change_id=?1",
        [change_id],
        |row| row.get(0),
    )?;
    ensure!(existing == 0, "live change already has decisions");
    let executed = if calibration.format_version == 2 {
        ensure!(
            calibration.executed_changes.len() == calibration.changes.len()
                && calibration
                    .executed_changes
                    .iter()
                    .zip(&calibration.changes)
                    .all(|(executed, old)| executed.change_id == old.change_id
                        && executed.pairs.len() == old.pairs.len()
                        && executed
                            .pairs
                            .iter()
                            .zip(&old.pairs)
                            .all(|(a, b)| a.scenario_id == b.scenario_id
                                && a.probability == b.probability
                                && a.regressed == b.regressed)),
            "executed risk pool differs from verified labels"
        );
        Some(calibrate_executed(
            &calibration.executed_changes,
            alpha,
            guarantee,
        )?)
    } else {
        refresh(
            (ledger_path, compiled, calibration_path),
            &source,
            &mut calibration,
            &frozen,
            &mut connection,
            change_id,
            alpha,
        )?;
        None
    };
    let policy = if let Some(executed) = &executed {
        Policy {
            alpha,
            lambda: executed.lambda,
            calibration_changes: executed.calibration_changes,
        }
    } else {
        calibrate(&calibration.changes, alpha)?
    };
    let pairs = if executed.is_some() {
        compile::compile_live_variant(ledger_path, change_id, compiled, false)?
    } else {
        compile::compile_live(ledger_path, change_id, compiled)?
    };
    let scores = if executed.is_some() {
        let cache = calibration_path.with_file_name(format!("{change_id}-noev-plan.json"));
        let binding = json!({"compiled_sha256": format!("{:x}",Sha256::digest(fs::read(compiled)?)), "calibration_sha256": format!("{:x}",Sha256::digest(&source)), "pairs": pairs});
        let scores: BTreeMap<String, Value> = if cache.exists() {
            let saved: Value = serde_json::from_slice(&fs::read(&cache)?)?;
            ensure!(
                saved["binding"] == binding
                    && saved["scores_sha256"]
                        == format!(
                            "{:x}",
                            Sha256::digest(serde_json::to_vec(&saved["scores"])?)
                        ),
                "no-evidence cache binding changed"
            );
            serde_json::from_value(saved["scores"].clone())?
        } else {
            let scores = pairs
                .iter()
                .map(|pair| Ok((pair.scenario_id.clone(), score_t2(pair, &calibration)?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            let mut file = File::create_new(&cache).context("create no-evidence plan cache")?;
            serde_json::to_writer(
                &mut file,
                &json!({"binding":binding,"scores_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&scores)?)),"scores":scores}),
            )?;
            file.sync_all()?;
            scores
        };
        ensure!(
            scores.len() == pairs.len()
                && pairs
                    .iter()
                    .all(|pair| scores.contains_key(&pair.scenario_id)),
            "no-evidence cache scenario mismatch"
        );
        Some(scores)
    } else {
        None
    };
    let evidence = compile::compile_live_variant(ledger_path, change_id, compiled, true)?;
    if let Some(scores) = &scores {
        let mut cheap_required = Vec::new();
        for pair in &pairs {
            let values: BTreeMap<String, BTreeMap<String, f64>> =
                serde_json::from_value(scores[&pair.scenario_id].clone())?;
            ensure!(
                values.len() == pair.questions.len(),
                "invalid cached question count"
            );
            for question in &pair.questions {
                let kind = question["id"]
                    .as_str()
                    .and_then(|id| id.split('/').nth(2))
                    .context("invalid compiled question")?;
                let map = values.get(kind).context("invalid cached semantic map")?;
                let options = question["options"]
                    .as_object()
                    .context("invalid options")?
                    .values()
                    .map(|v| v.as_str().unwrap_or(""))
                    .collect::<BTreeSet<_>>();
                ensure!(
                    map.keys().map(String::as_str).collect::<BTreeSet<_>>() == options
                        && map
                            .values()
                            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                        && (map.values().sum::<f64>() - 1.0).abs() <= 1.1e-7,
                    "invalid cached semantic distribution"
                );
            }
            if (LAMBDA_LOW..LAMBDA_HIGH).contains(
                &scores[&pair.scenario_id]["regressed"]["yes"]
                    .as_f64()
                    .context("invalid cached regression probability")?,
            ) && !evidence
                .iter()
                .any(|item| item.scenario_id == pair.scenario_id)
            {
                cheap_required.push(pair.scenario_id.clone());
            }
        }
        if !cheap_required.is_empty() {
            return Ok(DecideReport {
                change_id: change_id.into(),
                policy,
                decisions: Vec::new(),
                cheap_required,
            });
        }
    }
    let mut decisions = Vec::new();
    for pair in &pairs {
        let (tier, model_version, mut probabilities, noev_probability) =
            if let Some(scores) = &scores {
                let noev = scores[&pair.scenario_id]["regressed"]["yes"]
                    .as_f64()
                    .context("invalid cached regression probability")?;
                let uncertain = (LAMBDA_LOW..LAMBDA_HIGH).contains(&noev);
                let mut probabilities = if uncertain {
                    score_t2(
                        evidence
                            .iter()
                            .find(|item| item.scenario_id == pair.scenario_id)
                            .context("missing planned cheap evidence")?,
                        &calibration,
                    )?
                } else {
                    scores[&pair.scenario_id].clone()
                };
                probabilities["noev_probability"] = json!(noev);
                probabilities["selection_basis"] = json!(if executed.as_ref().unwrap().run_all {
                    "run_all_uncertifiable"
                } else if noev >= LAMBDA_HIGH {
                    "confident_high"
                } else if noev < LAMBDA_LOW {
                    "confident_low"
                } else {
                    "ev_threshold"
                });
                (
                    if uncertain { "T2" } else { "T2-noev" },
                    calibration.model_version.as_str(),
                    probabilities,
                    Some(noev),
                )
            } else {
                let t1 = cascade::predict_t1(&calibration.t1, &pair.features)?;
                if (0.1..=0.9).contains(&t1) {
                    (
                        "T2",
                        calibration.model_version.as_str(),
                        score_t2(pair, &calibration)?,
                        None,
                    )
                } else {
                    (
                        "T1",
                        calibration.t1["model_version"]
                            .as_str()
                            .context("missing T1 version")?,
                        json!({"regressed":{"yes":t1,"no":1.0-t1}}),
                        None,
                    )
                }
            };
        let probability = probabilities["regressed"]["yes"]
            .as_f64()
            .context("missing semantic regression probability")?;
        probabilities["policy"] = if let Some(executed) = &executed {
            serde_json::to_value(executed)?
        } else {
            serde_json::to_value(&policy)?
        };
        probabilities["calibration"] = json!({"model_version":calibration.model_version,"t1_model_version":calibration.t1["model_version"],"risk_change_ids":calibration.changes.iter().map(|change| &change.change_id).collect::<Vec<_>>()});
        let mut random_bytes = [0_u8; 8];
        random_bytes[..6].copy_from_slice(&Uuid::new_v4().as_bytes()[..6]);
        let draw = u64::from_le_bytes(random_bytes) as f64 / (1_u64 << 48) as f64;
        let (action, propensity, reason) =
            if let (Some(executed), Some(noev)) = (&executed, noev_probability) {
                if selected(
                    &ExecutedPair {
                        scenario_id: pair.scenario_id.clone(),
                        probability,
                        noev_probability: noev,
                        regressed: false,
                    },
                    executed,
                ) {
                    ("expensive", 1.0, "policy_above_lambda")
                } else if draw < 0.05 {
                    ("expensive", 0.05, "exploration")
                } else if noev < LAMBDA_LOW {
                    ("skip", 0.05, "cheap_skipped_confident_low")
                } else {
                    ("skip", 0.05, "policy_below_lambda")
                }
            } else {
                action(probability, policy.lambda, draw)
            };
        decisions.push(Decision {
            decision_id: Uuid::new_v4().to_string(),
            change_id: change_id.into(),
            scenario_id: pair.scenario_id.clone(),
            tier_used: tier.into(),
            model_version: model_version.into(),
            probabilities,
            lambda: policy.lambda,
            action: action.into(),
            propensity,
            reason: reason.into(),
        });
    }
    decisions.sort_by(|a, b| {
        b.probabilities["regressed"]["yes"]
            .as_f64()
            .unwrap()
            .total_cmp(&a.probabilities["regressed"]["yes"].as_f64().unwrap())
            .then_with(|| a.scenario_id.cmp(&b.scenario_id))
    });
    escalate(&mut decisions, policy.lambda, |decision| {
        let pair = evidence
            .iter()
            .find(|pair| pair.scenario_id == decision.scenario_id)
            .context("missing escalation pair")?;
        let evidence = explain::cheap_evidence(ledger_path, change_id, &decision.scenario_id)?;
        let prompt = compile::escalation_prompt(
            &serde_json::to_value(pair)?,
            &decision.probabilities,
            policy.lambda,
            &evidence,
        )?;
        explain::bridge(
            &json!({"operation":"escalate","prompt":prompt,"frames":evidence["frames"]}),
        )
    })?;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs()
        .to_string();
    let transaction = connection.transaction()?;
    let existing: i64 = transaction.query_row(
        "SELECT count(*) FROM decisions WHERE change_id=?1",
        [change_id],
        |row| row.get(0),
    )?;
    ensure!(
        existing == 0,
        "live change acquired decisions while scoring"
    );
    for decision in &decisions {
        ledger::insert_decision(
            &transaction,
            &ledger::Decision {
                decision_id: &decision.decision_id,
                change_id: &decision.change_id,
                scenario_id: &decision.scenario_id,
                tier_used: &decision.tier_used,
                model_version: &decision.model_version,
                probs_json: &serde_json::to_string(&decision.probabilities)?,
                lambda: decision.lambda,
                action: &decision.action,
                propensity: decision.propensity,
                reason: &decision.reason,
                created_at: &created,
            },
        )?;
    }
    transaction.commit().context("commit live decisions")?;
    Ok(DecideReport {
        change_id: change_id.into(),
        policy,
        decisions,
        cheap_required: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::{action, add_review, independently_complete, publish_refresh, RiskCalibration};
    use serde_json::json;
    use std::{collections::BTreeMap, fs};

    #[test]
    fn escalation_adds_only_two_highest_band_pairs_and_fails_loudly() {
        let mut decisions = (0..5)
            .map(|n| super::Decision {
                decision_id: n.to_string(),
                change_id: "c999".into(),
                scenario_id: format!("pusht-s{:02}", n + 1),
                tier_used: "T2".into(),
                model_version: "fixed".into(),
                probabilities: json!({"regressed":{"yes":0.79-n as f64*0.01}}),
                lambda: 0.8,
                action: "skip".into(),
                propensity: 0.05,
                reason: "policy_below_lambda".into(),
            })
            .collect::<Vec<_>>();
        let mut seen = Vec::new();
        super::escalate(&mut decisions,0.8,|decision| { seen.push(decision.scenario_id.clone()); Ok(json!({"model":"muse-spark-1.3-contributor","recommend":"run_expensive","rationale":"recorded evidence","cost_usd":null})) }).unwrap();
        assert_eq!(seen, vec!["pusht-s01", "pusht-s02"]);
        assert_eq!(
            decisions.iter().filter(|d| d.reason == "t3_added").count(),
            2
        );
        assert_eq!(
            decisions.iter().filter(|d| d.reason == "t3_capped").count(),
            3
        );
        decisions[0].action = "skip".into();
        assert!(super::escalate(&mut decisions, 0.8, |_| anyhow::bail!("provider error")).is_err());
    }

    #[test]
    fn independent_collection_requires_all_environment_scenarios() {
        let path =
            std::env::temp_dir().join(format!("brier-complete-{}.sqlite", uuid::Uuid::new_v4()));
        let connection = crate::ledger::init(&path).unwrap();
        connection.execute_batch("INSERT INTO changes VALUES('c200','robot','{}','now');
            INSERT INTO scenarios VALUES('robot-s01','robot','{}',0.8),('robot-s02','robot','{}',0.8);
            INSERT INTO labels VALUES('c200','robot-s01',1,'stall/timeout',1,'hidden');
            INSERT INTO runs VALUES('cheap1','c200','robot-s01','cheap',3,2,0.4,'raw',1.0),('full1','c200','robot-s01','expensive',20,2,0.4,'raw',1.0);").unwrap();
        assert!(!independently_complete(&connection, "c200").unwrap());
        connection
            .execute_batch(
                "INSERT INTO labels VALUES('c200','robot-s02',1,'stall/timeout',1,'hidden');
            INSERT INTO runs VALUES('cheap2','c200','robot-s02','cheap',3,2,0.4,'raw',1.0);",
            )
            .unwrap();
        assert!(!independently_complete(&connection, "c200").unwrap());
        connection.execute_batch("INSERT INTO runs VALUES('full2','c200','robot-s02','expensive',20,2,0.4,'raw',1.0);").unwrap();
        assert!(independently_complete(&connection, "c200").unwrap());
        drop(connection);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refreshed_artifact_detects_source_change_and_cleans_temporary_file() {
        let directory =
            std::env::temp_dir().join(format!("brier-refresh-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("risk.json");
        let updated = RiskCalibration {
            format_version: 1,
            model_version: "fixed".into(),
            t1: json!({}),
            temperatures: BTreeMap::new(),
            probability_calibration_change_ids: Vec::new(),
            changes: Vec::new(),
            executed_changes: Vec::new(),
        };
        fs::write(&path, b"newer caller content").unwrap();
        assert!(publish_refresh(&path, b"original content", &updated)
            .unwrap_err()
            .to_string()
            .contains("changed during refresh"));
        assert_eq!(fs::read(&path).unwrap(), b"newer caller content");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        publish_refresh(&path, b"newer caller content", &updated).unwrap();
        let actual: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(actual["model_version"], "fixed");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn threshold_equality_and_exploration_propensities_are_exact() {
        assert_eq!(
            action(0.4, 0.4, 0.9),
            ("expensive", 1.0, "policy_above_lambda")
        );
        assert_eq!(action(0.3, 0.4, 0.049), ("expensive", 0.05, "exploration"));
        assert_eq!(
            action(0.3, 0.4, 0.05),
            ("skip", 0.05, "policy_below_lambda")
        );
    }

    #[test]
    fn t3_can_only_add_evaluations_and_records_no_change_propensities() {
        for (action, propensity, reason) in [
            ("expensive", 1.0, "policy_above_lambda"),
            ("expensive", 0.05, "exploration"),
            ("skip", 0.05, "policy_below_lambda"),
        ] {
            let mut review = json!({"model":"muse-spark-1.3-contributor",
                "recommend":"no_change","rationale":"observed evidence","cost_usd":null});
            assert_eq!(
                add_review(action, propensity, reason, &review).unwrap(),
                (action.into(), propensity, "t3_no_change".into())
            );
            for recommendation in ["run_expensive", "run_more_cheap"] {
                review["recommend"] = json!(recommendation);
                assert_eq!(
                    add_review(action, propensity, reason, &review).unwrap(),
                    ("expensive".into(), 1.0, "t3_added".into())
                );
            }
            review["recommend"] = json!("skip");
            assert!(add_review(action, propensity, reason, &review).is_err());
        }
    }
}
