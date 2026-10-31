use anyhow::{ensure, Context, Result};
use reqwest::{blocking::Client, redirect, StatusCode, Url};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::Serialize;
use serde_json::{json, Value};
use std::{collections::BTreeSet, env, fs, path::Path, time::Duration};
use uuid::{Uuid, Variant};

#[derive(Serialize)]
pub struct ReportOutput {
    pub markdown: String,
    pub bundle: Option<Value>,
    pub verdict: String,
}

struct Logged {
    id: String,
    scenario: String,
    model: String,
    scores: String,
    lambda: f64,
    action: String,
    created: String,
}

struct Measured {
    id: String,
    episodes: i64,
    successes: i64,
    seconds: Option<f64>,
}

fn uuid(value: &str) -> Result<String> {
    let parsed = Uuid::parse_str(value).context("ledger IDs must be UUIDv4")?;
    ensure!(
        parsed.get_version_num() == 4 && parsed.get_variant() == Variant::RFC4122,
        "ledger IDs must be UUIDv4"
    );
    Ok(parsed.to_string())
}

fn timestamp(connection: &Connection, value: &str) -> Result<String> {
    let epoch = value.parse::<i64>().is_ok();
    ensure!(
        epoch || value.contains('T'),
        "ledger timestamps must be ISO timestamps or epoch seconds"
    );
    let query = if epoch {
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ',?1,'unixepoch')"
    } else {
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ',?1)"
    };
    connection
        .query_row(query, [value], |row| row.get::<_, Option<String>>(0))?
        .context("invalid ledger timestamp")
}

fn measured(connection: &Connection, change: &str, scenario: &str) -> Result<Option<Measured>> {
    let mut query = connection.prepare(
        "SELECT run_id,episodes,successes,wall_seconds FROM runs
        WHERE change_id=?1 AND scenario_id=?2 AND tier='expensive'",
    )?;
    let rows = query
        .query_map([change, scenario], |row| {
            Ok(Measured {
                id: row.get(0)?,
                episodes: row.get(1)?,
                successes: row.get(2)?,
                seconds: row.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    ensure!(rows.len() <= 1, "ambiguous expensive runs for {scenario}");
    if let Some(run) = rows.first() {
        ensure!(
            run.episodes > 0
                && (0..=run.episodes).contains(&run.successes)
                && run
                    .seconds
                    .is_none_or(|seconds| seconds.is_finite() && seconds >= 0.0),
            "invalid measured expensive run for {scenario}"
        );
    }
    Ok(rows.into_iter().next())
}

fn urls(explanation: &Value, key: &str) -> Result<Vec<String>> {
    let Some(value) = explanation.get(key) else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .with_context(|| format!("explanation.{key} must be an array"))?
        .iter()
        .map(|value| {
            let raw = value
                .as_str()
                .context("public evidence URLs must be strings")?;
            let parsed = Url::parse(raw).context("invalid public evidence URL")?;
            ensure!(
                parsed.scheme() == "https"
                    && parsed.username().is_empty()
                    && parsed.password().is_none(),
                "public evidence URLs must use HTTPS without credentials"
            );
            Ok(parsed.to_string())
        })
        .collect()
}

fn evidence_urls(
    explanation: &Value,
    key: &str,
    local_key: &str,
    root: Option<&str>,
) -> Result<Vec<String>> {
    let public = urls(explanation, key)?;
    let local = explanation
        .get(local_key)
        .map(|value| {
            value
                .as_array()
                .with_context(|| format!("explanation.{local_key} must be an array"))
        })
        .transpose()?;
    let mut refs = Vec::new();
    for value in local.into_iter().flatten() {
        let raw = value
            .as_str()
            .context("local evidence references must be strings")?;
        let parts: Vec<_> = raw.split('/').collect();
        ensure!(parts.len() >= 3 && parts[0] == "public" && matches!(parts[1],"keyframes"|"overlays")
            && parts.iter().all(|part| !part.is_empty() && *part != "." && *part != "..")
            && !raw.contains(['\\','?','#']) && !raw.chars().any(char::is_control), "local evidence must be a relative path under public/keyframes or public/overlays without traversal, query or fragment");
        refs.push(raw);
    }
    if !public.is_empty() {
        return Ok(public);
    }
    let Some(root) = root.and_then(|value| value.strip_prefix("gs://")) else {
        return Ok(Vec::new());
    };
    let (bucket, prefix) = root.split_once('/').unwrap_or((root, ""));
    ensure!(
        !bucket.is_empty()
            && bucket.bytes().all(|byte| byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || b"._-".contains(&byte))
            && bucket
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            && bucket
                .bytes()
                .last()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            && !prefix.contains(['\\', '?', '#'])
            && !prefix.chars().any(char::is_control),
        "invalid public GCS artifact root"
    );
    ensure!(
        !prefix.starts_with('/'),
        "GCS artifact prefix must not begin with an empty segment"
    );
    let prefix = prefix.strip_suffix('/').unwrap_or(prefix);
    let prefix: Vec<_> = if prefix.is_empty() {
        Vec::new()
    } else {
        prefix.split('/').collect()
    };
    ensure!(
        prefix
            .iter()
            .all(|part| !part.is_empty() && *part != "." && *part != ".."),
        "GCS artifact prefix must not contain empty or traversal segments"
    );
    refs.into_iter()
        .map(|raw| {
            let mut url = Url::parse("https://storage.googleapis.com")?;
            url.path_segments_mut()
                .map_err(|_| anyhow::anyhow!("invalid public GCS origin"))?
                .clear()
                .push(bucket)
                .extend(&prefix)
                .extend(raw.split('/'));
            Ok(url.to_string())
        })
        .collect()
}

fn md(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "\\|")
        .replace('`', "\\`")
        .replace(['\n', '\r'], " ")
}

fn endpoint(raw: &str) -> Result<Url> {
    let url = Url::parse(raw).context("invalid ingest URL")?;
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    ensure!(
        (url.scheme() == "https" || (url.scheme() == "http" && local))
            && url.username().is_empty()
            && url.password().is_none()
            && url.path() == "/api/ingest"
            && url.query().is_none()
            && url.fragment().is_none(),
        "push URL must be HTTPS POST /api/ingest without credentials, query or fragment"
    );
    Ok(url)
}

/// Write an authoritative check summary and optionally push one complete evaluation.
/// Skipped labels remain private; unknown counterfactual hours are reported as null.
pub fn report(
    ledger: &Path,
    change: &str,
    out: &Path,
    push: Option<&str>,
    suite_report: Option<&Path>,
) -> Result<ReportOutput> {
    ensure!(
        change.len() >= 4
            && change.starts_with('c')
            && change[1..].bytes().all(|b| b.is_ascii_digit())
            && change[1..].bytes().any(|b| b != b'0'),
        "change must start at c001 with at least three digits"
    );
    if out.exists() {
        let target = fs::canonicalize(out)?;
        ensure!(
            target != fs::canonicalize(ledger)?
                && suite_report
                    .is_none_or(|path| fs::canonicalize(path).is_ok_and(|path| path != target)),
            "report output must not replace an input artifact"
        );
    }
    let connection = Connection::open_with_flags(ledger, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    connection.execute_batch("BEGIN")?;
    let (environment, card, created): (String, String, String) = connection
        .query_row(
            "SELECT env,card_json,created_at FROM changes WHERE change_id=?1",
            [change],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .context("report change is missing from ledger")?;
    let env = match environment.as_str() {
        "gym_pusht/PushT-v0" | "pusht" => "pusht",
        "gym_aloha/AlohaTransferCube-v0" | "aloha_transfer_cube" => "aloha_transfer_cube",
        _ => anyhow::bail!("unsupported robot environment"),
    };
    let card: Value = serde_json::from_str(&card).context("invalid change card JSON")?;
    let title = card["description"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .context("change card requires a description")?;
    ensure!(card.is_object(), "change card must be an object");
    let mut query = connection.prepare(
        "SELECT decision_id,scenario_id,model_version,probs_json,lambda,action,created_at
        FROM decisions WHERE change_id=?1 ORDER BY scenario_id",
    )?;
    let decisions = query
        .query_map([change], |row| {
            Ok(Logged {
                id: row.get(0)?,
                scenario: row.get(1)?,
                model: row.get(2)?,
                scores: row.get(3)?,
                lambda: row.get(4)?,
                action: row.get(5)?,
                created: row.get(6)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut query = connection.prepare("SELECT scenario_id FROM scenarios WHERE env=?1")?;
    let expected = query
        .query_map([&environment], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<BTreeSet<_>>>()?;
    ensure!(
        !decisions.is_empty()
            && decisions.len() == expected.len()
            && decisions
                .iter()
                .map(|d| d.scenario.clone())
                .collect::<BTreeSet<_>>()
                == expected,
        "report requires exactly one decision for every scenario"
    );
    let first: Value = serde_json::from_str(&decisions[0].scores)?;
    let policy: crate::policy::Policy = serde_json::from_value(first["policy"].clone())
        .context("decisions require persisted policy provenance")?;
    let risk_ids = first["calibration"]["risk_change_ids"]
        .as_array()
        .context("decisions require persisted calibration provenance")?;
    let unique: BTreeSet<_> = risk_ids
        .iter()
        .map(|value| value.as_str().unwrap_or(""))
        .collect();
    ensure!(
        policy.alpha.is_finite()
            && (0.0..=1.0).contains(&policy.alpha)
            && policy.lambda.is_finite()
            && (0.0..=1.0).contains(&policy.lambda)
            && policy.calibration_changes > 0
            && risk_ids.len() == policy.calibration_changes
            && unique.len() == risk_ids.len()
            && !unique.contains("")
            && !unique.contains(change)
            && 1.0 / (policy.calibration_changes as f64 + 1.0) <= policy.alpha,
        "invalid persisted risk policy provenance"
    );
    for field in ["model_version", "t1_model_version"] {
        ensure!(
            first["calibration"][field]
                .as_str()
                .is_some_and(|value| !value.trim().is_empty()),
            "calibration provenance requires pinned model versions"
        );
    }
    let evaluation_id = uuid(&decisions[0].id)?;
    let created_at = timestamp(&connection, &decisions[0].created)?;
    let mut pairs = Vec::new();
    let mut failures = Vec::new();
    let mut failure_ids = BTreeSet::new();
    let mut models = BTreeSet::new();
    let (mut selected, mut completed, mut confirmed, mut pending) = (0, 0, 0, 0);
    let (mut total_seconds, mut selected_seconds, mut timed) = (0.0, 0.0, true);
    let mut table = String::new();
    let media_root = match env::var("BRIER_DATA") {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    for decision in &decisions {
        let scores: Value = serde_json::from_str(&decision.scores)?;
        ensure!(
            scores["policy"] == first["policy"]
                && scores["calibration"] == first["calibration"]
                && decision.lambda == policy.lambda
                && timestamp(&connection, &decision.created)? == created_at,
            "decision batch has inconsistent policy, calibration or timestamps"
        );
        ensure!(
            !decision.model.trim().is_empty(),
            "decision requires a model version"
        );
        models.insert(decision.model.as_str());
        let probability = scores["regressed"]["yes"]
            .as_f64()
            .context("missing semantic regression probability")?;
        let no = scores["regressed"]["no"]
            .as_f64()
            .context("missing semantic no-regression probability")?;
        ensure!(
            (0.0..=1.0).contains(&probability)
                && (0.0..=1.0).contains(&no)
                && (probability + no - 1.0).abs() <= 1.1e-7,
            "invalid regression probabilities"
        );
        ensure!(
            matches!(decision.action.as_str(), "expensive" | "skip"),
            "unknown decision action"
        );
        ensure!(
            decision.action != "skip" || probability < policy.lambda,
            "skipped decision violates its calibrated threshold"
        );
        let prefix = format!("{env}-s");
        ensure!(
            decision
                .scenario
                .strip_prefix(&prefix)
                .is_some_and(|suffix| suffix.len() == 2
                    && suffix.bytes().all(|b| b.is_ascii_digit())
                    && suffix != "00"),
            "scenario ID disagrees with environment"
        );
        let baseline: f64 = connection.query_row(
            "SELECT baseline_success FROM scenarios WHERE scenario_id=?1",
            [&decision.scenario],
            |row| row.get(0),
        )?;
        ensure!((0.0..=1.0).contains(&baseline), "invalid baseline success");
        let run = measured(&connection, change, &decision.scenario)?;
        let chosen = decision.action == "expensive";
        let seconds = run
            .as_ref()
            .filter(|r| r.episodes == 20)
            .and_then(|r| r.seconds);
        if let Some(seconds) = seconds {
            total_seconds += seconds;
            if chosen {
                selected_seconds += seconds;
            }
        } else {
            timed = false;
        }
        let mut result: Option<&str> = None;
        let mut after = None;
        if chosen {
            selected += 1;
            let label: Option<i64> = connection
                .query_row(
                    "SELECT regressed FROM labels WHERE change_id=?1 AND scenario_id=?2",
                    [change, &decision.scenario],
                    |row| row.get(0),
                )
                .optional()?;
            if let (Some(run), Some(label)) = (run.as_ref().filter(|run| run.episodes == 20), label)
            {
                ensure!(
                    label == 0 || label == 1,
                    "invalid authoritative regression label"
                );
                completed += 1;
                confirmed += usize::from(label == 1);
                result = Some(if label == 1 { "regressed" } else { "pass" });
                after = Some(run.successes as f64 / 20.0);
                let explanation: Option<String> = connection
                    .query_row(
                        "SELECT json FROM explanations WHERE run_id=?1",
                        [&run.id],
                        |row| row.get(0),
                    )
                    .optional()?;
                if let Some(raw) = explanation {
                    ensure!(
                        run.successes < 20,
                        "failure explanation attached to an entirely successful run"
                    );
                    let explanation: Value = serde_json::from_str(&raw)?;
                    ensure!(
                        explanation.is_object(),
                        "failure explanation must be an object"
                    );
                    let mut entries = Vec::new();
                    if let Some(episodes) = explanation.get("episodes") {
                        let episodes = episodes
                            .as_array()
                            .context("explanation.episodes must be an array")?;
                        ensure!(
                            !episodes.is_empty() && episodes.len() <= (20 - run.successes) as usize,
                            "episode explanations must not exceed the failed episode count"
                        );
                        for episode in episodes {
                            ensure!(
                                episode.is_object()
                                    && episode.get("keyframe_urls").is_some()
                                    && episode.get("overlay_urls").is_some(),
                                "episode explanation requires keyframe_urls and overlay_urls"
                            );
                            let id = episode["id"]
                                .as_str()
                                .context("episode explanation requires a persisted UUIDv4 id")?;
                            entries.push((uuid(id)?, episode));
                        }
                    } else {
                        entries.push((uuid(&run.id)?, &explanation));
                    }
                    for (id, explanation) in entries {
                        ensure!(
                            failure_ids.insert(id.clone()),
                            "duplicate failure explanation id"
                        );
                        failures.push(json!({"id":id,"pairId":uuid(&decision.id)?,
                            "keyframeUrls":evidence_urls(explanation,"keyframe_urls","local_keyframes",media_root.as_deref())?,"overlayUrls":evidence_urls(explanation,"overlay_urls","local_overlays",media_root.as_deref())?,
                            "explanation":explanation}));
                    }
                }
            } else {
                pending += 1;
            }
        }
        let outcome = match (result, after) {
            (Some(result), Some(rate)) => {
                format!("{} {:.2} → {:.2}", result.to_uppercase(), baseline, rate)
            }
            _ if chosen => "PENDING expensive run or label".to_string(),
            _ => "n/a".to_string(),
        };
        table.push_str(&format!(
            "| {} | {:.3} | {} | {} |\n",
            md(&decision.scenario),
            probability,
            decision.action,
            outcome
        ));
        pairs.push(json!({"id":uuid(&decision.id)?,"evaluationId":evaluation_id,"scenarioId":decision.scenario,
            "pRegress":probability,"action":decision.action,"result":result,"successBefore":baseline,"successAfter":after}));
    }
    let verdict = if confirmed > 0 {
        "block"
    } else if pending > 0 {
        "pending"
    } else {
        "pass"
    };
    ensure!(
        total_seconds.is_finite() && selected_seconds.is_finite(),
        "aggregate expensive timings overflow"
    );
    let hours_saved =
        (timed && total_seconds > 0.0).then(|| 100.0 * (1.0 - selected_seconds / total_seconds));
    let mut markdown = format!("## Brier: {} ({confirmed} regressions confirmed)\n\n{}\n\nGuarantee target: expected missed-regression fraction ≤ {:.1}% (λ = {:.3}, calibrated on {} changes), assuming exchangeable changes.\n\nExpensive evaluations: {completed} completed of {} scenarios; {selected} selected, {pending} pending.\nEvaluation-count reduction: {:.1}% vs run-all.\n",
        verdict.to_uppercase(), md(title), policy.alpha * 100.0, policy.lambda, policy.calibration_changes,
        decisions.len(), 100.0 * (1.0 - selected as f64 / decisions.len() as f64));
    if let Some(saved) = hours_saved {
        markdown.push_str(&format!(
            "Measured expensive hours: {:.4} selected / {:.4} run-all ({saved:.1}% saved).\n",
            selected_seconds / 3600.0,
            total_seconds / 3600.0
        ));
    } else {
        markdown.push_str(
            "Actual evaluation-hours savings: pending unmeasured counterfactual timings.\n",
        );
    }
    markdown.push_str(&format!(
        "\n| Scenario | p(regress) | Action | Result |\n| --- | --- | --- | --- |\n{table}"
    ));
    for failure in &failures {
        let scenario = pairs
            .iter()
            .find(|pair| pair["id"] == failure["pairId"])
            .and_then(|pair| pair["scenarioId"].as_str())
            .context("failure pair is missing")?;
        markdown.push_str(&format!("\n### Why {} failed\n", md(scenario)));
        for key in ["failure_mode", "cause_hypothesis", "suggested_check"] {
            if let Some(text) = failure["explanation"][key].as_str() {
                markdown.push_str(&format!("\n**{}**: {}\n", key.replace('_', " "), md(text)));
            }
        }
        for (key, label) in [("keyframeUrls", "Keyframe"), ("overlayUrls", "SAM overlay")] {
            for (index, url) in failure[key]
                .as_array()
                .context("invalid evidence URL array")?
                .iter()
                .enumerate()
            {
                markdown.push_str(&format!(
                    "\n[{label} {}](<{}>)\n",
                    index + 1,
                    url.as_str().context("invalid evidence URL")?
                ));
            }
        }
        if failure["keyframeUrls"]
            .as_array()
            .is_some_and(Vec::is_empty)
            || failure["overlayUrls"].as_array().is_some_and(Vec::is_empty)
        {
            markdown
                .push_str("\nEvidence publication: pending; public frame URLs are unavailable.\n");
        }
    }
    if confirmed > 0 && failures.is_empty() {
        markdown.push_str("\nFailure explanations: pending.\n");
    }
    let mut bundle = None;
    if pending == 0 {
        let mut value = json!({"change":{"id":change,"env":env,"title":title,"card":card,"prUrl":null,
            "createdAt":timestamp(&connection,&created)?},"evaluation":{"id":evaluation_id,"changeId":change,
            "verdict":verdict,"alpha":policy.alpha,"lambda":policy.lambda,"evalsRun":completed,"evalsTotal":decisions.len(),
            "hoursSavedPct":hours_saved,"modelVersion":models.into_iter().collect::<Vec<_>>().join(" + "),"createdAt":created_at},
            "pairs":pairs,"failures":failures});
        if let Some(path) = suite_report {
            let body: Value = serde_json::from_slice(&fs::read(path)?)?;
            ensure!(body.is_object(), "suite report must be a JSON object");
            value["report"] =
                json!({"id":evaluation_id,"suite":"robot","createdAt":created_at,"body":body});
        }
        bundle = Some(value);
    }
    connection.execute_batch("COMMIT")?;
    fs::write(out, &markdown).context("write check summary")?;
    let url = push.map(endpoint).transpose()?;
    if let Some(url) = url {
        let value = bundle
            .as_ref()
            .context("selected expensive results pending; report cannot be pushed")?;
        let token =
            env::var("BRIER_INGEST_TOKEN").context("BRIER_INGEST_TOKEN is required for --push")?;
        ensure!(
            !token.trim().is_empty(),
            "BRIER_INGEST_TOKEN must not be empty"
        );
        let response = Client::builder()
            .redirect(redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .build()?
            .post(url.clone())
            .bearer_auth(token)
            .json(value)
            .send()
            .context("POST /api/ingest failed")?;
        ensure!(
            response.status() == StatusCode::OK,
            "POST /api/ingest returned {}",
            response.status()
        );
        let ack: Value = response.json().context("invalid ingest acknowledgement")?;
        ensure!(
            ack["evaluationId"] == evaluation_id,
            "ingest acknowledged another evaluation"
        );
        let mut link = url;
        link.set_path(&format!("/runs/{evaluation_id}"));
        markdown.push_str(&format!("\n[View evaluation]({link})\n"));
        fs::write(out, &markdown).context("write acknowledged evaluation link")?;
    }
    Ok(ReportOutput {
        markdown,
        bundle,
        verdict: verdict.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_urls_escape_uploaded_object_segments_and_reject_unsafe_local_refs() {
        let explanation = json!({"keyframe_urls":[],"local_keyframes":["public/keyframes/push t/frame-é%20.png"]});
        assert_eq!(evidence_urls(&explanation,"keyframe_urls","local_keyframes",Some("gs://brier-test/ci outputs/測試/")).unwrap(),vec!["https://storage.googleapis.com/brier-test/ci%20outputs/%E6%B8%AC%E8%A9%A6/public/keyframes/push%20t/frame-%C3%A9%2520.png"]);
        assert!(evidence_urls(
            &explanation,
            "keyframe_urls",
            "local_keyframes",
            Some("/local/data")
        )
        .unwrap()
        .is_empty());
        assert!(
            evidence_urls(&explanation, "keyframe_urls", "local_keyframes", None)
                .unwrap()
                .is_empty()
        );
        let explicit = json!({"keyframe_urls":["https://example.com/published.png"],"local_keyframes":["public/keyframes/frame.png"]});
        assert_eq!(
            evidence_urls(
                &explicit,
                "keyframe_urls",
                "local_keyframes",
                Some("gs://brier-test/prefix")
            )
            .unwrap(),
            vec!["https://example.com/published.png"]
        );
        for path in [
            "/public/keyframes/frame.png",
            "public/keyframes/../secret",
            "public/keyframes/./frame.png",
            "public/keyframes//frame.png",
            "public/other/frame.png",
            "public/keyframes/frame.png?token=secret",
            "public/keyframes/frame.png#fragment",
            "public/keyframes/a\\b.png",
            "public/keyframes/line\nframe.png",
        ] {
            let invalid = json!({"keyframe_urls":[],"local_keyframes":[path]});
            assert!(
                evidence_urls(
                    &invalid,
                    "keyframe_urls",
                    "local_keyframes",
                    Some("gs://brier-test/prefix")
                )
                .is_err(),
                "accepted {path}"
            );
        }
        for root in [
            "gs:///prefix",
            "gs://user@bucket/prefix",
            "gs://bad bucket/prefix",
            "gs://bucket/a/../b",
            "gs://bucket/a//b",
            "gs://bucket/prefix//",
            "gs://bucket//",
            "gs://bucket/prefix?token=secret",
            "gs://bucket/prefix#fragment",
        ] {
            assert!(
                evidence_urls(&explanation, "keyframe_urls", "local_keyframes", Some(root))
                    .is_err(),
                "accepted {root}"
            );
        }
        let invalid = json!({"keyframe_urls":["http://example.com/frame.png"]});
        assert!(evidence_urls(
            &invalid,
            "keyframe_urls",
            "local_keyframes",
            Some("gs://brier-test/prefix")
        )
        .is_err());
    }
}
