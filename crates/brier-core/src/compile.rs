use anyhow::{ensure, Context, Result};
use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs::File, io::Write, path::Path};

// All neighbor labels and normalization scales come from other training changes.
const STATES: &str = "WITH
 changes AS (SELECT c.*, EXISTS(SELECT 1 FROM json_each(c.card_json,'$.families') f WHERE json_extract(f.value,'$.family') IN ('occlusion','chunk_len')) AS ood
   FROM main.changes c WHERE EXISTS(SELECT 1 FROM labels l WHERE l.change_id=c.change_id)),
 regular AS (SELECT change_id, row_number() OVER(PARTITION BY env ORDER BY change_id)-1 AS n,count(*) OVER(PARTITION BY env) AS total FROM changes WHERE NOT ood),
 splits AS (SELECT c.*, CASE WHEN ood THEN 'ood' WHEN n < total*3/5 THEN 'train'
   WHEN n < total*4/5 THEN 'calibration' ELSE 'test' END AS split FROM changes c LEFT JOIN regular USING(change_id)),
 features AS (SELECT c.change_id,json_extract(f.value,'$.family') AS key,1.0 AS value FROM changes c,json_each(c.card_json,'$.families') f UNION ALL
   SELECT c.change_id,json_extract(f.value,'$.family')||'.'||p.key,abs(p.value-CASE WHEN p.key IN ('scale','gain','contrast') THEN 1.0 ELSE 0.0 END)
   FROM changes c,json_each(c.card_json,'$.families') f,json_each(f.value,'$.params') p),
 eligible AS (SELECT l.change_id,l.scenario_id,h.change_id AS neighbor,h.regressed FROM labels l JOIN labels h ON h.scenario_id=l.scenario_id AND h.change_id<>l.change_id
   JOIN splits s ON s.change_id=h.change_id AND s.split='train'),
 scales AS (SELECT e.change_id,e.scenario_id,f.key,max(1.0,max(f.value)) AS scale FROM eligible e JOIN features f ON f.change_id=e.neighbor GROUP BY e.change_id,e.scenario_id,f.key),
 deltas AS (SELECT e.*, (coalesce(t.value,0)-coalesce(h.value,0))/coalesce(s.scale,1) AS d
   FROM eligible e CROSS JOIN (SELECT DISTINCT key FROM features) k
   LEFT JOIN features t ON t.change_id=e.change_id AND t.key=k.key
   LEFT JOIN features h ON h.change_id=e.neighbor AND h.key=k.key
   LEFT JOIN scales s ON s.change_id=e.change_id AND s.scenario_id=e.scenario_id AND s.key=k.key),
 distances AS (SELECT change_id,scenario_id,neighbor,regressed,sum(d*d) AS distance FROM deltas GROUP BY change_id,scenario_id,neighbor),
 ranked AS (SELECT *,row_number() OVER(PARTITION BY change_id,scenario_id ORDER BY distance,neighbor) AS rank FROM distances),
 history AS (SELECT change_id,scenario_id,avg(regressed) AS fragility,count(*) AS neighbors,
   sum(CASE WHEN rank<=5 THEN regressed ELSE 0 END) AS similar FROM ranked GROUP BY change_id,scenario_id)
 SELECT json_object('change',c.change_id,'scenario',l.scenario_id,'env',c.env,'card',json(c.card_json),
   'params',json(s.params_json),'baseline',s.baseline_success,'successes',r.successes,'reward',r.mean_reward,
   'regressed',l.regressed,'mode',l.failure_mode,'bucket',l.success_bucket,'split',c.split,
   'fragility',h.fragility,'neighbors',h.neighbors,'similar',h.similar)
 FROM labels l JOIN splits c USING(change_id) JOIN scenarios s USING(scenario_id)
 JOIN runs r ON r.change_id=l.change_id AND r.scenario_id=l.scenario_id AND r.tier='cheap' AND r.episodes=3
 JOIN history h ON h.change_id=l.change_id AND h.scenario_id=l.scenario_id ORDER BY c.change_id,l.scenario_id";

#[derive(Deserialize)]
struct State {
    change: String,
    scenario: String,
    env: String,
    card: Value,
    params: Value,
    baseline: f64,
    successes: i64,
    reward: f64,
    regressed: i64,
    mode: String,
    bucket: i64,
    split: String,
    fragility: f64,
    neighbors: usize,
    similar: usize,
}

/// Emits deterministic states with change-disjoint splits and exclusively training history.
pub fn compile(ledger: &Path, out: &Path) -> Result<()> {
    let connection = Connection::open_with_flags(ledger, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut query = connection.prepare(STATES)?;
    let rows = query.query_map([], |row| row.get::<_, String>(0))?;
    let states = rows
        .map(|row| Ok(serde_json::from_str::<State>(&row?)?))
        .collect::<Result<Vec<_>>>()?;
    let labels: i64 = connection.query_row("SELECT count(*) FROM labels", [], |r| r.get(0))?;
    ensure!(
        !states.is_empty() && states.len() as i64 == labels,
        "each label needs one cheap run and at least five other training changes"
    );
    let mut output = Vec::new();
    for state in states {
        ensure!(
            state.neighbors >= 5
                && (0.0..=1.0).contains(&state.baseline)
                && (0..=3).contains(&state.successes)
                && state.reward.is_finite(),
            "invalid evidence or fewer than five train neighbors"
        );
        let mut change = format!("CHANGE: id={}", state.change);
        let families = state.card["families"]
            .as_array()
            .context("missing families")?;
        for family in families {
            let name = family["family"].as_str().context("missing family")?;
            change.push_str(&format!(" family={name}"));
            let params = family["params"].as_object().context("missing params")?;
            for (key, value) in params {
                ensure!(value.as_f64().is_some(), "nonnumeric change parameter");
                change.push_str(&format!(" {key}={value}"));
            }
        }
        let description = state.card["description"]
            .as_str()
            .context("missing description")?;
        ensure!(
            !description.trim().is_empty() && !description.contains(['\n', '\r']),
            "description must be one nonempty line"
        );
        change.push_str(&format!(" desc={}", serde_json::to_string(description)?));
        let env = match state.env.as_str() {
            "gym_pusht/PushT-v0" => "pusht",
            "gym_aloha/AlohaTransferCube-v0" => "aloha_transfer_cube",
            _ => anyhow::bail!("unsupported environment"),
        };
        let mut scenario = format!("SCENARIO: env={env}");
        let params = state
            .params
            .as_object()
            .context("missing scenario params")?;
        for key in ["pose_bin", "lighting", "camera_jitter", "physics_jitter"] {
            if let Some(value) = params.get(key) {
                let value = match value {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => if *b { "on" } else { "off" }.into(),
                    _ => value.to_string(),
                };
                scenario.push_str(&format!(" {key}={value}"));
            }
        }
        for evidence in ["cheap", "no_cheap"] {
            for (question, text, choices) in [
                ("regressed", "Will the expensive evaluation regress by more than 15 points?", "yes|no"),
                ("failure_mode", "What will be the expensive evaluation failure mode?", "none|stall/timeout|overshoot/misalignment|grasp or drop failure|collision|oscillation"),
                ("success_bucket", "Which success-rate fifth (1 to 5) will the expensive evaluation reach?", "1|2|3|4|5")
            ] {
                let id = format!("{}/{}/{question}/{evidence}", state.change, state.scenario);
                let mut answers: Vec<_> = choices.split('|').collect();
                if state.split == "train" {
                    let mut hash = id.bytes().fold(14695981039346656037u64, |h, b| {
                        (h ^ u64::from(b)).wrapping_mul(1099511628211)
                    });
                    for i in (1..answers.len()).rev() {
                        hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1);
                        answers.swap(i, (hash % (i as u64 + 1)) as usize);
                    }
                }
                let options: BTreeMap<_, _> = answers.iter().enumerate()
                    .map(|(i, a)| (((b'A' + i as u8) as char).to_string(), *a)).collect();
                let target = match question {
                    "regressed" => if state.regressed == 1 { "yes" } else { "no" }.into(),
                    "failure_mode" => state.mode.clone(),
                    _ => state.bucket.to_string(),
                };
                let label = options.iter().find(|(_, a)| **a == target)
                    .context("invalid ledger label")?.0;
                let cheap = if evidence == "cheap" {
                    format!("CHEAP_TIER: success={}/3 mean_reward={:.2} (3 eps, 50% horizon)\n", state.successes, state.reward)
                } else {
                    String::new()
                };
                let rendered = options.iter().map(|(letter, answer)| format!("{letter}) {answer}"))
                    .collect::<Vec<_>>().join("  ");
                let prompt = format!("<system> You are Brier, a calibrated decision model for robot policy CI. Answer with one option letter.\n<state>\n{change}\n{scenario}\nBASELINE: success={:.2} (50 eps)\n{cheap}HISTORY: scenario_fragility={:.2}; similar_changes(k=5): {} regressed, {} passed\n</state>\n<question> {text}\n<options> {rendered}\n<answer>", state.baseline, state.fragility, state.similar, 5-state.similar);
                serde_json::to_writer(&mut output, &json!({"id": id, "split": state.split,
                    "question": question, "options": options, "prompt": prompt, "label": label}))?;
                output.push(b'\n');
            }
        }
    }
    let mut file = File::create_new(out).context("create new compiled JSONL")?;
    if let Err(error) = file.write_all(&output) {
        drop(file);
        std::fs::remove_file(out)?;
        return Err(error.into());
    }
    Ok(())
}
