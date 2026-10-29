use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

const SCHEMA: &str = "
CREATE TABLE changes (change_id TEXT PRIMARY KEY, env TEXT, card_json TEXT, created_at TEXT);
CREATE TABLE scenarios (scenario_id TEXT PRIMARY KEY, env TEXT, params_json TEXT, baseline_success REAL);
CREATE TABLE runs (run_id TEXT PRIMARY KEY, change_id TEXT, scenario_id TEXT, tier TEXT,
    episodes INT, successes INT, mean_reward REAL, uri TEXT, wall_seconds REAL);
CREATE TABLE decisions (decision_id TEXT PRIMARY KEY, change_id TEXT, scenario_id TEXT, tier_used TEXT,
    model_version TEXT, probs_json TEXT, lambda REAL, action TEXT,
    propensity REAL, reason TEXT, created_at TEXT);
CREATE TABLE labels (change_id TEXT, scenario_id TEXT, regressed INT, failure_mode TEXT,
    success_bucket INT, cause_family TEXT, PRIMARY KEY (change_id, scenario_id));
CREATE TABLE explanations (run_id TEXT PRIMARY KEY, model TEXT, json TEXT, cost_usd REAL);
";

pub struct Change<'a> {
    pub change_id: &'a str,
    pub env: &'a str,
    pub card_json: &'a str,
    pub created_at: &'a str,
}

pub struct Scenario<'a> {
    pub scenario_id: &'a str,
    pub env: &'a str,
    pub params_json: &'a str,
    pub baseline_success: f64,
}

pub struct Run<'a> {
    pub run_id: &'a str,
    pub change_id: &'a str,
    pub scenario_id: &'a str,
    pub tier: &'a str,
    pub episodes: i64,
    pub successes: i64,
    pub mean_reward: f64,
    pub uri: &'a str,
    pub wall_seconds: f64,
}

pub struct Decision<'a> {
    pub decision_id: &'a str,
    pub change_id: &'a str,
    pub scenario_id: &'a str,
    pub tier_used: &'a str,
    pub model_version: &'a str,
    pub probs_json: &'a str,
    pub lambda: f64,
    pub action: &'a str,
    pub propensity: f64,
    pub reason: &'a str,
    pub created_at: &'a str,
}

pub struct Label<'a> {
    pub change_id: &'a str,
    pub scenario_id: &'a str,
    pub regressed: bool,
    pub failure_mode: &'a str,
    pub success_bucket: i64,
    pub cause_family: &'a str,
}

pub struct Explanation<'a> {
    pub run_id: &'a str,
    pub model: &'a str,
    pub json: &'a str,
    pub cost_usd: f64,
}

/// Creates a new ledger; existing tables cause an error without changing their contents.
pub fn init(path: &Path) -> Result<Connection> {
    let mut connection =
        Connection::open(path).with_context(|| format!("open ledger {}", path.display()))?;
    let transaction = connection
        .transaction()
        .context("begin ledger initialization")?;
    transaction
        .execute_batch(SCHEMA)
        .context("initialize ledger schema")?;
    for (table, key) in [
        ("changes", "change_id = NEW.change_id"),
        ("scenarios", "scenario_id = NEW.scenario_id"),
        ("runs", "run_id = NEW.run_id"),
        ("decisions", "decision_id = NEW.decision_id"),
        (
            "labels",
            "change_id = NEW.change_id AND scenario_id = NEW.scenario_id",
        ),
        ("explanations", "run_id = NEW.run_id"),
    ] {
        transaction
            .execute_batch(&format!(
                "CREATE TRIGGER {table}_no_replace BEFORE INSERT ON {table}
             WHEN EXISTS (SELECT 1 FROM {table} WHERE ({key}) OR rowid = NEW.rowid)
             BEGIN SELECT RAISE(ABORT, 'ledger is append-only: duplicate primary key'); END;"
            ))
            .context("reject replacement of ledger rows")?;
        for operation in ["UPDATE", "DELETE"] {
            transaction
                .execute_batch(&format!(
                    "CREATE TRIGGER {table}_no_{operation} BEFORE {operation} ON {table}
                 BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;"
                ))
                .context("protect append-only ledger")?;
        }
    }
    transaction
        .commit()
        .context("commit ledger initialization")?;
    Ok(connection)
}

/// Inserts one change, rejecting an existing change ID.
pub fn insert_change(connection: &Connection, row: &Change<'_>) -> Result<()> {
    connection
        .execute(
            "INSERT INTO changes (change_id, env, card_json, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![row.change_id, row.env, row.card_json, row.created_at],
        )
        .context("insert change")?;
    Ok(())
}

/// Inserts one scenario, rejecting an existing scenario ID.
pub fn insert_scenario(connection: &Connection, row: &Scenario<'_>) -> Result<()> {
    connection
        .execute(
            "INSERT INTO scenarios (scenario_id, env, params_json, baseline_success)
         VALUES (?1, ?2, ?3, ?4)",
            params![
                row.scenario_id,
                row.env,
                row.params_json,
                row.baseline_success
            ],
        )
        .context("insert scenario")?;
    Ok(())
}

/// Inserts one run, rejecting an existing run ID.
pub fn insert_run(connection: &Connection, row: &Run<'_>) -> Result<()> {
    connection
        .execute(
            "INSERT INTO runs (run_id, change_id, scenario_id, tier, episodes, successes,
         mean_reward, uri, wall_seconds) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                row.run_id,
                row.change_id,
                row.scenario_id,
                row.tier,
                row.episodes,
                row.successes,
                row.mean_reward,
                row.uri,
                row.wall_seconds
            ],
        )
        .context("insert run")?;
    Ok(())
}

/// Inserts one decision, rejecting an existing decision ID.
pub fn insert_decision(connection: &Connection, row: &Decision<'_>) -> Result<()> {
    connection
        .execute(
            "INSERT INTO decisions (decision_id, change_id, scenario_id, tier_used, model_version,
         probs_json, lambda, action, propensity, reason, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                row.decision_id,
                row.change_id,
                row.scenario_id,
                row.tier_used,
                row.model_version,
                row.probs_json,
                row.lambda,
                row.action,
                row.propensity,
                row.reason,
                row.created_at
            ],
        )
        .context("insert decision")?;
    Ok(())
}

/// Stores a label already computed by `brier label`, rejecting an existing pair.
pub fn insert_label(connection: &Connection, row: &Label<'_>) -> Result<()> {
    connection
        .execute(
            "INSERT INTO labels (change_id, scenario_id, regressed, failure_mode, success_bucket,
         cause_family) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                row.change_id,
                row.scenario_id,
                row.regressed,
                row.failure_mode,
                row.success_bucket,
                row.cause_family
            ],
        )
        .context("insert label")?;
    Ok(())
}

/// Inserts one explanation, rejecting an existing run ID.
pub fn insert_explanation(connection: &Connection, row: &Explanation<'_>) -> Result<()> {
    connection
        .execute(
            "INSERT INTO explanations (run_id, model, json, cost_usd) VALUES (?1, ?2, ?3, ?4)",
            params![row.run_id, row.model, row.json, row.cost_usd],
        )
        .context("insert explanation")?;
    Ok(())
}
