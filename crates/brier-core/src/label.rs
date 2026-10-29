use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, ensure, Context, Result};
use brier_core::ledger;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::RowAccessor,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

const MODES: [&str; 6] = [
    "none",
    "collision",
    "oscillation",
    "grasp or drop failure",
    "overshoot/misalignment",
    "stall/timeout",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    changes: Vec<Change>,
    scenarios: Vec<Scenario>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Change {
    change_id: String,
    env: String,
    card: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    env: String,
    params: Value,
    seeds: Vec<i64>,
}
#[derive(Default)]
struct Run {
    seeds: BTreeSet<i64>,
    successes: i64,
    reward: f64,
    seconds: f64,
    failures: [usize; 6],
    uris: BTreeSet<PathBuf>,
}

fn family(card: &Value) -> Result<String> {
    let families = card["families"]
        .as_array()
        .context("card.families must be an array")?;
    ensure!(
        (1..=2).contains(&families.len()),
        "card needs one or two families"
    );
    ensure!(
        card["description"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty() && !s.contains(['\n', '\r'])),
        "card description must be one nonempty line"
    );
    let mut names = BTreeSet::new();
    for item in families {
        let name = item["family"].as_str().context("family must be a string")?;
        ensure!(
            [
                "obs_noise",
                "action_latency",
                "action_scale",
                "action_bias",
                "camera_shift",
                "occlusion",
                "brightness",
                "frame_drop",
                "chunk_len",
                "weight_noise",
                "benign"
            ]
            .contains(&name),
            "unknown change family {name}"
        );
        ensure!(
            item["params"].is_object() && names.insert(name),
            "invalid or duplicate family"
        );
    }
    ensure!(
        !names.contains("benign") || names.len() == 1,
        "benign must be the only family"
    );
    Ok(names.into_iter().collect::<Vec<_>>().join("+"))
}

fn failed_mode(terminal: &str, env: &str) -> Result<usize> {
    let state: Value = serde_json::from_str(terminal).context("invalid terminal_state JSON")?;
    let stats = &state["trajectory"];
    let metric = |key: &str| -> Result<f64> {
        let value = stats[key]
            .as_f64()
            .with_context(|| format!("failed episode requires trajectory.{key}"))?;
        ensure!(
            value.is_finite() && value >= 0.0,
            "trajectory.{key} must be finite and nonnegative"
        );
        Ok(value)
    };
    let collisions = metric("collision_steps")?;
    let changes = metric("action_direction_changes")?;
    let transitions = metric("action_transitions")?;
    let peak = metric("object_max_height")?;
    let final_height = metric("object_final_height")?;
    let nearest = metric("goal_distance_min")?;
    let final_distance = metric("goal_distance_final")?;
    ensure!(
        changes <= transitions && nearest <= final_distance + 1e-9,
        "inconsistent trajectory statistics"
    );
    // Precedence is physical collision, sustained action oscillation, dropped lifted cube,
    // moving away from the closest goal approach, then the remaining timeout/stall.
    Ok(if collisions > 0.0 {
        1
    } else if changes >= 4.0 && changes > 0.6 * transitions {
        2
    } else if env.contains("Aloha") && peak > 0.1 && peak - final_height > 0.05 {
        3
    } else if final_distance - nearest > if env.contains("PushT") { 25.0 } else { 0.05 } {
        4
    } else {
        5
    })
}

fn parquet_paths(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        fs::read_dir(directory).with_context(|| format!("read episodes {}", directory.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            parquet_paths(&path, paths)?;
        } else if path.extension().is_some_and(|s| s == "parquet") {
            paths.push(path);
        }
    }
    Ok(())
}

/// Atomically imports complete episode groups and computes their sole authoritative labels.
pub fn label(ledger_path: &Path, episodes: &Path, catalog_path: &Path) -> Result<()> {
    let catalog: Catalog =
        serde_json::from_reader(File::open(catalog_path)?).context("read rollout catalog")?;
    let changes: BTreeMap<_, _> = catalog
        .changes
        .iter()
        .map(|c| (c.change_id.as_str(), c))
        .collect();
    let scenarios: BTreeMap<_, _> = catalog
        .scenarios
        .iter()
        .map(|s| (s.scenario_id.as_str(), s))
        .collect();
    ensure!(
        changes.len() == catalog.changes.len() && scenarios.len() == catalog.scenarios.len(),
        "duplicate catalog IDs"
    );
    for change in changes.values() {
        family(&change.card)?;
    }
    for scenario in scenarios.values() {
        ensure!(
            scenario.params.is_object()
                && scenario.seeds.len() == 50
                && scenario.seeds.iter().all(|s| *s >= 0)
                && scenario.seeds.iter().collect::<BTreeSet<_>>().len() == 50,
            "scenario {} requires params and 50 distinct seeds",
            scenario.scenario_id
        );
    }
    let mut paths = Vec::new();
    parquet_paths(episodes, &mut paths)?;
    paths.sort();
    ensure!(
        !paths.is_empty(),
        "episodes directory contains no Parquet files"
    );
    let mut runs: BTreeMap<(String, String, String), Run> = BTreeMap::new();
    for path in paths {
        let reader = SerializedFileReader::new(File::open(&path)?)?;
        let mut row_count = 0;
        for row in reader.get_row_iter(None)? {
            let row = row?;
            let names: Vec<_> = row.get_column_iter().map(|(n, _)| n.as_str()).collect();
            ensure!(
                names
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
                "unexpected episode schema in {}",
                path.display()
            );
            let env = row.get_string(0)?;
            let change = changes
                .get(row.get_string(2)?.as_str())
                .context("episode change absent from catalog")?;
            let scenario = scenarios
                .get(row.get_string(3)?.as_str())
                .context("episode scenario absent from catalog")?;
            ensure!(
                env == &change.env && env == &scenario.env,
                "episode and catalog environments disagree"
            );
            let expected_policy = match env.as_str() {
                "gym_pusht/PushT-v0" => "lerobot/diffusion_pusht",
                "gym_aloha/AlohaTransferCube-v0" => "lerobot/act_aloha_sim_transfer_cube_human",
                _ => bail!("unsupported episode environment {env}"),
            };
            ensure!(
                row.get_string(1)? == expected_policy,
                "episode policy does not match environment"
            );
            let tier = row.get_string(4)?;
            let count = match tier.as_str() {
                "cheap" => 3,
                "expensive" => 20,
                "baseline" => 50,
                _ => bail!("unknown episode tier {tier}"),
            };
            let seed = row.get_long(5)?;
            ensure!(
                scenario.seeds[..count].contains(&seed),
                "episode seed is outside tier seed prefix"
            );
            let success = row.get_bool(6)?;
            let steps = row.get_long(7)?;
            let reward = row.get_double(9)?;
            let seconds = row.get_double(11)?;
            ensure!(
                steps > 0 && reward.is_finite() && seconds.is_finite() && seconds >= 0.0,
                "invalid episode steps/reward/time"
            );
            if tier == "baseline" {
                ensure!(
                    family(&change.card)? == "benign",
                    "baseline must use benign policy"
                );
            }
            let terminal = row.get_string(8)?;
            ensure!(
                serde_json::from_str::<Value>(terminal)?.is_object(),
                "terminal_state must be an object"
            );
            let mode = if success || tier == "cheap" {
                0
            } else {
                failed_mode(terminal, env)?
            };
            let run = runs
                .entry((
                    change.change_id.clone(),
                    scenario.scenario_id.clone(),
                    tier.clone(),
                ))
                .or_default();
            ensure!(
                run.seeds.insert(seed),
                "duplicate episode seed in a change/scenario/tier group"
            );
            run.successes += i64::from(success);
            run.reward += reward;
            run.seconds += seconds;
            run.failures[mode] += 1;
            run.uris.insert(path.clone());
            row_count += 1;
        }
        ensure!(row_count > 0, "empty episode Parquet {}", path.display());
    }
    let mut baselines = BTreeMap::new();
    let mut pairs = BTreeSet::new();
    for ((change, scenario, tier), run) in &runs {
        let count = match tier.as_str() {
            "cheap" => 3,
            "expensive" => 20,
            _ => 50,
        };
        ensure!(
            run.seeds.len() == count,
            "incomplete {change}/{scenario}/{tier}: expected {count} episodes"
        );
        if tier == "baseline" {
            ensure!(
                baselines.insert(scenario.as_str(), run.successes).is_none(),
                "multiple baselines for scenario {scenario}"
            );
        } else {
            pairs.insert((change.as_str(), scenario.as_str()));
        }
    }
    for (change, scenario) in &pairs {
        ensure!(
            baselines.contains_key(scenario),
            "missing 50-episode baseline for {scenario}"
        );
        for tier in ["cheap", "expensive"] {
            ensure!(
                runs.contains_key(&(change.to_string(), scenario.to_string(), tier.to_string())),
                "missing {tier} run for {change}/{scenario}"
            );
        }
    }
    let mut connection =
        Connection::open_with_flags(ledger_path, OpenFlags::SQLITE_OPEN_READ_WRITE)
            .context("open initialized ledger")?;
    let transaction = connection.transaction()?;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs()
        .to_string();
    for change in changes.values() {
        let json = serde_json::to_string(&change.card)?;
        let existing: Option<(String, String)> = transaction
            .query_row(
                "SELECT env,card_json FROM changes WHERE change_id=?1",
                [&change.change_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((env, card)) = existing {
            ensure!(
                env == change.env && serde_json::from_str::<Value>(&card)? == change.card,
                "catalog changes existing change {}",
                change.change_id
            );
        } else {
            ledger::insert_change(
                &transaction,
                &ledger::Change {
                    change_id: &change.change_id,
                    env: &change.env,
                    card_json: &json,
                    created_at: &created,
                },
            )?;
        }
    }
    for (scenario_id, successes) in &baselines {
        let scenario = scenarios[scenario_id];
        let json = serde_json::to_string(&scenario.params)?;
        let existing: Option<(String, String, f64)> = transaction
            .query_row(
                "SELECT env,params_json,baseline_success FROM scenarios WHERE scenario_id=?1",
                [scenario_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        if let Some((env, params, baseline)) = existing {
            ensure!(
                env == scenario.env
                    && serde_json::from_str::<Value>(&params)? == scenario.params
                    && (baseline - *successes as f64 / 50.0).abs() < 1e-12,
                "catalog changes existing scenario {scenario_id}"
            );
        } else {
            ledger::insert_scenario(
                &transaction,
                &ledger::Scenario {
                    scenario_id,
                    env: &scenario.env,
                    params_json: &json,
                    baseline_success: *successes as f64 / 50.0,
                },
            )?;
        }
    }
    for ((change, scenario, tier), run) in &runs {
        let existing: i64 = transaction.query_row(
            "SELECT count(*) FROM runs WHERE change_id=?1 AND scenario_id=?2 AND tier=?3",
            [change, scenario, tier],
            |r| r.get(0),
        )?;
        ensure!(
            existing == 0,
            "run already imported: {change}/{scenario}/{tier}"
        );
        let uri = if run.uris.len() == 1 {
            run.uris.first().unwrap().display().to_string()
        } else {
            serde_json::to_string(&run.uris)?
        };
        ledger::insert_run(
            &transaction,
            &ledger::Run {
                run_id: &Uuid::new_v4().to_string(),
                change_id: change,
                scenario_id: scenario,
                tier,
                episodes: run.seeds.len() as i64,
                successes: run.successes,
                mean_reward: run.reward / run.seeds.len() as f64,
                uri: &uri,
                wall_seconds: run.seconds,
            },
        )?;
    }
    for (change, scenario) in pairs {
        let run = &runs[&(change.to_string(), scenario.to_string(), "expensive".into())];
        let mode = (1..6)
            .max_by_key(|i| (run.failures[*i], std::cmp::Reverse(*i)))
            .unwrap();
        ledger::insert_label(
            &transaction,
            &ledger::Label {
                change_id: change,
                scenario_id: scenario,
                regressed: 5 * run.successes < 2 * baselines[scenario] - 15,
                failure_mode: if run.successes == 20 {
                    MODES[0]
                } else {
                    MODES[mode]
                },
                success_bucket: ((run.successes + 3) / 4).max(1),
                cause_family: &family(&changes[change].card)?,
            },
        )?;
    }
    transaction
        .commit()
        .context("commit labeled episode import")
}

#[cfg(test)]
mod tests {
    use super::{failed_mode, MODES};
    use serde_json::json;

    #[test]
    fn label_failure_modes_follow_measured_trajectory_and_precedence() {
        let base = json!({"collision_steps":0,"action_direction_changes":0,
            "action_transitions":10,"object_max_height":0.0,"object_final_height":0.0,
            "goal_distance_min":10.0,"goal_distance_final":10.0});
        for (env, key, value, expected) in [
            ("PushT", "collision_steps", 1.0, "collision"),
            ("PushT", "action_direction_changes", 8.0, "oscillation"),
            ("Aloha", "object_max_height", 0.2, "grasp or drop failure"),
            (
                "PushT",
                "goal_distance_final",
                40.0,
                "overshoot/misalignment",
            ),
            (
                "Aloha",
                "goal_distance_final",
                10.1,
                "overshoot/misalignment",
            ),
            ("PushT", "goal_distance_final", 10.0, "stall/timeout"),
        ] {
            let mut trajectory = base.clone();
            trajectory[key] = json!(value);
            let raw = json!({"trajectory": trajectory}).to_string();
            assert_eq!(MODES[failed_mode(&raw, env).unwrap()], expected);
        }
        let mut trajectory = base;
        trajectory["collision_steps"] = json!(1);
        trajectory["action_direction_changes"] = json!(8);
        assert_eq!(
            failed_mode(&json!({"trajectory": trajectory}).to_string(), "Aloha").unwrap(),
            1
        );
        trajectory["action_direction_changes"] = json!(11);
        assert!(failed_mode(&json!({"trajectory": trajectory}).to_string(), "PushT").is_err());
    }
}
