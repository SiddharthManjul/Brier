use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader},
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
use serde::{de::DeserializeOwned, Deserialize};
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
    #[serde(default)]
    baseline_success: Option<f64>,
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

/// Atomically imports complete tier groups, labelling only newly imported expensive results.
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
        ensure!(
            run.reward.is_finite() && run.seconds.is_finite(),
            "nonfinite aggregate run statistics"
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
    let mut connection =
        Connection::open_with_flags(ledger_path, OpenFlags::SQLITE_OPEN_READ_WRITE)
            .context("open initialized ledger")?;
    let transaction = connection.transaction()?;
    for scenario in baselines.keys() {
        let count: i64 = transaction.query_row(
            "SELECT count(*) FROM runs WHERE scenario_id=?1 AND tier='baseline'",
            [scenario],
            |row| row.get(0),
        )?;
        ensure!(count == 0, "baseline already imported for {scenario}");
    }
    for (change, scenario) in &pairs {
        if !baselines.contains_key(scenario) {
            let mut query = transaction.prepare(
                "SELECT change_id,episodes,successes,mean_reward,wall_seconds FROM runs
                 WHERE scenario_id=?1 AND tier='baseline'",
            )?;
            let existing = query
                .query_map([scenario], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, f64>(3)?,
                        row.get::<_, f64>(4)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            ensure!(
                existing.len() == 1
                    && existing[0].1 == 50
                    && (0..=50).contains(&existing[0].2)
                    && existing[0].3.is_finite()
                    && existing[0].4.is_finite()
                    && existing[0].4 >= 0.0,
                "missing or invalid 50-episode baseline for {scenario}"
            );
            let (env, card): (String, String) = transaction
                .query_row(
                    "SELECT env,card_json FROM changes WHERE change_id=?1",
                    [&existing[0].0],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .context("stored baseline change missing from ledger")?;
            ensure!(
                env == scenarios[scenario].env
                    && family(&serde_json::from_str(&card)?)? == "benign",
                "stored baseline metadata disagrees with scenario"
            );
            baselines.insert(*scenario, existing[0].2);
        }
        if runs.contains_key(&(change.to_string(), scenario.to_string(), "expensive".into()))
            && !runs.contains_key(&(change.to_string(), scenario.to_string(), "cheap".into()))
        {
            let mut query = transaction.prepare(
                "SELECT episodes,successes,mean_reward,wall_seconds FROM runs
                 WHERE change_id=?1 AND scenario_id=?2 AND tier='cheap'",
            )?;
            let existing = query
                .query_map([change, scenario], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            ensure!(
                existing.len() == 1
                    && existing[0].0 == 3
                    && (0..=3).contains(&existing[0].1)
                    && existing[0].2.is_finite()
                    && existing[0].3.is_finite()
                    && existing[0].3 >= 0.0,
                "missing or invalid three-episode cheap run for {change}/{scenario}"
            );
        }
    }
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
        let Some(run) = runs.get(&(change.to_string(), scenario.to_string(), "expensive".into()))
        else {
            continue;
        };
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

#[derive(Deserialize)]
struct GuideEpisode {
    env: String,
    change_id: String,
    scenario_id: String,
    tier: String,
    episode_idx: usize,
    seed: i64,
    success: bool,
    steps: i64,
    max_steps: i64,
    reward_sum: f64,
    wall_seconds: f64,
    action_flip_rate: f64,
    max_reward: f64,
    final_reward: f64,
    coverage_max: Option<f64>,
    block_disp_px: Option<f64>,
    table_contact_steps: Option<i64>,
}

fn jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let rows = BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(&line?)
                .with_context(|| format!("{} line {}", path.display(), index + 1))
        })
        .collect::<Result<Vec<T>>>()?;
    ensure!(!rows.is_empty(), "empty JSONL {}", path.display());
    Ok(rows)
}

fn guide_env(env: &str) -> Result<&'static str> {
    match env {
        "pusht" => Ok("gym_pusht/PushT-v0"),
        "aloha" => Ok("gym_aloha/AlohaTransferCube-v0"),
        _ => bail!("unsupported guide environment {env}"),
    }
}

fn guide_mode(row: &GuideEpisode) -> Result<usize> {
    ensure!(
        row.steps > 0
            && row.steps <= row.max_steps
            && row.reward_sum.is_finite()
            && row.wall_seconds.is_finite()
            && row.wall_seconds >= 0.0
            && (0.0..=1.0).contains(&row.action_flip_rate)
            && row.final_reward >= 0.0
            && row.final_reward <= row.max_reward
            && row.max_reward <= if row.env == "pusht" { 1.0 } else { 4.0 },
        "invalid episode measurements"
    );
    let coverage = row.coverage_max.unwrap_or(-1.0);
    let displacement = row.block_disp_px.unwrap_or(-1.0);
    let contacts = row.table_contact_steps.unwrap_or(-1);
    let aloha = row.env == "aloha";
    ensure!(
        if row.env == "pusht" {
            (0.0..=1.0).contains(&coverage)
                && displacement.is_finite()
                && displacement >= 0.0
                && row.table_contact_steps.is_none()
        } else {
            (0..=row.steps).contains(&contacts)
                && row.coverage_max.is_none()
                && row.block_disp_px.is_none()
        },
        "missing or invalid environment failure measurements"
    );
    let rules = [
        (row.success, 0),
        (row.action_flip_rate > 0.35, 2),
        (aloha && contacts >= 20, 1),
        (
            aloha && (row.max_reward >= 2.0 && row.final_reward < 2.0 || row.max_reward == 1.0),
            3,
        ),
        (
            aloha && row.max_reward == 3.0 || !aloha && coverage >= 0.5,
            4,
        ),
        (aloha || displacement < 20.0, 5),
        (true, 4),
    ];
    Ok(rules
        .into_iter()
        .find_map(|(matches, mode)| matches.then_some(mode))
        .unwrap())
}

fn guide_insert(connection: &Connection, table: &str, row: Value, keys: &[&str]) -> Result<()> {
    let object = row.as_object().unwrap();
    let cols: Vec<_> = object.keys().map(String::as_str).collect();
    let fields = cols.join(",");
    let extract = |name: &str| format!("json_extract(?1,'$.{name}')");
    let equal = |name: &str| format!("{name} IS {}", extract(name));
    let clauses: Vec<_> = keys.iter().map(|name| equal(name)).collect();
    let filter = clauses.join(" AND ");
    let clauses: Vec<_> = cols.iter().map(|name| equal(name)).collect();
    let identical = clauses.join(" AND ");
    let json = serde_json::to_string(&row)?;
    let sql = format!("SELECT count(*),coalesce(sum({identical}),0) FROM {table} WHERE {filter}");
    let (count, matches): (i64, i64) =
        connection.query_row(&sql, [&json], |r| Ok((r.get(0)?, r.get(1)?)))?;
    if count == 0 {
        let values: Vec<_> = cols.iter().map(|name| extract(name)).collect();
        let parameters = values.join(",");
        let sql = format!("INSERT INTO {table} ({fields}) SELECT {parameters}");
        connection.execute(&sql, [&json])?;
    } else {
        ensure!(
            count == 1 && matches == 1,
            "conflicting existing {table} row; ledger is append-only"
        );
    }
    Ok(())
}

/// Atomically imports guide JSONL, preserving identical rows and printing measured label rates.
pub fn label_jsonl(
    ledger_path: &Path,
    episodes: &Path,
    changes_path: &Path,
    scenarios_path: &Path,
) -> Result<()> {
    let mut changes = BTreeMap::new();
    for card in jsonl::<Value>(changes_path)? {
        let id = card["change_id"]
            .as_str()
            .context("missing change_id")?
            .to_owned();
        let name = card["env"].as_str().context("missing change env")?;
        let env = guide_env(name)?.to_owned();
        ensure!(
            card.as_object().is_some_and(|o| o.len() == 5)
                && card["seed"].as_u64().is_some()
                && id.starts_with('c')
                && id.len() >= 4
                && id[1..].bytes().all(|b| b.is_ascii_digit())
                && id[1..].parse::<u64>()? > 0,
            "invalid guide change card"
        );
        let mut validation = card.clone();
        validation["description"] = card["desc"].clone();
        family(&validation)?;
        let change = Change {
            change_id: id.clone(),
            env,
            card,
        };
        ensure!(changes.insert(id, change).is_none(), "duplicate change ID");
    }
    let mut scenarios = BTreeMap::new();
    for mut scenario in jsonl::<Scenario>(scenarios_path)? {
        let prefix = format!("{}-s", scenario.env);
        scenario.env = guide_env(&scenario.env)?.to_owned();
        let rate = scenario.baseline_success;
        ensure!(
            scenario.scenario_id.starts_with(&prefix)
                && scenario.params.is_object()
                && scenario.seeds.len() == 50
                && scenario.seeds.iter().all(|seed| *seed >= 0)
                && scenario.seeds.iter().collect::<BTreeSet<_>>().len() == 50
                && rate.is_some_and(|value| (0.0..=1.0).contains(&value)),
            "invalid guide scenario"
        );
        ensure!(
            scenarios
                .insert(scenario.scenario_id.clone(), scenario)
                .is_none(),
            "duplicate scenario ID"
        );
    }
    let mut runs: BTreeMap<(String, String, String), Run> = BTreeMap::new();
    for row in jsonl::<GuideEpisode>(episodes)? {
        let scenario = scenarios
            .get(&row.scenario_id)
            .context("unknown episode scenario")?;
        ensure!(
            guide_env(&row.env)? == scenario.env,
            "episode environment disagrees with scenario"
        );
        let count = match row.tier.as_str() {
            "baseline" => 50,
            "cheap" => 3,
            "expensive" => 20,
            _ => bail!("unsupported episode tier"),
        };
        if row.tier == "baseline" {
            ensure!(row.change_id == "base", "baseline must use base policy");
        } else {
            ensure!(
                changes
                    .get(&row.change_id)
                    .is_some_and(|change| change.env == scenario.env),
                "unknown change or mismatched episode env"
            );
        }
        ensure!(
            row.episode_idx < count && scenario.seeds[row.episode_idx] == row.seed,
            "episode seed/index is not the fixed tier prefix"
        );
        let mode = guide_mode(&row)?;
        let run = runs
            .entry((row.change_id, row.scenario_id, row.tier))
            .or_default();
        ensure!(run.seeds.insert(row.seed), "duplicate episode seed");
        run.successes += i64::from(row.success);
        run.reward += row.reward_sum;
        run.seconds += row.wall_seconds;
        run.failures[mode] += 1;
    }
    for ((_, _, tier), run) in &runs {
        ensure!(
            run.seeds.len()
                == match tier.as_str() {
                    "baseline" => 50,
                    "cheap" => 3,
                    _ => 20,
                }
                && run.reward.is_finite()
                && run.seconds.is_finite(),
            "incomplete or nonfinite guide group"
        );
    }
    for scenario in scenarios.values() {
        let key = (
            "base".into(),
            scenario.scenario_id.clone(),
            "baseline".into(),
        );
        let baseline = runs.get(&key).context("missing 50-episode base run")?;
        ensure!(
            scenario.baseline_success == Some(baseline.successes as f64 / 50.0),
            "scenario baseline_success differs from measured episodes"
        );
    }
    for (change, scenario, tier) in runs.keys().filter(|key| key.0 != "base") {
        let other = if tier == "cheap" {
            "expensive"
        } else {
            "cheap"
        };
        let key = (change.clone(), scenario.clone(), other.into());
        ensure!(runs.contains_key(&key), "pair lacks cheap or expensive run");
    }
    let root =
        std::env::var("BRIER_DATA").context("BRIER_DATA is required to identify episode keys")?;
    ensure!(!root.trim().is_empty(), "BRIER_DATA must not be empty");
    let mut connection =
        Connection::open_with_flags(ledger_path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    let tx = connection.transaction()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs()
        .to_string();
    for change in changes.values() {
        let sql = "SELECT created_at FROM changes WHERE change_id=?1";
        let id = &change.change_id;
        let created: Option<String> = tx.query_row(sql, [id], |row| row.get(0)).optional()?;
        let values = serde_json::json!({"change_id":change.change_id,"env":change.env,
            "card_json":serde_json::to_string(&change.card)?,"created_at":created.unwrap_or_else(|| now.clone())});
        guide_insert(&tx, "changes", values, &["change_id"])?;
    }
    for scenario in scenarios.values() {
        let values = serde_json::json!({"scenario_id":scenario.scenario_id,"env":scenario.env,
            "params_json":serde_json::to_string(&scenario.params)?,"baseline_success":scenario.baseline_success.unwrap()});
        guide_insert(&tx, "scenarios", values, &["scenario_id"])?;
    }
    for ((change, scenario, tier), run) in &runs {
        let key = format!("{change}:{scenario}:{tier}");
        let env = if scenarios[scenario].env.contains("PushT") {
            "pusht"
        } else {
            "aloha"
        };
        let uri = format!(
            "{}/episodes/env={env}/tier={tier}/change={change}/{scenario}.parquet",
            root.trim_end_matches('/')
        );
        let values = serde_json::json!({"run_id":Uuid::new_v5(&Uuid::NAMESPACE_URL,key.as_bytes()).to_string(),
            "change_id":change,"scenario_id":scenario,"tier":tier,"episodes":run.seeds.len(),"successes":run.successes,
            "mean_reward":run.reward/run.seeds.len() as f64,"uri":uri,"wall_seconds":run.seconds});
        guide_insert(&tx, "runs", values, &["change_id", "scenario_id", "tier"])?;
    }
    for ((change, scenario, _), run) in runs.iter().filter(|(key, _)| key.2 == "expensive") {
        let baseline = &runs[&("base".into(), scenario.clone(), "baseline".into())];
        let regressed = 5 * run.successes < 2 * baseline.successes - 15;
        let priority = [2, 1, 3, 4, 5];
        let mode = if regressed {
            *priority
                .iter()
                .enumerate()
                .max_by_key(|(rank, mode)| (run.failures[**mode], std::cmp::Reverse(*rank)))
                .unwrap()
                .1
        } else {
            0
        };
        let cause = &changes[change].card["families"][0]["family"];
        let values = serde_json::json!({"change_id":change,"scenario_id":scenario,"regressed":regressed,
            "failure_mode":MODES[mode],"success_bucket":(run.successes/4+1).min(5),"cause_family":cause});
        guide_insert(&tx, "labels", values, &["change_id", "scenario_id"])?;
    }
    let summary: String = tx.query_row(r#"
        WITH grouped AS (
          SELECT 'per_family' category,cause_family name,regressed FROM labels UNION ALL
          SELECT 'per_scenario',scenario_id,regressed FROM labels UNION ALL
          SELECT 'per_env',CASE c.env WHEN 'gym_pusht/PushT-v0' THEN 'pusht' ELSE 'aloha' END,l.regressed
            FROM labels l JOIN changes c USING(change_id)),
        rates AS (SELECT category,name,count(*) pairs,avg(regressed) rate FROM grouped GROUP BY category,name ORDER BY category,name),
        distributions AS (SELECT category,json_group_object(name,json_object('pairs',pairs,'regressed_rate',rate)) value FROM rates GROUP BY category)
        SELECT json_object('pairs',count(*),'regressed_rate',avg(regressed),
          'per_family',json((SELECT value FROM distributions WHERE category='per_family')),
          'per_scenario',json((SELECT value FROM distributions WHERE category='per_scenario')),
          'per_env',json((SELECT value FROM distributions WHERE category='per_env')),
          'benign_false_regression_rate',(SELECT avg(regressed) FROM labels WHERE cause_family='benign'),
          'failure_modes',json((SELECT json_group_object(failure_mode,n) FROM (SELECT failure_mode,count(*) n FROM labels GROUP BY failure_mode ORDER BY failure_mode)))) FROM labels
        "#, [], |row| row.get(0))?;
    tx.commit()?;
    println!("{summary}");
    Ok(())
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
