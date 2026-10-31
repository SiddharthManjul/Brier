use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs::File, path::Path};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub id: String,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Family {
    pub family: String,
    pub params: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    env: String,
    policy: Checkpoint,
    description: String,
    families: Vec<Family>,
    #[serde(default)]
    logging: Value,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Identity {
    pub env: String,
    pub policy: Checkpoint,
    pub effective_families: Vec<Family>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IdentityProof {
    pub before: Identity,
    pub after: Identity,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CardReport {
    pub change_id: String,
    pub env: String,
    pub policy: Checkpoint,
    pub card: Value,
    pub t0_identical: bool,
    pub identity: IdentityProof,
}

fn read_snapshot(path: &Path) -> Result<Snapshot> {
    let snapshot: Snapshot = serde_yaml::from_reader(File::open(path)?)
        .with_context(|| format!("read policy snapshot {}", path.display()))?;
    let expected_policy = match snapshot.env.as_str() {
        "gym_pusht/PushT-v0" => "lerobot/diffusion_pusht",
        "gym_aloha/AlohaTransferCube-v0" => "lerobot/act_aloha_sim_transfer_cube_human",
        _ => anyhow::bail!("unsupported policy environment"),
    };
    ensure!(
        snapshot.policy.id == expected_policy,
        "policy checkpoint does not match the supported environment"
    );
    ensure!(
        snapshot.policy.revision.len() == 40
            && snapshot
                .policy
                .revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "policy revision must be an immutable 40-character hexadecimal checkpoint commit"
    );
    ensure!(
        snapshot.logging.is_null() || snapshot.logging.is_object(),
        "logging metadata must be an object"
    );
    crate::cascade::change_features(
        &json!({"description":snapshot.description,"families":snapshot.families}),
    )?;
    for family in &snapshot.families {
        for (key, value) in &family.params {
            let number = value
                .as_f64()
                .context("change parameters must be numeric")?;
            if matches!(
                key.as_str(),
                "k" | "dx" | "dy" | "x" | "y" | "width" | "height" | "n_action_steps"
            ) {
                ensure!(value.as_i64().is_some(), "{key} must be an integer");
            }
            if matches!(key.as_str(), "sigma" | "gain" | "contrast" | "x" | "y") {
                ensure!(number >= 0.0, "{key} must be nonnegative");
            }
            if matches!(
                key.as_str(),
                "k" | "scale" | "width" | "height" | "n_action_steps"
            ) {
                ensure!(number > 0.0, "{key} must be positive");
            }
            if key == "p" {
                ensure!(
                    (0.0..=1.0).contains(&number),
                    "p must be between zero and one"
                );
            }
        }
    }
    Ok(snapshot)
}

fn identity(snapshot: &Snapshot) -> Identity {
    let effective_families = snapshot
        .families
        .iter()
        .filter(|family| {
            !(family.family == "benign"
                || matches!(family.family.as_str(), "obs_noise" | "weight_noise")
                    && family.params.get("sigma").and_then(Value::as_f64) == Some(0.0))
        })
        .cloned()
        .collect();
    Identity {
        env: snapshot.env.clone(),
        policy: snapshot.policy.clone(),
        effective_families,
    }
}

/// Builds a validated change card and compares pinned checkpoint/no-op identities.
/// Description and logging metadata cannot affect identity. Perturbation families
/// retain their order, and non-noop changes never receive the T0 shortcut.
pub fn card(before: &Path, after: &Path, change_id: &str) -> Result<CardReport> {
    let number = change_id
        .strip_prefix('c')
        .context("change id must start with c")?;
    ensure!(
        number.len() >= 3
            && number.bytes().all(|byte| byte.is_ascii_digit())
            && number
                .parse::<u64>()
                .context("change id exceeds the supported seed range")?
                > 0,
        "change id must contain at least three digits and start at c001"
    );
    let before = read_snapshot(before)?;
    let after = read_snapshot(after)?;
    let before_identity = identity(&before);
    let after_identity = identity(&after);
    let t0_identical =
        before_identity == after_identity && before_identity.effective_families.is_empty();
    Ok(CardReport {
        change_id: change_id.into(),
        env: after.env,
        policy: after.policy,
        card: json!({"description":after.description,"families":after.families}),
        t0_identical,
        identity: IdentityProof {
            before: before_identity,
            after: after_identity,
        },
    })
}
