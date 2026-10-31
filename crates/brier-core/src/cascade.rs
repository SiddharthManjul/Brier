use anyhow::{ensure, Context, Result};
use serde_json::Value;

pub const FEATURE_NAMES: [&str; 37] = [
    "family_obs_noise",
    "family_action_latency",
    "family_action_scale",
    "family_action_bias",
    "family_camera_shift",
    "family_occlusion",
    "family_brightness",
    "family_frame_drop",
    "family_chunk_len",
    "family_weight_noise",
    "family_benign",
    "obs_noise_sigma",
    "action_latency_k",
    "action_scale_scale",
    "action_bias_drift",
    "camera_shift_dx",
    "camera_shift_dy",
    "camera_shift_angle",
    "occlusion_x",
    "occlusion_y",
    "occlusion_width",
    "occlusion_height",
    "brightness_gain",
    "brightness_contrast",
    "frame_drop_p",
    "chunk_len_n_action_steps",
    "weight_noise_sigma",
    "env_is_aloha",
    "pose_bin",
    "lighting_low",
    "camera_jitter",
    "physics_jitter",
    "cheap_success",
    "cheap_mean_reward",
    "baseline_success",
    "scenario_fragility",
    "similar_regressed_rate",
];

const FAMILIES: [(&str, &[&str]); 11] = [
    ("obs_noise", &["sigma"]),
    ("action_latency", &["k"]),
    ("action_scale", &["scale"]),
    ("action_bias", &["drift"]),
    ("camera_shift", &["dx", "dy", "angle"]),
    ("occlusion", &["x", "y", "width", "height"]),
    ("brightness", &["gain", "contrast"]),
    ("frame_drop", &["p"]),
    ("chunk_len", &["n_action_steps"]),
    ("weight_noise", &["sigma"]),
    ("benign", &[]),
];

pub struct NumericState<'a> {
    pub card: &'a Value,
    pub env: &'a str,
    pub params: &'a Value,
    pub cheap_successes: i64,
    pub cheap_reward: f64,
    pub baseline: f64,
    pub fragility: f64,
    pub similar_regressed: usize,
}

pub(crate) fn change_features(card: &Value) -> Result<Vec<f64>> {
    let normalized;
    let card = if card.get("desc").is_some() {
        ensure!(
            card.as_object().is_some_and(|o| o.len() == 5)
                && ["change_id", "env", "families", "desc", "seed"]
                    .iter()
                    .all(|key| card.get(key).is_some())
                && matches!(card["env"].as_str(), Some("pusht" | "aloha"))
                && card["seed"].as_u64().is_some()
                && card["change_id"]
                    .as_str()
                    .and_then(|id| id.strip_prefix('c'))
                    .is_some_and(|id| id.len() >= 3
                        && id.bytes().all(|b| b.is_ascii_digit())
                        && id.bytes().any(|b| b != b'0')),
            "invalid guide change card metadata"
        );
        let mut value =
            serde_json::json!({"description": card["desc"], "families": card["families"]});
        for entry in value["families"]
            .as_array_mut()
            .context("missing change families")?
        {
            let family = entry["family"]
                .as_str()
                .context("missing family name")?
                .to_owned();
            let params = entry["params"]
                .as_object()
                .context("missing change parameters")?;
            ensure!(
                params
                    .iter()
                    .all(|(key, value)| value.as_f64().is_some_and(|n| n.is_finite()
                        && (!matches!(key.as_str(), "k" | "shift" | "n_action_steps")
                            || value.as_u64().is_some())
                        && (!matches!(key.as_str(), "sigma" | "gain" | "contrast" | "b")
                            || n >= 0.0)
                        && (!matches!(
                            key.as_str(),
                            "k" | "s" | "shift" | "n_action_steps" | "fraction"
                        ) || n > 0.0)
                        && (!matches!(key.as_str(), "p" | "fraction")
                            || (0.0..=1.0).contains(&n)))),
                "invalid guide change parameter bounds"
            );
            let number = |key: &str| -> Result<f64> {
                params
                    .get(key)
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite())
                    .context("missing finite guide parameter")
            };
            entry["params"] = match family.as_str() {
                "action_scale" | "action_bias" => {
                    let key = if family == "action_scale" { "s" } else { "b" };
                    let n = number(key)?;
                    ensure!(params.len() == 1, "invalid guide scale or bias");
                    if key == "s" {
                        serde_json::json!({"scale":n})
                    } else {
                        serde_json::json!({"drift":n})
                    }
                }
                "camera_shift" => {
                    let shift = number("shift")?;
                    ensure!(params.len() == 2, "invalid guide camera shift");
                    serde_json::json!({"dx":shift,"dy":shift,"angle":number("angle")?})
                }
                "occlusion" => {
                    let fraction = number("fraction")?;
                    ensure!(params.len() == 1, "invalid guide occlusion");
                    // Guide occlusion features encode normalized side lengths; location is seeded.
                    serde_json::json!({"x":0.0,"y":0.0,"width":fraction,"height":fraction})
                }
                _ => entry["params"].clone(),
            };
        }
        normalized = value;
        &normalized
    } else {
        card
    };
    let object = card.as_object().context("change card must be an object")?;
    ensure!(
        object.len() == 2 && object.contains_key("description"),
        "invalid change card fields"
    );
    let description = card["description"]
        .as_str()
        .context("missing change description")?;
    ensure!(
        !description.trim().is_empty() && !description.contains(['\r', '\n']),
        "invalid change description"
    );
    let entries = card["families"]
        .as_array()
        .context("missing change families")?;
    ensure!(
        (1..=2).contains(&entries.len()),
        "one or two change families required"
    );
    let mut features = vec![0.0; 27];
    for entry in entries {
        let name = entry["family"].as_str().context("missing family name")?;
        let index = FAMILIES
            .iter()
            .position(|(family, _)| *family == name)
            .context("unknown change family")?;
        ensure!(features[index] == 0.0, "duplicate change family");
        features[index] = 1.0;
        let params = entry["params"]
            .as_object()
            .context("missing change parameters")?;
        ensure!(
            entry.as_object().is_some_and(|o| o.len() == 2)
                && params.len() == FAMILIES[index].1.len(),
            "invalid family fields"
        );
        let offset = 11
            + FAMILIES[..index]
                .iter()
                .map(|(_, keys)| keys.len())
                .sum::<usize>();
        for (parameter, key) in FAMILIES[index].1.iter().enumerate() {
            let value = params
                .get(*key)
                .and_then(Value::as_f64)
                .context("missing numeric change parameter")?;
            ensure!(value.is_finite(), "nonfinite change parameter");
            let neutral = f64::from(matches!(*key, "scale" | "gain" | "contrast"));
            features[offset + parameter] = (value - neutral).abs();
        }
    }
    ensure!(
        features[10] == 0.0 || entries.len() == 1,
        "benign must be the only family"
    );
    Ok(features)
}

/// Builds the exact numeric feature order exported by Python's T1 training.
pub fn features(state: &NumericState<'_>) -> Result<Vec<f64>> {
    let mut values = change_features(state.card)?;
    let aloha = match state.env {
        "gym_pusht/PushT-v0" => false,
        "gym_aloha/AlohaTransferCube-v0" => true,
        _ => anyhow::bail!("unsupported environment"),
    };
    let params = state
        .params
        .as_object()
        .context("scenario params must be an object")?;
    let mut pose = params
        .get("pose_bin")
        .and_then(Value::as_u64)
        .context("invalid pose_bin")?;
    if state.card.get("desc").is_some() {
        ensure!(
            (1..=4).contains(&pose),
            "guide pose_bin must be one through four"
        );
        ensure!(
            state.card["env"] == if aloha { "aloha" } else { "pusht" },
            "guide change/scenario environment mismatch"
        );
        pose -= 1;
    }
    let lighting = params
        .get("lighting")
        .and_then(Value::as_str)
        .context("invalid lighting")?;
    let camera = params
        .get("camera_jitter")
        .and_then(Value::as_bool)
        .context("invalid camera_jitter")?;
    let physics = match params.get("physics_jitter") {
        Some(value) => value.as_bool().context("invalid physics_jitter")?,
        None => {
            ensure!(!aloha, "ALOHA requires physics_jitter");
            false
        }
    };
    ensure!(
        pose < 4
            && matches!(lighting, "normal" | "low")
            && params.len() == 3 + usize::from(params.contains_key("physics_jitter")),
        "invalid scenario parameters"
    );
    ensure!(
        (0..=3).contains(&state.cheap_successes)
            && state.cheap_reward.is_finite()
            && (0.0..=1.0).contains(&state.baseline)
            && (0.0..=1.0).contains(&state.fragility)
            && state.similar_regressed <= 5,
        "invalid numeric evidence"
    );
    values.extend([
        f64::from(aloha),
        pose as f64,
        f64::from(lighting == "low"),
        f64::from(camera),
        f64::from(physics),
        state.cheap_successes as f64 / 3.0,
        state.cheap_reward,
        state.baseline,
        state.fragility,
        state.similar_regressed as f64 / 5.0,
    ]);
    Ok(values)
}

/// Evaluates numeric LightGBM JSON and the exported isotonic interpolation knots.
pub fn predict_t1(artifact: &Value, features: &[f64]) -> Result<f64> {
    ensure!(
        artifact["format_version"] == 1,
        "unsupported T1 artifact format"
    );
    let names = artifact["feature_names"]
        .as_array()
        .context("missing feature_names")?;
    ensure!(
        names.len() == FEATURE_NAMES.len() && names.iter().zip(FEATURE_NAMES).all(|(a, b)| a == b),
        "T1 feature order differs from the engine"
    );
    ensure!(
        features.len() == FEATURE_NAMES.len() && features.iter().all(|value| value.is_finite()),
        "T1 features must be finite and match feature_names"
    );
    ensure!(
        artifact["booster"]["objective"] == "binary sigmoid:1",
        "T1 requires binary sigmoid:1"
    );
    let trees = artifact["booster"]["tree_info"]
        .as_array()
        .context("missing T1 trees")?;
    ensure!(!trees.is_empty(), "T1 requires at least one tree");
    let mut score = 0.0;
    for tree in trees {
        let mut node = &tree["tree_structure"];
        loop {
            if let Some(value) = node.get("leaf_value") {
                score += value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .context("invalid T1 leaf")?;
                break;
            }
            ensure!(
                node["decision_type"] == "<=",
                "T1 supports numeric <= splits only"
            );
            let index = node["split_feature"]
                .as_u64()
                .context("invalid T1 split feature")?;
            let value = features
                .get(usize::try_from(index)?)
                .context("T1 split feature out of range")?;
            let threshold = node["threshold"]
                .as_f64()
                .filter(|v| v.is_finite())
                .context("invalid T1 threshold")?;
            node = node
                .get(if *value <= threshold {
                    "left_child"
                } else {
                    "right_child"
                })
                .context("missing T1 tree child")?;
        }
    }
    ensure!(score.is_finite(), "nonfinite T1 tree sum");
    let number_array = |key: &str| -> Result<Vec<f64>> {
        artifact["isotonic"][key]
            .as_array()
            .context("missing isotonic knots")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .context("invalid isotonic knot")
            })
            .collect()
    };
    let x = number_array("x")?;
    let y = number_array("y")?;
    ensure!(
        !x.is_empty()
            && x.len() == y.len()
            && x.windows(2).all(|w| w[0] < w[1])
            && y.windows(2).all(|w| w[0] <= w[1])
            && y.iter().all(|v| (0.0..=1.0).contains(v)),
        "invalid isotonic knots"
    );
    let exponential = (-score.abs()).exp();
    let probability = if score >= 0.0 {
        1.0 / (1.0 + exponential)
    } else {
        exponential / (1.0 + exponential)
    };
    let upper = x.partition_point(|value| *value <= probability);
    Ok(if upper == 0 {
        y[0]
    } else if upper == x.len() {
        y[y.len() - 1]
    } else {
        y[upper - 1]
            + (y[upper] - y[upper - 1]) * (probability - x[upper - 1]) / (x[upper] - x[upper - 1])
    })
}
