use brier_core::cascade::{features, predict_t1, NumericState, FEATURE_NAMES};
use serde_json::{json, Value};

fn artifact(tree: Value) -> Value {
    json!({"format_version": 1, "feature_names": FEATURE_NAMES.as_slice(),
        "booster": {"objective": "binary sigmoid:1", "tree_info": [{"tree_structure": tree}]},
        "isotonic": {"x": [0.0, 1.0], "y": [0.1, 0.9]}})
}

#[test]
fn numeric_features_preserve_training_order_magnitudes_and_precision() {
    let card = json!({"description": "adjust images and actions", "families": [
        {"family": "action_scale", "params": {"scale": 0.8}},
        {"family": "brightness", "params": {"gain": 0.7, "contrast": 1.2}}]});
    let params =
        json!({"pose_bin": 3, "lighting": "low", "camera_jitter": true, "physics_jitter": false});
    let result = features(&NumericState {
        card: &card,
        env: "gym_aloha/AlohaTransferCube-v0",
        params: &params,
        cheap_successes: 2,
        cheap_reward: 0.123456789,
        baseline: 0.9,
        fragility: 0.25,
        similar_regressed: 3,
    })
    .unwrap();
    assert_eq!(result.len(), 37);
    assert_eq!(result[2], 1.0);
    assert_eq!(result[6], 1.0);
    assert!((result[13] - 0.2).abs() < 1e-12);
    assert!((result[22] - 0.3).abs() < 1e-12);
    assert!((result[23] - 0.2).abs() < 1e-12);
    assert_eq!(
        &result[27..],
        &[
            1.0,
            3.0,
            1.0,
            1.0,
            0.0,
            2.0 / 3.0,
            0.123456789,
            0.9,
            0.25,
            0.6
        ]
    );
}

#[test]
fn exported_trees_use_inclusive_splits_stable_sigmoid_and_isotonic_interpolation() {
    let model = artifact(
        json!({"split_feature": 12, "decision_type": "<=", "threshold": 2.0,
        "left_child": {"leaf_value": -2.0}, "right_child": {"leaf_value": 2.0}}),
    );
    let mut input = vec![0.0; 37];
    input[12] = 2.0;
    let lower = predict_t1(&model, &input).unwrap();
    assert!((lower - (0.1 + 0.8 / (1.0 + 2.0_f64.exp()))).abs() < 1e-12);
    input[12] = 2.001;
    let upper = predict_t1(&model, &input).unwrap();
    assert!((upper - (0.1 + 0.8 / (1.0 + (-2.0_f64).exp()))).abs() < 1e-12);
}

#[test]
fn isotonic_clips_endpoints_and_supports_constant_single_knot_models() {
    let mut model = artifact(json!({"leaf_value": 1000.0}));
    model["isotonic"] = json!({"x": [0.2, 0.8], "y": [0.3, 0.7]});
    assert_eq!(predict_t1(&model, &[0.0; 37]).unwrap(), 0.7);
    model["booster"]["tree_info"][0]["tree_structure"]["leaf_value"] = json!(-1000.0);
    assert_eq!(predict_t1(&model, &[0.0; 37]).unwrap(), 0.3);
    model["isotonic"] = json!({"x": [0.5], "y": [0.6]});
    assert_eq!(predict_t1(&model, &[0.0; 37]).unwrap(), 0.6);
}

#[test]
fn malformed_models_and_nonfinite_inputs_fail_explicitly() {
    let model = artifact(json!({"leaf_value": 0.0}));
    let mut input = [0.0; 37];
    input[0] = f64::NAN;
    assert!(predict_t1(&model, &input).is_err());
    let mut wrong_order = model.clone();
    wrong_order["feature_names"][0] = json!("unknown");
    assert!(predict_t1(&wrong_order, &[0.0; 37]).is_err());
    let mut bad_knots = model.clone();
    bad_knots["isotonic"] = json!({"x": [0.5, 0.5], "y": [0.3, 0.7]});
    assert!(predict_t1(&bad_knots, &[0.0; 37]).is_err());
    let categorical =
        artifact(json!({"split_feature": 0, "decision_type": "==", "threshold": "1||2"}));
    assert!(predict_t1(&categorical, &[0.0; 37]).is_err());
}
