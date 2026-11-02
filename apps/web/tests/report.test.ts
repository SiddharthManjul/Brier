import assert from "node:assert/strict";
import test from "node:test";
import { metric, readStrategies } from "../lib/report.ts";

test("metric distinguishes measured zero from absent data and rejects invalid types", () => {
  assert.equal(metric(0), 0);
  assert.equal(metric(undefined), null);
  assert.equal(metric(null), null);
  assert.throws(() => metric("0.5"), /finite number/);
});

test("readStrategies uses test metrics when combined and OOD measurements differ", () => {
  const strategies = readStrategies({ strategies: { t1: {
    ece: 0.9, brier: 0.8, nll: 3, auroc: 0.2, eval_hours_saved: 0.4,
    test_metrics: { ece: 0.03, brier: 0.07, nll: 0.3, auroc: 0.94,
      reliability_bins: [{ count: 4, mean_probability: 0.2, observed_regression: 0.25 }] },
    ood_metrics: { ece: 0.7 }, guarantee_splits: 18,
    reliability_bins: [{ count: 40, mean_probability: 0.95, observed_regression: 0.99 }],
    threshold_curve: [
      { eval_hours_saved: 0.8, missed_regression_fraction: null },
      { eval_hours_saved: 0.4, missed_regression_fraction: 0.05 },
    ],
  } } });
  assert.equal(strategies[0].metrics.ece, 0.03);
  assert.equal(strategies[0].metrics.brier, 0.07);
  assert.equal(strategies[0].metrics.nll, 0.3);
  assert.equal(strategies[0].metrics.auroc, 0.94);
  assert.equal(strategies[0].metrics.eval_hours_saved, 0.4);
  assert.equal(strategies[0].oodEce, 0.7);
  assert.equal(strategies[0].metrics.guarantee_splits, 18);
  assert.deepEqual(strategies[0].reliability, [{ predicted: 0.2, observed: 0.25 }]);
  assert.deepEqual(strategies[0].curve, [{ saved: 40, missed: 5 }]);
  assert.deepEqual(readStrategies({}), []);
  assert.throws(() => readStrategies({ strategies: { t1: { test_metrics: { reliability_bins: false } } } }), /array/);
  assert.throws(() => readStrategies({ strategies: { t1: { guarantee_splits: 20.5 } } }), /integer/);
});

test("combined-only reports cannot masquerade as measured test calibration", () => {
  const [strategy] = readStrategies({ strategies: { legacy: {
    ece: 0.01, brier: 0.02, nll: 0.03, auroc: 0.99, ood_metrics: { ece: 0.04 },
    reliability_bins: [{ count: 4, mean_probability: 0.8, observed_regression: 0.75 }],
  } } });
  assert.deepEqual([strategy.metrics.ece, strategy.metrics.brier, strategy.metrics.nll, strategy.metrics.auroc],
    [null, null, null, null]);
  assert.equal(strategy.oodEce, 0.04);
  assert.deepEqual(strategy.reliability, []);
  assert.throws(() => readStrategies({ strategies: { invalid: { test_metrics: false } } }), /object/);
});

test("readStrategies rejects malformed values for every dashboard metric before rendering", () => {
  for (const field of ["eval_hours_saved", "missed_regression_fraction", "change_recall",
    "p50_latency_seconds", "p95_latency_seconds", "cost_usd_per_change"]) {
    for (const value of ["pending", false, [], {}, NaN, Infinity]) {
      assert.throws(() => readStrategies({ strategies: { measured: { [field]: value } } }),
        `${field} must reject ${String(value)}`);
    }
  }
  for (const [field, value] of [
    ["eval_hours_saved", 1.01], ["missed_regression_fraction", -0.01],
    ["missed_regression_fraction", 1.01], ["change_recall", -0.01], ["change_recall", 1.01],
    ["p50_latency_seconds", -0.01], ["p95_latency_seconds", -0.01], ["cost_usd_per_change", -0.01],
  ] as const) {
    assert.throws(() => readStrategies({ strategies: { measured: { [field]: value } } }),
      `${field} must reject out-of-range ${value}`);
  }
  for (const [field, value] of [["ece", -0.01], ["ece", 1.01], ["brier", -0.01],
    ["brier", 1.01], ["auroc", -0.01], ["auroc", 1.01], ["nll", -0.01]] as const) {
    assert.throws(() => readStrategies({ strategies: { measured: { test_metrics: { [field]: value } } } }));
  }
  assert.throws(() => readStrategies({ strategies: { measured: { ood_metrics: { ece: 1.01 } } } }));
});

test("readStrategies preserves nullable measurements and legitimate cost overruns", () => {
  const [strategy] = readStrategies({ strategies: { measured: {
    eval_hours_saved: -0.2, missed_regression_fraction: null, change_recall: null,
    p50_latency_seconds: 0, p95_latency_seconds: 400, cost_usd_per_change: 120,
    test_metrics: { ece: 0, brier: 1, nll: 40, auroc: null }, ood_metrics: { ece: null },
    threshold_curve: [{ eval_hours_saved: -0.2, missed_regression_fraction: 0.05 }],
  } } });
  assert.equal(strategy.metrics.eval_hours_saved, -0.2);
  assert.equal(strategy.metrics.missed_regression_fraction, null);
  assert.equal(strategy.metrics.change_recall, null);
  assert.equal(strategy.metrics.p50_latency_seconds, 0);
  assert.equal(strategy.metrics.p95_latency_seconds, 400);
  assert.equal(strategy.metrics.cost_usd_per_change, 120);
  assert.equal(strategy.metrics.nll, 40);
  assert.equal(strategy.metrics.auroc, null);
  assert.equal(strategy.oodEce, null);
  assert.deepEqual(strategy.curve, [{ saved: -20, missed: 5 }]);
});

test("readGuide3 keeps failed diagnostics separate and rejects inconsistent risk or gate claims", async () => {
  const { readGuide3 } = await import("../lib/report.ts");
  const make = () => ({ guide3: {
    guarantee: "change", guarantee_denominator: "all changes; conditional regressing-change misses are measured separately",
    heldout_changes: 42, t0_credit_verified: false, wave_size: 4,
    alphas: { "0.05": {
      production_policy: { alpha: .05, lambda: .228, risk_bound: .0465, calibration_changes: 42,
        guarantee: "change", guarantee_metric: "unconditional_change_miss", certified: true, run_all: false },
      splits: Array.from({ length: 20 }, (_, seed) => ({ seed })), splits_within_alpha: 14,
      required_splits_within_alpha: 18, replay_gate_passed: false,
      summary: { eval_hours_saved: { mean: .72, worst: .63 },
        unconditional_change_miss_rate: { mean: .04, worst: .27 },
        conditional_regressing_change_miss_rate: { mean: .058, worst: .43 },
        actual_scenario_miss_rate: { mean: .65, worst: .74 },
        savings_breakdown_mean_baseline_fraction: { t0_saved_seconds: 0,
          adaptive_cheap_saved_seconds: .008, cheap_overhead_seconds: .067,
          threshold_saved_seconds: .37, early_stop_saved_seconds: .43,
          exploration_selected_overhead_seconds: .011 } },
    } },
  } });
  assert.equal(readGuide3({}), null);
  const report = readGuide3(make())!;
  assert.equal(report.alphas[0].passed, false);
  assert.equal(report.alphas[0].within, 14);
  assert.equal(report.alphas[0].allChangeMiss.mean, .04);
  assert.equal(report.alphas[0].conditionalMiss.mean, .058);
  assert.equal(report.alphas[0].scenarioMiss.mean, .65);
  const mutate = (path: string[], value: unknown) => {
    const input = make();
    let target: Record<string, unknown> = input;
    for (const key of path.slice(0, -1)) target = target[key] as Record<string, unknown>;
    target[path.at(-1)!] = value;
    assert.throws(() => readGuide3(input), `must reject ${path.join(".")}`);
  };
  const row = ["guide3", "alphas", "0.05"];
  for (const [field, value] of [["replay_gate_passed", "false"], ["replay_gate_passed", true],
    ["splits_within_alpha", 21], ["required_splits_within_alpha", 0], ["splits", []]] as const) {
    mutate([...row, field], value);
  }
  for (const [field, value] of [["lambda", NaN], ["risk_bound", .1], ["run_all", true],
    ["certified", "true"], ["guarantee_metric", "conditional_change_miss"], ["calibration_changes", 49]] as const) {
    mutate([...row, "production_policy", field], value);
  }
  mutate([...row, "summary", "eval_hours_saved", "mean"], "pending");
  mutate([...row, "summary", "actual_scenario_miss_rate", "worst"], -.1);
  mutate([...row, "summary", "conditional_regressing_change_miss_rate", "mean"], 2);
  mutate([...row, "summary", "savings_breakdown_mean_baseline_fraction", "t0_saved_seconds"], .3);
});
