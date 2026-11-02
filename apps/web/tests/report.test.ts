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
