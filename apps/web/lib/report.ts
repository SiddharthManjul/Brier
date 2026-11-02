import type { Json } from "../db/schema.ts";

export type Strategy = {
  name: string;
  metrics: Record<string, Json>;
  oodEce: number | null;
  reliability: { predicted: number; observed: number }[];
  curve: { saved: number; missed: number }[];
};

/** Read optional measured numbers without inventing values for absent metrics. */
export function metric(value: Json | undefined): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("Report metric must be a finite number");
  return value;
}

function record(value: Json | undefined, path: string): Record<string, Json> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
  return value;
}

function boundedMetric(value: Json | undefined, name: string, min: number, max = Infinity): number | null {
  const measured = metric(value);
  if (measured !== null && (measured < min || measured > max)) {
    throw new Error(`${name} must be between ${min} and ${max}`);
  }
  return measured;
}

/** Read test metrics and chart points while keeping OOD calibration separate. */
export function readStrategies(body: Record<string, Json>): Strategy[] {
  if (body.strategies === undefined) return [];
  return Object.entries(record(body.strategies, "strategies")).map(([name, value]) => {
    const metrics = record(value, `strategies.${name}`);
    const test = metrics.test_metrics == null ? {} : record(metrics.test_metrics, "test_metrics");
    const oodEce = metrics.ood_metrics == null ? null
      : boundedMetric(record(metrics.ood_metrics, "ood_metrics").ece, "OOD ECE", 0, 1);
    const guarantee = metric(metrics.guarantee_splits);
    if (guarantee !== null && (!Number.isInteger(guarantee) || guarantee < 0 || guarantee > 20)) {
      throw new Error("guarantee_splits must be an integer in [0, 20]");
    }
    const reliability = test.reliability_bins;
    const curve = metrics.threshold_curve;
    if (reliability !== undefined && !Array.isArray(reliability)) throw new Error("reliability_bins must be an array");
    if (curve !== undefined && !Array.isArray(curve)) throw new Error("threshold_curve must be an array");
    return {
      name, metrics: {
        ...metrics,
        ece: boundedMetric(test.ece, "Test ECE", 0, 1),
        brier: boundedMetric(test.brier, "Test Brier score", 0, 1),
        nll: boundedMetric(test.nll, "Test NLL", 0),
        auroc: boundedMetric(test.auroc, "Test AUROC", 0, 1),
        eval_hours_saved: boundedMetric(metrics.eval_hours_saved, "Evaluation hours saved", -Infinity, 1),
        missed_regression_fraction: boundedMetric(metrics.missed_regression_fraction, "Missed regression fraction", 0, 1),
        change_recall: boundedMetric(metrics.change_recall, "Change recall", 0, 1),
        p50_latency_seconds: boundedMetric(metrics.p50_latency_seconds, "p50 latency", 0),
        p95_latency_seconds: boundedMetric(metrics.p95_latency_seconds, "p95 latency", 0),
        cost_usd_per_change: boundedMetric(metrics.cost_usd_per_change, "Cost per change", 0),
      }, oodEce,
      reliability: (reliability ?? []).map((value) => {
        const point = record(value, "reliability bin");
        const predicted = metric(point.mean_probability), observed = metric(point.observed_regression);
        if (predicted === null || observed === null || predicted < 0 || predicted > 1 || observed < 0 || observed > 1) {
          throw new Error("Reliability bins require probabilities in [0, 1]");
        }
        return { predicted, observed };
      }),
      curve: (curve ?? []).flatMap((value) => {
        const point = record(value, "threshold curve point");
        const saved = boundedMetric(point.eval_hours_saved, "Threshold evaluation hours saved", -Infinity, 1);
        const missed = boundedMetric(point.missed_regression_fraction, "Threshold missed regression fraction", 0, 1);
        if (saved === null || missed === null) return [];
        return [{ saved: saved * 100, missed: missed * 100 }];
      }).sort((a, b) => a.saved - b.saved),
    };
  });
}

export type Guide3Diagnostics = {
  guarantee: "change" | "scenario";
  denominator: string;
  heldoutChanges: number;
  t0Verified: boolean;
  waveSize: number;
  alphas: {
    alpha: number; lambda: number; certified: boolean; runAll: boolean; riskBound: number;
    within: number; total: number; required: number; passed: boolean;
    savings: { mean: number | null; worst: number | null };
    allChangeMiss: { mean: number | null; worst: number | null };
    conditionalMiss: { mean: number | null; worst: number | null };
    scenarioMiss: { mean: number | null; worst: number | null };
    breakdown: Record<string, number>;
  }[];
};

/** Read policy-as-executed diagnostics without treating a failed replay as validation. */
export function readGuide3(body: Record<string, Json>): Guide3Diagnostics | null {
  if (body.guide3 === undefined) return null;
  const root = record(body.guide3, "guide3");
  const guarantee = root.guarantee;
  if (guarantee !== "change" && guarantee !== "scenario") throw new Error("Guide 3 guarantee must be change or scenario");
  if (root.guarantee_denominator !== "all changes; conditional regressing-change misses are measured separately") {
    throw new Error("Guide 3 guarantee denominator is required");
  }
  function required(value: Json | undefined, name: string, min = 0, max = 1): number {
    const result = boundedMetric(value, name, min, max);
    if (result === null) throw new Error(`${name} is required`);
    return result;
  }
  function count(value: Json | undefined, name: string, max = Infinity): number {
    const result = required(value, name, 0, max);
    if (!Number.isInteger(result)) throw new Error(`${name} must be an integer`);
    return result;
  }
  function boolean(value: Json | undefined, name: string): boolean {
    if (typeof value !== "boolean") throw new Error(`${name} must be boolean`);
    return value;
  }
  function range(value: Json | undefined, name: string, savings = false) {
    const row = record(value, name);
    if (!Object.hasOwn(row, "mean") || !Object.hasOwn(row, "worst")) throw new Error(`${name} requires mean and worst`);
    const mean = boundedMetric(row.mean, `${name}.mean`, savings ? -Infinity : 0, 1);
    const worst = boundedMetric(row.worst, `${name}.worst`, savings ? -Infinity : 0, 1);
    if ((mean === null) !== (worst === null) || (mean !== null && worst !== null &&
      (savings ? worst > mean : worst < mean))) throw new Error(`${name} has inconsistent mean/worst measurements`);
    return { mean, worst };
  }
  const heldoutChanges = count(root.heldout_changes, "Guide 3 held-out changes");
  if (heldoutChanges < 2) throw new Error("Guide 3 requires at least two held-out changes");
  const t0Verified = boolean(root.t0_credit_verified, "Guide 3 T0 verification");
  const waveSize = count(root.wave_size, "Guide 3 wave size");
  if (waveSize !== 4) throw new Error("Guide 3 requires four-job waves");
  const rows = Object.entries(record(root.alphas, "guide3.alphas"));
  if (!rows.length) throw new Error("Guide 3 alpha diagnostics are required");
  const alphas = rows.map(([key, value]) => {
    const row = record(value, `guide3.alphas.${key}`);
    const policy = record(row.production_policy, "production_policy");
    const alpha = required(policy.alpha, "Guide 3 alpha");
    if (!key.trim() || Number(key) !== alpha || policy.guarantee !== guarantee ||
      policy.guarantee_metric !== (guarantee === "change" ? "unconditional_change_miss" : "mean_scenario_miss") ||
      count(policy.calibration_changes, "Guide 3 calibration changes") !== heldoutChanges) {
      throw new Error("Guide 3 policy identity or risk denominator disagrees");
    }
    const lambda = required(policy.lambda, "Guide 3 lambda");
    const riskBound = required(policy.risk_bound, "Guide 3 risk bound");
    const certified = boolean(policy.certified, "Guide 3 certification");
    const runAll = boolean(policy.run_all, "Guide 3 run-all fallback");
    if (certified === runAll || (certified && riskBound > alpha + 1e-12) || (runAll && lambda !== 0)) {
      throw new Error("Guide 3 certification and fallback disagree");
    }
    if (!Array.isArray(row.splits) || row.splits.length !== 20) throw new Error("Guide 3 requires twenty replay splits");
    const total = row.splits.length;
    const within = count(row.splits_within_alpha, "Guide 3 splits within alpha", total);
    const minimum = count(row.required_splits_within_alpha, "Guide 3 required splits", total);
    const passed = boolean(row.replay_gate_passed, "Guide 3 replay gate");
    if (minimum !== 18 || passed !== (within >= minimum)) throw new Error("Guide 3 replay gate disagrees with its counts");
    const summary = record(row.summary, "Guide 3 summary");
    const breakdown = Object.fromEntries(Object.entries(record(summary.savings_breakdown_mean_baseline_fraction,
      "Guide 3 savings breakdown")).map(([name, value]) => [name, required(value, `Guide 3 ${name}`)]));
    for (const name of ["t0_saved_seconds", "adaptive_cheap_saved_seconds", "cheap_overhead_seconds",
      "threshold_saved_seconds", "early_stop_saved_seconds", "exploration_selected_overhead_seconds"]) {
      if (breakdown[name] === undefined) throw new Error(`Guide 3 savings breakdown requires ${name}`);
    }
    if (!t0Verified && breakdown.t0_saved_seconds !== 0) throw new Error("Unverified T0 savings cannot be credited");
    return { alpha, lambda, certified, runAll, riskBound, within, total, required: minimum, passed,
      savings: range(summary.eval_hours_saved, "Guide 3 savings", true),
      allChangeMiss: range(summary.unconditional_change_miss_rate, "Guide 3 all-change misses"),
      conditionalMiss: range(summary.conditional_regressing_change_miss_rate, "Guide 3 conditional misses"),
      scenarioMiss: range(summary.actual_scenario_miss_rate, "Guide 3 scenario misses"), breakdown };
  }).sort((a, b) => a.alpha - b.alpha);
  if (new Set(alphas.map((row) => row.alpha)).size !== alphas.length) throw new Error("Guide 3 alpha values must be unique");
  return { guarantee, denominator: root.guarantee_denominator, heldoutChanges, t0Verified, waveSize, alphas };
}
