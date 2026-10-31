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

/** Read test metrics and chart points while keeping OOD calibration separate. */
export function readStrategies(body: Record<string, Json>): Strategy[] {
  if (body.strategies === undefined) return [];
  return Object.entries(record(body.strategies, "strategies")).map(([name, value]) => {
    const metrics = record(value, `strategies.${name}`);
    const test = metrics.test_metrics == null ? {} : record(metrics.test_metrics, "test_metrics");
    const oodEce = metrics.ood_metrics == null ? null : metric(record(metrics.ood_metrics, "ood_metrics").ece);
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
        ...metrics, ece: metric(test.ece), brier: metric(test.brier),
        nll: metric(test.nll), auroc: metric(test.auroc),
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
        const saved = metric(point.eval_hours_saved), missed = metric(point.missed_regression_fraction);
        if (saved === null || missed === null) return [];
        if (saved < 0 || saved > 1 || missed < 0 || missed > 1) throw new Error("Threshold curve fractions are out of range");
        return [{ saved: saved * 100, missed: missed * 100 }];
      }).sort((a, b) => a.saved - b.saved),
    };
  });
}
