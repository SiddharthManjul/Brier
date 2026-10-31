import { timingSafeEqual } from "node:crypto";
import type { changes, evaluations, failures, Json, pairs, reports } from "../db/schema.ts";

export type IngestBundle = {
  change: typeof changes.$inferInsert;
  evaluation: typeof evaluations.$inferInsert;
  pairs: (typeof pairs.$inferInsert)[];
  failures: (typeof failures.$inferInsert)[];
  report?: typeof reports.$inferInsert;
};
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function object(value: unknown, keys: string[], path: string, optional: string[] = []) {
  if (value === null || typeof value !== "object" || Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Object.prototype) throw new Error(`${path} must be an object`);
  const row = value as Record<string, unknown>;
  const allowed = new Set([...keys, ...optional]);
  if (keys.some((key) => !Object.hasOwn(row, key)) || Object.keys(row).some((key) => !allowed.has(key))) {
    throw new Error(`${path} has missing or unexpected fields`);
  }
  return row;
}

function text(value: unknown, path: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${path} must be a nonempty string`);
  return value;
}

function uuid(value: unknown, path: string): string {
  const result = text(value, path);
  if (!UUID.test(result)) throw new Error(`${path} must be a UUIDv4`);
  return result.toLowerCase();
}

function number(value: unknown, path: string, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${path} must be a finite number between ${min} and ${max}`);
  }
  return value;
}

function count(value: unknown, path: string): number {
  const result = number(value, path, 0, 2_147_483_647);
  if (!Number.isInteger(result)) throw new Error(`${path} must be an integer`);
  return result;
}

function nullableProbability(value: unknown, path: string): number | null {
  return value === null ? null : number(value, path, 0, 1);
}

function date(value: unknown, path: string): string {
  const raw = text(value, path);
  const match = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?Z$/.exec(raw);
  const parsed = new Date(raw);
  const normalized = match ? `${match[1]}.${(match[2] ?? "").padEnd(3, "0")}Z` : "";
  if (!match || !Number.isFinite(parsed.valueOf()) || parsed.toISOString() !== normalized) {
    throw new Error(`${path} must be a valid UTC ISO timestamp`);
  }
  return normalized;
}

function url(value: unknown, path: string): string {
  const raw = text(value, path);
  let parsed: URL;
  try { parsed = new URL(raw); } catch { throw new Error(`${path} must be an HTTPS URL without credentials`); }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password) {
    throw new Error(`${path} must be an HTTPS URL without credentials`);
  }
  return raw;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
  return value;
}

function jsonObject(value: unknown, path: string): Record<string, Json> {
  const seen = new Set<object>();
  function visit(current: unknown): void {
    if (current === null || typeof current === "string" || typeof current === "boolean") return;
    if (typeof current === "number" && Number.isFinite(current)) return;
    if (typeof current !== "object" || current === null || seen.has(current)) {
      throw new Error(`${path} must contain finite, acyclic JSON values`);
    }
    if (!Array.isArray(current) && Object.getPrototypeOf(current) !== Object.prototype) {
      throw new Error(`${path} must contain plain JSON objects`);
    }
    seen.add(current);
    for (const child of Array.isArray(current) ? current : Object.values(current)) visit(child);
    seen.delete(current);
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error(`${path} must be a JSON object`);
  visit(value);
  return value as Record<string, Json>;
}

/** Check the bearer token; missing server configuration throws. */
export function authenticateIngest(header: string | null, configuredToken: string | undefined): boolean {
  if (!configuredToken?.trim()) throw new Error("BRIER_INGEST_TOKEN is required");
  const expected = Buffer.from(`Bearer ${configuredToken}`);
  const received = Buffer.from(header ?? "");
  return expected.length === received.length && timingSafeEqual(expected, received);
}

/** Validate one complete evaluation without creating IDs, labels, or prompts. */
export function validateIngest(value: unknown): IngestBundle {
  const root = object(value, ["change", "evaluation", "pairs", "failures"], "bundle", ["report"]);
  const c = object(root.change, ["id", "env", "title", "card", "prUrl", "createdAt"], "change");
  const change = {
    id: text(c.id, "change.id"), env: text(c.env, "change.env"),
    title: text(c.title, "change.title"), card: jsonObject(c.card, "change.card"),
    prUrl: c.prUrl === null ? null : url(c.prUrl, "change.prUrl"), createdAt: date(c.createdAt, "change.createdAt"),
  };
  if (!/^c\d{3,}$/.test(change.id) || /^c0+$/.test(change.id)) throw new Error("change.id must start at c001 with at least three digits");
  if (!/^[a-z][a-z0-9_]*$/.test(change.env)) throw new Error("change.env is invalid");
  const e = object(root.evaluation, ["id", "changeId", "verdict", "alpha", "lambda", "evalsRun", "evalsTotal",
    "hoursSavedPct", "modelVersion", "createdAt"], "evaluation");
  if (e.changeId !== change.id) throw new Error("evaluation.changeId must match change.id");
  if (e.verdict !== "pass" && e.verdict !== "block") throw new Error("evaluation.verdict must be pass or block");
  const evaluation: IngestBundle["evaluation"] = {
    id: uuid(e.id, "evaluation.id"), changeId: change.id, verdict: e.verdict,
    alpha: number(e.alpha, "evaluation.alpha", 0, 1), lambda: number(e.lambda, "evaluation.lambda", 0, 1),
    evalsRun: count(e.evalsRun, "evaluation.evalsRun"), evalsTotal: count(e.evalsTotal, "evaluation.evalsTotal"),
    hoursSavedPct: e.hoursSavedPct === null ? null : number(e.hoursSavedPct, "evaluation.hoursSavedPct", 0, 100),
    modelVersion: text(e.modelVersion, "evaluation.modelVersion"), createdAt: date(e.createdAt, "evaluation.createdAt"),
  };
  const pairRows = array(root.pairs, "pairs").map((value, index) => {
    const path = `pairs[${index}]`;
    const p = object(value, ["id", "evaluationId", "scenarioId", "pRegress", "action", "result", "successBefore", "successAfter"], path);
    const evaluationId = uuid(p.evaluationId, `${path}.evaluationId`);
    const scenarioId = text(p.scenarioId, `${path}.scenarioId`);
    const prefix = `${change.env}-s`;
    if (evaluationId !== evaluation.id) throw new Error(`${path}.evaluationId must match evaluation.id`);
    if (!scenarioId.startsWith(prefix) || !/^(0[1-9]|[1-9]\d)$/.test(scenarioId.slice(prefix.length))) {
      throw new Error(`${path}.scenarioId must match the change environment`);
    }
    return {
      id: uuid(p.id, `${path}.id`), evaluationId, scenarioId,
      pRegress: number(p.pRegress, `${path}.pRegress`, 0, 1), action: text(p.action, `${path}.action`),
      result: p.result === null ? null : text(p.result, `${path}.result`),
      successBefore: nullableProbability(p.successBefore, `${path}.successBefore`),
      successAfter: nullableProbability(p.successAfter, `${path}.successAfter`),
    };
  });
  const pairIds = new Set(pairRows.map((pair) => pair.id));
  if (pairIds.size !== pairRows.length) throw new Error("pairs contain duplicate IDs");
  if (new Set(pairRows.map((pair) => pair.scenarioId)).size !== pairRows.length) throw new Error("pairs contain duplicate scenario IDs");
  if (evaluation.evalsTotal !== pairRows.length || evaluation.evalsRun !== pairRows.filter((pair) => pair.result !== null).length) {
    throw new Error("evaluation counts must match the supplied pairs and completed results");
  }
  const failureRows = array(root.failures, "failures").map((value, index) => {
    const path = `failures[${index}]`;
    const f = object(value, ["id", "pairId", "keyframeUrls", "overlayUrls", "explanation"], path);
    const pairId = uuid(f.pairId, `${path}.pairId`);
    if (!pairIds.has(pairId)) throw new Error(`${path}.pairId must identify a supplied pair`);
    return {
      id: uuid(f.id, `${path}.id`), pairId,
      keyframeUrls: array(f.keyframeUrls, `${path}.keyframeUrls`).map((v, i) => url(v, `${path}.keyframeUrls[${i}]`)),
      overlayUrls: array(f.overlayUrls, `${path}.overlayUrls`).map((v, i) => url(v, `${path}.overlayUrls[${i}]`)),
      explanation: jsonObject(f.explanation, `${path}.explanation`),
    };
  });
  if (new Set(failureRows.map((failure) => failure.id)).size !== failureRows.length) throw new Error("failures contain duplicate IDs");
  const bundle: IngestBundle = { change, evaluation, pairs: pairRows, failures: failureRows };
  if (Object.hasOwn(root, "report")) {
    const r = object(root.report, ["id", "suite", "createdAt", "body"], "report");
    if (r.suite !== "robot" && r.suite !== "software") throw new Error("report.suite must be robot or software");
    bundle.report = {
      id: uuid(r.id, "report.id"), suite: r.suite,
      createdAt: date(r.createdAt, "report.createdAt"), body: jsonObject(r.body, "report.body"),
    };
  }
  return bundle;
}
