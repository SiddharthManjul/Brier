import assert from "node:assert/strict";
import test from "node:test";
import { POST } from "../app/api/ingest/route.ts";
import { getDb } from "../lib/db.ts";
import { authenticateIngest, validateIngest } from "../lib/ingest.ts";

function fixture() {
  const evaluationId = "10000000-0000-4000-8000-000000000001";
  const pairId = "20000000-0000-4000-8000-000000000001";
  const createdAt = "2026-10-29T00:00:00.000Z";
  return {
    change: { id: "c001", env: "pusht", title: "Friction change", card: { friction: 0.5 },
      prUrl: "https://github.com/example/repo/pull/1", createdAt },
    evaluation: { id: evaluationId, changeId: "c001", verdict: "block", alpha: 0.1, lambda: 0.5,
      evalsRun: 1, evalsTotal: 1, hoursSavedPct: 0 as number | null, modelVersion: "base@revision", createdAt },
    pairs: [{ id: pairId, evaluationId, scenarioId: "pusht-s01", pRegress: 0.8,
      action: "run", result: "fail", successBefore: 1, successAfter: 0.2 }],
    failures: [{ id: "30000000-0000-4000-8000-000000000001", pairId,
      keyframeUrls: ["https://storage.googleapis.com/example/frame.png"], overlayUrls: [],
      explanation: { summary: "The object was dropped." } }],
    report: { id: "40000000-0000-4000-8000-000000000001", suite: "robot", createdAt,
      body: { strategies: {} } },
  };
}

function restore(name: string, previous: string | undefined) {
  if (previous === undefined) delete process.env[name];
  else process.env[name] = previous;
}

test("validateIngest preserves a complete valid bundle", () => {
  const input = fixture();
  assert.deepEqual(validateIngest(input), input);
  input.evaluation.hoursSavedPct = null;
  assert.equal(validateIngest(input).evaluation.hoursSavedPct, null);
  input.change.id = "c123456789012345678";
  input.evaluation.changeId = input.change.id;
  assert.equal(validateIngest(input).change.id, input.change.id);
});

test("validateIngest identifies malformed values, relationships and counts", () => {
  const cases: [string, (v: ReturnType<typeof fixture>) => void][] = [
    ["change", (v) => { Object.assign(v.change, { extra: true }); }],
    ["change", (v) => { Reflect.deleteProperty(v.change, "title"); }],
    ["change.id", (v) => { v.change.id = "c000"; }],
    ["change.createdAt", (v) => { v.change.createdAt = "2026-02-30T00:00:00Z"; }],
    ["change.card", (v) => { v.change.card.friction = NaN; }],
    ["evaluation.changeId", (v) => { v.evaluation.changeId = "c002"; }],
    ["evaluation.alpha", (v) => { v.evaluation.alpha = -0.1; }],
    ["pairs[0].evaluationId", (v) => { v.pairs[0].evaluationId = "invalid"; }],
    ["pairs[0].scenarioId", (v) => { v.pairs[0].scenarioId = "aloha-s01"; }],
    ["pairs[0].pRegress", (v) => { v.pairs[0].pRegress = Infinity; }],
    ["counts", (v) => { v.evaluation.evalsRun = 0; }],
    ["failures[0].pairId", (v) => { v.failures[0].pairId = "20000000-0000-4000-8000-000000000002"; }],
  ];
  for (const [path, mutate] of cases) {
    const input = fixture();
    mutate(input);
    assert.throws(() => validateIngest(input), (error: unknown) =>
      error instanceof Error && error.message.includes(path));
  }
});

test("validateIngest rejects duplicate pair IDs and scenario IDs", () => {
  for (const duplicateId of [true, false]) {
    const input = fixture();
    input.pairs.push({ ...input.pairs[0], id: duplicateId ? input.pairs[0].id
      : "20000000-0000-4000-8000-000000000002" });
    assert.throws(() => validateIngest(input), /duplicate/);
  }
});

test("validateIngest requires HTTPS media URLs without credentials", () => {
  for (const address of ["javascript:alert(1)", "http://example.com/frame.png",
    "https://user:secret@example.com/frame.png"]) {
    const input = fixture();
    input.failures[0].keyframeUrls[0] = address;
    assert.throws(() => validateIngest(input), /keyframeUrls\[0\]/);
  }
});

test("authenticateIngest requires the exact configured bearer token", () => {
  assert.equal(authenticateIngest("Bearer test-token", "test-token"), true);
  assert.equal(authenticateIngest("Bearer wrong-token", "test-token"), false);
  assert.equal(authenticateIngest("Bearer best-token", "test-token"), false);
  assert.equal(authenticateIngest(null, "test-token"), false);
  assert.throws(() => authenticateIngest(null, undefined), /BRIER_INGEST_TOKEN/);
});

test("getDb fails before connecting when DATABASE_URL is missing", (t) => {
  const previous = process.env.DATABASE_URL;
  t.after(() => restore("DATABASE_URL", previous));
  delete process.env.DATABASE_URL;
  assert.throws(() => getDb(), /DATABASE_URL/);
});

test("POST separates authorization, JSON and configuration failures", async (t) => {
  const previous = process.env.BRIER_INGEST_TOKEN;
  t.after(() => restore("BRIER_INGEST_TOKEN", previous));
  process.env.BRIER_INGEST_TOKEN = "test-token";
  assert.equal((await POST(new Request("http://localhost/api/ingest"))).status, 401);
  const invalid = await POST(new Request("http://localhost/api/ingest", {
    method: "POST", headers: { authorization: "Bearer test-token" }, body: "{",
  }));
  assert.equal(invalid.status, 400);
  assert.deepEqual(await invalid.json(), { error: "Request body must be valid JSON" });
  const missingFields = await POST(new Request("http://localhost/api/ingest", {
    method: "POST", headers: { authorization: "Bearer test-token" }, body: "{}",
  }));
  assert.equal(missingFields.status, 400);
  assert.match((await missingFields.json()).error, /bundle/);
  delete process.env.BRIER_INGEST_TOKEN;
  assert.equal((await POST(new Request("http://localhost/api/ingest"))).status, 500);
});

test("POST awaits one scoped batch with equivalent replay statements", async (t) => {
  const token = process.env.BRIER_INGEST_TOKEN;
  const database = process.env.DATABASE_URL;
  t.after(() => { restore("BRIER_INGEST_TOKEN", token); restore("DATABASE_URL", database); });
  process.env.BRIER_INGEST_TOKEN = "test-token";
  process.env.DATABASE_URL = "postgresql://test:test@unit-test.invalid/brier";
  type Statement = { sql: string; params: unknown[] };
  const batches: Statement[][] = [];
  const batch = t.mock.method(Object.getPrototypeOf(getDb()), "batch",
    async (queries: { toSQL(): Statement }[]) => { batches.push(queries.map((q) => q.toSQL())); return []; });
  const input = fixture();
  const request = () => new Request("http://localhost/api/ingest", {
    method: "POST", headers: { authorization: "Bearer test-token" }, body: JSON.stringify(input),
  });
  for (let index = 0; index < 2; index++) {
    const response = await POST(request());
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { evaluationId: input.evaluation.id });
  }
  assert.equal(batch.mock.callCount(), 2);
  assert.deepEqual(batches[0], batches[1]);
  assert.deepEqual(batches[0].map((q) => q.sql.match(/^(?:insert into|delete from) "([^"]+)"/)?.[1]),
    ["changes", "evaluations", "failures", "pairs", "pairs", "failures", "reports"]);
  for (const query of batches[0].filter((q) => q.sql.startsWith("delete from"))) {
    assert.deepEqual(query.params, [input.evaluation.id]);
  }
  batch.mock.mockImplementation(async () => { throw new Error("hidden-database-secret"); });
  const failed = await POST(request());
  assert.equal(failed.status, 500);
  assert.equal(JSON.stringify(await failed.json()).includes("hidden-database-secret"), false);
});
