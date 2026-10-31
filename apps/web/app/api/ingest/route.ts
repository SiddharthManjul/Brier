import { eq, inArray, sql } from "drizzle-orm";
import { changes, evaluations, failures, pairs, reports } from "../../../db/schema.ts";
import { getDb } from "../../../lib/db.ts";
import { authenticateIngest, validateIngest } from "../../../lib/ingest.ts";
import type { IngestBundle } from "../../../lib/ingest.ts";

export const runtime = "nodejs";

/** Atomically replace one evaluation and its pairs after authenticating the caller. */
export async function POST(request: Request): Promise<Response> {
  let authorized: boolean;
  try {
    authorized = authenticateIngest(request.headers.get("authorization"), process.env.BRIER_INGEST_TOKEN);
  } catch {
    return Response.json({ error: "BRIER_INGEST_TOKEN is required" }, { status: 500 });
  }
  if (!authorized) return Response.json({ error: "Unauthorized" }, { status: 401 });
  let bundle: IngestBundle;
  try {
    bundle = validateIngest(await request.json());
  } catch (error) {
    const message = error instanceof SyntaxError ? "Request body must be valid JSON"
      : error instanceof Error ? error.message : "Invalid ingest bundle";
    return Response.json({ error: message }, { status: 400 });
  }
  try {
    const db = getDb();
    const mutations = [
      db.insert(changes).values(bundle.change).onConflictDoUpdate({ target: changes.id, set: bundle.change }),
      db.insert(evaluations).values(bundle.evaluation).onConflictDoUpdate({
        target: evaluations.id,
        set: { ...bundle.evaluation, changeId: sql<string>`case when ${evaluations.changeId} = ${bundle.change.id}
          then ${evaluations.changeId} else null end` },
      }),
      db.delete(failures).where(inArray(failures.pairId,
        db.select({ id: pairs.id }).from(pairs).where(eq(pairs.evaluationId, bundle.evaluation.id)))),
      db.delete(pairs).where(eq(pairs.evaluationId, bundle.evaluation.id)),
    ] as const;
    // Neon HTTP executes the batch as one non-interactive transaction.
    await db.batch([
      ...mutations,
      ...(bundle.pairs.length ? [db.insert(pairs).values(bundle.pairs)] : []),
      ...(bundle.failures.length ? [db.insert(failures).values(bundle.failures)] : []),
      ...(bundle.report ? [db.insert(reports).values(bundle.report).onConflictDoUpdate({
        target: reports.id,
        set: { ...bundle.report, suite: sql<"robot" | "software">`case when ${reports.suite} = ${bundle.report.suite}
          then ${reports.suite} else null end` },
      })] : []),
    ]);
  } catch {
    return Response.json({ error: "Database ingest failed; check DATABASE_URL and the database schema" }, { status: 500 });
  }
  return Response.json({ evaluationId: bundle.evaluation.id });
}
