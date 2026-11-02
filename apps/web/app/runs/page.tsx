import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { desc, eq, inArray } from "drizzle-orm";
import { ArrowUpRight } from "lucide-react";
import { changes, evaluations } from "@/db/schema";
import { getDb } from "@/lib/db";
import { AppShell } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Pending } from "@/components/pending";

export const metadata: Metadata = { title: "Runs" };
export const dynamic = "force-dynamic";

/** List the latest fifty measured evaluations with an environment filter. */
export default async function Runs({ searchParams }: { searchParams: Promise<{ env?: string }> }) {
  const { env } = await searchParams;
  if (env !== undefined && env !== "" && env !== "pusht" && env !== "aloha_transfer_cube") notFound();
  const selected = env === "pusht" || env === "aloha_transfer_cube" ? env : "";
  const connected = Boolean(process.env.DATABASE_URL);
  const rows = connected ? await getDb().select({ evaluation: evaluations, change: changes })
    .from(evaluations).innerJoin(changes, eq(evaluations.changeId, changes.id))
    .where(selected ? eq(changes.env, selected) : inArray(changes.env, ["pusht", "aloha_transfer_cube"]))
    .orderBy(desc(evaluations.createdAt)).limit(50) : [];
  const blocked = rows.filter(({ evaluation }) => evaluation.verdict === "block").length;
  const run = rows.reduce((sum, { evaluation }) => sum + evaluation.evalsRun, 0);
  const total = rows.reduce((sum, { evaluation }) => sum + evaluation.evalsTotal, 0);
  const measuredSavings = rows.flatMap(({ evaluation }) => evaluation.hoursSavedPct === null ? [] : [evaluation.hoursSavedPct]);
  const averageSaved = measuredSavings.length ? measuredSavings.reduce((a, b) => a + b, 0) / measuredSavings.length : null;
  return <AppShell active="runs"><div className="page">
    <header className="page-heading">
      <div className="page-intro"><h1>Runs</h1><p>Every policy change Brier evaluated, newest first.</p></div>
      <nav className="inline-flex rounded-md border bg-white p-1 text-xs" aria-label="Filter runs by environment">
        {[["", "All"], ["pusht", "PushT"], ["aloha_transfer_cube", "ALOHA"]].map(([value, label]) => <Link key={label} href={value ? `/runs?env=${value}` : "/runs"} aria-current={selected === value ? "page" : undefined} className={`rounded px-3 py-2 font-medium ${selected === value ? "bg-primary text-white" : "hover:bg-accent hover:text-primary"}`}>{label}</Link>)}
      </nav>
    </header>
    <div className="status-strip"><span className="status-dot" aria-hidden="true" /><strong>{rows.length ? "Recorded risk targets" : "Evaluation ledger"}</strong><span className="text-muted-foreground">{rows.length ? "Targets are calibrated under exchangeable changes." : "Real decisions and their evidence, as they are received."}</span>{rows[0] && <span className="ml-auto font-mono text-[10px] text-muted-foreground">latest α {rows[0].evaluation.alpha.toFixed(3)} · λ {rows[0].evaluation.lambda.toFixed(3)}</span>}</div>
    <section className="stats-grid" aria-label="Summary of the evaluations shown">
      <div className="stat"><p className="eyebrow">Evaluations shown</p><p className="metric">{connected ? rows.length : "—"}</p><p className="stat-note">latest 50 matching policy changes</p></div>
      <div className="stat"><p className="eyebrow">Blocked changes</p><p className="metric">{connected ? blocked : "—"}</p><p className="stat-note">blocked verdicts in this view</p></div>
      <div className="stat"><p className="eyebrow">Expensive evaluations</p><p className="metric">{connected ? <>{run}<span className="text-lg"> / {total}</span></> : "—"}</p><p className="stat-note">completed across the runs shown</p></div>
      <div className="stat stat-accent"><p className="eyebrow">Eval-hours saved</p><p className="metric">{averageSaved === null ? "—" : `${averageSaved.toFixed(1)}%`}</p><p className="stat-note">{measuredSavings.length ? `mean across ${measuredSavings.length} measured runs` : "awaiting measured rollout durations"}</p></div>
    </section>
    {rows.length ? <section className="table-panel" aria-labelledby="evaluations-heading"><div className="table-panel-heading"><h2 id="evaluations-heading">Evaluations</h2><small className="font-mono">latest {rows.length}</small></div><Table>
      <TableHeader><TableRow><TableHead>Change</TableHead><TableHead>Environment</TableHead><TableHead>Verdict</TableHead><TableHead>Expensive evals</TableHead><TableHead>Hours saved</TableHead><TableHead>Date · UTC</TableHead><TableHead><span className="sr-only">Open run</span></TableHead></TableRow></TableHeader>
      <TableBody>{rows.map(({ evaluation, change }) => <TableRow key={evaluation.id}>
        <TableCell className="min-w-52"><Link className="font-medium hover:text-primary" href={`/runs/${evaluation.id}`}>{change.title}</Link><div className="mt-1.5 font-mono text-[10px] text-muted-foreground">{change.id}</div></TableCell>
        <TableCell className="text-[11px]">{change.env === "pusht" ? "PushT" : "ALOHA"}</TableCell>
        <TableCell><Badge variant="outline" className={`status-${evaluation.verdict}`}>{evaluation.verdict.toUpperCase()}</Badge></TableCell>
        <TableCell><div className="probability"><span className="probability-track" aria-hidden="true"><span className="probability-fill" style={{ width: `${evaluation.evalsTotal ? evaluation.evalsRun / evaluation.evalsTotal * 100 : 0}%` }} /></span><span className="font-mono">{evaluation.evalsRun} / {evaluation.evalsTotal}</span></div></TableCell>
        <TableCell className="font-mono text-[11px]">{evaluation.hoursSavedPct === null ? "Pending" : `${evaluation.hoursSavedPct.toFixed(1)}%`}</TableCell>
        <TableCell className="font-mono text-[10px] text-muted-foreground">{new Date(evaluation.createdAt).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}</TableCell>
        <TableCell><Link href={`/runs/${evaluation.id}`} className="subtle-link p-2" aria-label={`Open evaluation for ${change.title}`}><ArrowUpRight className="size-3.5" aria-hidden="true" /></Link></TableCell>
      </TableRow>)}</TableBody>
    </Table></section> : <Pending title={connected ? "No evaluations here yet" : "Connect your evaluation ledger"}>{connected ? "No runs match this environment. Completed evaluations appear here after their real results are ingested." : "Connect the platform database and ingest a completed evaluation to see decisions, scenario outcomes, and failure evidence."}</Pending>}
    <p className="data-note">Summary values describe only the runs displayed above. Hours saved are based on measured rollout durations; absent measurements remain pending.</p>
  </div></AppShell>;
}
