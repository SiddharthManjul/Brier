import type { Metadata } from "next";
import { desc, eq } from "drizzle-orm";
import { ArrowUpRight } from "lucide-react";
import { reports } from "@/db/schema";
import type { Json } from "@/db/schema";
import { getDb } from "@/lib/db";
import { metric, readGuide3, readStrategies } from "@/lib/report";
import { AppShell } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Pending } from "@/components/pending";
import { ModelCharts } from "@/components/model-charts";

export const metadata: Metadata = { title: "Models" };
export const dynamic = "force-dynamic";

function display(value: Json | undefined, percent = false): string {
  const measured = metric(value);
  return measured === null ? "Pending" : percent ? `${(measured * 100).toFixed(1)}%` : measured.toFixed(3);
}

/** Show held-out robot model measurements, calibration, and evaluation tradeoffs. */
export default async function Models() {
  const [report] = process.env.DATABASE_URL ? await getDb().select().from(reports)
    .where(eq(reports.suite, "robot")).orderBy(desc(reports.createdAt)).limit(1) : [];
  const strategies = report ? readStrategies(report.body) : [];
  const diagnostic = report ? readGuide3(report.body) : null;
  const featured = strategies.find(({ name }) => name === "Full cascade") ?? strategies[0];
  const ece = featured ? metric(featured.metrics.ece) : null;
  const saved = featured ? metric(featured.metrics.eval_hours_saved) : null;
  const missed = featured ? metric(featured.metrics.missed_regression_fraction) : null;
  const coverage = featured ? metric(featured.metrics.guarantee_splits) : null;
  return <AppShell active="models"><div className="page">
    <header className="page-heading"><div className="page-intro"><h1>Models</h1><p>How well calibrated the decision models are, from measured robot replay.</p></div><Button variant="outline" size="sm" asChild><a href="https://github.com/SiddharthManjul/Brier#readme">Evaluation guide<ArrowUpRight aria-hidden="true" /></a></Button></header>
    <div className="flex items-center gap-3 border-b pb-3"><span className="border-b-2 border-primary pb-3 -mb-3.5 text-xs font-medium text-primary">Robot CI</span><Badge variant="outline" className="status-pending">MEASURED REPORTS</Badge></div>
    {diagnostic && <section className="table-panel" aria-labelledby="guide3-heading">
      <div className="table-panel-heading"><h2 id="guide3-heading">Guide 3 · policy diagnostics</h2><Badge variant="outline" className="status-pending">REPLAY DIAGNOSTIC</Badge></div>
      <div className="border-b p-5 text-xs leading-7"><p>{diagnostic.heldoutChanges} independent held-out changes · {diagnostic.guarantee}-level risk · {diagnostic.waveSize}-job waves. Costs include cheap evaluations, expensive evaluations and exploration.</p>
        <p>CRC denominator: {diagnostic.denominator}. {diagnostic.guarantee === "change" ? <>The change-level bound measures missed regressing changes divided by <strong>all changes</strong>.</> : <>Scenario-level CRC bounds the mean per-change fraction of regressed scenarios not selected. Selected scenarios run without early stop.</>} Conditional misses among regressing changes are measured separately.</p>
        {diagnostic.alphas.some((row) => !row.passed) && <p className="mt-2 font-medium text-destructive">Replay acceptance failed for one or more alpha values. These are diagnostic savings; they do not establish a qualified savings claim.</p>}
      </div>
      <Table><TableHeader><TableRow><TableHead>α / level</TableHead><TableHead>Production λ</TableHead><TableHead>CRC bound</TableHead><TableHead>Replay gate</TableHead><TableHead>Eval-hours saved · mean / worst</TableHead><TableHead>Missed / all changes · mean / worst</TableHead><TableHead>Missed / regressing changes · mean / worst</TableHead><TableHead>Scenario misses · mean / worst</TableHead></TableRow></TableHeader>
        <TableBody>{diagnostic.alphas.map((row) => <TableRow key={row.alpha}>
          <TableCell>{display(row.alpha, true)} / {diagnostic.guarantee}</TableCell><TableCell className="font-mono text-[10px]">{row.lambda.toFixed(6)}</TableCell>
          <TableCell>{display(row.riskBound, true)} · {row.certified ? "CRC feasible" : "uncertified; run all"}</TableCell>
          <TableCell className={row.passed ? "" : "text-destructive"}>{row.within} / {row.total} · {row.passed ? "passed" : "failed"} (requires {row.required})</TableCell>
          {[row.savings, row.allChangeMiss, row.conditionalMiss, row.scenarioMiss].map((value, index) => <TableCell key={index} className="font-mono text-[10px]">{display(value.mean, true)} / {display(value.worst, true)}</TableCell>)}
        </TableRow>)}</TableBody>
      </Table>
      <div className="border-t p-5"><p className="eyebrow">Cost accounting · mean share of run-all expensive cost</p>
        {diagnostic.alphas.map((row) => <div key={row.alpha} className="mt-4 text-xs leading-7"><strong>α {display(row.alpha, true)}</strong><p>T0 credit: {display(row.breakdown.t0_saved_seconds, true)}{diagnostic.t0Verified ? " (verified)" : " (not verified; no credit)"} · adaptive cheap cost avoided: {display(row.breakdown.adaptive_cheap_saved_seconds, true)} · threshold cost avoided: {display(row.breakdown.threshold_saved_seconds, true)} · early-stop cost avoided: {display(row.breakdown.early_stop_saved_seconds, true)}</p><p>Cheap cost used: {display(row.breakdown.cheap_overhead_seconds, true)} · exploration cost added: {display(row.breakdown.exploration_selected_overhead_seconds, true)}. Adaptive cheap savings are relative to running every cheap evaluation; the total above already includes all overhead.</p></div>)}
      </div>
    </section>}
    <div><h2 className="text-lg">Frozen held-out test ablation</h2><p className="mt-1 text-xs text-muted-foreground">The original model comparison below remains separate from pooled Guide 3 policy diagnostics.</p></div>
    <section className="stats-grid" aria-label="Measured strategy summary">
      <div className="stat"><p className="eyebrow">Test calibration · ECE</p><p className="metric">{ece === null ? "—" : ece.toFixed(3)}</p><p className="stat-note">{featured?.name ?? (diagnostic ? "No frozen test ablation in this report" : "No report connected")} · target ≤ 0.05</p></div>
      <div className="stat stat-accent"><p className="eyebrow">Eval-hours saved</p><p className="metric">{saved === null ? "—" : `${(saved * 100).toFixed(1)}%`}</p><p className="stat-note">{saved === null ? "awaiting measured replay" : "versus running every scenario"}</p></div>
      <div className="stat"><p className="eyebrow">Splits within target</p><p className="metric">{coverage === null ? "—" : <>{coverage}<span className="text-lg"> / 20</span></>}</p><p className="stat-note">empirical risk-target coverage</p></div>
      <div className="stat"><p className="eyebrow">Missed regressions</p><p className="metric">{missed === null ? "—" : `${(missed * 100).toFixed(1)}%`}</p><p className="stat-note">{missed === null ? "awaiting measured replay" : "observed missed-regression fraction"}</p></div>
    </section>
    {featured && <div className="status-strip"><span className="status-dot" aria-hidden="true" /><span>Summary strategy: <strong>{featured.name}</strong>.{ece !== null && ece > 0.05 ? " Test calibration error is above the 0.05 project target." : " Compare all strategies and held-out results below."}</span></div>}
    {strategies.length ? <>
      <ModelCharts strategies={strategies.map(({ name, reliability, curve }) => ({ name, reliability, curve }))} />
      <section className="table-panel" aria-labelledby="ablation-heading"><div className="table-panel-heading"><h2 id="ablation-heading">Ablation</h2><small className="font-mono">held-out test + OOD</small></div><Table><TableHeader><TableRow>
        <TableHead>Strategy</TableHead><TableHead>Test ECE</TableHead><TableHead>AUROC</TableHead><TableHead>Eval-hours saved</TableHead><TableHead>Change recall</TableHead><TableHead>Splits within α</TableHead><TableHead>OOD ECE</TableHead><TableHead>Test Brier</TableHead><TableHead>Test NLL</TableHead><TableHead>Missed</TableHead><TableHead>p50 / p95 · s</TableHead><TableHead>Cost / change · $</TableHead>
      </TableRow></TableHeader><TableBody>{strategies.map(({ name, metrics, oodEce }) => <TableRow key={name}>
        <TableCell className={`min-w-48 font-medium ${name === featured?.name ? "text-primary" : ""}`}>{name}</TableCell>
        <TableCell className="font-mono text-[10px]">{display(metrics.ece)}</TableCell><TableCell className="font-mono text-[10px]">{display(metrics.auroc)}</TableCell>
        <TableCell className="font-mono text-[10px]">{display(metrics.eval_hours_saved, true)}</TableCell><TableCell className="font-mono text-[10px]">{display(metrics.change_recall, true)}</TableCell><TableCell className="font-mono text-[10px]">{metrics.guarantee_splits == null ? "Pending" : `${metric(metrics.guarantee_splits)} / 20`}</TableCell>
        <TableCell className="font-mono text-[10px]">{oodEce === null ? "Pending" : oodEce.toFixed(3)}</TableCell><TableCell className="font-mono text-[10px]">{display(metrics.brier)}</TableCell><TableCell className="font-mono text-[10px]">{display(metrics.nll)}</TableCell>
        <TableCell className="font-mono text-[10px]">{display(metrics.missed_regression_fraction, true)}</TableCell><TableCell className="font-mono text-[10px]">{display(metrics.p50_latency_seconds)} / {display(metrics.p95_latency_seconds)}</TableCell><TableCell className="font-mono text-[10px]">{display(metrics.cost_usd_per_change)}</TableCell>
      </TableRow>)}</TableBody></Table></section>
      <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]"><section className="rounded-md border bg-white p-6"><p className="eyebrow">Read the evidence</p><h2 className="mt-3 text-2xl">Calibration before confidence.</h2><p className="mt-3 text-xs leading-7 text-muted-foreground">Test and OOD metrics are held apart. A lower calibration error does not by itself establish a missed-regression bound. Risk targets assume exchangeable changes, with replay coverage reported as an empirical check.</p></section><section className="rounded-md border bg-white p-6"><p className="eyebrow">Report provenance</p><dl className="mt-4 space-y-3 text-[11px]"><div className="flex justify-between gap-4 border-b pb-3"><dt className="text-muted-foreground">Suite</dt><dd>Robot CI</dd></div><div className="flex justify-between gap-4 border-b pb-3"><dt className="text-muted-foreground">Strategies</dt><dd className="font-mono">{strategies.length}</dd></div><div className="flex justify-between gap-4"><dt className="text-muted-foreground">Received · UTC</dt><dd className="font-mono">{new Date(report!.createdAt).toLocaleString("en-GB", { timeZone: "UTC" })}</dd></div></dl></section></div>
      <p className="data-note">Missing measurements remain pending. Calibration bins, curves, and summary values come only from the latest ingested robot report.</p>
    </> : <Pending title={diagnostic ? "Frozen test ablation not included" : "Your model measurements belong here"}>{diagnostic ? "The latest report contains pooled policy diagnostics. Its measurements are not substituted for the frozen test comparison." : "Connect the database and ingest the measured robot replay report to see calibration plots, held-out results, and evaluation savings. Training results are not substituted for replay measurements."}</Pending>}
  </div></AppShell>;
}
