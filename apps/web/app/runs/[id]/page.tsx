import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { asc, eq } from "drizzle-orm";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import { changes, evaluations, failures, pairs } from "@/db/schema";
import { getDb } from "@/lib/db";
import { AppShell } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Pending } from "@/components/pending";

export const metadata: Metadata = { title: "Evaluation details" };
export const dynamic = "force-dynamic";

/** Show one recorded decision, its scenarios, and the evidence behind observed failures. */
export default async function Run({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id)) notFound();
  if (!process.env.DATABASE_URL) return <AppShell active="runs"><div className="page"><Link className="subtle-link" href="/runs"><ArrowLeft className="size-3" aria-hidden="true" />All runs</Link><Pending title="Connect the evaluation ledger">This run will be available once the platform database is connected and its measured evaluation is ingested.</Pending></div></AppShell>;
  const db = getDb();
  const [row] = await db.select({ evaluation: evaluations, change: changes }).from(evaluations)
    .innerJoin(changes, eq(evaluations.changeId, changes.id)).where(eq(evaluations.id, id)).limit(1);
  if (!row) notFound();
  const scenarios = await db.select().from(pairs).where(eq(pairs.evaluationId, id)).orderBy(asc(pairs.scenarioId));
  const gallery = await db.select({ failure: failures, scenarioId: pairs.scenarioId }).from(failures)
    .innerJoin(pairs, eq(failures.pairId, pairs.id)).where(eq(pairs.evaluationId, id)).orderBy(asc(pairs.scenarioId));
  const { evaluation, change } = row;
  const confirmed = scenarios.filter((scenario) => scenario.result === "regressed" || scenario.result === "fail").length;
  return <AppShell active="runs"><div className="page">
    <header className="page-heading"><div className="page-intro"><Link className="subtle-link text-muted-foreground" href="/runs"><ArrowLeft className="size-3" aria-hidden="true" />Runs / {change.id}</Link><h1>{change.title}</h1><p><span className="font-mono text-[11px]">{change.env === "pusht" ? "PushT" : "ALOHA transfer cube"}</span> · {new Date(evaluation.createdAt).toLocaleString("en-GB", { timeZone: "UTC" })} UTC</p></div>{change.prUrl && <Button variant="outline" size="sm" asChild><a href={change.prUrl}>View pull request<ArrowUpRight aria-hidden="true" /></a></Button>}</header>
    <section className="run-summary" aria-label="Evaluation decision summary">
      <div><p className="eyebrow">Verdict</p><span className="value"><Badge variant="outline" className={`status-${evaluation.verdict}`}>{evaluation.verdict.toUpperCase()}</Badge></span></div>
      <div><p className="eyebrow">Expensive evals</p><span className="value font-mono">{evaluation.evalsRun} of {evaluation.evalsTotal}</span></div>
      <div><p className="eyebrow">Regressions</p><span className="value">{confirmed} confirmed</span></div>
      <div><p className="eyebrow">Hours saved</p><span className="value text-primary">{evaluation.hoursSavedPct === null ? "Pending" : `${evaluation.hoursSavedPct.toFixed(1)}%`}</span></div>
      <div><p className="eyebrow">Risk target</p><span className="value font-mono">α {(evaluation.alpha * 100).toFixed(1)}% · λ {evaluation.lambda.toFixed(3)}</span></div>
    </section>
    <section className="rounded-md border bg-white p-5" aria-labelledby="risk-heading"><div className="section-heading"><h2 id="risk-heading">p(regress) by scenario</h2><div className="flex flex-wrap gap-4 text-[10px] text-muted-foreground"><span>■ Regressed</span><span>● Completed</span><span>○ Not run</span></div></div>
      <div className="mt-5 overflow-x-auto"><svg className="block min-w-[480px] w-full" viewBox="0 0 820 145" role="img" aria-label="Scenario regression probabilities and the configured decision threshold">
        <line x1="30" y1="85" x2="790" y2="85" stroke="var(--border)" />
        {[0, 0.25, 0.5, 0.75, 1].map((value) => <g key={value}><line x1={30 + value * 760} y1="81" x2={30 + value * 760} y2="90" stroke="var(--border)" /><text x={30 + value * 760} y="121" textAnchor="middle" fill="var(--muted-foreground)" fontSize="9" fontFamily="monospace">{value.toFixed(2)}</text></g>)}
        <line x1={30 + evaluation.lambda * 760} y1="20" x2={30 + evaluation.lambda * 760} y2="96" stroke="var(--primary)" strokeDasharray="3 4" /><text x={Math.min(730, 36 + evaluation.lambda * 760)} y="16" fill="var(--primary)" fontSize="9" fontFamily="monospace">λ {evaluation.lambda.toFixed(3)}</text>
        {scenarios.map((scenario, index) => {
          const x = 30 + scenario.pRegress * 760, y = 85 - index % 5 * 11;
          const regressed = scenario.result === "regressed" || scenario.result === "fail";
          return <g key={scenario.id}><title>{scenario.scenarioId}: {(scenario.pRegress * 100).toFixed(1)}% predicted regression, {scenario.result ?? "not run"}</title>{regressed ? <rect x={x - 4} y={y - 4} width="8" height="8" fill="var(--primary)" /> : <circle cx={x} cy={y} r="4" fill={scenario.result === null ? "white" : "var(--primary)"} stroke={scenario.result === null ? "var(--chart-2)" : "var(--primary)"} />}</g>;
        })}
      </svg></div>
      <p className="data-note">Horizontal position shows the predicted probability. Vertical offsets separate overlapping points. The threshold targets expected missed regressions under exchangeable changes and a calibrated policy.</p>
    </section>
    <section className="table-panel" aria-labelledby="scenarios-heading"><div className="table-panel-heading"><h2 id="scenarios-heading">Scenarios</h2><small className="font-mono">{scenarios.length} scenarios · ordered by identifier</small></div><Table><TableHeader><TableRow>
      <TableHead>Scenario</TableHead><TableHead>p(regress)</TableHead><TableHead>Action</TableHead><TableHead>Result</TableHead><TableHead>Baseline success</TableHead><TableHead>Changed success</TableHead>
    </TableRow></TableHeader><TableBody>{scenarios.map((scenario) => <TableRow key={scenario.id}>
      <TableCell className="font-mono text-[10px]">{scenario.scenarioId}</TableCell><TableCell><div className="probability"><span className="probability-track" aria-hidden="true"><span className="probability-fill" style={{ width: `${scenario.pRegress * 100}%` }} /></span><span className="font-mono">{scenario.pRegress.toFixed(3)}</span></div></TableCell>
      <TableCell><Badge variant="outline" className={scenario.action === "skip" ? "status-pending" : "status-pass"}>{scenario.action.toUpperCase()}</Badge></TableCell><TableCell>{scenario.result === null ? <span className="text-[10px] text-muted-foreground">not run</span> : <Badge variant="outline" className={scenario.result === "regressed" || scenario.result === "fail" ? "status-block" : "status-pass"}>{scenario.result.toUpperCase()}</Badge>}</TableCell>
      <TableCell className="font-mono text-[10px]">{scenario.successBefore === null ? "Pending" : `${(scenario.successBefore * 100).toFixed(1)}%`}</TableCell><TableCell className="font-mono text-[10px]">{scenario.successAfter === null ? "Pending" : `${(scenario.successAfter * 100).toFixed(1)}%`}</TableCell>
    </TableRow>)}</TableBody></Table></section>
    <section className="space-y-4" aria-labelledby="failures-heading"><div className="section-heading"><h2 id="failures-heading" className="!font-serif !text-2xl">Failure evidence</h2><span className="data-note">Observed frames · masks · explanations</span></div>
      {gallery.length ? gallery.map(({ failure, scenarioId }) => <Card key={failure.id} className="shadow-none">
        <CardHeader className="flex-row items-center justify-between gap-3"><CardTitle className="text-sm">Why {scenarioId} failed</CardTitle><Badge variant="outline">OBSERVED EVIDENCE</Badge></CardHeader>
        <CardContent><div className="evidence-grid"><div><div className="evidence-frames">
          {failure.keyframeUrls.map((url, index) => <figure className="evidence-frame" key={`frame-${index}`}><Image unoptimized src={url} width={720} height={480} alt={`${scenarioId} observed failure keyframe ${index + 1}`} /><figcaption>Raw keyframe {index + 1}</figcaption></figure>)}
          {failure.overlayUrls.map((url, index) => <figure className="evidence-frame" key={`overlay-${index}`}><Image unoptimized src={url} width={720} height={480} alt={`${scenarioId} SAM mask overlay ${index + 1}`} /><figcaption>SAM overlay {index + 1}</figcaption></figure>)}
        </div>{!failure.keyframeUrls.length && !failure.overlayUrls.length && <p className="data-note">No image evidence was attached to this explanation.</p>}<p className="data-note mt-4">Only images attached to this evaluation are shown. Explanations describe the supplied evidence and may contain hypotheses.</p></div><dl className="evidence-explanation">{Object.entries(failure.explanation).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{typeof value === "string" ? value : JSON.stringify(value, null, 2)}</dd></div>)}</dl></div></CardContent>
      </Card>) : <Pending title="No failure evidence attached">Observed failure keyframes, SAM overlays, and explanations appear here when they are attached to this evaluation.</Pending>}
    </section>
    <div className="rounded-md border bg-white px-5 py-4"><p className="eyebrow">Decision provenance</p><p className="data-note mt-3 break-all">Model: <span className="font-mono">{evaluation.modelVersion}</span></p><p className="data-note mt-2 break-all">Evaluation: <span className="font-mono">{evaluation.id}</span></p></div>
  </div></AppShell>;
}
