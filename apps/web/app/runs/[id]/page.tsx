import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { asc, eq } from "drizzle-orm";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import { changes, evaluations, failures, pairs } from "@/db/schema";
import { getDb } from "@/lib/db";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Pending } from "@/components/pending";

export const metadata: Metadata = { title: "Evaluation details" };
export const dynamic = "force-dynamic";

/** Show one evaluation, its risk target, scenarios, and observed failure evidence. */
export default async function Run({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id)) notFound();
  if (!process.env.DATABASE_URL) return <div className="page"><Pending title="Evaluation data pending">This run will be available after the platform database is connected and the evaluation is ingested.</Pending></div>;
  const db = getDb();
  const [row] = await db.select({ evaluation: evaluations, change: changes }).from(evaluations)
    .innerJoin(changes, eq(evaluations.changeId, changes.id)).where(eq(evaluations.id, id)).limit(1);
  if (!row) notFound();
  const scenarios = await db.select().from(pairs).where(eq(pairs.evaluationId, id)).orderBy(asc(pairs.scenarioId));
  const gallery = await db.select({ failure: failures, scenarioId: pairs.scenarioId }).from(failures)
    .innerJoin(pairs, eq(failures.pairId, pairs.id)).where(eq(pairs.evaluationId, id)).orderBy(asc(pairs.scenarioId));
  const { evaluation, change } = row;
  return <div className="page">
    <Link className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary" href="/runs"><ArrowLeft className="size-4" aria-hidden="true" />All runs</Link>
    <header className="flex flex-wrap items-start justify-between gap-5"><div className="page-intro">
      <p className="eyebrow">{change.id} / {change.env}</p><h1>{change.title}</h1>
      <p className="break-all font-mono text-xs text-muted-foreground">{evaluation.id}</p>
    </div><Badge variant="outline" className={`status-${evaluation.verdict} px-4 py-2 text-base`}>{evaluation.verdict.toUpperCase()}</Badge></header>
    <Card className="shadow-none"><CardHeader><p className="eyebrow">Configured risk target</p>
      <CardTitle>Missed-regression loss ≤ {(evaluation.alpha * 100).toFixed(1)}%</CardTitle>
      <CardDescription>Threshold λ = {evaluation.lambda.toFixed(3)}. This target assumes exchangeable changes and a calibrated policy.</CardDescription>
    </CardHeader><CardContent className="flex flex-wrap gap-8 border-t pt-5">
      <div><p className="metric">{evaluation.evalsRun}<span className="text-base text-muted-foreground"> / {evaluation.evalsTotal}</span></p><p className="mt-1 text-xs text-muted-foreground">expensive evaluations run</p></div>
      <div><p className="metric">{evaluation.hoursSavedPct === null ? "Pending" : `${evaluation.hoursSavedPct.toFixed(1)}%`}</p><p className="mt-1 text-xs text-muted-foreground">evaluation hours saved</p></div>
      <div className="max-w-sm space-y-2 text-xs text-muted-foreground"><p className="break-all">Model: {evaluation.modelVersion}</p><p>{new Date(evaluation.createdAt).toLocaleString("en-GB", { timeZone: "UTC" })} UTC</p>
        {change.prUrl && <a className="inline-flex items-center gap-1 text-primary hover:underline" href={change.prUrl}>View policy PR<ArrowUpRight className="size-3" aria-hidden="true" /></a>}
      </div>
    </CardContent></Card>
    <section className="space-y-4"><h2>Scenario outcomes</h2>
      <div className="rounded-lg border bg-card"><Table><TableHeader><TableRow>
        <TableHead>Scenario</TableHead><TableHead>p(regress)</TableHead><TableHead>Action</TableHead><TableHead>Result</TableHead><TableHead>Baseline success</TableHead><TableHead>Changed success</TableHead>
      </TableRow></TableHeader><TableBody>{scenarios.map((scenario) => <TableRow key={scenario.id}>
        <TableCell className="font-mono">{scenario.scenarioId}</TableCell><TableCell className="font-mono">{(scenario.pRegress * 100).toFixed(1)}%</TableCell>
        <TableCell>{scenario.action}</TableCell><TableCell>{scenario.result ?? "Not run"}</TableCell>
        <TableCell>{scenario.successBefore === null ? "Pending" : `${(scenario.successBefore * 100).toFixed(1)}%`}</TableCell>
        <TableCell>{scenario.successAfter === null ? "Pending" : `${(scenario.successAfter * 100).toFixed(1)}%`}</TableCell>
      </TableRow>)}</TableBody></Table></div>
    </section>
    <section className="space-y-4"><div className="page-intro"><p className="eyebrow">Observed evidence</p><h2>Failure gallery</h2></div>
      {gallery.length ? gallery.map(({ failure, scenarioId }) => <Card key={failure.id} className="shadow-none">
        <CardHeader><CardTitle className="font-mono text-base">{scenarioId}</CardTitle></CardHeader>
        <CardContent className="space-y-6"><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {failure.keyframeUrls.map((url, index) => <figure key={`frame-${index}`}><Image unoptimized src={url} width={720} height={480} alt={`${scenarioId} failure keyframe ${index + 1}`} className="h-auto w-full rounded border" /><figcaption className="mt-2 text-xs text-muted-foreground">Keyframe {index + 1}</figcaption></figure>)}
          {failure.overlayUrls.map((url, index) => <figure key={`overlay-${index}`}><Image unoptimized src={url} width={720} height={480} alt={`${scenarioId} SAM overlay ${index + 1}`} className="h-auto w-full rounded border" /><figcaption className="mt-2 text-xs text-muted-foreground">SAM overlay {index + 1}</figcaption></figure>)}
        </div><dl className="space-y-4">{Object.entries(failure.explanation).map(([key, value]) => <div key={key}>
          <dt className="mb-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">{key.replaceAll("_", " ")}</dt>
          <dd className="break-words whitespace-pre-wrap text-sm leading-relaxed">{typeof value === "string" ? value : JSON.stringify(value, null, 2)}</dd>
        </div>)}</dl></CardContent>
      </Card>) : <Pending title="No failure evidence attached">Failure keyframes, SAM overlays, and explanations appear here when attached to an evaluation.</Pending>}
    </section>
  </div>;
}
