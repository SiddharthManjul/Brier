import type { Metadata } from "next";
import { desc, eq } from "drizzle-orm";
import { reports } from "@/db/schema";
import type { Json } from "@/db/schema";
import { getDb } from "@/lib/db";
import { metric, readStrategies } from "@/lib/report";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Pending } from "@/components/pending";
import { ModelCharts } from "@/components/model-charts";

export const metadata: Metadata = { title: "Models" };
export const dynamic = "force-dynamic";

function display(value: Json | undefined, percent = false): string {
  const measured = metric(value);
  return measured === null ? "Pending" : percent ? `${(measured * 100).toFixed(1)}%` : measured.toFixed(3);
}

/** Present measured robot ablations, calibration bins, and evaluation tradeoffs. */
export default async function Models() {
  const [report] = process.env.DATABASE_URL ? await getDb().select().from(reports)
    .where(eq(reports.suite, "robot")).orderBy(desc(reports.createdAt)).limit(1) : [];
  const strategies = report ? readStrategies(report.body) : [];
  return <div className="page">
    <header className="page-intro"><p className="eyebrow">Measurement over intuition</p><h1>Models</h1>
      <p className="text-muted-foreground">Robot CI ablations from held-out evaluations. Savings are read alongside misses and calibration.</p>
      <Badge variant="outline">Robot CI</Badge>
    </header>
    {strategies.length ? <>
      <section className="space-y-4"><div className="flex flex-wrap items-baseline justify-between gap-2"><h2>Ablation results</h2>
        <p className="text-xs text-muted-foreground">Report received {new Date(report!.createdAt).toLocaleString("en-GB", { timeZone: "UTC" })} UTC</p>
      </div><div className="rounded-lg border bg-card"><Table><TableHeader><TableRow>
        <TableHead>Strategy</TableHead><TableHead>Test ECE</TableHead><TableHead>OOD ECE</TableHead><TableHead>Test Brier</TableHead><TableHead>Test NLL</TableHead><TableHead>Test AUROC</TableHead>
        <TableHead>Hours saved</TableHead><TableHead>Missed</TableHead><TableHead>Change recall</TableHead><TableHead>Splits within α</TableHead><TableHead>p50 / p95 (s)</TableHead><TableHead>Cost / change ($)</TableHead>
      </TableRow></TableHeader><TableBody>{strategies.map(({ name, metrics, oodEce }) => <TableRow key={name}>
        <TableCell className="font-medium">{name}</TableCell>
        <TableCell className="font-mono">{display(metrics.ece)}</TableCell><TableCell className="font-mono">{oodEce === null ? "Pending" : oodEce.toFixed(3)}</TableCell>
        <TableCell className="font-mono">{display(metrics.brier)}</TableCell><TableCell className="font-mono">{display(metrics.nll)}</TableCell><TableCell className="font-mono">{display(metrics.auroc)}</TableCell>
        <TableCell className="font-mono">{display(metrics.eval_hours_saved, true)}</TableCell><TableCell className="font-mono">{display(metrics.missed_regression_fraction, true)}</TableCell>
        <TableCell className="font-mono">{display(metrics.change_recall, true)}</TableCell><TableCell className="font-mono">{metrics.guarantee_splits == null ? "Pending" : `${metric(metrics.guarantee_splits)} / 20`}</TableCell>
        <TableCell className="font-mono">{display(metrics.p50_latency_seconds)} / {display(metrics.p95_latency_seconds)}</TableCell><TableCell className="font-mono">{display(metrics.cost_usd_per_change)}</TableCell>
      </TableRow>)}</TableBody></Table></div>
        <p className="text-xs leading-relaxed text-muted-foreground">Missing measurements remain pending. Risk targets assume exchangeable changes; replay split counts are empirical checks of that assumption.</p>
      </section>
      <ModelCharts strategies={strategies.map(({ name, reliability, curve }) => ({ name, reliability, curve }))} />
    </> : <Pending title="Model measurements pending">The ablation table and charts will be populated from the latest measured robot report. Cloud rollouts, training, and evaluation have not supplied results yet.</Pending>}
  </div>;
}
