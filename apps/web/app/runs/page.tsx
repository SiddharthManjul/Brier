import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { desc, eq, inArray } from "drizzle-orm";
import { changes, evaluations } from "@/db/schema";
import { getDb } from "@/lib/db";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Pending } from "@/components/pending";

export const metadata: Metadata = { title: "Runs" };
export const dynamic = "force-dynamic";

/** List at most fifty evaluations with one environment filter. */
export default async function Runs({ searchParams }: { searchParams: Promise<{ env?: string }> }) {
  const { env } = await searchParams;
  if (env !== undefined && env !== "" && env !== "pusht" && env !== "aloha_transfer_cube") notFound();
  const selected = env === "pusht" || env === "aloha_transfer_cube" ? env : "";
  const rows = process.env.DATABASE_URL ? await getDb().select({ evaluation: evaluations, change: changes })
    .from(evaluations).innerJoin(changes, eq(evaluations.changeId, changes.id))
    .where(selected ? eq(changes.env, selected) : inArray(changes.env, ["pusht", "aloha_transfer_cube"]))
    .orderBy(desc(evaluations.createdAt)).limit(50) : [];
  return <div className="page">
    <div className="flex flex-wrap items-end justify-between gap-5">
      <div className="page-intro"><p className="eyebrow">Evaluation ledger</p><h1>Runs</h1><p className="text-muted-foreground">The latest 50 decisions and the evaluation time behind them.</p></div>
      <form method="get" className="flex items-end gap-2">
        <div className="space-y-2"><label htmlFor="env" className="text-xs font-medium">Environment</label>
          <NativeSelect name="env" id="env" defaultValue={selected}>
            <NativeSelectOption value="">All environments</NativeSelectOption>
            <NativeSelectOption value="pusht">PushT</NativeSelectOption>
            <NativeSelectOption value="aloha_transfer_cube">AlohaTransferCube</NativeSelectOption>
          </NativeSelect></div>
        <Button variant="outline" type="submit">Apply</Button>
      </form>
    </div>
    {rows.length ? <div className="rounded-lg border bg-card"><Table>
      <TableHeader><TableRow><TableHead>Change</TableHead><TableHead>Environment</TableHead><TableHead>Verdict</TableHead><TableHead>Evaluations</TableHead><TableHead>Hours saved</TableHead><TableHead>Date (UTC)</TableHead></TableRow></TableHeader>
      <TableBody>{rows.map(({ evaluation, change }) => <TableRow key={evaluation.id}>
        <TableCell><Link className="font-medium underline-offset-4 hover:underline" href={`/runs/${evaluation.id}`}>{change.title}</Link><div className="mt-1 font-mono text-xs text-muted-foreground">{change.id}</div></TableCell>
        <TableCell>{change.env === "pusht" ? "PushT" : "AlohaTransferCube"}</TableCell>
        <TableCell><Badge variant="outline" className={`status-${evaluation.verdict}`}>{evaluation.verdict.toUpperCase()}</Badge></TableCell>
        <TableCell className="font-mono tabular-nums">{evaluation.evalsRun} / {evaluation.evalsTotal}</TableCell>
        <TableCell className="font-mono tabular-nums">{evaluation.hoursSavedPct === null ? "Pending" : `${evaluation.hoursSavedPct.toFixed(1)}%`}</TableCell>
        <TableCell className="text-muted-foreground">{new Date(evaluation.createdAt).toLocaleString("en-GB", { timeZone: "UTC" })}</TableCell>
      </TableRow>)}</TableBody>
    </Table></div> : <Pending title="No evaluations available">{process.env.DATABASE_URL ? "No runs match this environment. Completed evaluations will appear here after ingest." : "Evaluation data is pending platform database setup and the first completed run."}</Pending>}
  </div>;
}
