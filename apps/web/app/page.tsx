import Link from "next/link";
import { desc, eq } from "drizzle-orm";
import { ArrowRight, FlaskConical, GitPullRequest, ShieldCheck } from "lucide-react";
import { reports } from "@/db/schema";
import { getDb } from "@/lib/db";
import { metric, readStrategies } from "@/lib/report";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Pending } from "@/components/pending";

export const dynamic = "force-dynamic";

/** Present the project and the latest measured robot report. */
export default async function Home() {
  const [latest] = process.env.DATABASE_URL ? await getDb().select().from(reports)
    .where(eq(reports.suite, "robot")).orderBy(desc(reports.createdAt)).limit(1) : [];
  const strategies = latest ? readStrategies(latest.body) : [];
  return <div className="page">
    <section className="grid gap-10 py-8 lg:grid-cols-[1.3fr_1fr] lg:py-14">
      <div className="space-y-6">
        <Badge variant="outline" className="font-mono text-xs">CALIBRATED EVALUATION / ROBOT CI</Badge>
        <h1 className="max-w-3xl font-serif text-5xl leading-[1.05] font-normal sm:text-6xl">Every change.<br />The right evaluations.</h1>
        <p className="max-w-xl text-lg leading-relaxed text-muted-foreground">Brier ranks robot policy checks, runs the expensive cases, and keeps the evidence behind each pass or block.</p>
        <div className="flex flex-wrap gap-3">
          <Button asChild><Link href="/runs">Inspect evaluations<ArrowRight aria-hidden="true" /></Link></Button>
          <Button variant="outline" asChild><Link href="/models">View measured results</Link></Button>
        </div>
      </div>
      <Card className="self-center shadow-none">
        <CardHeader><p className="eyebrow">Latest robot report</p><CardTitle>Evidence, before claims.</CardTitle>
          <CardDescription>{latest ? `Received ${new Date(latest.createdAt).toLocaleDateString("en-GB", { timeZone: "UTC" })}` : "Awaiting the first measured report."}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {strategies.length ? <>
            <div className="metric">{strategies.length}<span className="ml-3 text-sm font-sans font-normal text-muted-foreground">measured strategies</span></div>
            <div className="space-y-3 border-t pt-4">{strategies.map(({ name, metrics }) => {
              const saved = metric(metrics.eval_hours_saved), missed = metric(metrics.missed_regression_fraction);
              return <div key={name} className="flex flex-wrap justify-between gap-2 text-sm"><span className="font-medium">{name}</span><span className="text-muted-foreground">
                {saved === null ? "Savings pending" : `${(saved * 100).toFixed(1)}% hours saved`}
                {missed === null ? " · miss rate pending" : ` · ${(missed * 100).toFixed(1)}% missed`}
              </span></div>;
            })}</div>
          </> : <p className="text-sm leading-relaxed text-muted-foreground">Rollouts and model evaluation are pending. Savings, calibration, and risk measurements will appear after a report is ingested.</p>}
        </CardContent>
      </Card>
    </section>
    <section aria-label="Evaluation process" className="grid gap-4 md:grid-cols-3">
      {[
        { icon: GitPullRequest, title: "Describe the change", text: "A structured policy change is evaluated against fixed simulation scenarios." },
        { icon: FlaskConical, title: "Spend evaluation time deliberately", text: "Cheap rollouts and calibrated predictions select the expensive checks." },
        { icon: ShieldCheck, title: "Inspect the evidence", text: "Review scenario outcomes, failure frames, and explanations alongside the risk target." },
      ].map(({ icon: Icon, title, text }) => <Card key={title} className="shadow-none"><CardHeader>
        <Icon className="mb-3 size-5 text-primary" aria-hidden="true" /><CardTitle className="text-base">{title}</CardTitle><CardDescription className="leading-relaxed">{text}</CardDescription>
      </CardHeader></Card>)}
    </section>
    <section className="space-y-4"><div className="page-intro"><p className="eyebrow">See a complete decision</p><h2>From policy change to an explained result.</h2></div>
      <Pending title="Demo video pending">A recorded walkthrough will be linked when the end-to-end robot evaluation has been completed.</Pending>
    </section>
  </div>;
}
