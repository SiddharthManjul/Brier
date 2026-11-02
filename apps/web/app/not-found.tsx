import Link from "next/link";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";

/** Keep missing routes inside the evaluation workspace with a clear way back. */
export default function NotFound() {
  return <AppShell active="runs"><div className="page">
    <div className="empty-panel min-h-[65vh]">
      <div className="max-w-md text-center">
        <p className="eyebrow mb-5">404 · Outside the suite</p>
        <h1>This page landed off the diagonal.</h1>
        <p className="mt-5 text-sm leading-7 text-muted-foreground">We couldn&apos;t find that page. Your evaluations and their evidence are still in the workspace.</p>
        <div className="mt-7 flex flex-wrap justify-center gap-3">
          <Button asChild><Link href="/runs"><ArrowLeft aria-hidden="true" />Back to runs</Link></Button>
          <Button asChild variant="outline"><Link href="/">Brier home<ArrowUpRight aria-hidden="true" /></Link></Button>
        </div>
      </div>
    </div>
  </div></AppShell>;
}
