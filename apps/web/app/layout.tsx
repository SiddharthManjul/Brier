import type { Metadata } from "next";
import Link from "next/link";
import { Activity, ArrowUpRight } from "lucide-react";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Brier · Robot CI", template: "%s · Brier" },
  description: "Measured robot evaluations, calibrated decisions, and the evidence behind every pass or block.",
};
export const runtime = "nodejs";

/** Frame the four public platform pages with shared navigation. */
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className="min-h-screen antialiased">
    <a className="sr-only focus:not-sr-only focus:block focus:p-3" href="#main">Skip to content</a>
    <header className="border-b border-border/70">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-5 sm:px-8">
        <Link href="/" className="flex items-center gap-3 text-xl font-semibold tracking-tight">
          <Activity className="size-6 text-primary" aria-hidden="true" />Brier
          <span className="border-l pl-3 text-xs font-medium tracking-wide text-muted-foreground">ROBOT CI</span>
        </Link>
        <nav aria-label="Main navigation" className="flex items-center gap-5 text-sm font-medium">
          <Link className="hover:text-primary" href="/">Overview</Link>
          <Link className="hover:text-primary" href="/runs">Runs</Link>
          <Link className="hover:text-primary" href="/models">Models</Link>
          <a className="flex items-center gap-1 text-muted-foreground hover:text-primary" href="https://github.com/SiddharthManjul/Brier">
            GitHub<ArrowUpRight className="size-3" aria-hidden="true" />
          </a>
        </nav>
      </div>
    </header>
    <main id="main">{children}</main>
    <footer className="mx-auto flex max-w-7xl flex-wrap justify-between gap-3 border-t px-4 py-6 text-xs text-muted-foreground sm:px-8">
      <span>Simulation evidence. Measured decisions.</span><span>PushT · AlohaTransferCube</span>
    </footer>
  </body></html>;
}
