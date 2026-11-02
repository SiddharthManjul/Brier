import Link from "next/link";
import { ArrowUpRight, BookOpen, ChartNoAxesCombined, House, List, Workflow } from "lucide-react";

/** Frame the evaluation workspace with the four scoped routes and current navigation. */
export function AppShell({ active, children }: { active: "runs" | "models"; children: React.ReactNode }) {
  return <div className="app-shell">
    <aside className="app-sidebar">
      <Link className="brand" href="/" aria-label="Brier home"><span className="brand-mark" aria-hidden="true" />brier</Link>
      <div className="workspace-label"><span className="workspace-icon"><Workflow className="size-4" aria-hidden="true" /></span><div>Robot CI<small>Simulation workspace</small></div></div>
      <nav className="app-nav" aria-label="Main navigation">
        <Link href="/"><House aria-hidden="true" />Overview</Link>
        <Link href="/runs" aria-current={active === "runs" ? "page" : undefined}><List aria-hidden="true" />Runs</Link>
        <Link href="/models" aria-current={active === "models" ? "page" : undefined}><ChartNoAxesCombined aria-hidden="true" />Models</Link>
      </nav>
      <div className="sidebar-footer">
        <a href="https://github.com/SiddharthManjul/Brier#readme"><BookOpen aria-hidden="true" />Documentation<ArrowUpRight className="ml-auto size-3" aria-hidden="true" /></a>
        <div className="sidebar-note">PushT · ALOHA transfer cube<br />Every decision, with its evidence.</div>
      </div>
    </aside>
    <div className="app-main"><main id="main">{children}</main><footer className="app-footer"><span>Simulation evidence. Measured decisions.</span><a href="https://github.com/SiddharthManjul/Brier">Built in the open ↗</a></footer></div>
  </div>;
}
