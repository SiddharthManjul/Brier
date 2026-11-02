import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { Card } from "@/components/ui/card";

/** Show a purposeful empty state while keeping absent measurements explicit. */
export function Pending({ title, children }: { title: string; children: React.ReactNode }) {
  return <Card className="empty-panel shadow-none" role="status"><div>
    <div className="empty-mark" aria-hidden="true"><i /><i /><i /><i /></div>
    <h2>{title}</h2><p>{children}</p>
    <Link className="subtle-link mt-5" href="https://github.com/SiddharthManjul/Brier#readme">Read the setup guide<ArrowUpRight className="size-3" aria-hidden="true" /></Link>
  </div></Card>;
}
