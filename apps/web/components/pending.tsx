import { Clock3 } from "lucide-react";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

/** Show an explicit pending state without fabricated measurements. */
export function Pending({ title, children }: { title: string; children: React.ReactNode }) {
  return <Card className="border-dashed shadow-none"><CardHeader>
    <CardTitle className="flex items-center gap-2 text-base"><Clock3 className="size-4 text-muted-foreground" aria-hidden="true" />{title}</CardTitle>
    <CardDescription className="max-w-2xl leading-relaxed">{children}</CardDescription>
  </CardHeader></Card>;
}
