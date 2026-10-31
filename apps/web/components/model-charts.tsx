"use client";

import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { Strategy } from "@/lib/report";

/** Plot observed calibration bins and threshold tradeoffs without client fetching. */
export function ModelCharts({ strategies }: {
  strategies: Pick<Strategy, "name" | "reliability" | "curve">[];
}) {
  const config: ChartConfig = Object.fromEntries(strategies.map((strategy, index) => [
    `m${index}`, { label: strategy.name, color: `var(--chart-${index % 5 + 1})` },
  ]));
  const reliabilityPresent = strategies.some((strategy) => strategy.reliability.length > 0);
  const curvesPresent = strategies.some((strategy) => strategy.curve.length > 0);
  return <div className="grid gap-6 lg:grid-cols-2">
    <Card className="shadow-none"><CardHeader><CardTitle>Reliability</CardTitle>
      <CardDescription>Predicted regression probability versus observed regression. Quantile bins keep tied scores together.</CardDescription>
    </CardHeader><CardContent>
      {reliabilityPresent ? <ChartContainer config={config} className="h-80 w-full">
        <LineChart margin={{ top: 12, right: 12, bottom: 24, left: 8 }}>
          <CartesianGrid vertical={false} /><XAxis type="number" dataKey="predicted" domain={[0, 1]} tickLine={false} label={{ value: "Predicted probability", position: "insideBottom", offset: -16 }} />
          <YAxis type="number" domain={[0, 1]} tickLine={false} width={40} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <ChartLegend content={<ChartLegendContent nameKey="name" />} />
          {strategies.map((strategy, index) => strategy.reliability.length > 0 && <Line
            key={strategy.name} data={strategy.reliability} type="linear" dataKey="observed"
            name={`m${index}`} stroke={`var(--color-m${index})`} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false}
          />)}
        </LineChart>
      </ChartContainer> : <p className="py-16 text-center text-sm text-muted-foreground">Reliability bins pending measured predictions.</p>}
      <p className="mt-3 text-xs text-muted-foreground">Dashed line: perfect calibration. Vertical axis: observed regression probability.</p>
    </CardContent></Card>
    <Card className="shadow-none"><CardHeader><CardTitle>Missed regressions vs. evaluation hours</CardTitle>
      <CardDescription>Each point is an observed threshold. Lower miss rate and higher hours saved are better.</CardDescription>
    </CardHeader><CardContent>
      {curvesPresent ? <ChartContainer config={config} className="h-80 w-full">
        <LineChart margin={{ top: 12, right: 12, bottom: 24, left: 8 }}>
          <CartesianGrid vertical={false} /><XAxis type="number" dataKey="saved" domain={[0, 100]} tickLine={false} label={{ value: "Evaluation hours saved (%)", position: "insideBottom", offset: -16 }} />
          <YAxis type="number" domain={[0, 100]} tickLine={false} width={40} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <ChartLegend content={<ChartLegendContent nameKey="name" />} />
          {strategies.map((strategy, index) => strategy.curve.length > 0 && <Line
            key={strategy.name} data={strategy.curve} type="linear" dataKey="missed"
            name={`m${index}`} stroke={`var(--color-m${index})`} strokeWidth={2} dot={false} isAnimationActive={false}
          />)}
        </LineChart>
      </ChartContainer> : <p className="py-16 text-center text-sm text-muted-foreground">Threshold curves pending measured evaluations.</p>}
      <p className="mt-3 text-xs text-muted-foreground">Vertical axis: missed regression fraction (%). Undefined miss rates are not plotted.</p>
    </CardContent></Card>
  </div>;
}
