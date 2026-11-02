import Link from "next/link";
import { desc, eq } from "drizzle-orm";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { reports } from "@/db/schema";
import { getDb } from "@/lib/db";
import { metric, readStrategies } from "@/lib/report";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export const dynamic = "force-dynamic";
const github = "https://github.com/SiddharthManjul/Brier";

function Illustration({ kind = "robot" }: { kind?: "robot" | "grid" | "gate" | "cascade" | "bars" }) {
  return <svg viewBox="0 0 420 300" fill="none" role="img" aria-label={kind === "robot" ? "Isometric illustration of a robot gripper and a blue cube" : `${kind} decision process illustration`}>
    <g stroke="var(--muted-foreground)" strokeWidth="0.9" strokeLinejoin="round">
      <path d="M58 195 212 108 370 197 217 285Z" fill="white" /><path d="m58 195 0 8 159 90 153-88v-8M217 285v8" />
      {[0, 1, 2, 3].map((i) => <g key={i} stroke="var(--border)"><path d={`M${88 + i * 31} ${178 - i * 17.5}l158 89`} /><path d={`M${89 + i * 32} ${213 + i * 18}l154-87`} /></g>)}
      {kind === "robot" || kind === "gate" ? <>
        <path d="m127 174 16-9 16 9-16 9Z" fill="var(--muted)" /><path d="M127 174v-101l16-9 16 9v101l-16 9Z" fill="white" /><path d="M143 83v100l16-9V74Z" fill="var(--accent)" />
        <path d="m119 68 17-10 119 68-17 10Z" fill="white" /><path d="m119 68 119 68v13L119 81Z" fill="var(--muted)" /><path d="m238 136 17-10v13l-17 10Z" fill="var(--accent)" />
        <path d="m212 137 12 7v28l-12-7Z" fill="white" /><path d="m224 144 12-7v28l-12 7Z" fill="var(--accent)" /><path d="m212 137 12-7 12 7-12 7Z" fill="white" />
        <path d="m213 169-8 5v15l8 5M234 169l8 5v15l-8 5" /><path d="m275 218 14-8V133l-14 8Z" fill="var(--accent)" /><path d="m262 133 13 8v77l-13-8Z" fill="white" /><path d="m254 126 20-12 26 15-20 12Z" fill="white" /><path d="m254 126 26 15v12l-26-15Z" fill="var(--muted)" /><path d="m280 141 20-12v12l-20 12Z" fill="var(--accent)" />
        <path d="m205 212 20-12 20 12-20 12Z" fill="var(--primary)" /><path d="m205 212 20 12v23l-20-12Z" fill="var(--primary)" /><path d="m225 224 20-12v23l-20 12Z" fill="#1b2ac0" />
        <path d="M224 180v17M225 251v12" stroke="var(--primary)" strokeDasharray="3 3" />
        {kind === "gate" && <path d="m175 174 45 26 45-26" stroke="var(--primary)" strokeDasharray="3 4" />}
      </> : kind === "grid" ? <>
        {[0, 1, 2].flatMap((row) => [0, 1, 2, 3].map((col) => {
          const x = 93 + col * 39 + row * 37, y = 191 - col * 22 + row * 21;
          return <g key={`${row}-${col}`}><path d={`m${x} ${y} 29-17 29 17-29 17Z`} fill="white" /><path d={`m${x + 3} ${y} 26-14 25 14-25 14Z`} stroke="var(--border)" />{(row + col) % 3 === 0 && <path d={`m${x + 23} ${y} 6-3 6 3-6 3Z`} stroke="var(--primary)" fill="var(--primary)" />}</g>;
        }))}
      </> : kind === "bars" ? <>
        {[0, 1, 2, 3].map((i) => {
          const x = 102 + i * 58, y = 221 - i * 19, h = 23 + i * 25;
          return <g key={i}><path d={`m${x} ${y - h} 21-12 21 12-21 12Z`} fill="var(--primary)" /><path d={`m${x} ${y - h} 21 12v${h}l-21-12Z`} fill="#3249ff" /><path d={`m${x + 21} ${y - h + 12} 21-12v${h}l-21 12Z`} fill="#1b2ac0" /></g>;
        })}
      </> : <>
        {["CHEAP", "SCORE", "DECIDE", "RUN"].map((label, i) => <g key={label} transform={`translate(${95 + i * 52} ${170 + i * 15})`}><path d="m0 0 31-18 43 25-31 18Z" fill={i === 3 ? "var(--primary)" : "white"} /><path d="m0 0 43 25v10L0 10ZM43 25l31-18v10L43 35Z" fill={i === 3 ? "#1b2ac0" : "var(--muted)"} /><text x="22" y="9" stroke="none" fill={i === 3 ? "white" : "var(--muted-foreground)"} fontSize="6" fontFamily="monospace" transform="rotate(30 22 9)">{label}</text></g>)}
      </>}
    </g>
  </svg>;
}

/** Present the decision engine and connect its design to measured evaluation evidence. */
export default async function Home() {
  const [latest] = process.env.DATABASE_URL ? await getDb().select().from(reports)
    .where(eq(reports.suite, "robot")).orderBy(desc(reports.createdAt)).limit(1) : [];
  const strategies = latest ? readStrategies(latest.body) : [];
  return <div className="landing">
    <div className="build-banner">Brier is being built in the open for the Meta Global AI Developer Hackathon.<a href={github}>Follow the build →</a></div>
    <header className="landing-frame landing-nav">
      <Link className="brand" href="/" aria-label="Brier home"><span className="brand-mark" aria-hidden="true" />brier</Link>
      <nav aria-label="Product navigation"><a href="#product">Product</a><a href="#features">Features</a><a href="#compare">Compare</a><a href="#faq">FAQ</a><a href={`${github}#readme`}>Docs</a></nav>
      <div className="landing-nav-actions"><a href={github}>GitHub</a><Button asChild><Link href="/runs">Open workspace<ArrowUpRight aria-hidden="true" /></Link></Button></div>
    </header>
    <main id="main">
      <div className="landing-frame">
        <section className="landing-hero">
          <span className="eyebrow hero-label">The decision layer for robot CI</span>
          <h1>Know the result<br />before you run it.</h1>
          <p className="hero-copy">Predict which simulated rollouts need a closer look. Run the ones that matter, and keep the evidence behind every skip.</p>
          <div className="hero-actions"><Button asChild><Link href="/runs">Open workspace<ArrowRight aria-hidden="true" /></Link></Button><Button variant="outline" asChild><Link href="/models">See measured results</Link></Button></div>
          <p className="hero-note">Simulation only · no hardware needed to start</p>
        </section>
        <section className="landing-preview dotted" aria-label="The Brier evaluation workflow">
          <div className="preview-window">
            <div className="preview-topbar"><span className="window-dots" aria-hidden="true"><i /><i /><i /></span>Policy change → calibrated decision → evaluation evidence</div>
            <div className="preview-content">
              <div className="preview-evidence">
                <div className="preview-title"><span className="brand"><span className="brand-mark" aria-hidden="true" />brier / robot-ci</span><Badge variant="outline" className="status-pending">{latest ? "REPORT CONNECTED" : "AWAITING REPORT"}</Badge></div>
                <p className="preview-heading">Every decision leaves a trail.</p>
                <p className="preview-description">{latest ? `Measured robot report received ${new Date(latest.createdAt).toLocaleDateString("en-GB", { timeZone: "UTC" })}.` : "Your real evaluations appear here once the first report is connected. No sample results are substituted."}</p>
                <div className="preview-path" aria-hidden="true">{Array.from({ length: 16 }, (_, i) => <i key={i} />)}</div>
                {strategies.length ? strategies.slice(0, 3).map(({ name, metrics }) => {
                  const saved = metric(metrics.eval_hours_saved);
                  return <div className="preview-stage" key={name}><span>{name}</span><code>{saved === null ? "Savings pending" : `${(saved * 100).toFixed(1)}% hours saved`}</code></div>;
                }) : <>
                  <div className="preview-stage"><span>Predict regression risk</span><code>Calibrated scores</code></div>
                  <div className="preview-stage"><span>Choose the evaluation</span><code>RUN / SKIP</code></div>
                  <div className="preview-stage"><span>Explain observed failures</span><code>Frames + evidence</code></div>
                </>}
                <Link className="subtle-link mt-5" href={latest ? "/models" : "/runs"}>{latest ? "Inspect the measured report" : "Inspect the evaluation ledger"}<ArrowRight className="size-3" aria-hidden="true" /></Link>
              </div>
              <figure className="preview-art dotted"><figcaption className="preview-art-label"><span>From simulation to evidence</span><span className="font-mono">ALOHA transfer cube</span></figcaption><Illustration /><div className="preview-caption">Architecture illustration. Actual failure frames and SAM overlays are attached to each evaluated run.</div></figure>
            </div>
          </div>
        </section>
        <div className="technology-strip" aria-label="Brier model and simulation stack">{["Built on", "Muse Spark", "Muse Glimmer", "SAM 3.1", "LeRobot", "PushT", "ALOHA"].map((name) => <span key={name}>{name}</span>)}</div>
        <section className="landing-section problem-grid" id="product">
          <div className="problem-copy"><p className="eyebrow">The problem</p><h2>Simulation CI is<br />the new bottleneck.</h2><p>Every policy change fans out into dozens of rollouts. Each takes time and compute. Many confirm what you already knew.</p></div>
          <div className="problem-options"><p className="eyebrow">One change. A suite of scenarios.</p>{[
            ["all", "Run everything", "Every scenario runs, whether or not it changes the answer."],
            ["sample", "Sample by hand", "Fewer evaluations, with an unknown chance of missing a regression."],
            ["brier", "Brier", "Choose evaluations using measured probabilities and an explicit risk target."],
          ].map(([approach, title, description]) => <div className="comparison-option" data-approach={approach} key={approach}><h3>{title}</h3><div className="scenario-blocks" aria-hidden="true">{Array.from({ length: 16 }, (_, i) => <i key={i} />)}</div><p>{description}</p></div>)}<p className="data-note mt-5">Conceptual comparison. Actual savings and misses are reported in Models.</p></div>
        </section>
        <section className="landing-section" id="how-it-works">
          <div className="landing-section-head"><div><p className="eyebrow">How it works</p><h2>Predict. Decide. Run.</h2></div><p>A calibrated decision engine sits between your policy change and your simulator.</p></div>
          <div className="process-grid">{[
            { kind: "grid" as const, step: "01", title: "Predict", text: "Read the policy change and cheap rollout evidence. Estimate the chance of a regression in each scenario." },
            { kind: "gate" as const, step: "02", title: "Decide", text: "Apply the calibrated risk threshold. Choose which expensive evaluations to run and which to skip." },
            { kind: "cascade" as const, step: "03", title: "Run and explain", text: "Run selected scenarios. Inspect failure frames, masks, and explanations alongside the observed outcome." },
          ].map(({ kind, step, title, text }) => <div className="process-step" key={step}><div className="process-art dotted"><Illustration kind={kind} /></div><p className="eyebrow">Step {step}</p><h3>{title}</h3><p>{text}</p></div>)}</div>
        </section>
        <section className="landing-section" id="features">
          <div className="landing-section-head"><div><p className="eyebrow">Features</p><h2>Everything a skip<br />needs to be trusted.</h2></div><p>Each decision carries a probability, a risk target, and a way to check it.</p></div>
          <div className="feature-grid">
            <div className="feature"><div className="feature-art dotted"><Illustration kind="bars" /></div><h3>Calibrated probabilities</h3><p>Predictions are checked against observed regressions using log loss, Brier score, and calibration error.</p></div>
            <div className="feature"><div className="feature-art dotted"><div className="bound-art"><p className="eyebrow">Expected missed regressions</p><div className="bound-line" aria-hidden="true">{Array.from({ length: 20 }, (_, i) => <i key={i} />)}</div><div className="bound-labels"><span>0%</span><strong>α target</strong><span>100%</span></div></div></div><h3>A bound on every skip</h3><p>Conformal risk control selects a threshold against a configured missed-regression target.</p></div>
            <div className="feature"><div className="feature-art dotted"><div className="tier-list">{[["T0", "Rules", "no model call"], ["T1", "LightGBM", "CPU"], ["T2", "Glimmer LoRA", "GPU"], ["T3", "Muse Spark", "adds evaluations"]].map(([tier, name, note]) => <div key={tier}><code>{tier}</code><span>{name}</span><small>{note}</small></div>)}</div></div><h3>Cheapest model first</h3><p>Start with rules and a lightweight model. Escalate harder decisions through Glimmer and Muse Spark.</p></div>
            <div className="feature"><div className="feature-art dotted"><div className="decision-tiles"><div><span>Confident, below threshold</span><strong>SKIP</strong></div><div><span>Near the decision boundary</span><strong>ESCALATE</strong></div><div><span>Likely regression</span><strong>RUN</strong></div></div></div><h3>Escalates when unsure</h3><p>Close calls can add a full evaluation. The reasoning model never removes a required check.</p></div>
            <div className="feature"><div className="feature-art dotted"><div className="decision-tiles"><div><span>Selected evaluation</span><strong>RUN</strong></div><div><span>Exploration selection</span><strong>RUN · EXPLORE</strong></div><div><span>Decision recorded</span><strong>LEDGER</strong></div></div></div><h3>Checks its own skips</h3><p>A share of skipped scenarios still runs through exploration, keeping the feedback measurable.</p></div>
          </div>
        </section>
        <section className="landing-section" id="integrations">
          <div className="landing-section-head"><div><p className="eyebrow">Where it runs</p><h2>In the pull request, or<br />anywhere you call it.</h2></div><p>The CLI, CI workflow, and decision endpoint share one evaluation engine.</p></div>
          <div className="integration-panel dotted"><div className="integration-grid"><div className="integration-copy"><Badge variant="outline">CLI + ROBOT CI</Badge><h3 className="mt-4">The evidence stays with the change.</h3><p>Compile the state, choose evaluations, and produce a report with probabilities, scenario outcomes, and failure evidence.</p><a className="subtle-link" href={`${github}#readme`}>Read the integration guide<ArrowUpRight className="size-3" aria-hidden="true" /></a></div><figure className="terminal"><figcaption>brier / command line</figcaption><pre><code><span>$</span> brier compile --help{"\n"}<span>$</span> brier decide --help{"\n"}<span>$</span> brier explain --help{"\n"}<span>$</span> brier report --help</code></pre></figure></div></div>
        </section>
        <section className="landing-section" id="compare">
          <div className="landing-section-head"><div><p className="eyebrow">Compare</p><h2>More deliberate than<br />running it all.</h2></div><p>Compare how evaluation strategies spend compute and account for uncertainty.</p></div>
          <div className="compare-table"><table><thead><tr><th aria-label="Capability" /><th>Run everything</th><th>Sample by hand</th><th>Brier</th></tr></thead><tbody>{[
            ["Evaluation selection", "Every scenario", "Whatever you pick", "Risk-ranked scenarios"],
            ["Missed-regression risk", "No evaluation skipped", "Unmeasured", "Explicit calibrated target"],
            ["Why a rollout failed", "Logs", "Logs", "Frames, masks, and evidence"],
            ["Decision audit trail", "Run results", "Varies by workflow", "Predictions, actions, and outcomes"],
          ].map(([label, ...cells]) => <tr key={label}><td>{label}</td>{cells.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody></table></div>
        </section>
      </div>
      <section className="guarantee-band" aria-labelledby="guarantee-title"><div className="landing-frame"><div className="guarantee-head"><p className="eyebrow">The risk target</p><h2 id="guarantee-title">Skip with a bound, not a hunch.</h2></div><div className="guarantee-grid"><div><span className="guarantee-number">α</span><p>An explicit target for expected missed regressions.</p></div><div><span className="guarantee-number">0</span><p>Expensive rollout minutes spent on a skipped scenario.</p></div><div><span className="guarantee-number">4</span><p>Decision tiers, starting with the cheapest check.</p></div></div><p className="guarantee-footnote">Risk control assumes exchangeable changes and valid calibration. Measured replay results and target coverage are shown separately in Models.</p></div></section>
      <div className="landing-frame">
        <section className="landing-section" id="roadmap"><div className="landing-section-head"><div><p className="eyebrow">Roadmap</p><h2>Robots first. Every<br />slow loop next.</h2></div><p>One decision engine. Measured evidence for each new use.</p></div><div className="roadmap-grid">{[["NOW", "Robot CI", "PushT and ALOHA simulation suites. Inspect policy changes and their evaluation evidence."], ["NEXT", "Software CI", "The next application for the decision engine: expensive test suites for code changes."], ["LATER", "Your loop", "Slow evaluation loops where the cost and risk of a skipped check can be measured."]].map(([tag, title, text]) => <div key={title}><Badge variant={tag === "NOW" ? "default" : "outline"}>{tag}</Badge><h3>{title}</h3><p>{text}</p></div>)}</div></section>
        <section className="landing-section faq-grid" id="faq"><div className="faq-intro"><p className="eyebrow">FAQ</p><h2>Questions,<br />answered.</h2><p>Start with the decision engine. Inspect the measurements behind it.</p></div><div className="faq-items">{[
          ["What is a calibrated decision model?", "A model whose probabilities are checked against observed outcomes. Brier uses measured prediction quality and held-out calibration to choose RUN, SKIP, or escalation."],
          ["What does a missed-regression target mean?", "It sets the expected fraction of regressions the policy may miss across exchangeable changes. It is a statistical target under stated assumptions, not a promise that an individual change cannot be missed. Inspect replay coverage in Models."],
          ["Do I need a robot?", "No. The current evaluation suites run in simulation using PushT and ALOHA transfer cube. No physical robot is required."],
          ["Which models does it use?", "The cascade combines T0 rules, T1 LightGBM, a Glimmer LoRA adapter, and Muse Spark for deeper checks. SAM 3.1 provides masks for failure-frame evidence."],
          ["Can the reasoning model skip evaluations?", "No. Muse Spark can add evaluations for uncertain cases. It cannot remove scenarios that the calibrated policy requires."],
          ["Where can I inspect the results?", "Open Models for ingested, measured replay results and calibration plots. Open Runs for decisions, selected scenarios, and attached failure evidence. Missing measurements stay clearly marked."],
          ["Is it open source?", "Yes. The source, command-line workflow, and setup instructions are available in the Brier GitHub repository."],
        ].map(([question, answer], i) => <details key={question} open={i === 0}><summary>{question}</summary><p>{answer}</p></details>)}</div></section>
      </div>
      <section className="guarantee-band"><div className="landing-frame final-cta"><div><p className="eyebrow">Know what needs to run</p><h2>Spend compute<br />where the answer<br />is unknown.</h2><p>Start with the evaluation ledger. Follow each decision back to its evidence.</p><div className="hero-actions"><Button asChild><Link href="/runs">Open workspace<ArrowRight aria-hidden="true" /></Link></Button><Button variant="outline" asChild><a href={github}>View on GitHub<ArrowUpRight aria-hidden="true" /></a></Button></div></div><div className="diagonal-art"><svg viewBox="0 0 420 230" role="img" aria-label="A diagonal of decision markers"><path d="M35 200H385M35 200V25" stroke="#ffffff55" /><path d="m55 185 300-145" stroke="#ffffff66" strokeDasharray="2 12" />{[0, 1, 2, 3, 4].map((i) => <rect x={70 + i * 64} y={172 - i * 31} width="7" height="7" key={i} fill={i === 2 ? "none" : "white"} stroke="white" />)}<text x="35" y="221" fill="#dce0ff" fontFamily="monospace" fontSize="9">predicted →</text><text x="45" y="29" fill="#dce0ff" fontFamily="monospace" fontSize="9">observed ↑</text></svg></div></div></section>
    </main>
    <footer className="landing-frame landing-footer"><Link className="brand" href="/" aria-label="Brier home"><span className="brand-mark" aria-hidden="true" />brier</Link><nav aria-label="Footer navigation"><a href="#product">Product</a><Link href="/runs">Runs</Link><Link href="/models">Models</Link><a href={`${github}#readme`}>Docs</a><a href={github}>GitHub ↗</a></nav><p>Simulation evidence. Measured decisions.</p></footer>
  </div>;
}
