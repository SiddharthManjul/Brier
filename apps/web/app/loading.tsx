/** Announce server-rendered data loading without showing invented measurements. */
export default function Loading() {
  return <main id="main" className="grid min-h-screen place-items-center bg-muted px-6" aria-busy="true">
    <div className="w-full max-w-md border border-border bg-background p-10" role="status">
      <span className="brand"><span className="brand-mark" aria-hidden="true" />brier</span>
      <p className="eyebrow mt-10">Reading the evaluation ledger</p>
      <h1 className="mt-4 text-3xl">Bringing the evidence together.</h1>
      <p className="mt-5 text-sm leading-7 text-muted-foreground">Loading measured runs, decisions and model results.</p>
    </div>
  </main>;
}
