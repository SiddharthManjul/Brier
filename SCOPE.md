# SCOPE (frozen; only Brooklyn edits this file)

CLI commands, and no others: brier ledger init | label | compile | card | decide | replay | explain | report [--push]
HTTP endpoints, and no others: POST /decide (serve/modal_app.py), POST /api/ingest (apps/web)
Workflows, and no others: check.yml, rollouts.yml, robot-ci.yml
Platform: apps/web, Next.js App Router + TypeScript + shadcn/ui + Drizzle + Postgres (Neon), hosted on Vercel
Platform pages, and no others: /, /runs, /runs/[id], /models
Envs: PushT, AlohaTransferCube (AlohaInsertion only if T06 timing allows)
Models: T0 rules, T1 LightGBM + isotonic, T2 Glimmer + LoRA, T3 Muse Spark (Contributor), SAM 3.1 for failure frames
Phase 2 (only after the Nov 2 gate): software CI adapter + replay on public repos, shown as a tab on /models

Out of scope: auth, accounts, settings or admin pages, GitHub App, billing, a second UI library, Vercel-specific SDKs,
Edge runtime, databases beyond SQLite (engine) and Postgres (platform), Docker beyond Modal images, Kubernetes,
Terraform, real hardware, ZK proofs, chip CI
