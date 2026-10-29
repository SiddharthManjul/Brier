# AGENTS.md: rules for every Codex task in this repo

## Mission
Implement exactly the current task card. Nothing more. SCOPE.md is the outer boundary; the task card is the inner one.

## Hard rules
1. Touch only the files listed under "Files" in the task card. If another file must change, stop and explain why in your final message. Do not change it.
2. Add no dependency the card does not list. Never add a framework.
3. Add no feature, flag, config option, CLI subcommand, endpoint, page, environment variable or abstraction that the card does not name. No plugin systems, registries, base classes with one implementation or "future-proofing".
4. No refactors, renames or formatting changes outside the task.
5. Smallest diff that passes the acceptance commands. Stay under the card's line budget; if you cannot, stop and say so.
6. Each new public function gets one focused test that the acceptance commands run.
7. No mock modes, demo modes, silent fallbacks or retries the card does not ask for. Fail loudly with a clear error.
8. Prompt text is produced only by `brier compile` (Rust). Python never formats prompts.
9. Labels are computed only by `brier label`.
10. Secrets and paths come from env vars: META_API_KEY, BRIER_DATA (gs:// or local), MODAL_*, DATABASE_URL, BRIER_INGEST_TOKEN. Never hardcode them.
11. No new top-level files or directories.
12. Docstrings on public functions only. Do not edit README.md unless the card says so.
13. When a choice is unclear, take the simpler option and list the assumption in your final message.

## Definition of done
- The card's acceptance commands pass.
- `make check` passes: cargo fmt --check, cargo clippy -- -D warnings, cargo test, ruff check, pytest -q, and in apps/web: pnpm lint, pnpm typecheck, pnpm build.
- Final message lists: files changed, commands run and their results, assumptions, anything skipped.

## Layout
- crates/brier-core/  Rust: ledger, adapter trait, state compiler, cascade, policy, CLI, report
- py/brier_sim/       envs, policies, change library, scenarios, rollout runner
- py/brier_ml/        datasets, T1, prompt-format loader, finetune, calibrate, eval
- py/brier_explain/   keyframes, SAM 3.1 client, Muse Spark client
- serve/modal_app.py  vLLM + LoRA endpoint: POST /decide
- .github/workflows/  check.yml, rollouts.yml, robot-ci.yml
- apps/web/           Next.js + TS + shadcn + Drizzle: 4 pages, POST /api/ingest
- data/               gitignored; mirrors gs://brier-data

## Conventions
- Rust: anyhow for errors, serde for JSON, clap for CLI, blocking reqwest. No async unless the card says so.
- Python 3.11: type hints, plain functions over classes, pathlib, logging (no print in library code).
- apps/web: Next.js App Router, TypeScript strict, server components by default, client components only for charts. shadcn/ui components only (lucide icons). Drizzle with the Neon serverless driver (drizzle-orm/neon-http) for every query, so the app runs on Vercel now and Cloudflare later; one db/schema.ts and one lib/db.ts; queries live in the page that uses them. Node runtime only, no Vercel-specific SDKs. No auth, settings, admin, search, theme toggle, i18n or analytics.
- IDs: change `c001`, scenario `<env>-s01`, run and decision ids are UUIDv4.
- Data files: Parquet for episodes, JSONL for compiled states, SQLite for the engine ledger, Postgres for the platform.
