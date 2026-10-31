.PHONY: check test replay

PYTHON := $(if $(wildcard .venv/bin/python3.11),.venv/bin/python3.11,python3.11)
export PATH := $(CURDIR)/.venv/bin:$(PATH)

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings
	cargo test
	$(PYTHON) -m ruff check
	$(PYTHON) -m pytest -q
	pnpm --dir apps/web lint
	pnpm --dir apps/web typecheck
	pnpm --dir apps/web test
	pnpm --dir apps/web build

test:
	cargo test
	$(PYTHON) -m pytest -q
	pnpm --dir apps/web test

replay:
	: "$${BRIER_DATA:?BRIER_DATA must be set to the local replay data directory}"
	cargo run --locked --bin brier -- replay --ledger "$${BRIER_DATA}/ledger.sqlite" --compiled "$${BRIER_DATA}/compiled.jsonl" --scores "$${BRIER_DATA}/replay-scores.json" --out "$${BRIER_DATA}/report.json"
