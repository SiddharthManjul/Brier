.PHONY: check test

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings
	cargo test
	python3.11 -m ruff check
	python3.11 -m pytest -q

test:
	cargo test
	python3.11 -m pytest -q
