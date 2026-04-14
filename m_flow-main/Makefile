.PHONY: quickstart up down build logs status test lint docs-build docs-serve sync-benchmarks

# ── Quick Start ──────────────────────────────────────────────────────────────

quickstart:
	@./quickstart.sh

# ── Deployment ───────────────────────────────────────────────────────────────

up:
	docker compose --profile ui up -d

down:
	docker compose --profile ui down

build:
	docker compose --profile ui build

logs:
	docker compose logs -f --tail=100

status:
	docker compose ps

restart:
	docker compose --profile ui restart

# ── Development ──────────────────────────────────────────────────────────────

test:
	PYTHONPATH=. pytest m_flow/tests/unit/ -v --tb=short

lint:
	ruff check m_flow/ --fix
	ruff format m_flow/

# ── Documentation ────────────────────────────────────────────────────────────

docs-build:
	source .venv-docs/bin/activate && mkdocs build --strict

docs-serve:
	source .venv-docs/bin/activate && mkdocs serve

sync-benchmarks:
	cp benchmarks/README.md docs/benchmarks/README.md
	cp benchmarks/hotpotqa/README.md docs/benchmarks/hotpotqa/README.md
	cp benchmarks/evolving/README.md docs/benchmarks/evolving/README.md
