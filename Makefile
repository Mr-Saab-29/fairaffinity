PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help check compile test quality-ci smoke-mlops monitoring governance retrain-policy ci-local docker-config orchestrate

help:
	@echo "Available targets:"
	@echo "  make check         - End-to-end local health check (compile, tests, gates, mlops smoke)"
	@echo "  make ci-local      - CI-like checks (compile, quality gates if artifacts exist, tests)"
	@echo "  make orchestrate   - Full pipeline orchestration run"
	@echo "  make docker-config - Validate docker compose config"

check: compile test quality-ci smoke-mlops
	@echo "[OK] Local check completed."

ci-local: compile quality-ci test
	@echo "[OK] CI-like checks completed."

compile:
	$(PYTHON) -m compileall -q src dashboards tests

test:
	$(PYTHON) -m pytest -q

quality-ci:
	@if [ -f data/processed/interactions.parquet ] || [ -f data/processed/interactions_collapsed_c_p_d_s.parquet ]; then \
		$(PYTHON) -m src.cli.data_quality_gates --stage ci --strict; \
	else \
		echo "Skipping data quality gates: no processed artifacts found."; \
	fi

smoke-mlops: monitoring governance retrain-policy
	@echo "[OK] MLOps smoke checks completed."

monitoring:
	$(PYTHON) -m src.cli.monitoring_snapshot --window-hours 24

governance:
	$(PYTHON) -m src.cli.experiment_governance --name make_smoke

retrain-policy:
	$(PYTHON) -m src.cli.retrain_policy

docker-config:
	@docker compose config >/dev/null
	@echo "[OK] docker compose config is valid."

orchestrate:
	$(PYTHON) -m src.cli.orchestrate_pipeline --promote
