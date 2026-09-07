PYTHON ?= python

.PHONY: install install-dev test lint lint-arch lint-deps lint-quality build setup-env start-server teardown-env

install: ## [Local development] Upgrade pip and install the open_clip package.
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .

install-dev: ## [Local development] Install test requirements.
	$(PYTHON) -m pip install -r requirements-test.txt

test: ## [Local development] Run pytest. The tests/ directory must exist.
	$(PYTHON) -m pytest -x -s -v tests

lint: lint-arch ## [Local development] Run architecture and quality linters.

lint-arch: lint-deps lint-quality ## [Architecture] Run all harness linters.

lint-deps: ## [Architecture] Validate package dependency direction.
	$(PYTHON) scripts/lint_deps.py

lint-quality: ## [Quality] Report size, secret, debug, and TODO findings.
	$(PYTHON) scripts/lint_quality.py

build: ## [Local development] Syntax-compile all Python source.
	$(PYTHON) -m compileall -q src my_code inference.py measure_throughput.py

setup-env: ## [Harness] Set up the local environment.
	bash harness/scripts/setup-env.sh

start-server: ## [Harness] Run the library smoke check.
	bash harness/scripts/start-server.sh

teardown-env: ## [Harness] Remove harness runtime artifacts.
	bash harness/scripts/teardown-env.sh
