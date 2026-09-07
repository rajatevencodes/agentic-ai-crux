SHELL := /bin/zsh

UV ?= uv
PYTHON_VERSION := 3.13

.DEFAULT_GOAL := help
.PHONY: help init sync notebook check

help: ## Show available commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init: ## Install Python, create the local environment file, and sync dependencies.
	$(UV) python install $(PYTHON_VERSION)
	@test -f .env || cp .env.example .env
	$(UV) sync
	@echo "Agentic AI Crux is ready. Add OPENROUTER_API_KEY to .env, choose a model in notebooks/starter.ipynb, then run: make notebook"

sync: ## Create or update the UV environment.
	$(UV) sync

notebook: ## Open the Agentic AI Crux starter notebook in JupyterLab.
	$(UV) run python -m jupyterlab notebooks/starter.ipynb --ServerApp.open_browser=True

check: ## Validate the lockfile, Python setup, and clean notebook structure.
	$(UV) lock --check
	$(UV) run python -m py_compile notebooks/common.py
	$(UV) run python -m jupyterlab --version
	$(UV) run python -c 'import json, pathlib; path = pathlib.Path("notebooks/starter.ipynb"); notebook = json.loads(path.read_text()); code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]; assert all(not cell.get("outputs") and cell.get("execution_count") is None for cell in code_cells), "Clear notebook outputs and execution counts before publishing"; [compile("".join(cell["source"]), f"{path}:cell-{index}", "exec") for index, cell in enumerate(code_cells, 1)]; print(f"Validated {len(code_cells)} clean Python code cells in {path}")'
