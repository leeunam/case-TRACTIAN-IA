# Makefile — TRACTIAN × Inteli
# Comandos somente para os componentes que existem hoje.
#
# Uso:
#   make setup   # prepara API, agente, demo, frontend e dados
#   make demo    # inicia os quatro processos da demonstração
#   make test    # executa todas as suítes e o build
#   make eval    # runner/checks locais, sem credenciais
#   make stop    # encerra todos os processos locais
#   make smoke-groq # compara modelos Groq com dados sintéticos (opt-in)
#
# Variáveis (override: make VAR=valor):
PYTHON ?= python3
API_PORT ?= 8000
DEMO_PORT ?= 8100
FRONTEND_PORT ?= 5173
ROOT := $(abspath $(dir $(MAKEFILE_LIST)))
API_VENV := $(ROOT)/api/.venv
API_PY := $(API_VENV)/bin/python
AGENT_VENV := $(ROOT)/agent/.venv
AGENT_PY := $(AGENT_VENV)/bin/python
DEMO_VENV := $(ROOT)/demo/.venv
DEMO_PY := $(DEMO_VENV)/bin/python
PID_DIR := $(ROOT)/.run
EVAL_PROVIDER ?= groq
EVAL_OUTPUT_DIR ?= $(PID_DIR)/evaluation/tractian-eval-v5
EVAL_LABELS ?= $(EVAL_OUTPUT_DIR)/human-labels.json
EVAL_SCORES ?= $(EVAL_OUTPUT_DIR)/judge-scores.json
MAKEFLAGS += --no-print-directory

.DEFAULT_GOAL := help

.PHONY: help setup deps data up up-api demo up-demo up-worker up-frontend stop logs test test-e2e smoke-groq smoke-slack eval eval-live eval-providers eval-judges eval-label-template eval-calibrate eval-layers clean clean-data

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: deps data ## Cria as venvs, instala dependências e gera os dados

deps: ## Cria as venvs, instala API/agente/demo e o frontend
	@command -v uv >/dev/null 2>&1 || { echo "Instale o uv: https://docs.astral.sh/uv/"; exit 1; }
	@if [ ! -x "$(API_PY)" ]; then cd $(ROOT)/api && uv venv --python $(PYTHON); fi
	@cd $(ROOT)/api && uv pip install --python "$(API_PY)" -e ".[dev]"
	@if [ ! -x "$(AGENT_PY)" ]; then cd $(ROOT)/agent && uv venv --python $(PYTHON); fi
	@cd $(ROOT)/agent && uv pip install --python "$(AGENT_PY)" -e ".[dev]"
	@if [ ! -x "$(DEMO_PY)" ]; then cd $(ROOT)/demo && uv venv --python $(PYTHON); fi
	@cd $(ROOT)/demo && uv pip install --python "$(DEMO_PY)" -e ".[dev]"
	@cd $(ROOT)/frontend && npm install --legacy-peer-deps
	@echo "✓ dependências instaladas para API, agente, demo e frontend"

data: ## Gera data/*.parquet, agent-input/ e eval/
	@cd $(ROOT)/api && $(API_PY) -m seed_data
	@cd $(ROOT)/api && $(API_PY) -m package_material
	@echo "✓ dados gerados"

up: up-api ## Inicia a API industrial em background
	@echo "✓ Swagger: http://localhost:$(API_PORT)/docs"
	@echo "  use 'make stop' para parar e 'make logs' para acompanhar"

define wait_up
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -s -o /dev/null http://localhost:$(1) && break; sleep 1; \
	done
endef

up-api: ## Inicia somente a API industrial em background
	@mkdir -p $(PID_DIR)
	@cd $(ROOT)/api && $(API_PY) -m uvicorn app.main:app --host 127.0.0.1 --port $(API_PORT) \
		> $(PID_DIR)/api.log 2>&1 & echo $$! > $(PID_DIR)/api.pid
	$(call wait_up,$(API_PORT))
	@curl -s -o /dev/null -w "✓ API industrial em :$(API_PORT) (HTTP %{http_code})\n" http://localhost:$(API_PORT)/docs \
		|| echo "✗ API não subiu — veja $(PID_DIR)/api.log"

demo: up-api up-demo up-worker up-frontend ## Inicia API, backend, worker e frontend da demonstração
	@echo "✓ Central de casos: http://127.0.0.1:$(FRONTEND_PORT)"
	@echo "  use 'make stop' para encerrar todos os processos"

up-demo: ## Inicia somente a fachada FastAPI da demonstração
	@mkdir -p $(PID_DIR)
	@if [ -f $(ROOT)/.env ]; then set -a; . $(ROOT)/.env; set +a; fi; \
		cd $(ROOT)/demo && $(DEMO_PY) -m uvicorn tractian_demo.app:app --host 127.0.0.1 --port $(DEMO_PORT) \
		> $(PID_DIR)/demo.log 2>&1 & echo $$! > $(PID_DIR)/demo.pid
	$(call wait_up,$(DEMO_PORT))
	@curl -s -o /dev/null -w "✓ Backend demo em :$(DEMO_PORT) (HTTP %{http_code})\n" http://127.0.0.1:$(DEMO_PORT)/v1/demo/config \
		|| { echo "✗ Backend demo não subiu — veja $(PID_DIR)/demo.log"; exit 1; }

up-worker: ## Inicia worker do agente e da outbox Slack
	@mkdir -p $(PID_DIR)
	@if [ -f $(ROOT)/.env ]; then set -a; . $(ROOT)/.env; set +a; fi; \
		cd $(ROOT)/demo && $(DEMO_PY) -m tractian_demo.run_worker \
		> $(PID_DIR)/worker.log 2>&1 & echo $$! > $(PID_DIR)/worker.pid
	@sleep 1
	@kill -0 $$(cat $(PID_DIR)/worker.pid) 2>/dev/null \
		&& echo "✓ Worker ao vivo iniciado" \
		|| { echo "✗ Worker não iniciou — configure .env e veja $(PID_DIR)/worker.log"; exit 1; }

up-frontend: ## Inicia somente a SPA Vite
	@mkdir -p $(PID_DIR)
	@cd $(ROOT)/frontend && VITE_DEMO_API_URL=http://127.0.0.1:$(DEMO_PORT) npm run dev -- --port $(FRONTEND_PORT) \
		> $(PID_DIR)/frontend.log 2>&1 & echo $$! > $(PID_DIR)/frontend.pid
	$(call wait_up,$(FRONTEND_PORT))
	@curl -s -o /dev/null -w "✓ Frontend em :$(FRONTEND_PORT) (HTTP %{http_code})\n" http://127.0.0.1:$(FRONTEND_PORT) \
		|| { echo "✗ Frontend não subiu — veja $(PID_DIR)/frontend.log"; exit 1; }

stop: ## Encerra API, backend demo, worker e frontend
	@for name in frontend worker demo; do \
		if [ -f $(PID_DIR)/$$name.pid ]; then \
			kill $$(cat $(PID_DIR)/$$name.pid) 2>/dev/null && echo "✓ $$name encerrado" || true; \
			rm -f $(PID_DIR)/$$name.pid; \
		fi; \
	done
	@if [ -f $(PID_DIR)/api.pid ]; then \
		kill $$(cat $(PID_DIR)/api.pid) 2>/dev/null && echo "✓ API encerrada" || true; \
		rm -f $(PID_DIR)/api.pid; \
	fi
	@-pkill -f "uvicorn app.main:app --host 127.0.0.1 --port $(API_PORT)" 2>/dev/null || true
	@-pkill -f "uvicorn tractian_demo.app:app --host 127.0.0.1 --port $(DEMO_PORT)" 2>/dev/null || true
	@-pkill -f "tractian_demo.run_worker" 2>/dev/null || true
	@-pkill -f "vite --host 127.0.0.1" 2>/dev/null || true

logs: ## Mostra o log da API
	@tail -f $(PID_DIR)/api.log $(PID_DIR)/demo.log $(PID_DIR)/worker.log $(PID_DIR)/frontend.log 2>/dev/null || echo "Sem logs — execute 'make demo'."

test: ## Executa todas as suítes, build e fluxo no Chrome
	@cd $(ROOT)/api && $(API_PY) -m pytest -q
	@cd $(ROOT)/agent && $(AGENT_PY) -m pytest -q
	@cd $(ROOT)/demo && $(DEMO_PY) -m pytest -q
	@cd $(ROOT)/frontend && npm test
	@cd $(ROOT)/frontend && npm run build
	@cd $(ROOT)/frontend && npm run test:e2e

test-e2e: ## Executa somente o fluxo Playwright da central
	@cd $(ROOT)/frontend && npm run test:e2e

smoke-groq: ## Compara modelos Groq com dados sintéticos; requer GROQ_API_KEY
	@cd $(ROOT)/agent && $(AGENT_PY) -m tractian_agent.groq_smoke

smoke-slack: ## Envia uma notificação segura em cada canal Slack (opt-in)
	@set -a; . $(ROOT)/.env; set +a; cd $(ROOT)/demo && \
		$(DEMO_PY) -m tractian_demo.slack_smoke

eval: ## Executa 17 casos x 2, checks e lote humano sem LLM/rede
	@cd $(ROOT)/agent && $(AGENT_PY) -m tractian_agent.evaluation offline \
		--root $(ROOT) --output-dir $(EVAL_OUTPUT_DIR)

eval-live: up-api ## Executa o agente real; requer .env e provider configurado
	@set -a; . $(ROOT)/.env; set +a; cd $(ROOT)/agent && \
		$(AGENT_PY) -m tractian_agent.evaluation live --root $(ROOT) \
		--provider $(EVAL_PROVIDER) --api-base-url http://127.0.0.1:$(API_PORT) \
		--output-dir $(EVAL_OUTPUT_DIR)

eval-providers: ## Compara Groq x NVIDIA NIM com configuração congelada
	@set -a; . $(ROOT)/.env; set +a; cd $(ROOT)/agent && \
		$(AGENT_PY) -m tractian_agent.evaluation providers --root $(ROOT)

eval-judges: ## Avalia offline o relatório existente; não chama o agente
	@set -a; . $(ROOT)/.env; set +a; cd $(ROOT)/agent && \
		$(AGENT_PY) -m tractian_agent.evaluation judges --root $(ROOT) \
		--provider $(EVAL_PROVIDER) \
		--programmatic-report $(EVAL_OUTPUT_DIR)/programmatic-report.json \
		--output $(EVAL_OUTPUT_DIR)/judge-report.json \
		--scores-output $(EVAL_SCORES) \
		--comparison-output $(EVAL_OUTPUT_DIR)/evaluation-layers.json

eval-label-template: ## Gera template cego para especialista da TRACTIAN
	@cd $(ROOT)/agent && $(AGENT_PY) -m tractian_agent.evaluation labels-template \
		--packet $(EVAL_OUTPUT_DIR)/blind-review-packet.json \
		--output $(EVAL_LABELS)

eval-calibrate: ## Calcula calibracao com 20-30 rotulos humanos cegos
	@cd $(ROOT)/agent && $(AGENT_PY) -m tractian_agent.evaluation calibrate \
		--labels $(EVAL_LABELS) --scores $(EVAL_SCORES) \
		--output $(EVAL_OUTPUT_DIR)/calibration-report.json

eval-layers: ## Compara checks contra checks + juizes nos mesmos runs
	@cd $(ROOT)/agent && $(AGENT_PY) -m tractian_agent.evaluation layers \
		--programmatic-report $(EVAL_OUTPUT_DIR)/programmatic-report.json \
		--judge-report $(EVAL_OUTPUT_DIR)/judge-report.json \
		--output $(EVAL_OUTPUT_DIR)/evaluation-layers.json

clean-data: ## Apaga dados gerados; regenere com make data
	@rm -rf data agent-input eval/expected-paths.json
	@echo "✓ dados apagados"

clean: stop clean-data ## Encerra a API e apaga dados, venvs e arquivos temporários
	@rm -rf $(API_VENV) $(AGENT_VENV) $(DEMO_VENV) $(ROOT)/frontend/node_modules $(PID_DIR)
	@echo "✓ ambiente limpo"
