.PHONY: audit test test-source test-private check check-source check-private \
        status repro-check repro-smoke derive-configs compare-e4 \
        ui-install ui-api ui-web ui-build ui-serve ui-check ui-private-model \
        ui-install-e2 ui-api-e2 ui-check-e2 ui-e2-model \
        ui-up ui-up-sidecar ui-wait-sidecar ui-down ui-restart ui-status ui-logs \
        e1 e1-train e1-val e1-freeze e1-test \
        e2 e2-train e2-val e2-freeze e2-test \
        e3-custom e3-custom-train e3-custom-val e3-custom-freeze e3-custom-test \
        e4 e4-train e4-val e4-freeze e4-test knowledge-preflight \
        compare error-analysis error-analysis-summary all

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DELETE_ON_ERROR:

export PYTHONPATH := $(CURDIR)/code

# The three environments are not interchangeable: .conda/eval has neither torch nor
# fairseq, .conda/fairseq has no transformers/peft/bitsandbytes, .conda/llm has no
# fairseq. Each stage must run under the env that owns its dependencies.
EVAL_PYTHON ?= conda run --no-capture-output -p $(CURDIR)/.conda/eval python
EVAL         := $(EVAL_PYTHON) -m mt_pipeline
FAIRSEQ      := conda run --no-capture-output -p $(CURDIR)/.conda/fairseq python -m mt_pipeline
LLM          := conda run --no-capture-output -p $(CURDIR)/.conda/llm python -m mt_pipeline
UI_PYTHON    ?= conda run --no-capture-output -p $(CURDIR)/.conda/e1-ui python
UI_E2_PYTHON ?= conda run --no-capture-output -p $(CURDIR)/.conda/e2-ui python
UI_NPM       := npm --prefix ui

# RUN_ROOT empty => canonical tree. Set it to send a whole rerun into a parallel
# tree so the graded artifacts in checkpoint/, predictions/ and metrics/ survive:
#   make all RUN_ROOT=runs/rerun-$$(date +%F-%H%M%S)
RUN_ROOT ?=

ifeq ($(strip $(RUN_ROOT)),)
  E1_CONFIG := configs/e1_fairseq.yaml
  E2_CONFIG := configs/e2_qwen3_qlora.yaml
  E3_CONFIG := configs/e3_custom_fairseq_knowledge.yaml
  E4_CONFIG := configs/e4_qwen3_knowledge.yaml
  METRICS_DIR := metrics
  WORK_DIR := work
  PRED_DIR := predictions
  ANALYSIS_DIR := error_analysis
else
  E1_CONFIG := $(RUN_ROOT)/configs/e1_fairseq.yaml
  E2_CONFIG := $(RUN_ROOT)/configs/e2_qwen3_qlora.yaml
  E3_CONFIG := $(RUN_ROOT)/configs/e3_custom_fairseq_knowledge.yaml
  E4_CONFIG := $(RUN_ROOT)/configs/e4_qwen3_knowledge.yaml
  METRICS_DIR := $(RUN_ROOT)/metrics
  WORK_DIR := $(RUN_ROOT)/work
  PRED_DIR := $(RUN_ROOT)/predictions
  ANALYSIS_DIR := $(RUN_ROOT)/error_analysis
endif

E1_ID := e1_fairseq_vi_zh_v1
E2_ID := e2_qwen3_8b_qlora_vi_zh_v1
E3_ID := e3_custom_fairseq_knowledge_vi_zh_v1
E4_ID := e4_qwen3_knowledge_vi_zh_v1

# The stored comparisons used seed 42; the CLI default is 12345. Never omit this.
COMPARE_SEED := 42
# Specification section 12 requires 100 shared test sentences.
ERROR_SAMPLE_SIZE := 100

# ---------------------------------------------------------------- housekeeping

audit:
	$(EVAL) data-audit --config configs/dataset.yaml --output $(METRICS_DIR)/data_audit.json

test: test-source

test-source:
	$(EVAL_PYTHON) -m pytest -q \
	  -m "not private_data"

test-private:
	$(EVAL_PYTHON) -m pytest -q

status:
	$(EVAL) project-status

check: check-source

check-source: test-source

check-private: test-private
	$(EVAL) data-audit --config configs/dataset.yaml --output /tmp/group10_data_audit.json
	$(MAKE) repro-check

# ------------------------------------------------------------- rerun scaffolding

# When RUN_ROOT is set the experiment configs are derived copies with their output
# directories rebased; when it is empty $(E*_CONFIG) names the checked-in config and
# this rule never fires. Every stage below takes its config as a prerequisite so the
# derivation actually runs.
$(RUN_ROOT)/configs/%.yaml: configs/%.yaml
	@mkdir -p $(RUN_ROOT)/configs
	$(EVAL) derive-run-config --config configs/$*.yaml --run-root $(RUN_ROOT)

derive-configs: $(E1_CONFIG) $(E2_CONFIG) $(E3_CONFIG) $(E4_CONFIG)

# ------------------------------------------------------------------------- E1

e1-train: $(E1_CONFIG)
	$(FAIRSEQ) train --config $(E1_CONFIG)

e1-val: $(E1_CONFIG)
	$(FAIRSEQ) predict --config $(E1_CONFIG) --split val
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E1_ID).val.jsonl \
	                 --output $(METRICS_DIR)/$(E1_ID).val.json

e1-freeze: $(E1_CONFIG)
	$(EVAL) freeze-selection --config $(E1_CONFIG) \
	  --validation-predictions $(PRED_DIR)/$(E1_ID).val.jsonl \
	  --validation-metrics $(METRICS_DIR)/$(E1_ID).val.json

e1-test: $(E1_CONFIG)
	$(FAIRSEQ) predict --config $(E1_CONFIG) --split test
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E1_ID).test.jsonl \
	                 --output $(METRICS_DIR)/$(E1_ID).test.json

e1: e1-train e1-val e1-freeze e1-test

# ------------------------------------------------------------------------- E2
# QLoRA owns its transcript after its safety guards pass. External `tee` used to
# truncate the canonical log before the no-overwrite guard could stop the run.

e2-train: $(E2_CONFIG)
	$(LLM) train --config $(E2_CONFIG)

e2-val: $(E2_CONFIG)
	$(LLM) predict --config $(E2_CONFIG) --split val
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E2_ID).val.jsonl \
	                 --output $(METRICS_DIR)/$(E2_ID).val.json

e2-freeze: $(E2_CONFIG)
	$(EVAL) freeze-selection --config $(E2_CONFIG) \
	  --validation-predictions $(PRED_DIR)/$(E2_ID).val.jsonl \
	  --validation-metrics $(METRICS_DIR)/$(E2_ID).val.json

e2-test: $(E2_CONFIG)
	$(LLM) predict --config $(E2_CONFIG) --split test
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E2_ID).test.jsonl \
	                 --output $(METRICS_DIR)/$(E2_ID).test.json

e2: e2-train e2-val e2-freeze e2-test

# ------------------------------------------------------------- E3 (custom)
# Not the official lab knowledge_v2 method; see docs/EXPERIMENTS.md.

e3-custom-train: $(E3_CONFIG)
	$(FAIRSEQ) train --config $(E3_CONFIG)

e3-custom-val: $(E3_CONFIG)
	$(FAIRSEQ) predict --config $(E3_CONFIG) --split val
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E3_ID).val.jsonl \
	                 --output $(METRICS_DIR)/$(E3_ID).val.json

e3-custom-freeze: $(E3_CONFIG)
	$(EVAL) freeze-selection --config $(E3_CONFIG) \
	  --validation-predictions $(PRED_DIR)/$(E3_ID).val.jsonl \
	  --validation-metrics $(METRICS_DIR)/$(E3_ID).val.json

e3-custom-test: $(E3_CONFIG)
	$(FAIRSEQ) predict --config $(E3_CONFIG) --split test
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E3_ID).test.jsonl \
	                 --output $(METRICS_DIR)/$(E3_ID).test.json

e3-custom: e3-custom-train e3-custom-val e3-custom-freeze e3-custom-test

# ------------------------------------------------- E4 (Qwen + knowledge, bonus ext.)
# Not the named knowledge_v2 bonus (that is Fairseq + knowledge_v2 = E3); this is an
# extension testing whether the same retrieval helps the much stronger LLM.

knowledge-preflight: $(E4_CONFIG)
	$(LLM) knowledge-preflight --config $(E4_CONFIG) --candidates 8,16,24,32 \
	  --output $(METRICS_DIR)/e4_knowledge_preflight.json

e4-train: $(E4_CONFIG)
	$(LLM) train --config $(E4_CONFIG)

e4-val: $(E4_CONFIG)
	$(LLM) predict --config $(E4_CONFIG) --split val
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E4_ID).val.jsonl \
	                 --output $(METRICS_DIR)/$(E4_ID).val.json

e4-freeze: $(E4_CONFIG)
	$(EVAL) freeze-selection --config $(E4_CONFIG) \
	  --validation-predictions $(PRED_DIR)/$(E4_ID).val.jsonl \
	  --validation-metrics $(METRICS_DIR)/$(E4_ID).val.json

e4-test: $(E4_CONFIG)
	$(LLM) predict --config $(E4_CONFIG) --split test
	$(EVAL) evaluate --predictions $(PRED_DIR)/$(E4_ID).test.jsonl \
	                 --output $(METRICS_DIR)/$(E4_ID).test.json

e4: e4-train e4-val e4-freeze e4-test

# -------------------------------------------------------------- local MT UI

# Defaults so the UI targets run without exporting anything by hand. Every one
# is `?=`, so an explicit environment value still wins. Set MT_E2_URL empty to
# run E1 only:  make ui-api MT_E2_URL=
MT_CHECKPOINT_PATH ?= $(CURDIR)/checkpoint/$(E1_ID)/checkpoint_best.pt
MT_DATA_BIN_PATH   ?= $(CURDIR)/work/$(E1_ID)/data-bin
MT_DEVICE          ?= cpu
MT_PORT            ?= 8000
MT_E2_PORT         ?= 8001
MT_E2_DEVICE       ?= cuda
MT_E2_URL          ?= http://127.0.0.1:$(MT_E2_PORT)
export MT_CHECKPOINT_PATH MT_DATA_BIN_PATH MT_DEVICE MT_PORT
export MT_E2_PORT MT_E2_DEVICE MT_E2_URL

# Backgrounded services are launched through the interpreter/binary itself, not
# `conda run` or `npm run`: those fork a child, so the PID we record would not
# be the server we later kill. Each service below is a single process.
UI_PYTHON_BIN    ?= $(CURDIR)/.conda/e1-ui/bin/python
UI_E2_PYTHON_BIN ?= $(CURDIR)/.conda/e2-ui/bin/python
UI_VITE          := $(CURDIR)/ui/node_modules/.bin/vite
UI_RUN_DIR       := $(CURDIR)/.run
UI_WEB_PORT      ?= 5173
UI_LOG_LINES     ?= 40
UI_STOP_TIMEOUT  ?= 15
WITH_E2          ?= 1

# $(1) service name, $(2) command
define ui_start
	@mkdir -p $(UI_RUN_DIR)
	@if [ -s "$(UI_RUN_DIR)/$(1).pid" ] && kill -0 "$$(cat $(UI_RUN_DIR)/$(1).pid)" 2>/dev/null; then \
	   printf '  %-8s already running (pid %s)\n' "$(1)" "$$(cat $(UI_RUN_DIR)/$(1).pid)"; \
	 else \
	   nohup $(2) > "$(UI_RUN_DIR)/$(1).log" 2>&1 & \
	   echo $$! > "$(UI_RUN_DIR)/$(1).pid"; \
	   printf '  %-8s started (pid %s)\n' "$(1)" "$$(cat $(UI_RUN_DIR)/$(1).pid)"; \
	 fi
endef

# $(1) service name, $(2) URL, $(3) seconds, $(4) 1 = required
define ui_wait
	@for i in $$(seq 1 $(3)); do \
	   if curl -sf -m 2 -o /dev/null "$(2)"; then printf '  %-8s ready\n' "$(1)"; exit 0; fi; \
	   sleep 1; \
	 done; \
	 printf '  %-8s NOT ready after %ss - see %s\n' "$(1)" "$(3)" "$(UI_RUN_DIR)/$(1).log"; \
	 [ "$(4)" = "1" ] && exit 1 || exit 0
endef

# One command for the whole desk: E1 gateway, E2 sidecar and Vite, backgrounded
# with pidfiles under .run/. Equivalent to ui-api + ui-api-e2 + ui-web in three
# terminals.  make ui-up WITH_E2=0   starts E1 and the browser UI only.
ui-up:
	@echo "Starting the local MT desk"
	$(call ui_start,gateway,$(UI_PYTHON_BIN) -m mt_pipeline.serving)
	@if [ "$(WITH_E2)" != "1" ]; then \
	   printf '  %-8s disabled (WITH_E2=0)\n' sidecar; \
	 elif [ ! -x "$(UI_E2_PYTHON_BIN)" ]; then \
	   printf '  %-8s skipped (.conda/e2-ui missing - run: make ui-install-e2)\n' sidecar; \
	 else $(MAKE) --no-print-directory ui-up-sidecar; fi
	$(call ui_start,web,$(UI_VITE) ui --port $(UI_WEB_PORT))
	@echo
	$(call ui_wait,gateway,http://127.0.0.1:$(MT_PORT)/api/health,60,1)
	@if [ -s "$(UI_RUN_DIR)/sidecar.pid" ]; then \
	   $(MAKE) --no-print-directory ui-wait-sidecar; fi
	$(call ui_wait,web,http://127.0.0.1:$(UI_WEB_PORT)/,30,1)
	@echo
	@echo "  open  http://127.0.0.1:$(UI_WEB_PORT)"
	@echo "  logs  make ui-logs     stop  make ui-down     state  make ui-status"

ui-up-sidecar:
	$(call ui_start,sidecar,$(UI_E2_PYTHON_BIN) -m mt_pipeline.serving.e2_main)

# The sidecar is optional: a missing GPU must not fail the whole desk.
ui-wait-sidecar:
	$(call ui_wait,sidecar,http://127.0.0.1:$(MT_E2_PORT)/api/health,30,0)

# Waits for each process to actually exit before returning. SIGTERM returns
# immediately, but the sidecar has ~7GB of VRAM to release, and ui-restart would
# otherwise race it and fail to bind the port.
ui-down:
	@stopped=0; \
	 for name in web sidecar gateway; do \
	   f="$(UI_RUN_DIR)/$$name.pid"; [ -s "$$f" ] || continue; \
	   pid=$$(cat "$$f"); \
	   if kill -0 "$$pid" 2>/dev/null; then \
	     kill "$$pid" 2>/dev/null; \
	     for i in $$(seq 1 $(UI_STOP_TIMEOUT)); do \
	       kill -0 "$$pid" 2>/dev/null || break; sleep 1; \
	     done; \
	     if kill -0 "$$pid" 2>/dev/null; then \
	       kill -9 "$$pid" 2>/dev/null; \
	       printf '  %-8s force-killed after %ss (pid %s)\n' "$$name" "$(UI_STOP_TIMEOUT)" "$$pid"; \
	     else \
	       printf '  %-8s stopped (pid %s)\n' "$$name" "$$pid"; \
	     fi; \
	     stopped=1; \
	   fi; \
	   rm -f "$$f"; \
	 done; \
	 [ "$$stopped" = "1" ] || echo "  nothing was running"

ui-restart:
	@$(MAKE) --no-print-directory ui-down
	@$(MAKE) --no-print-directory ui-up

ui-status:
	@for name in gateway sidecar web; do \
	   f="$(UI_RUN_DIR)/$$name.pid"; \
	   if [ -s "$$f" ] && kill -0 "$$(cat $$f)" 2>/dev/null; then \
	     printf '  %-8s running (pid %s)\n' "$$name" "$$(cat $$f)"; \
	   else printf '  %-8s stopped\n' "$$name"; fi; \
	 done
	@echo
	@curl -sf -m 3 http://127.0.0.1:$(MT_PORT)/api/health 2>/dev/null \
	  | python3 -c 'import json,sys;\
d = json.load(sys.stdin);\
print("  health:  " + d["status"]);\
[print("    %-3s %-12s %-5s %s" % (m["key"], m["status"], m["device"], m["model_id"])) for m in d["models"]]' \
	  || echo "  health:  gateway unreachable on port $(MT_PORT)"

ui-logs:
	@for name in gateway sidecar web; do \
	   f="$(UI_RUN_DIR)/$$name.log"; [ -f "$$f" ] || continue; \
	   echo "---- $$name ----"; tail -n $(UI_LOG_LINES) "$$f"; echo; \
	 done


ui-install:
	conda env create -p $(CURDIR)/.conda/e1-ui -f environments/e1-ui-macos.yml
	$(UI_NPM) install

ui-api:
	$(UI_PYTHON) -m mt_pipeline.serving

ui-web:
	$(UI_NPM) run dev

ui-build:
	$(UI_NPM) run build

ui-serve: ui-build
	$(UI_PYTHON) -m mt_pipeline.serving

ui-check:
	$(UI_PYTHON) -m pytest -q ui/backend_tests/test_serving.py \
	  ui/backend_tests/test_serving_models.py \
	  ui/backend_tests/test_remote_translator.py \
	  ui/backend_tests/test_lazy_translator.py \
	  ui/backend_tests/test_e2_translator.py
	$(UI_NPM) run check

ui-private-model:
	$(UI_PYTHON) -m pytest -q ui/backend_tests/test_serving_private_model.py -m private_model

# E1 and E2 cannot share a process: fairseq 0.12.2 needs numpy<2 while the Qwen
# stack needs numpy 2.x. E2 therefore runs as a sidecar in its own environment,
# cloned from .conda/llm so it keeps the exact stack that produced the adapter
# (environments/llm.yml pins pytorch 2.5.1; the env actually holds 2.6.0+cu124).
# The clone hardlinks from the shared package cache, so it costs little disk.
ui-install-e2:
	conda create -y -p $(CURDIR)/.conda/e2-ui --clone $(CURDIR)/.conda/llm
	conda run --no-capture-output -p $(CURDIR)/.conda/e2-ui python -m pip install \
	  fastapi==0.141.1 uvicorn==0.52.4

ui-api-e2:
	$(UI_E2_PYTHON) -m mt_pipeline.serving.e2_main

ui-check-e2:
	$(UI_E2_PYTHON) -m pytest -q ui/backend_tests/test_e2_sidecar.py

ui-e2-model:
	$(UI_E2_PYTHON) -m pytest -q ui/backend_tests/test_e2_translator_model.py -m private_e2_model

# ------------------------------------------------------- comparisons + analysis

compare:
	$(EVAL) compare --baseline $(PRED_DIR)/$(E1_ID).test.jsonl \
	                --candidate $(PRED_DIR)/$(E2_ID).test.jsonl \
	                --seed $(COMPARE_SEED) --output $(METRICS_DIR)/e1_vs_e2.test.json
	$(EVAL) compare --baseline $(PRED_DIR)/$(E1_ID).test.jsonl \
	                --candidate $(PRED_DIR)/$(E3_ID).test.jsonl \
	                --seed $(COMPARE_SEED) --output $(METRICS_DIR)/e1_vs_e3_custom.test.json
	$(EVAL) compare --baseline $(PRED_DIR)/$(E3_ID).test.jsonl \
	                --candidate $(PRED_DIR)/$(E2_ID).test.jsonl \
	                --seed $(COMPARE_SEED) --output $(METRICS_DIR)/e3_custom_vs_e2.test.json

# Does the same retrieval help the strong model? Only runs once E4 exists.
compare-e4:
	$(EVAL) compare --baseline $(PRED_DIR)/$(E2_ID).test.jsonl \
	                --candidate $(PRED_DIR)/$(E4_ID).test.jsonl \
	                --seed $(COMPARE_SEED) --output $(METRICS_DIR)/e2_vs_e4.test.json

error-analysis:
	$(EVAL) prepare-error-analysis \
	  --predictions $(PRED_DIR)/$(E1_ID).test.jsonl \
	                $(PRED_DIR)/$(E2_ID).test.jsonl \
	                $(PRED_DIR)/$(E3_ID).test.jsonl \
	  --sample-size $(ERROR_SAMPLE_SIZE) \
	  --output $(ANALYSIS_DIR)/manual_sample.jsonl

error-analysis-summary:
	$(EVAL) summarize-error-analysis \
	  --annotations $(ANALYSIS_DIR)/manual_sample.jsonl \
	  --output $(ANALYSIS_DIR)/summary.json

# `all` is a fresh-RUN_ROOT target. On the canonical tree e2-train trips the
# no-resume guard and e2-test trips the frozen-selection gate, both by design.
all: audit e1 e2 e3-custom compare error-analysis status

# ------------------------------------------------------------------ repro gate

# Verifies the frozen selections, the stored metrics, and the dataset hashes
# against what is on disk right now.
repro-check:
	$(EVAL) repro-check --scratch-dir /tmp/group10_repro/metrics \
	                    --output /tmp/group10_repro/repro_check.json

# Trains the smoke config twice into two isolated trees and asserts the runs agree.
# --gpu forces cuda+fp16 so the runs take the same path as E1/E3; a CPU fp32 double
# train would exercise none of the CUDA determinism switches and pass vacuously.
repro-smoke:
	@rm -rf /tmp/group10_repro/runA /tmp/group10_repro/runB
	$(EVAL) derive-run-config --config configs/smoke_fairseq.yaml \
	  --run-root /tmp/group10_repro/runA --gpu
	$(EVAL) derive-run-config --config configs/smoke_fairseq.yaml \
	  --run-root /tmp/group10_repro/runB --gpu
	$(FAIRSEQ) train --config /tmp/group10_repro/runA/configs/smoke_fairseq.yaml
	$(FAIRSEQ) train --config /tmp/group10_repro/runB/configs/smoke_fairseq.yaml
	$(FAIRSEQ) predict --config /tmp/group10_repro/runA/configs/smoke_fairseq.yaml --split val
	$(FAIRSEQ) predict --config /tmp/group10_repro/runB/configs/smoke_fairseq.yaml --split val
	$(FAIRSEQ) compare-runs \
	  --baseline-checkpoint /tmp/group10_repro/runA/checkpoint/smoke_fairseq_vi_zh/checkpoint_best.pt \
	  --candidate-checkpoint /tmp/group10_repro/runB/checkpoint/smoke_fairseq_vi_zh/checkpoint_best.pt \
	  --baseline-predictions /tmp/group10_repro/runA/predictions/smoke_fairseq_vi_zh.val.jsonl \
	  --candidate-predictions /tmp/group10_repro/runB/predictions/smoke_fairseq_vi_zh.val.jsonl \
	  --output /tmp/group10_repro/compare_runs.json
