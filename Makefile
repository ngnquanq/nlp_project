.PHONY: audit test test-source test-private check check-source check-private \
        status repro-check repro-smoke derive-configs compare-e4 \
        ui-install ui-api ui-web ui-build ui-serve ui-check ui-private-model \
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
	$(UI_PYTHON) -m pytest -q ui/backend_tests/test_serving.py
	$(UI_NPM) run check

ui-private-model:
	$(UI_PYTHON) -m pytest -q ui/backend_tests/test_serving_private_model.py -m private_model

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
