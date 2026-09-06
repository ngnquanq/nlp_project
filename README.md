# Group 10 — Vietnamese to Classical Chinese Machine Translation

A reproducible machine-translation pipeline that translates modern Vietnamese
into Classical Chinese (Hán văn cổ), built on a small, domain-specific parallel
corpus drawn from *Đại Việt sử ký toàn thư*.

> **Restricted data.** The lab corpus is not public. Keep this repository and all
> Kaggle datasets private. The raw `zh-vi/` corpus and the unrelated `en-vi/`
> directory are excluded from version control, as is the course specification
> (`PROJECT_SPECIFICATION_v1.3.docx`). Read
> [`docs/GITHUB_PUBLISHING.md`](docs/GITHUB_PUBLISHING.md) before staging any source.

## Table of Contents

* [Introduction](#introduction)
    * [Project Goal](#project-goal)
    * [Data Source](#data-source)
* [Repository Structure](#repository-structure)
* [High-level System Architecture](#high-level-system-architecture)
* [Guide to Install and Run Code](#guide-to-install-and-run-code)
    * [Briefly Introduce about the CI Process](#briefly-introduce-about-the-ci-process)
    * [Requirements](#requirements)
    * [Setup the Environment Variables](#setup-the-environment-variables)
        * [Private Artifact Paths](#private-artifact-paths)
        * [Gateway Runtime](#gateway-runtime)
        * [E2 Sidecar](#e2-sidecar)
    * [Setup the Infrastructure](#setup-the-infrastructure)
    * [Run the Experiment Pipeline](#run-the-experiment-pipeline)
        * [Audit the Dataset](#audit-the-dataset)
        * [E1 — Fairseq Transformer Baseline](#e1--fairseq-transformer-baseline)
        * [E2 — Qwen3-8B QLoRA](#e2--qwen3-8b-qlora)
        * [E3-custom — Fairseq with Knowledge Retrieval](#e3-custom--fairseq-with-knowledge-retrieval)
        * [Compare Systems and Prepare Error Analysis](#compare-systems-and-prepare-error-analysis)
        * [Verify Reproducibility](#verify-reproducibility)
        * [Rerun Without Overwriting Graded Artifacts](#rerun-without-overwriting-graded-artifacts)
    * [Navigate to the System's Components](#navigate-to-the-systems-components)
        * [Translation Desk (React + FastAPI Gateway)](#translation-desk-react--fastapi-gateway)
        * [E2 Sidecar Service](#e2-sidecar-service)
        * [Generated Artifacts](#generated-artifacts)
* [Experiment Policy](#experiment-policy)
* [Further Development](#further-development)

## Introduction

### Project Goal

Translate Vietnamese source sentences into Classical Chinese and measure, with
statistical evidence, how far a fine-tuned large language model outperforms a
Transformer trained from scratch on a corpus of roughly twenty thousand pairs.

Three experiments are implemented and completed:

| ID | System | Test BLEU | Test chrF++ |
|---|---|---:|---:|
| **E1** | Transformer trained from scratch with Fairseq | 28.63 | 30.58 |
| **E2** | Qwen3-8B adapted with 4-bit QLoRA | 40.77 | 40.87 |
| **E3-custom** | Fairseq with source-side Hán character hints | 29.62 | 31.41 |

E2 beats E1 by **+12.14 BLEU** (95% CI [10.22, 13.77], *p* = 0.002). E3-custom
improves on E1 by +0.99 BLEU (*p* = 0.092), which is not significant at 0.05.
All figures come from 510 held-out test pairs scored with SacreBLEU 2.6.0, and
each test run was generated only after its validation selection was frozen. See
[`docs/EXECUTION_STATUS.md`](docs/EXECUTION_STATUS.md) for the full record.

The named bonus experiment is Fairseq combined with the lab's `knowledge_v2`
integration (E3). That integration has not been supplied, so the repository
implements a separately labelled custom retrieval variant instead.

### Data Source

The corpus is a manually aligned Vietnamese ↔ Classical Chinese parallel set
built from *Đại Việt sử ký toàn thư*, supplied through the course and stored
under the ignored `zh-vi/` directory.

| Split | Pairs | Notes |
|---|---:|---|
| Train | 19,218 | includes 1,596 repeated pairs |
| Validation | 510 | manually aligned |
| Test | 510 | manually aligned |

No cross-split source or pair leakage was detected. Six train target rows
contain the replacement character `�`; their IDs are listed in the generated
data audit. A retrieval knowledge file (`zh-vi/knowledge.json`) is present and
feeds E3-custom. Full documentation lives in [`docs/DATASET.md`](docs/DATASET.md).

## Repository Structure

```
.
├── code/mt_pipeline/          # the pipeline package — one CLI, all stages
│   ├── cli.py                 #   subcommand definitions and argument parsing
│   ├── data.py                #   corpus loading, audit, hashing, profiling
│   ├── fairseq_*.py           #   E1 / E3-custom training and generation
│   ├── llm_runner.py          #   E2 QLoRA training and generation
│   ├── knowledge.py           #   source-side retrieval for E3-custom
│   ├── evaluation.py          #   SacreBLEU, chrF++, paired bootstrap
│   ├── freeze.py              #   locks checkpoint + config after validation
│   ├── repro_check.py         #   re-verifies frozen selections and metrics
│   └── serving/               #   FastAPI gateway and E2 sidecar
├── configs/                   # one YAML per experiment, plus dataset + smoke
├── environments/              # Conda specs for the three isolated stacks
├── tests/                     # pytest suite (source-only and private markers)
├── ui/                        # React translation desk + backend tests
├── docs/                      # dataset, pipeline, experiments, status, policy
├── notebooks/                 # exploratory notebooks
├── report/  slides/           # LaTeX sources for the deliverables
├── Error_Analysis.py          # heuristic auto-labeler, run by the Kaggle notebook
├── make_submission_bundle.py  # assembles the private artifact archive for submission
├── Makefile                   # high-level wrappers over the CLI commands below
└── .github/workflows/         # source-only CI gate
```

Generated at run time and ignored by git:

```
checkpoint/<experiment_id>/    # model weights and adapters
work/<experiment_id>/          # data-bin, run manifests, frozen selections, logs
predictions/*.jsonl            # schema-validated per-sentence output
metrics/*.json                 # corpus metrics and paired comparisons
error_analysis/                # the shared 100-sentence annotation sheet
runs/                          # parallel trees produced by RUN_ROOT reruns
zh-vi/  en-vi/                 # restricted corpora, never committed
```

## High-level System Architecture

A one-time corpus audit gates the data; each experiment then moves through the
same six stages, and the CLI enforces the ordering: **test generation stays locked
until the validation selection is frozen.**

```
   zh-vi/  (restricted corpus) ──► [eval] data-audit ──► metrics/data_audit.json
        │                           one-time gate, driven by configs/dataset.yaml
        │
        └───────────────┬───────────── configs/<experiment>.yaml
                        ▼
   [fairseq|llm]  train                    ──►  checkpoint/<id>/
                        │
                        ▼
   [fairseq|llm]  predict --split val      ──►  predictions/<id>.val.jsonl
                        │
                        ▼
   [eval]      evaluate                    ──►  metrics/<id>.val.json
                        │
                        ▼
   [eval]      freeze-selection            ──►  work/<id>/selection_frozen.json
                        │
   ═══ test stays sealed until the validation selection is frozen ═══
                        │
                        ▼
   [fairseq|llm]  predict --split test     ──►  predictions/<id>.test.jsonl
                        │
                        ▼
   [eval]      evaluate                    ──►  metrics/<id>.test.json
                        │
      E1 ───────────────┼─────────────── E2 ─── E3-custom
                        ▼
   [eval]      compare                     ──►  metrics/<a>_vs_<b>.test.json
   [eval]      prepare-error-analysis      ──►  error_analysis/manual_sample.jsonl
```

The tag in brackets is the Conda environment that owns each stage.

Three Conda environments back these stages and **are not interchangeable**:

| Environment | Owns | Used by |
|---|---|---|
| `.conda/eval` | SacreBLEU 2.6.0, sacremoses, PyYAML, pytest — no torch, no fairseq | `data-audit`, `evaluate`, `freeze-selection`, `compare`, `prepare-error-analysis`, `project-status`, `repro-check` |
| `.conda/fairseq` | fairseq 0.12.2, numpy<2 | `train` / `predict` for E1 and E3-custom |
| `.conda/llm` | transformers, peft, bitsandbytes, numpy 2.x | `train` / `predict` for E2 |

The same split shapes the serving layer. Fairseq and the Qwen stack cannot share
a process, so the translation desk runs as a gateway plus an optional sidecar:

```
  browser ──► Vite dev server (:5173)
                    │
                    ▼
        FastAPI gateway (:8000)  ── E1 fairseq model, in process
                    │
                    └── forwards E2 requests ──► E2 sidecar (:8001)
                                                 Qwen3-8B + QLoRA adapter
```

## Guide to Install and Run Code

### Briefly Introduce about the CI Process

There is no deployment pipeline; CI is a single source-only correctness gate at
[`.github/workflows/source-ci.yml`](.github/workflows/source-ci.yml). On every
push and pull request it checks out the repo on `ubuntu-latest`, installs Python
3.10 and `requirements-eval.txt`, and runs:

```bash
make check-source EVAL_PYTHON=python
# actual command:
python -m pytest -q -m "not private_data"
```

The `not private_data` marker is what makes this safe to run on GitHub: the gate
needs no restricted corpus, no model downloads, and no GPU. The heavier gates —
`check-private` and `repro-check` — are local-only, because they read the private
inputs and the saved artifacts.

### Requirements

* **Python 3.10** and **Conda** (Miniconda or Anaconda).
* An **NVIDIA GPU with CUDA** to *train* E1, E2 and E3-custom. E2 was trained on
  a single GPU using NF4 4-bit quantisation and needs roughly 7 GB of VRAM to
  serve. E1 decoding runs on CPU (that is the default in `.env.example`, and
  `environments/e1-ui-macos.yml` exists for Apple Silicon); evaluation,
  comparison and error analysis are CPU-only throughout.
* **Node.js and npm**, only if you want the browser translation desk.
* The private corpus in `zh-vi/`. Without it, only the source-only test gate runs.

### Setup the Environment Variables

Copy the template and fill in absolute paths. Never commit the resolved values.

```bash
cp .env.example .env
```

**Nothing auto-loads this file.** `code/mt_pipeline/serving/settings.py` reads
`os.environ` directly, so export the values into the shell that starts the
services:

```bash
set -a; source .env; set +a
```

The `make ui-*` targets sidestep this — they define every variable with `?=` and
export it, so `make ui-up` works out of the box and an explicit environment value
still wins. You need `.env` when running the services by hand, or to override a
path. These variables affect only the serving layer; the training and evaluation
pipeline is driven entirely by the YAML files in `configs/`.

#### Private Artifact Paths

Point the gateway at the E1 checkpoint and its Fairseq `data-bin`. Both stay
outside the repository.

```bash
MT_CHECKPOINT_PATH=/absolute/path/to/checkpoint/e1_fairseq_vi_zh_v1/checkpoint_best.pt
MT_DATA_BIN_PATH=/absolute/path/to/work/e1_fairseq_vi_zh_v1/data-bin
```

#### Gateway Runtime

```bash
MT_DEVICE=cpu     # E1 runs on CPU on Apple Silicon; set cuda on Linux/NVIDIA
MT_PORT=8000
```

#### E2 Sidecar

Leave `MT_E2_URL` unset to run the desk with E1 only. Setting it makes the
gateway forward E2 requests to the sidecar.

```bash
# read by the gateway
MT_E2_URL=http://127.0.0.1:8001
MT_E2_TIMEOUT_SECONDS=900
MT_E2_HEALTH_TTL_SECONDS=5

# read by the sidecar
MT_E2_PORT=8001
MT_E2_CONFIG_PATH=configs/e2_qwen3_qlora.yaml
MT_E2_MAX_PROMPT_TOKENS=512
MT_E2_DEVICE=cuda
```

`MT_E2_ADAPTER_PATH` and `MT_E2_MANIFEST_PATH` default to the paths recorded in
the E2 config; override them only to relocate those artifacts.

### Setup the Infrastructure

Create the three environments and put the package on the import path. **Every
raw `python -m mt_pipeline` invocation below needs `PYTHONPATH`** — the Makefile
exports it for you, so a bare shell will fail with `ModuleNotFoundError` without it.

The command blocks below use `conda activate`, which assumes a shell where
`conda init` has run. If it has not, use the Makefile's form instead:
`conda run --no-capture-output -p .conda/<env> python -m mt_pipeline ...`.

```bash
export PYTHONPATH="$PWD/code"

conda env create -p .conda/eval    -f environments/eval.yml      # Python 3.10
conda env create -p .conda/fairseq -f environments/fairseq.yml
conda env create -p .conda/llm     -f environments/llm.yml
```

`requirements-llm.lock.txt` records the exact direct package versions resolved by
the completed E2 run; use it with the pinned model revision when recreating that
software stack. `requirements-llm.txt` is the looser spec for fresh environments.

Before accepting any result, record the environment in the run directory:

```bash
conda env export --from-history > work/<experiment_id>/conda_env.yml
python -m pip freeze              > work/<experiment_id>/pip_freeze.txt
```

### Run the Experiment Pipeline

Each `make` target is a thin wrapper. The raw commands are given below so the
pipeline runs without `make` — note that a single target often **switches
environments mid-stage**, because generation and scoring live in different stacks.

If you prefer the wrappers, the whole project is:

```bash
make audit && make e1 && make e2 && make e3-custom && make compare && make error-analysis && make status
# or simply:  make all
```

#### Audit the Dataset

Validates the corpus, records SHA-256 hashes, and writes the profile.

```bash
conda activate "$PWD/.conda/eval"          # make audit
python -m mt_pipeline data-audit \
  --config configs/dataset.yaml \
  --output metrics/data_audit.json
```

#### E1 — Fairseq Transformer Baseline

```bash
# make e1  (= e1-train → e1-val → e1-freeze → e1-test)
ID=e1_fairseq_vi_zh_v1
CFG=configs/e1_fairseq.yaml

conda activate "$PWD/.conda/fairseq"
python -m mt_pipeline train   --config $CFG
python -m mt_pipeline predict --config $CFG --split val

conda activate "$PWD/.conda/eval"
python -m mt_pipeline evaluate \
  --predictions predictions/$ID.val.jsonl \
  --output      metrics/$ID.val.json
python -m mt_pipeline freeze-selection --config $CFG \
  --validation-predictions predictions/$ID.val.jsonl \
  --validation-metrics     metrics/$ID.val.json

conda activate "$PWD/.conda/fairseq"
python -m mt_pipeline predict --config $CFG --split test

conda activate "$PWD/.conda/eval"
python -m mt_pipeline evaluate \
  --predictions predictions/$ID.test.jsonl \
  --output      metrics/$ID.test.json
```

#### E2 — Qwen3-8B QLoRA

Identical stage order; `train` and `predict` move to the `.conda/llm`
environment while scoring stays in `.conda/eval`.

```bash
# make e2
ID=e2_qwen3_8b_qlora_vi_zh_v1
CFG=configs/e2_qwen3_qlora.yaml

conda activate "$PWD/.conda/llm"
python -m mt_pipeline train   --config $CFG
python -m mt_pipeline predict --config $CFG --split val

conda activate "$PWD/.conda/eval"
python -m mt_pipeline evaluate \
  --predictions predictions/$ID.val.jsonl \
  --output      metrics/$ID.val.json
python -m mt_pipeline freeze-selection --config $CFG \
  --validation-predictions predictions/$ID.val.jsonl \
  --validation-metrics     metrics/$ID.val.json

conda activate "$PWD/.conda/llm"
python -m mt_pipeline predict --config $CFG --split test

conda activate "$PWD/.conda/eval"
python -m mt_pipeline evaluate \
  --predictions predictions/$ID.test.jsonl \
  --output      metrics/$ID.test.json
```

#### E3-custom — Fairseq with Knowledge Retrieval

Same environment split as E1 — retrieval from `zh-vi/knowledge.json` runs on the
source side inside `train` and `predict`, so no extra stage is needed.

```bash
# make e3-custom
ID=e3_custom_fairseq_knowledge_vi_zh_v1
CFG=configs/e3_custom_fairseq_knowledge.yaml

conda activate "$PWD/.conda/fairseq"
python -m mt_pipeline train   --config $CFG
python -m mt_pipeline predict --config $CFG --split val

conda activate "$PWD/.conda/eval"
python -m mt_pipeline evaluate \
  --predictions predictions/$ID.val.jsonl \
  --output      metrics/$ID.val.json
python -m mt_pipeline freeze-selection --config $CFG \
  --validation-predictions predictions/$ID.val.jsonl \
  --validation-metrics     metrics/$ID.val.json

conda activate "$PWD/.conda/fairseq"
python -m mt_pipeline predict --config $CFG --split test

conda activate "$PWD/.conda/eval"
python -m mt_pipeline evaluate \
  --predictions predictions/$ID.test.jsonl \
  --output      metrics/$ID.test.json
```

#### Compare Systems and Prepare Error Analysis

Paired bootstrap uses **1,000 samples at seed 42**. The CLI default is 12345, so
omitting `--seed` will not reproduce the stored numbers.

```bash
conda activate "$PWD/.conda/eval"          # make compare
E1=predictions/e1_fairseq_vi_zh_v1.test.jsonl
E2=predictions/e2_qwen3_8b_qlora_vi_zh_v1.test.jsonl
E3=predictions/e3_custom_fairseq_knowledge_vi_zh_v1.test.jsonl

python -m mt_pipeline compare --baseline $E1 --candidate $E2 --seed 42 \
  --output metrics/e1_vs_e2.test.json
python -m mt_pipeline compare --baseline $E1 --candidate $E3 --seed 42 \
  --output metrics/e1_vs_e3_custom.test.json
python -m mt_pipeline compare --baseline $E3 --candidate $E2 --seed 42 \
  --output metrics/e3_custom_vs_e2.test.json
```

The specification requires 100 shared test sentences for manual annotation:

```bash
python -m mt_pipeline prepare-error-analysis \
  --predictions $E1 $E2 $E3 \
  --sample-size 100 \
  --output error_analysis/manual_sample.jsonl     # make error-analysis

python -m mt_pipeline summarize-error-analysis \
  --annotations error_analysis/manual_sample.jsonl \
  --output      error_analysis/summary.json       # make error-analysis-summary
```

Two further read-only commands:

```bash
python -m mt_pipeline project-status               # make status
python -m mt_pipeline rescore-moses \
  --prediction-dir predictions --metrics-dir metrics \
  --output-dir metrics/moses                       # make rescore-moses
```

`rescore-moses` re-scores saved predictions with the `moses-char-v1` tokenisation
without retraining anything; see [`docs/MOSES_EVALUATION.md`](docs/MOSES_EVALUATION.md).

#### Verify Reproducibility

```bash
conda activate "$PWD/.conda/eval"
python -m pytest -q -m "not private_data"          # make check-source
python -m pytest -q                                # the private-data tests too

python -m mt_pipeline repro-check \
  --scratch-dir /tmp/group10_repro/metrics \
  --output      /tmp/group10_repro/repro_check.json   # make repro-check
```

`make check-private` chains all three: the full pytest run, a `data-audit` into a
scratch path, and `repro-check`. It needs the private inputs and the saved
artifacts, but neither model downloads nor a GPU.

`repro-check` re-verifies the frozen selections, the stored metrics and the
dataset hashes against what is on disk right now; it reports `skipped` for
artifacts that are absent. `make repro-smoke` goes further — it trains the smoke
config twice into two isolated trees under `/tmp/group10_repro/` and asserts the
runs agree on both weights and translations. It requires a GPU.

**Scope of the determinism guarantee.** `repro-smoke` covers the Fairseq path
(E1 and E3-custom) only, and only at smoke scale — 2 layers, 64 dim, 2 updates —
not E1/E3 at full size. Fairseq runs under `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(warn_only=True)`; `warn_only` means an op
without a deterministic kernel warns and proceeds rather than aborting, so check
`work/<id>/train.log` for such warnings before calling a full-scale run
bit-reproducible. **E2 has no equivalent guarantee**: it trains in-process, seeded
only through `set_seed`, over NF4 bitsandbytes kernels. For E2 the reproducible
artifact is the saved adapter plus its manifest, not a repeatable retrain.

Training also refuses to resume silently. Fairseq defaults `--restore-file` to
`checkpoint_last.pt`, which used to warm-start a re-run from the previous attempt
and produce a different model from the same command. Both backends now stop unless
the config sets `resume: true`, and QLoRA additionally requires and records an
exact `work/<id>/trainer/checkpoint-*` resume source.

#### Rerun Without Overwriting Graded Artifacts

The published E1, E2 and E3-custom results are final and are not to be
regenerated. The canonical tree defends itself: there, `e2-train` refuses to
overwrite the existing adapter and `e2-test` refuses to generate against a frozen
selection that no longer verifies. Only read-only targets and `repro-check` are
meant to run there.

If you nevertheless need a fresh run — for a new experiment, not to replace these
numbers — redirect the entire run into a parallel tree:

```bash
make all RUN_ROOT=runs/rerun-$(date +%F-%H%M%S)
```

`RUN_ROOT` rebases `configs/`, `work/`, `predictions/`, `metrics/` and
`error_analysis/` under that directory, so `checkpoint/`, `predictions/`,
`metrics/` and `error_analysis/` in the canonical tree are untouched. The raw
equivalent derives each config first:

```bash
conda activate "$PWD/.conda/eval"
python -m mt_pipeline derive-run-config \
  --config configs/e1_fairseq.yaml --run-root runs/my-rerun
```

### Navigate to the System's Components

#### Translation Desk (React + FastAPI Gateway)

A local translation desk serves both E1 and E2. The interface and API are safe to
publish; the private checkpoint and `data-bin` stay external and are injected
through the environment variables above.

```bash
make ui-install     # conda env create -p .conda/e1-ui -f environments/e1-ui-macos.yml
                    # npm --prefix ui install

make ui-up          # starts gateway + sidecar + Vite, backgrounded with pidfiles in .run/
make ui-status      # what is running
make ui-logs        # tail the service logs
make ui-down        # stop everything, waiting for the sidecar to release VRAM
```

Then open <http://127.0.0.1:5173>. Use `make ui-up WITH_E2=0` for an E1-only desk.

To run the services in the foreground instead, in separate terminals:

```bash
.conda/e1-ui/bin/python -m mt_pipeline.serving        # gateway  :8000
.conda/e2-ui/bin/python -m mt_pipeline.serving.e2_main # sidecar :8001
npm --prefix ui run dev                                # web     :5173
```

Backgrounded services are launched through the interpreter binary rather than
`conda run` or `npm run`, because those fork a child and the recorded PID would
not be the server later killed.

#### E2 Sidecar Service

E2 cannot share a process with E1: fairseq 0.12.2 needs `numpy<2` while the Qwen
stack needs `numpy 2.x`. The sidecar therefore runs in its own environment,
cloned from `.conda/llm` so it keeps the exact stack that produced the adapter.

```bash
make ui-install-e2  # conda create -p .conda/e2-ui --clone .conda/llm
                    # + pip install fastapi==0.141.1 uvicorn==0.52.4
make ui-check-e2    # sidecar contract tests
```

The sidecar is optional by design — a missing GPU must not fail the whole desk.
Architecture, setup and operating instructions are in [`docs/MT_UI.md`](docs/MT_UI.md).

#### Generated Artifacts

| Component | Location |
|---|---|
| Corpus metrics and paired comparisons | `metrics/*.json` |
| Moses-tokenised rescoring | `metrics/moses/` |
| Per-sentence predictions | `predictions/*.jsonl` |
| Frozen selections and run manifests | `work/<experiment_id>/` |
| Model weights and adapters | `checkpoint/<experiment_id>/` |
| Shared 100-sentence annotation sheet | `error_analysis/manual_sample.jsonl` |
| Heuristic error labels and summary | `error_analysis/automatic_{labels.csv,summary.json}` |
| Heuristic labelling workbooks | `Error_Analysis_{PreLabeled,Labeled}.xlsx` |
| Training logs | `work/<experiment_id>/train.log` |

These are ignored by git. Complete the templates in `report/`, `slides/` and
`error_analysis/` only from saved outputs.

To hand the artifacts over, build the private archive rather than zipping directories
by hand — the script selects exactly the files the specification asks for and refuses
to run if restricted material reaches the manifest:

```bash
python3 make_submission_bundle.py --dry-run   # print the manifest, write nothing
python3 make_submission_bundle.py             # group10_submission_bundle.zip
```

It carries the three experiments' checkpoints and adapters, every validation/test
prediction and metric, the paired comparisons, run manifests, frozen selections,
`docs/DATASET.md`, and the error-analysis sheet and workbooks. Share it through the
course-approved private channel; restrict the link to the recipient rather than
using an anyone-with-link share.

## Experiment Policy

* E1 and E2 use only the official train split for optimisation, and validation
  only for checkpoint selection.
* Test is used once, after configurations are frozen.
* E1 preserves the corpus tokenisation and its rare characters.
* E2 trains on natural unspaced Chinese but stores a character-spaced
  `prediction_scored` value so metrics remain exactly comparable across systems.
* The official E3 ID stays blocked until the lab `knowledge_v2` code is supplied.
* E3-custom uses source-side retrieval from `zh-vi/knowledge.json` under the
  distinct ID `e3_custom_fairseq_knowledge_vi_zh_v1`. **It must not be reported
  as the official method.**

## Further Development

* **Official E3.** The lab `knowledge_v2` integration is still unavailable. Once
  supplied, it slots into `configs/e3_fairseq_knowledge.yaml` and runs through
  the same stage chain.
* **Manual error analysis.** All 300 rows in
  `error_analysis/manual_sample.jsonl` — 100 shared test sentences × 3 systems —
  are still `annotation_status: PENDING`. Until
  two annotators label and adjudicate them, no error-rate or entity/date accuracy
  claim can be made. Guidelines are in `error_analysis/GUIDELINES.md`.
* **Deliverables.** The LaTeX sources in `report/` and `slides/` need to be
  populated from the saved metrics and compiled to PDF.
* **Full-scale determinism.** The reproducibility guarantee currently covers the
  Fairseq path at smoke scale. Extending it to full-scale E1/E3 runs, and finding
  a workable determinism story for QLoRA, are both open.

Run `python -m mt_pipeline --help` for the complete command reference, and see
[`docs/PIPELINE.md`](docs/PIPELINE.md) and
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for design detail.
