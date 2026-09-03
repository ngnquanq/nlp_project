# Group 10 — Vietnamese to Classical Chinese MT

This repository implements the Group 10 pipeline from the private course specification (`PROJECT_SPECIFICATION_v1.3.docx`, intentionally excluded from version control). The core corpus is *Đại Việt sử ký toàn thư* and the direction is Vietnamese → Classical Chinese.

The lab data is restricted. Keep the repository and all Kaggle datasets private. The raw `zh-vi/` and unrelated `en-vi/` directories are excluded from version control.

## Current status

E1, E2, and the separately labeled custom E3 have completed locally. Their checkpoints, frozen selections, 510-row validation/test predictions, metrics, and paired bootstrap comparisons are present. Manual error annotations and the final report/slides PDFs remain pending.

See [`docs/EXECUTION_STATUS.md`](docs/EXECUTION_STATUS.md) for the executed results, artifact paths, and remaining deliverables.

Known input facts:

- Train: 19,218 aligned pairs, including 1,596 repeated pairs.
- Validation/test: 510 manually aligned pairs each.
- No cross-split source or pair leakage was detected.
- Six train target rows contain `�`; see the generated data audit for their IDs.
- The knowledge file is present. The official lab E3 integration is unavailable; a separately labeled custom knowledge-retrieval E3 is implemented.

## Setup

Evaluation and local tooling use Conda with Python 3.10:

```bash
conda env create -p .conda/eval -f environments/eval.yml
conda activate "$PWD/.conda/eval"
export PYTHONPATH="$PWD/code"
```

Fairseq and QLoRA use separate Conda environments because their dependency stacks differ:

```bash
conda env create -p .conda/fairseq -f environments/fairseq.yml
conda env create -p .conda/llm -f environments/llm.yml
```

`requirements-llm.lock.txt` records the direct Python package versions resolved by
the completed E2 run. Use it with the pinned model revision when recreating that
software stack; keep `requirements-llm.txt` for compatible fresh environments.

Record `conda env export --from-history` and `python -m pip freeze` in the corresponding run directory before accepting a result.

## Workflow

The Makefile drives every stage and selects the right Conda environment per stage;
the three environments are not interchangeable. Each experiment runs
train → predict val → evaluate → freeze-selection → predict test → evaluate, and
test generation stays locked until the validation selection is frozen.

```bash
make audit                 # dataset validation, hashes, profile

make e1                    # Fairseq baseline          (.conda/fairseq)
make e2                    # Qwen3-8B QLoRA            (.conda/llm)
make e3-custom             # custom knowledge retrieval (.conda/fairseq)

make compare               # three paired bootstraps, seed 42
make error-analysis        # 100-sentence shared annotation sheet (spec §12)
make status                # completed vs missing graded artifacts
```

E4 is an extension, not the named bonus — the bonus is Fairseq + knowledge_v2 (E3).
It applies the same retrieval to the much stronger LLM:

```bash
make knowledge-preflight   # token/runtime cost per candidate budget — run this first
make e4                    # Qwen3-8B QLoRA + knowledge hints  (.conda/llm)
make compare-e4            # E2 vs E4 paired bootstrap, seed 42
```

Individual stages are available as `make e1-train`, `e1-val`, `e1-freeze`,
`e1-test`, and the same pattern for `e2-*` and `e3-custom-*`. `make all` chains
everything in order.

### Reruns never overwrite graded artifacts

`make all` writes to `checkpoint/`, `predictions/`, `metrics/` and
`error_analysis/`. To re-run an experiment without touching results already
produced, set `RUN_ROOT` and the whole run is redirected into a parallel tree:

```bash
make all RUN_ROOT=runs/rerun-$(date +%F-%H%M%S)
```

On the canonical tree `make all` deliberately stops: `e2-train` refuses to
overwrite the existing adapter and `e2-test` refuses to generate against a frozen
selection that no longer verifies. Only read-only targets and `make repro-check`
run there.

### Reproducibility

```bash
make check-source          # CI-safe tests; no restricted corpus required
make check-private         # all tests + data audit + saved-artifact verification
make repro-check           # frozen selections, stored metrics, dataset hashes
make repro-smoke           # trains the smoke config twice, asserts the runs agree
```

`repro-smoke` derives two isolated run trees, forces CUDA+fp16 so they take the
same runtime path as E1/E3, trains and decodes both, then compares model weights
and translation content. Run-scoped state is excluded from the comparison by
design: wall-clock timers and absolute paths inside the checkpoint, Fairseq's
uninitialised `embed_positions._float_tensor` device-tracking buffer, and the
`checkpoint` path recorded in each prediction row.

**Scope of the determinism guarantee.** `repro-smoke` covers the Fairseq path
(E1 and E3) only, and it covers the smoke model — 2 layers at 64 dim for 2 updates
— not E1/E3 at full scale. Fairseq training and decoding run under
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(warn_only=True)`; `warn_only` means an operation
without a deterministic kernel warns and proceeds rather than aborting, so check
`work/<id>/train.log` for such warnings before claiming a full-scale run is
bit-reproducible. E2 (QLoRA) has no equivalent guarantee: it trains in-process,
seeded only via `set_seed`, over NF4 bitsandbytes kernels. For E2 the reproducible
artifact is the saved adapter plus its manifest, not a repeatable retrain.

Training refuses to resume silently. Fairseq defaults `--restore-file` to
`checkpoint_last.pt`, so re-running training used to warm-start from the previous
attempt and produce a different model from the same command. Both backends stop
unless the config sets `resume: true`; QLoRA additionally requires and records an
exact `work/<id>/trainer/checkpoint-*` resume source.

Use `python -m mt_pipeline --help` for all options. `make check` aliases the
source-only CI gate. `make check-private` adds the restricted-corpus tests, data
audit, and `repro-check`; it needs neither model downloads nor a GPU but does need
the private inputs and saved artifacts. `repro-check` reports `skipped` for absent
generated artifacts; `repro-smoke` requires a GPU.

## Local translation UI

The `feature/e1-mt-web-ui` branch adds a local React/FastAPI translation desk for
the E1 model. The interface and API are safe to publish, while the private
checkpoint and Fairseq `data-bin` remain external and are injected with environment
variables. The desk serves both E1 and E2 — E2 runs as a separate sidecar process
because the Fairseq and Qwen dependency stacks are mutually exclusive. See
[`docs/MT_UI.md`](docs/MT_UI.md) for the architecture, setup and operating instructions.

## Experiment policy

- E1 and E2 use only the official train split for optimization and validation for checkpoint selection.
- Test is used once configurations are frozen.
- E1 preserves the corpus tokenization and rare characters.
- E2 trains on natural unspaced Chinese but stores a character-spaced `prediction_scored` value for exactly comparable metrics.
- The official E3 ID remains blocked until lab `knowledge_v2` code is supplied.
- The custom E3 uses source-side retrieval from `zh-vi/knowledge.json` under the distinct ID `e3_custom_fairseq_knowledge_vi_zh_v1`; it must not be reported as the official method.

## Required outputs

Real runs populate `checkpoint/`, `predictions/`, and `metrics/`. These generated artifacts are ignored by default; submit them through the course-approved private channel or provide private links and SHA-256 hashes. Complete the templates in `report/`, `slides/`, and `error_analysis/` only from saved outputs. Follow [`docs/GITHUB_PUBLISHING.md`](docs/GITHUB_PUBLISHING.md) before staging any source.
