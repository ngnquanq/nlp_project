# Reproducibility audit and specification gap analysis — 2026-08-17

> Historical audit. The Git hygiene, QLoRA logging, E4 preflight dependency,
> annotation extension, and model-revision findings below were remediated on
> 2026-08-20. Artifact-integrity caveats remain historical facts.

Two questions: are E1, E2, and the custom E3 reproducible through the Makefile, and what
still separates this workspace from what `PROJECT_SPECIFICATION_v1.3.docx` asks for.

Method: static analysis of the Makefile, `code/mt_pipeline/`, `configs/`, and every stored
artifact, manifest, hash, and training log; `make -n` dry runs of `e1`, `e2`, `e3-custom`,
and `all RUN_ROOT=…`; one live `make repro-check`, which writes only under `/tmp`. No
training was run. No configuration was changed. No stored result was altered.

Frozen-result constraint: nothing recorded here proposes retraining or editing an
experiment configuration.

## 1. Verdict

"Reproducible" is three separate questions, and the three experiments score differently on
each.

| | Re-runnable from `make` | Re-derivable — same numbers from a clean run | Verifiable — numbers provable from stored artifacts |
|---|---|---|---|
| E1 Fairseq | yes, via `RUN_ROOT` (dry run only) | **no — warm-started run** | full |
| E2 Qwen3-8B QLoRA | yes, via `RUN_ROOT` (dry run only); environment drift | no, and never claimed | partial — freeze link broken, raw log destroyed |
| E3 custom knowledge | yes, via `RUN_ROOT` (dry run only) | plausible, unproven at full scale | full, plus a pinned regression test |

The re-runnable column rests on `make -n` resolution plus the environments and corpus
being present. No rerun was executed, so `derive-run-config` emitting a valid YAML and
`prepare_fairseq` rebuilding `data-bin` in a fresh tree are inferred, not observed.

The pipeline machinery is strong. The failures are specific and local.

### What holds up

- `make -n e1`, `e2`, `e3-custom` all resolve: train → predict val → evaluate →
  freeze-selection → predict test → evaluate, each stage under the environment that owns
  its dependencies. All three environments exist with the expected stacks: `.conda/fairseq`
  fairseq 0.12.2 / torch 1.13.1, `.conda/llm` transformers 4.57.6 / peft 0.20.0 /
  bitsandbytes 0.50.1 / torch 2.5.1, `.conda/eval` sacrebleu 2.6.0.
- `make all RUN_ROOT=runs/probe` rebases every writable path. Only `work_dir`,
  `checkpoint_dir`, and `prediction_dir` are relocated (`reruns.py:26`); every other
  path-bearing config field — `dataset_config`, `knowledge.file` — is a read-only input
  and correctly left pointing at the shared tree. No path exists by which a `RUN_ROOT`
  rerun would write into or read stale data from the canonical `checkpoint/`,
  `predictions/`, or `metrics/`.
- `make repro-check` passes live. All six stored metric files (E1, E2, E3 × val, test)
  recompute exactly from the saved predictions, and every dataset and knowledge SHA-256
  still matches `zh-vi/` on disk.
- Provenance capture is automatic and more complete than `README.md:38` implies. Each
  `work/<id>/run_manifest.json` records the full config dict and its SHA-256, dataset
  fingerprints, `conda list --json` (134 packages for E2), `pip freeze` (65), platform,
  hostname, accelerator, and for E2 `resolved_model_revision:
  b968826d9c46dd6066d109eabc6255188de91218` — the actual Hugging Face commit behind
  `revision: main`.
- All eight prediction files are exactly 510 rows and carry the seven fields §13 requires,
  plus a per-row `decoding_config`. `evaluate` refuses to score if `prediction_scored`
  drifts from canonical character-spaced form (`evaluation.py:34-37`).
- The no-resume guards work. `make e1-train` and `make e2-train` on the canonical tree
  abort rather than silently clobber. 34 tests across 9 files, including
  `test_fairseq_augmentation_matches_recorded_e3_hashes`, which re-derives E3's
  knowledge-augmented source text from the live `zh-vi/knowledge.json` and asserts it
  still hashes to the values pinned in
  `work/e3_custom_fairseq_knowledge_vi_zh_v1/knowledge_augmentation.json`.
- Neither Fairseq training log contains a single non-deterministic-op warning, so the
  `warn_only=True` caveat in `README.md:104-107` did not bite at full scale. The absence
  is meaningful rather than an artifact of logging: `run_logged` runs the subprocess with
  `stderr=subprocess.STDOUT` (`runtime.py:39-45`), so PyTorch's `warnings`-module output
  would have landed in those logs had any operation lacked a deterministic kernel.

### Blocker 1 — E1's reported checkpoint was warm-started

`work/e1_fairseq_vi_zh_v1/train.log:268`:

```
2026-08-15 16:04:51 | INFO | fairseq.trainer | Loaded checkpoint
  .../checkpoint/e1_fairseq_vi_zh_v1/checkpoint_last.pt (epoch 5 @ 579 updates)
```

The reported E1 model is the product of two chained runs: an interrupted first attempt,
preserved as `train.interrupted_epoch_checkpoints.log` and ending in a `KeyboardInterrupt`
inside `fairseq/utils.py:113`, and a resumed second run that early-stopped at epoch 171 /
24,791 updates on patience 20. A single clean `make e1-train` will not reproduce BLEU
28.63. The guard that now prevents this (`ensure_clean_checkpoint_dir`,
`fairseq_runner.py:102-115`) was added after E1 ran.

This is a Limitations disclosure, not a fabrication. The run happened and both logs
survive. But the one-command claim is false for E1 and must not be made.

E3 is clean by contrast — no `Loaded checkpoint` line appears anywhere in its log.

### Blocker 2 — E2's frozen-selection chain is broken, though less badly than the code comments claim

Three different SHA-256 values for `configs/e2_qwen3_qlora.yaml` appear in this
workspace's own provenance records:

| Moment | config SHA-256 |
|---|---|
| Train time — `run_manifest.json`, `created_at` 2026-08-15T22:18:28Z | `d4057d535fd9bf91…` |
| Freeze and test generation — `selection_frozen.json` | `ca68e42cd1c80300…` — lost, no version control |
| File on disk today | `d4057d535fd9bf91…` |

The configuration was edited after training, frozen and used for test generation in that
intermediate state, then edited back to its training-time content. `ensure_selection_frozen`
(`freeze.py:75-76`) therefore raises `Experiment config changed after validation selection
was frozen` today, and `test_generation_authorized` cannot be re-verified.

`repro_check.py:25-28` states that "the config bytes behind BLEU 40.77 are unrecoverable."
That is too pessimistic. Verified against the artifacts:

- `run_manifest.json` archives the complete training config dict, and it is
  field-for-field identical to the file on disk today, `max_sequence_length: 768`
  included. The training configuration behind the reported numbers is recoverable.
- The `decoding_config` in `selection_frozen.json`, the `decoding:` block in the current
  config, and the `decoding_config` recorded in each of the 510 test prediction rows are
  all identical: `num_beams: 4`, `do_sample: false`, `max_new_tokens_cap: 512`,
  `output_length_multiplier: 2`, `output_length_offset: 32`. §10's decoding configuration
  requirement is independently provable.
- `adapter_config_sha256` in the manifest still matches the live adapter.

The residual unknown is narrow and specific: **`prompt` and `model.*` as they stood at
test-generation time are recorded nowhere.** Those are the fields to disclose as
unverifiable — not the configuration as a whole.

Do not repair this by re-running `make e2-freeze`. It would hash today's configuration and
stamp `test_generation_authorized: true` with a timestamp after the test predictions,
converting broken provenance into apparent validity. That is fabrication under §14.
Disclosure is the only honest remedy.

### Blocker 3 — the Makefile destroyed E2's real training log

`work/e2_qwen3_8b_qlora_vi_zh_v1/train.log` is 1,409 bytes and contains only a
`RuntimeError` traceback from the adapter-overwrite guard, dated 2026-08-16 20:14.
`Makefile:111` pipes training through `tee …/train.log`, and tee truncates the file before
the guard inside the Python process can fire. A later `make e2-train` attempt therefore
erased the record of the 20,328-second run. The same exposure applies to `e2-val`,
`e2-test`, and all four E4 targets.

E1 and E3 are immune for a structural reason worth copying: `train_fairseq` calls
`ensure_clean_checkpoint_dir` (`fairseq_runner.py:121-122`) before `run_logged` opens the
log (`fairseq_runner.py:182-187`), so the same 20:14 attempt aborted E1 without touching
its log — its mtime is still Aug 15 16:56. Only `work/e1_fairseq_vi_zh_v1/text/*.{vi,zh}`
were harmlessly regenerated; `data-bin` was untouched.

Surviving quantitative evidence for E2: `training_history.json` (57 records, best
`eval_loss` 1.4322, final `train_loss` 1.5388, `train_runtime` 20,328.3 s),
`trainer/trainer_state.json`, and `run_manifest.json`'s `train_metrics`. The
`docs/EXPERIMENTS.md:43` artifact contract — "training log and real checkpoint hash" — is
therefore partially violated. The fix changes no result.

### Blocker 4 — the Makefile does not drive every stage for anyone else

`README.md:42` claims "The Makefile drives every stage." For any machine other than this
one it does not:

- **There is no git repository and no `.gitignore`**, despite `README.md:5` and
  `README.md:133` describing files as excluded from version control or ignored by default.
  Every manifest records `git_revision: null`. The code that produced every artifact was
  never committed and cannot be diffed — which is exactly why E2's lost intermediate
  config state is unrecoverable. Corroborating evidence of an uncontrolled machine
  migration: `metrics/smoke_fairseq_vi_zh.*.json` still records a macOS path under
  `/Users/…`, unlike every other metrics file.
- No target acquires `zh-vi/` (100 MB, restricted) or builds the three conda environments.
- `environments/llm.yml` pins by range — `transformers>=4.51,<5`, `peft>=0.15,<1`,
  `bitsandbytes>=0.45,<1` — so a fresh install today will not match the recorded 4.57.6 /
  0.20.0 / 0.50.1.
- `configs/e2_qwen3_qlora.yaml` uses `revision: main`, which resolves to a different
  commit over time than the recorded `b968826d9c46…`.

Scope note: §13 mandates *submitting* code, configs, dataset description, predictions,
metrics, and a checkpoint or link. It does not require a grader-executable rerun. This
blocker counts against the README's own claim, not against the specification.

### Minor defects

- `Makefile:161` — `knowledge-preflight` is the only stage target missing its
  `$(E4_CONFIG)` prerequisite, so under `RUN_ROOT` it fails on a missing file unless
  `derive-configs` ran first.
- `Makefile:54` — `ERROR_SAMPLE_SIZE := 100   # spec §12 requires…` leaks the trailing
  spaces into the value, visible in dry-run output as `--sample-size 100    `. Harmless.
- `summarize-error-analysis` exists (`cli.py:88`) and `error_analysis/GUIDELINES.md` step
  4 requires running it, but no Makefile target wraps it.
- `repro_check.py:29-36` hardcodes `EXPECTED_FREEZE_STATE["configs/e2_qwen3_qlora.yaml"]
  = False`, asserting that the E2 defect persists. `make repro-check` exiting 0 does not
  mean everything verifies, and the gate would begin failing if E2 were ever legitimately
  repaired. A reader who checks only the exit code will be misled.
- `README.md:38` instructs the group to record `conda env export --from-history` and
  `pip freeze` manually. `base_manifest()` (`io_utils.py:84-129`) already captures a
  superset automatically.

## 2. Gap analysis against the specification

Rubric, §17: E1 20%, E2 25%, evaluation 15%, error analysis 15%, report and presentation
10%, reproducibility 10%, bonus E3 +10% on top.

| § | Requirement | Status |
|---|---|---|
| §7/§8 | E1 Fairseq baseline, required | Done. Test BLEU 28.63, chrF++ 30.58. Needs the warm-start noted in Limitations |
| §7/§8 | E2 LLM adaptation, required | Done. Qwen3-8B NF4 QLoRA, test BLEU 40.77, chrF++ 40.87. Needs the provenance caveat disclosed |
| §17 | Bonus: Fairseq + official `knowledge_v2`, +10% | **Not earned.** `configs/e3_fairseq_knowledge.yaml` is `status: blocked`; `cli.py:27-29` refuses to run the backend. The completed `e3_custom_fairseq_knowledge_vi_zh_v1` is a separate method this repo's own docs say must not be presented as the official one |
| §10 | Report experiment_id, model, dataset, training config, checkpoint, decoding config | E1 and E3 fully. E2 recoverable from `run_manifest.json` plus the prediction rows, but not from `selection_frozen.json` |
| §11 | SacreBLEU with a unified Moses tokenizer across all four groups | **Nuanced, not a violation.** SacreBLEU 2.6.0 exposes no `moses` tokenizer; `13a` is the closest, its own docstring reading "equivalent to mteval-v13a." The substantive issue is upstream: `normalize.py:20-23` inserts a space between every codepoint, so ZH-target BLEU is character-level by construction, close to `tok:zh`, and not on the same numeric scale as word-level BLEU over EN or VI targets — which is what §18's cross-direction matrix assumes. One methodology paragraph, not a metric change |
| §11 | chrF++ | Done. `chrF2++`, `nc:6 nw:2`, every metrics file |
| §11 | Every metric records version and signature | Done, e.g. `nrefs:1\|case:mixed\|eff:no\|tok:13a\|smooth:exp\|version:2.6.0` |
| §11 | COMET / xCOMET-lite | Optional ("có thể sử dụng"). Absence is not a gap |
| §12 | 100 test sentences labeled with the seven-label taxonomy | **Not started.** `error_analysis/manual_sample.jsonl` holds the right sample — 300 records, 100 shared test IDs × 3 systems, taxonomy and dual-annotator fields in place — but all 300 rows are `annotation_status: PENDING` with empty label arrays. No `error_analysis/summary.json`. The largest recoverable gap at 15% |
| §12 | Group 09/10: `ENTITY_ERROR` and `NUMBER_OR_TIME_ERROR` tallies *are* the domain-specific evaluation | Falls out of §12 once annotation is done. The spec explicitly says not to build a separate automated terminology or entity checker |
| §13 | Submit code, configs, dataset description, predictions, metrics, checkpoint or link | All present locally. The checkpoints (2.3 GB, 182 MB, 2.6 GB) need a private link plus SHA-256 under the lab data policy |
| §13 | Each prediction stores the seven named fields | Verified on E1 and E2 rows |
| §14 | Understand the AI-written code; never report unrun or unverified results | `AI_USAGE.md` is a maintained dated log and self-discloses all three integrity caveats. Keep it that way |
| §15 | `report.pdf` | Missing. `report/report.md` is a template; every section is `TBD` or instructions |
| §15 | `slides.pdf` or `.pptx` | Missing. `slides/slides.md` is a bullet outline |
| §16 | Nine-section report structure | Satisfied in form — headings match exactly. Unsatisfied in substance |
| §17 | Reproducibility, 10% | Machinery is strong and honestly documented, but E1's warm start, E2's broken freeze link, the absent version control, and the destroyed E2 log all cut against it |
| §5/§6 | The spec's own "Phụ lục" for the Lục Vân Tiên procedure does not exist in the .docx | Irrelevant to this direction — that is the Group 07/08 bonus. Noted for completeness |
| — | The spec states no deadline, submission channel, or page and slide limits | Ask the instructor |

### Documentation inconsistencies, zero result impact

1. `docs/EXPERIMENTS.md:33` says the custom E3 "Uses E1's seed, architecture, effective
   batch, training budget." True at configuration level, false at realized level. E1 ran
   171 epochs / 24,791 updates / batch 132.5, early-stopping on patience 20; E3 ran 87
   epochs / 50,000 updates / batch 33.3, because appending up to 64 knowledge hints
   lengthens sources so fewer sentences fit a 1,000-token batch. **The E1-versus-E3
   comparison is therefore not budget-matched**, and its +0.99 BLEU (p = 0.092, not
   significant at 0.05) carries that confound. State it in the report.
2. `docs/EXECUTION_STATUS.md:33` says "Shared 100-sentence annotation sheet." It is 100,
   which is what §12 requires. Stale line from before the sample was raised.
3. `error_analysis/manual_sample.jsonl` is JSON Lines, not CSV. Misleading extension.
4. `README.md:5`, `README.md:38`, and `README.md:133` describe version-control exclusion
   and manual environment recording that do not match reality.
5. `metrics/e4_knowledge_preflight.json` is a cost estimate only. E4 has never run — no
   `work/e4_qwen3_knowledge_vi_zh_v1/`, no checkpoint, no predictions, no metrics. Nothing
   may be reported for E4.

## 3. Recommendations

Priority order by grade impact. Nothing here retrains a model or edits an experiment
configuration.

**P0 — the 15% currently sitting empty (§12).** Annotate the 300 rows in
`error_analysis/manual_sample.jsonl` per `error_analysis/GUIDELINES.md`: pilot ten rows
jointly, label independently, then adjudicate. Run `summarize-error-analysis` to produce
`error_analysis/summary.json`, and report the `ENTITY_ERROR` and `NUMBER_OR_TIME_ERROR`
tallies — per §12 that tally *is* Group 10's domain-specific evaluation. The labels are
human judgment and must come from the group.

**P1 — the report and slides (§15, §16, 10%).** Fill `report/report.md`'s nine scaffolded
sections from `metrics/*.json` and `docs/EXECUTION_STATUS.md`, then compile `report.pdf`
and `slides.pdf`. Three disclosures belong in Limitations:

- E1's checkpoint was resumed from an interrupted run (`train.log:268`). Its BLEU is valid
  but not reproducible from a single command.
- E2's frozen-selection record references a configuration state that no longer exists.
  State precisely what is verifiable — the training configuration and full config dict in
  `run_manifest.json` matching the file on disk, and the decoding configuration identical
  across the frozen record, the config, and all 510 prediction rows — and what is not:
  `prompt` and `model.*` at test-generation time.
- E1 versus E3-custom is not budget-matched, with the numbers above; the +0.99 BLEU is not
  significant at p = 0.092.

Add a §11 methodology paragraph: SacreBLEU 2.6.0 has no Moses tokenizer, and `tok:13a`
over character-spaced Chinese is character-level BLEU, so ZH-target scores are not on the
same scale as word-level EN and VI scores in the §18 matrix. Record E2's resolved
revision `b968826d9c46…` and the exact `packages` versions from `run_manifest.json`.

**P2 — the confirmed defects, all result-neutral.**

- Stop `tee` from truncating an existing log (`Makefile:111`, `:116`, `:127`, and the E4
  equivalents). The guard cannot protect a file tee has already emptied. Cleanest fix:
  have `train_qlora` and `predict_qlora` write their own logs through the existing
  `run_logged` path, matching E1 and E3, so the guard aborts before any file is opened.
  Minimal fix: redirect to a timestamped filename.
- Add `$(E4_CONFIG)` as a prerequisite to `knowledge-preflight` (`Makefile:161`).
- Add an `error-analysis-summary` target wrapping `summarize-error-analysis`.
- Move the `ERROR_SAMPLE_SIZE` comment off the assignment line (`Makefile:54`).
- Make `repro-check` state loudly that E2's entry is an expected failure, so exit code 0
  cannot be misread, and note that the assertion inverts once E2 is repaired.

**P3 — an honest, re-runnable submission bundle.**

- `git init` plus a real `.gitignore` excluding `zh-vi/`, `en-vi/`, `checkpoint/`,
  `predictions/`, `metrics/`, `work/`, `runs/`, and `.conda/`, then commit. Future
  manifests stop recording `git_revision: null`.
- Correct the five documentation inconsistencies above.
- Rename `manual_sample.jsonl` to `.jsonl`, updating `error_analysis.py`, its tests, and the
  Makefile together.
- Add a `SUBMISSION.md` listing every §15 deliverable with its SHA-256 and the private
  links for the three checkpoint bundles.
- Ask the instructor about the missing deadline and submission channel, and whether the
  blocked official `knowledge_v2` code can still be supplied. That is +10%.

**Out of scope.** Retraining anything. Editing the E1, E2, or E3 configs. Re-running
`make e2-freeze`. Editing `evaluation.py::_metrics` — `check_stored_metrics`
(`repro_check.py:100-109`) requires exact metric metadata and signatures while allowing
only `1e-12` absolute score drift, so any supplementary metric must write to a separate
path. Building an
automated terminology or entity checker, which §12 explicitly does not want. Reporting
E4 as a result.

## 4. Verification

This audit is static analysis plus one live `repro-check`. Two gates a grader runs first
were not executed here:

```bash
make test          # AI_USAGE.md:14 claims 34 passing; 34 collected, not run
make repro-check   # expect ok:true, with E2's freeze entry as an expected failure
make status        # expect report_pdf, slides_pdf, error_analysis_summary all false
```

`make repro-smoke` needs a GPU and covers only the Fairseq path at smoke scale — 2 layers
at 64 dim for 2 updates. It does not cover E1 or E3 at full scale and never covers E2. Do
not run any write target on the canonical tree; `RUN_ROOT` exists for that.
