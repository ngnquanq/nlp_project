# Code and artifact audit against specification v1.3

Checked 2026-09-06 against the private `PROJECT_SPECIFICATION_v1.3.docx`.
Scope: Group 10 code, configurations, tests, and local experiment artifacts.
PDF/report content and presentation deliverables are excluded from this review.
The findings below are follow-up work, not additional fixes in the Moses commit.

## Coverage already present

| Specification | Current evidence |
|---|---|
| Sections 4/8: required core corpus and E1/E2 | Frozen 19,218/510/510 splits, dataset hashes, E1 checkpoint and E2 adapter, and 510-row validation/test predictions exist. The fresh data audit and dataset fingerprint checks pass. |
| Sections 10/13: experiment and prediction metadata | Configs, run manifests, checkpoint paths, decoding settings, and all seven required prediction fields are present; `schema.py` also enforces ordering and unique IDs. Historical provenance limitations are listed below. |
| Section 11: Moses BLEU and chrF++ | New `moses-char-v1` evaluation uses Sacremoses 0.2.0 followed by SacreBLEU 2.6.0 with `tokenize=none`. Full external preprocessing metadata is recorded. All eight rescored files and three bootstrap comparisons are numerically unchanged. |
| Section 12: error-analysis tooling | Shared 100-ID sampling, all seven labels, entity/time percentages, and rejection of non-adjudicated rows are implemented. The validator has the confirmed edge cases below. |
| Section 14: AI disclosure | `docs/AI_USAGE.md` records development assistance; human review remains the group's responsibility. |

## Remaining gaps, in priority order

1. **Annotation validation can accept invalid completed sheets** (section 12).
   `code/mt_pipeline/error_analysis.py:validate_labels` accepts an empty list and
   repeated labels. `summarize_annotations` counts duplicate rows but checks only
   the number of unique IDs. In-memory fixtures demonstrated that a duplicated
   row yields 101% OMISSION, repeated labels yield 200%, and 100 rows with empty
   labels are accepted as a complete summary. Reject empty/repeated labels and
   duplicate `(experiment_id, sample_id)` rows before counting. No real annotation
   files were edited during these checks.

2. **Frozen validation evidence is recorded but not fully enforced** (reproducibility
   hardening supporting sections 10/13). `freeze_selection` records validation
   prediction/metric hashes, but `ensure_selection_frozen` checks only the config,
   checkpoint, and decoding settings. It does not recheck those validation hashes.
   Also bind the supplied validation metrics to the actual prediction SHA-256 when
   creating the freeze. Preserve historical records rather than rewriting them to
   make a check pass.

3. **E2's historical freeze no longer matches its current configuration** (sections
   10/13). The live reproducibility gate reports this as an expected failure via
   `EXPECTED_FREEZE_STATE`, so aggregate `ok=true` does not mean the E2 chain is
   intact. E1/E3 manifests also have `git_revision=null`; E1's log records a resume
   from epoch 5/update 579. Current weights and saved scores remain usable, but
   complete training provenance is not established. Recover authentic historical
   inputs if available; otherwise keep this limitation explicit and use an isolated
   new run if a fully verified training chain is required.

4. **The error-analysis execution is unfinished** (section 12). The local sheet
   contains 100 distinct test IDs for each of E1/E2/custom E3, totaling 300 rows;
   all are `PENDING`, none has adjudicated labels, and no summary exists. This
   requires human annotation, followed by `summarize-error-analysis`; model
   retraining and an additional automatic entity detector are not required.

5. **Cross-group Moses settings remain unconfirmed** (sections 11/18). The
   specification provides no shared command or explicit Chinese segmentation/
   escaping settings. The implementation preserves Chinese character spacing and
   disables escaping. Share the exact protocol in `MOSES_EVALUATION.md` with the
   other groups; the local implementation alone cannot verify their configuration.

## Optional work and validation limits

- Official E3 (`knowledge_v2`) is blocked by missing integration code/configuration
  in `configs/e3_fairseq_knowledge.yaml`. It is bonus work; custom E3 is separate.
- COMET/xCOMET and E4 are not required. Their absence is not a core code gap.
- `make check-private EVAL_PYTHON=.conda/eval/bin/python` passed all 54 tests, the
  data audit, and the reproducibility gate under its documented expectations.
  Corpus scoring, legacy notebook restore fixtures, and notebook syntax were
  checked. Full GPU retraining and a remote Kaggle session were not run.
- Private generated metrics, predictions, checkpoints, and annotations remain
  outside Git and need to accompany source through the permitted private channel.
