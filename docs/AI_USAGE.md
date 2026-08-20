# AI Usage Disclosure

AI assistance was used to inspect the project specification and corpus structure, design the repository layout, and implement the initial data-audit, training, prediction, evaluation, and testing code.

Before submission, append a dated log for every further material use:

| Date | Tool/model | Task | Human verification |
|---|---|---|---|
| 2026-08-15 | OpenAI Codex | Initial pipeline implementation | Code and tests must be reviewed and GPU runs verified by the group |
| 2026-08-16 | Claude (Claude Code) | Reproducibility audit of E1/E2/E3: re-derived all 12 stored metrics from saved predictions and re-ran the three paired bootstraps at seed 42 | Reproduced numbers match `metrics/*.json` exactly; group to confirm the reported findings |
| 2026-08-16 | Claude (Claude Code) | Found E1's checkpoint was warm-started from an interrupted run (`train.log:268`), and E2's config was edited 12 min after its test predictions, breaking its frozen-selection chain | Both confirmed from saved logs and hashes; group to decide how to report the E2 caveat |
| 2026-08-16 | Claude (Claude Code) | Found E1 vs E3 is not budget-matched (E1 171 epochs / 24,791 updates / batch 132.5; E3 87 epochs / 50,000 updates / batch 33.3) | Derived from both `train.log` files; group to state the confound in the report |
| 2026-08-16 | Claude (Claude Code) | Makefile pipeline for all experiments, per-stage Conda env selection, E2 logging parity, `--seed 42` on comparisons | `make -n` dry-runs and a full smoke chain executed; group to review before long runs |
| 2026-08-16 | Claude (Claude Code) | Determinism and safety code: `fairseq_train.py`, `reruns.py`, `repro_check.py`, no-resume guards in both trainers, annotation-overwrite guard | 34 unit tests pass; two smoke trainings verified bitwise-identical weights and translations |
| 2026-08-16 | Claude (Claude Code) | Compliance review against `PROJECT_SPECIFICATION_v1.3.docx`; error-analysis sample raised from 50 to the required 100 sentences (§12) | Verified the 100-sample retains all 50 previously sampled sentences; group assigns all labels |
| 2026-08-16 | Claude (Claude Code) | E4 knowledge-augmented QLoRA: `knowledge.py` split into retrieval core plus per-backend renderers, `qlora_knowledge` backend, `knowledge-preflight` cost measurement | E3's Fairseq output pinned byte-identical by regression test; group must run and validate E4 itself |

The group remains responsible for understanding the code, reviewing generated commands, validating every metric against saved predictions, checking citations, and ensuring that no result or interpretation is fabricated.

