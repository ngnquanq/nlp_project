# Moses evaluation migration

New `evaluate` and `compare` calls default to `moses-char-v1`. The historical
`13a-char-v1` protocol remains available for reproducing saved experiment evidence.
This changes evaluation only: training tokenization, model weights, decoding,
prediction JSONL fields, dataset splits, and manual annotations are unchanged.

## Exact scoring contract

1. Validate the prediction schema and its existing `prediction_scored` field.
2. Normalize Chinese hypotheses and references to NFC, remove whitespace, and put
   one space between Unicode code points (the existing `score_form_chinese`).
3. For BLEU only, apply `sacremoses==0.2.0` `MosesTokenizer(lang="zh")` with
   `escape=False`, `aggressive_dash_splits=False`, and `return_str=True`.
4. Score with SacreBLEU 2.6.0: `tokenize="none"`, mixed case, exponential smoothing,
   and `effective_order=False`. This prevents double tokenization.
5. Keep chrF++ on the original character-spaced representation with `char_order=6`,
   `word_order=2`, `beta=2`, `lowercase=False`, and `whitespace=False`.

The BLEU JSON includes a `preprocessing` object recording the protocol, Moses
implementation/version, language, input representation, and tokenizer options.
Always publish that object together with the SacreBLEU signature (`tok:none`);
the signature alone does not describe external preprocessing. Bootstrap comparisons
use the same preprocessing, applied once before resampling, and record both
prediction-file hashes. No model package or GPU is needed for evaluation.

Sacremoses is a Python port of Moses's tokenizer; `13a` implements mteval-v13a,
which is a different tokenization algorithm. Sources:
[Sacremoses](https://github.com/hplt-project/sacremoses),
[Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl),
[SacreBLEU](https://github.com/mjpost/sacrebleu#languages--preprocessing).

## Rescore existing experiments

```bash
.conda/eval/bin/python -m pip install -r requirements-eval.txt
make rescore-moses
make repro-check
```

`rescore-moses` first checks each available original score against its saved
prediction hash and recorded evaluation protocol. It writes new metrics for all
saved prediction files into `metrics/moses/`, reruns the comparisons described by
the original metric files using their seeds and sample counts, and writes
`metrics/moses/migration_summary.json` with numerical deltas and provenance.
Original metrics, predictions, and frozen-selection records are not rewritten.
Use this command for migration; `make e1`, `make e2`, and prediction stages perform
model work and are unnecessary here. Generated results remain private and ignored
by Git, matching the existing artifact policy.

For one saved prediction file, set `PYTHONPATH=code` and run:

```bash
.conda/eval/bin/python -m mt_pipeline evaluate \
  --predictions predictions/e1_fairseq_vi_zh_v1.test.jsonl \
  --output metrics/moses/e1_fairseq_vi_zh_v1.test.json
```

To reproduce the old protocol, add `--protocol 13a-char-v1` and use a separate
output path. New training runs can still write metrics at their configured root;
their BLEU metadata identifies the new protocol. `repro-check` detects the stored
protocol and checks both original and migrated corpus metrics.

## Notebook and checkpoint compatibility

The consolidated root notebook verifies restored metrics under their recorded
protocol. Its evaluation section rescores all three systems into `metrics/moses/`,
uses those results for tables and figures, and includes that directory in the final
private archive. Exported metric tables include signatures and Moses metadata.
Historical stage archives retain their original selection evidence. The thin
notebooks inherit Moses scoring through the CLI and updated dependency files.
Rebuild private source bundles after code changes; remote Kaggle clones require a
committed/pushed revision containing these changes before they can use them. The
consolidated notebook selects `REPO_REF="feature/moses-substitute"` on this branch.

Fairseq's training-time BLEU and checkpoint selection still use the original
character-spaced 13a metric. Historical checkpoints are being rescored, not
reselected. Qwen retains its native model tokenizer. Existing frozen config and
checkpoint checks continue unchanged, including pre-existing provenance failures.

## Interpretation and cross-group alignment

The specification names Moses but does not provide a shared script or specify
Chinese segmentation/escaping settings. This implementation explicitly retains
Chinese character segmentation; Moses alone does not generally segment compact
Chinese into useful lexical tokens. Share this exact protocol with the other
groups/instructor before describing the shared matrix as fully aligned. Do not
apply Chinese character spacing to English or Vietnamese targets. A common
tokenizer does not make BLEU directly comparable across target languages.

This migration changes how translations are scored, not their quality. chrF++
should remain identical; BLEU differences must be measured from the saved outputs.

## Verified local migration — 2026-09-06

The rescore completed in 228.34 seconds on the existing CPU evaluation environment.
All eight saved validation/test prediction files (E1, E2, custom E3, and smoke;
510 rows each) produced exactly unchanged BLEU and chrF++ scores. All numerical
statistics in the three 1,000-resample, seed-42 comparisons were also identical.
The new BLEU signatures and preprocessing metadata identify the Moses protocol.

| Test experiment | Historical BLEU | Moses BLEU | chrF++ (unchanged) |
|---|---:|---:|---:|
| E1 Fairseq | 28.626792 | 28.626792 | 30.579882 |
| E2 Qwen3 QLoRA | 40.770122 | 40.770122 | 40.868183 |
| Custom E3 | 29.620977 | 29.620977 | 31.409753 |

Evidence: `metrics/moses/migration_summary.json` and its referenced metric files.
Before/after SHA-256 checks confirmed that the original predictions, root metric
JSON files, and frozen-selection records were unchanged. All 54 project tests and
the data audit passed. The reproducibility gate passed its recorded expectations;
E2's pre-existing frozen-config mismatch remains explicitly reported, not repaired
by this evaluation migration. Notebook code cells were syntax-checked and legacy
restore validation was exercised with fixtures; a full Kaggle training session was
not run.
