# Local execution status - 2026-08-16

This file records only results backed by the current workspace artifacts.

## Completed experiments

| Experiment | Validation BLEU | Validation chrF++ | Test BLEU | Test chrF++ |
|---|---:|---:|---:|---:|
| E1 Fairseq | 29.37 | 31.12 | 28.63 | 30.58 |
| E2 Qwen3-8B QLoRA | 41.06 | 41.20 | 40.77 | 40.87 |
| Custom E3 Fairseq knowledge retrieval | 31.04 | 32.72 | 29.62 | 31.41 |

All validation and test metrics use 510 aligned examples and SacreBLEU 2.6.0. Each test run was generated only after its validation selection was frozen.

E2 used Qwen3-8B with NF4 QLoRA. Its packing window was reduced from 1024 to 768 after local memory preflight; the longest encoded training example was 481 tokens, so this change truncated no examples. Training completed three epochs in 20,328 seconds. The best trainer validation loss was 1.4322.

The official lab E3 remains blocked because the `knowledge_v2` integration is unavailable. The completed custom E3 is a separate source-side retrieval experiment and must not be presented as the official method.

## Paired comparisons

Paired bootstrap comparisons use 1,000 samples with seed 42 on the held-out test set.

- E2 minus E1: +12.14 BLEU, 95% interval [10.22, 13.77], p = 0.002.
- Custom E3 minus E1: +0.99 BLEU, 95% interval [-0.14, 2.16], p = 0.092. This improvement is not statistically significant at 0.05.
- E2 minus custom E3: +11.15 BLEU, 95% interval [9.26, 12.77], p = 0.002.

## Artifact locations

- Metrics: `metrics/*.json`
- Predictions: `predictions/*.jsonl`
- Frozen selections and manifests: `work/<experiment_id>/`
- Model outputs: `checkpoint/<experiment_id>/`
- Shared 100-sentence annotation sheet: `error_analysis/manual_sample.jsonl`

## Remaining deliverables

- Complete the manual error labels in `error_analysis/manual_sample.jsonl` and generate the summary.
- Populate the final report and slides from the saved metrics and comparisons.
- Compile `report.pdf` and `slides.pdf`.
