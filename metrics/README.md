# Metrics

Generated JSON metrics and full SacreBLEU signatures belong here. Only report numbers that can be regenerated from a saved prediction JSONL file.

`make rescore-moses` writes migrated scores, bootstrap comparisons, and a provenance
summary to `moses/`, preserving original files. Include BLEU's `preprocessing` object
with its `tok:none` signature. See [Moses evaluation](../docs/MOSES_EVALUATION.md).
