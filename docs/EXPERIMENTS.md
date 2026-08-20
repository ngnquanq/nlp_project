# Experiment registry

## Required experiments

### `e1_fairseq_vi_zh_v1`

- Conventional 6-layer encoder/decoder Transformer in Fairseq.
- Direct Vietnamese word and Chinese character vocabularies with frequency threshold 1.
- One 1,000-token GPU batch with four-step accumulation, preserving an effective batch of about 4,000 tokens on the local RTX 3060.
- Best validation-BLEU checkpoint; deterministic beam-7 decoding.

Source of truth: `configs/e1_fairseq.yaml`.

### `e2_qwen3_8b_qlora_vi_zh_v1`

- `Qwen/Qwen3-8B` with thinking disabled.
- 4-bit NF4 QLoRA, response-only loss, and packed training examples.
- Best validation-loss adapter; deterministic beam-4 decoding.

Source of truth: `configs/e2_qwen3_qlora.yaml`.

## Bonus experiments

### `e3_fairseq_knowledge_vi_zh_v1`

Status: blocked until the official lab `knowledge_v2` integration code and configuration are supplied. Do not substitute a newly invented architecture under this ID.

### `e3_custom_fairseq_knowledge_vi_zh_v1`

- Custom, clearly labeled source-side retrieval experiment; it is not the missing lab `knowledge_v2` method.
- Matches Vietnamese source forms against pronunciation and query fields in `zh-vi/knowledge.json`.
- Appends at most 64 unique target-character hints ranked by match count, training-target frequency, and Unicode order.
- Uses E1's seed, architecture, effective batch, training budget, and deterministic beam-7 decoding.

Source of truth: `configs/e3_custom_fairseq_knowledge.yaml`.

## Artifact contract

Every completed experiment must have:

- immutable configuration and dataset hashes;
- resolved environment and hardware metadata;
- training log and real checkpoint hash;
- 510 ordered validation predictions and metrics;
- a frozen-selection record created before test generation;
- 510 ordered test predictions and metric signatures;
- a private checkpoint or adapter link for submission.

