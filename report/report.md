# Vietnamese-to-Classical-Chinese Machine Translation for Ancient Texts

> Drafting template. Replace every `TBD` only from verified saved artifacts. Do not report unexecuted results.

## 1. Introduction

Describe the historical MT task, Group 10 direction, research questions, and contributions.

## 2. Dataset

Document provenance, confidentiality, frozen 19,218/510/510 splits, silver-versus-gold status, tokenization, duplicate pairs, six replacement-character rows, vocabulary statistics, and leakage audit. Cite `metrics/data_audit.json`.

## 3. Methods

### E1 Fairseq

Describe the exact configuration in `configs/e1_fairseq.yaml` and how it adapts the supplied low-resource batch-size paper.

### E2 Qwen3-8B QLoRA

Describe the prompt, non-thinking chat template, NF4 quantization, LoRA targets, response-only loss, and deterministic decoding from `configs/e2_qwen3_qlora.yaml`.

### E3 Knowledge bonus

Report only if the official integration passes its gate and the experiment actually runs; otherwise state why it was not attempted.

## 4. Experimental Setup

Add hardware, wall-clock time, seeds, resolved package versions, checkpoint-selection policy, decoding settings, and artifact hashes from run manifests.

## 5. Results

| Experiment | SacreBLEU | chrF++ | Checkpoint |
|---|---:|---:|---|
| E1 Fairseq | TBD | TBD | TBD |
| E2 Qwen3 QLoRA | TBD | TBD | TBD |

Use the executed Moses results in `metrics/moses/`, including paired bootstrap results.
Copy both metric signatures and BLEU's external `preprocessing` metadata. Describe
the retained Chinese character segmentation and distinguish historical Fairseq
checkpoint-selection BLEU from final Moses evaluation; see `docs/MOSES_EVALUATION.md`.

## 6. Error Analysis

Report adjudicated counts over the same 100 seeded test sentences for both systems, emphasizing entity and number/time errors. Include representative correct, mistranslated, omitted, and unsupported-added examples.

## 7. Discussion

Explain differences between conventional NMT and LLM adaptation without treating reference wording as the only valid translation.

## 8. Limitations

Cover corpus size, self-aligned training data, repeated pairs, missing glyphs, punctuation-free text, single references, metric limitations for Classical Chinese, compute budget, and any unavailable E3 code.

## 9. Conclusion

Summarize only verified findings.
