# Vietnamese → Classical Chinese MT for Ancient Texts

## Problem and contribution

- Group 10 direction and historical domain
- Fairseq versus Qwen3 QLoRA
- Reproducible evaluation and manual domain-error analysis

## Dataset

- 19,218 / 510 / 510 frozen pairs
- Silver train; gold validation/test
- Data-quality and confidentiality notes

## E1 Fairseq

- Architecture and low-resource batch recipe
- Training and decoding configuration

## E2 Qwen3 QLoRA

- Non-thinking prompt format
- 4-bit NF4 and LoRA setup

## Results

Insert the verified scores and paired-bootstrap comparison from `metrics/moses/`.
Include BLEU's Moses preprocessing metadata alongside both metric signatures and
state that Chinese character segmentation is retained.

## Error analysis

Insert adjudicated counts and examples from the same 100 test sentences.

## Discussion and limitations

Contrast system behavior and state corpus, metric, compute, and E3 limitations.

## Reproducibility

Show configs, private checkpoints, prediction schema, hashes, and one-command evaluation.
