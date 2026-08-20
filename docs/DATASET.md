# Dataset card — DVSKTT Vietnamese ↔ Classical Chinese

## Intended use

The supplied historical parallel corpus is used by Group 10 in the Vietnamese → Classical Chinese direction. It is restricted lab research data and must not be redistributed through public repositories, Kaggle datasets, or model/data hubs.

## Splits and provenance labels

| Split | Pairs | Role | Quality label |
|---|---:|---|---|
| Train | 19,218 | Optimization | Core silver, self-aligned |
| Validation | 510 | Checkpoint/config selection | Core gold, manually aligned |
| Test | 510 | Final evaluation only | Core gold, manually aligned |

The labels above follow the filenames and supplied project documentation. Stable IDs are assigned in original row order, for example `test-000001`.

## Representation

- Vietnamese is lower-case and whitespace tokenized.
- Classical Chinese is whitespace separated at character level.
- All three splits are punctuation-free.
- Fairseq consumes the supplied tokenization directly.
- Qwen consumes natural unspaced Chinese targets; evaluation converts every model output to the same character-spaced representation.

## Verified data-quality facts

- The parallel side counts match in every split and there are no blank lines.
- Train has 1,596 duplicate pairs and 44 source strings associated with multiple target strings.
- Validation and test have no duplicate pairs.
- There are no exact source, target, or pair overlaps across train/validation/test.
- Six train target rows contain the Unicode replacement character `�`. They are identified in the private generated audit and are not guessed or silently corrected.
- `knowledge.json` contains 16,000 character-keyed entries. The private audit reports exact target-character coverage and uncovered characters.

Run `make audit` to regenerate hashes and all statistics. The generated JSON is kept private because its issue records include affected corpus lines.

## Leakage and modification policy

- Raw files are immutable inputs.
- Test references are never used for training, prompt construction, data augmentation, or checkpoint selection.
- Exact duplicates remain in required core experiments for comparability.
- Any instructor-supplied correction receives new hashes and triggers reruns of affected experiments.
- Auxiliary `en-vi/` data is out of scope and unused.

