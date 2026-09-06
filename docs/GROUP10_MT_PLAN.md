# Group 10: Vietnamese to Classical Chinese MT

## Goal

Deliver reproducible Vietnamese-to-Classical-Chinese machine translation experiments for *Đại Việt sử ký toàn thư*: E1 Fairseq, E2 Qwen3-8B QLoRA, shared evaluation, manual analysis of 100 test sentences, and all required submission artifacts.

## Current data findings

- Frozen core splits contain 19,218 train, 510 validation, and 510 test pairs.
- Train is self-aligned core/silver data; validation and test are manually aligned gold data.
- No pair or source overlap was found between splits.
- Train contains 1,596 duplicate pairs and six Chinese lines containing the replacement character `�`. These remain unchanged in the official experiments and are reported as limitations.
- `zh-vi/knowledge.json` has 16,000 entries and covers 5,013 of 5,458 distinct corpus target characters. E3 remains gated because the official `knowledge_v2` integration code and configuration are missing.

## Implementation sequence

1. Audit and hash immutable data; create stable sample IDs and reproducible derived representations.
2. Run E1 with the supplied Vietnamese word tokens and Chinese character tokens using a low-resource Fairseq Transformer.
3. Run E2 with `Qwen/Qwen3-8B`, non-thinking chat formatting, and 4-bit QLoRA.
4. Normalize both systems to the same character-spaced scoring representation; apply Moses and score with SacreBLEU `tokenize=none`, retaining the existing chrF++ inputs. See `docs/MOSES_EVALUATION.md`; historical `13a` scores remain reproducible.
5. Sample the same 100 test IDs for both systems, annotate the required multi-label taxonomy, adjudicate disagreements, and summarize entity and number/time errors.
6. Produce the README, configurations, predictions, metrics, private checkpoint links/checksums, report, slides, and AI usage disclosure.
7. Keep the official E3 blocked; after E1/E2 pass, run the separately labeled custom source-side retrieval experiment from `knowledge.json`.

## Completion gates

- Raw data is never changed or published.
- Test predictions are generated only after model/checkpoint selection is frozen on validation.
- Each real experiment has a config, run manifest, logs, checkpoint, 510 ordered test predictions, metric signatures, and hashes.
- No result is reported unless it was actually generated and can be traced to saved artifacts.
