# Pipeline: preprocessing, experiment runs, validation

What the code actually does, end to end, with the numbers the completed runs produced.
Written to be lifted into the report's *Data*, *Experimental setup*, and *Evaluation*
sections.

"Validation" covers both senses, and §3 separates them deliberately:

- **model selection** — how the val split chose each reported checkpoint;
- **verification** — the gates that prove the reported numbers came from the artifacts on
  disk (schema checks, the freeze chain, `repro-check`, determinism).

Every claim is cited to a `file:line`, a log line, or a stored hash. Numbers come from
`metrics/data_audit.json`, `work/<id>/run_manifest.json`, `work/<id>/preprocess.log`,
`work/<id>/train.log`, and `metrics/*.json` as they stand on disk today. Integrity caveats
that change how these numbers should be *interpreted* live in `docs/REPRO_AUDIT.md`; this
document describes the mechanism and points at them where relevant.

---

## 0. How the pipeline is wired

One Python package, one CLI, one Makefile. Every stage is
`python -m mt_pipeline <subcommand>` (`code/mt_pipeline/cli.py`), and the Makefile's only
job is to run each subcommand under the right environment with the right paths.

### 0.1 Three environments, not interchangeable

`Makefile:15-20`. This is the first thing a grader trips on: running a stage under the
wrong env fails on a missing import, not a wrong number.

| Make variable | Env | Holds | Owns these stages |
|---|---|---|---|
| `$(EVAL)` | `.conda/eval` | `sacrebleu` 2.6.0, PyYAML, pytest — **no** torch, **no** fairseq | `data-audit`, `evaluate`, `compare`, `freeze-selection`, `prepare-error-analysis`, `summarize-error-analysis`, `derive-run-config`, `repro-check`, `project-status`, `pytest` |
| `$(FAIRSEQ)` | `.conda/fairseq` | `fairseq` 0.12.2, `torch` 1.13.1 — no transformers | E1/E3 `train`, E1/E3 `predict` |
| `$(LLM)` | `.conda/llm` | `transformers` 4.57.6, `peft` 0.20.0, `bitsandbytes` 0.50.1, `accelerate` 1.14.0, `datasets` 4.8.5, `torch` 2.5.1 | E2/E4 `train`, E2/E4 `predict`, `knowledge-preflight` |

Scoring is deliberately isolated in the smallest env: the metric numbers depend on
`sacrebleu` alone and can be recomputed without a GPU or either training stack.

The three envs are built from `environments/{eval,fairseq,llm}.yml` into `.conda/<name>`;
no Makefile target creates them, so this is a manual prerequisite. **`llm.yml` pins by
range** (`transformers>=4.51,<5`, `peft>=0.15,<1`, `bitsandbytes>=0.45,<1`,
`accelerate>=1.7,<2`, `datasets>=3.6,<5`), so a fresh install will *not* reproduce the
versions E2 actually ran under — 4.57.6 / 0.20.0 / 0.50.1 / 1.14.0 / 4.8.5, recorded in
`work/e2_.../run_manifest.json` → `packages`. Pin from the manifest, not from the YAML, when
reproducing E2. (`torch` and `pytorch-cuda` *are* pinned exactly: 2.5.1 / cu124.)

Hardware for every completed run: one **NVIDIA GeForce RTX 3060, 12 GB**, CUDA 12.4
(`work/e2_.../run_manifest.json` → `accelerator`).

### 0.2 One command per experiment, four ordered stages

```
make e1          = e1-train → e1-val → e1-freeze → e1-test
make e2          = e2-train → e2-val → e2-freeze → e2-test
make e3-custom   = e3-custom-train → … → e3-custom-test
make all         = audit → e1 → e2 → e3-custom → compare → error-analysis → status
```

The order is enforced in code, not only by the Makefile: `predict --split test` calls
`ensure_selection_frozen` before it loads a checkpoint
(`fairseq_runner.py:226-228`, `llm_runner.py:291-294`). Test data cannot be decoded until
validation selection has been frozen. See §3.2.

### 0.3 Reruns without touching graded artifacts

`RUN_ROOT` (`Makefile:22-45`) rebases `configs/`, `work/`, `checkpoint/`, `predictions/`,
`metrics/` and `error_analysis/` into a parallel tree:

```
make all RUN_ROOT=runs/rerun-$(date +%F-%H%M%S)
```

`derive-run-config` (`reruns.py:26`) relocates exactly the three writable path fields —
`work_dir`, `checkpoint_dir`, `prediction_dir`. `dataset_config` and `knowledge.file` stay
pointed at the read-only inputs. No write target on the canonical tree is needed to
re-run anything.

---

## 1. Data preprocessing

### 1.1 The corpus and its splits

Single source of truth: `configs/dataset.yaml`. Direction is **vi → zh** (Vietnamese to
Classical Chinese), corpus `dvsktt_vi_zh_v1`, from *Đại Việt sử ký toàn thư*.

| Split | Quality | Pairs | File stem |
|---|---|---:|---|
| train | `core_silver_self_aligned` | 19,218 | `DVSKTT_self_aligned_cn_sv_vi_20240927_cleaned_official.no_punct.line.train.{vi,cn}` |
| val | `core_gold_manual` | 510 | `DVSKTT_manually_aligned_sent_cleaned_gold_eval.20240829.no_punct.val.{vi,cn}` |
| test | `core_gold_manual` | 510 | `DVSKTT_manually_aligned_sent_cleaned_gold_eval.20240829.no_punct.test.{vi,cn}` |

Two properties of the supplied files matter for everything downstream:

- **Punctuation-free** (`no_punct` in the filenames). The audit confirms it empirically:
  **zero** Unicode-punctuation characters on either side of any split
  (`data_audit.json` → `splits.*.{source,target}.punctuation` is `{}`). No punctuation
  handling exists in the pipeline because there is no punctuation.
- **Pre-tokenized, space-separated.** Vietnamese is word/syllable-segmented; Chinese is
  **one character per token**. Nothing in the pipeline re-tokenizes. This is the single
  most consequential fact about the evaluation (§3.4).

Training data is *silver* (self-aligned); val and test are *gold* (manually aligned). The
report should not present train and eval quality as equivalent.

### 1.2 Loading, normalization, and the three text forms

`read_split` (`data.py:44-63`) is the only reader. Per split it:

1. reads both sides and **hard-fails on a length mismatch** (`data.py:48-51`);
2. **hard-fails if the line count differs from `expected_lines`** in the config
   (`data.py:52-54`) — a silently swapped or truncated corpus file cannot enter a run;
3. assigns `sample_id = f"{split}-{index:06d}"`, **1-based** (`data.py:57`, `enumerate(…, 1)`);
4. applies `normalize_whitespace` to both sides: **Unicode NFC**, then collapse all
   whitespace runs to one space, then strip (`normalize.py:11-12`).

Nothing else is altered. No lowercasing, no punctuation stripping, no length filtering, no
deduplication — the corpus enters training as supplied.

Each row then exposes three forms of the target, and keeping them distinct is what makes
the metrics well-defined (`data.py:16-29`):

| Form | Definition | Used for |
|---|---|---|
| `reference_tokenized` | the corpus line, whitespace-normalized: `如 使 越 人 …` | Fairseq training/eval target |
| `reference` | `detokenize_chinese` — all whitespace removed: `如使越人…` | human-readable; the `reference` field in predictions; the E2 training target |
| `reference_scored` | `score_form_chinese` — re-spaced, one codepoint per token | **the scoring input** for BLEU and chrF++ |

`score_form_chinese` (`normalize.py:20-23`) is a join-then-respace round trip, so it is
lossless only if every corpus target token is a single codepoint. **Verified, not
assumed:** across all 20,238 rows of train + val + test, `score_form_chinese(reference_tokenized)`
equals `reference_tokenized` exactly, and no row contains a multi-codepoint target token.
The scored reference *is* the corpus reference, byte for byte.

### 1.3 The data audit — `make audit`

`audit_dataset` (`data.py:96-179`) → `metrics/data_audit.json`. It hashes, profiles, and
looks for the specific failure modes that would invalidate the whole comparison.

**Sizes and lexical profile** (tokens are space-separated; Chinese tokens = characters):

| | train src (vi) | train tgt (zh) | val src | val tgt | test src | test tgt |
|---|---:|---:|---:|---:|---:|---:|
| tokens | 401,997 | 357,806 | 10,365 | 8,908 | 10,539 | 8,982 |
| types | 3,607 | 5,386 | 1,641 | 1,771 | 1,660 | 1,789 |
| hapax types | 507 | 1,137 | 549 | 699 | 563 | 725 |
| tokens/line (mean) | 20.92 | 18.62 | 20.32 | 17.47 | 20.66 | 17.61 |
| tokens/line (p95 / max) | 49 / 199 | 44 / 196 | 46 / 70 | 38 / 66 | 49 / 90 | 41 / 79 |

Sentences are short (~21 source tokens mean) and the max of 199 sits far under the
256-position encoder limit, so **no example is truncated by `max_source_positions`** in E1.

**Contamination check — clean.** All three pairwise overlaps (train×val, train×test,
val×test) are **0 shared pairs, 0 shared sources, 0 shared targets**
(`data_audit.json` → `overlap`). Reported test scores are not inflated by leakage.

**Internal duplication — train only.** 1,596 duplicate pairs, 1,653 duplicate source
lines, 1,635 duplicate target lines, and **44 sources with more than one distinct target**
(`data_audit.json` → `splits.train`). Val and test are 0 on all four counts. The
duplicates are left in place; the 44 ambiguous sources are an upper bound on what any
model can get right from the source alone, and are worth one sentence in the report.

**Data defects — 6 rows.** Six train-side target lines contain the U+FFFD replacement
character (`�`), i.e. characters lost before the corpus reached us:
`train-000182`, `train-000235`, `train-000243`, `train-001128`, `train-001668`,
`train-013331`. All six are on the target side, all in train, none in val/test. They are
surfaced in `report["issues"]` but not removed. Zero blank lines and zero
edge-whitespace lines anywhere.

**Split hashes** (SHA-256, first 12 hex; full values in `data_audit.json` and in every
`run_manifest.json`):

| Split | source | target |
|---|---|---|
| train | `66a58dabc6c2` | `bf38525cedbc` |
| val | `b2035392b9dc` | `577351ae1fc8` |
| test | `26fd46122eab` | `79e2a7b8bc5c` |

`dataset_fingerprint` (`data.py:186-198`) recomputes these six hashes plus the knowledge
hash at every train, and `repro-check` re-verifies them against each manifest (§3.6).

### 1.4 The knowledge resource

`zh-vi/knowledge.json`, SHA-256 `68dc862354581cd2…`. 16,000 character entries, each with
all four top-level fields (`character_coverage`, `glyph_metadata`, `lexical_retrieval`,
`pronunciation_lookup`).

**Coverage of the corpus:** the target side of the three splits uses 5,458 distinct
characters; 5,013 have a knowledge entry → **91.85 %**. The 445 uncovered characters are
listed in full in `data_audit.json` → `knowledge.uncovered_characters` (they are rare
Nôm/variant forms: `㐌 㐮 㓂 㓙 㔫 …`). That 8.15 % ceiling is the honest upper bound on how
much retrieval can possibly help.

### 1.5 Backend-specific preprocessing

The three experiments diverge here and nowhere earlier: all read the same rows through
`read_split`, then each renders them for its own trainer.

#### 1.5.1 E1 — Fairseq binarization

`prepare_fairseq` (`fairseq_runner.py:28-99`):

1. Write `work/<id>/text/{train,valid,test}.{vi,zh}` — source as normalized, target as
   `reference_tokenized` (character-spaced). Note the split rename: **`val` → `valid`**,
   which is what Fairseq's `--validpref` and `--gen-subset valid` expect.
2. Run `fairseq-preprocess` with `--thresholdsrc 1 --thresholdtgt 1 --workers 4`, logged
   to `work/<id>/preprocess.log`.
3. If `data-bin/` already exists it is **reused** unless `--force` (`fairseq_runner.py:72-75`),
   so re-running preparation is cheap and does not silently rebuild vocabularies.

Frequency threshold 1 means **no subword model and no vocabulary pruning** — full word
vocabulary on the Vietnamese side, full character vocabulary on the Chinese side.
Dictionaries are built from `--trainpref` only, so val/test contribute nothing to the
vocabulary and their OOV rate is a genuine held-out measurement:

| | dictionary | train tokens | val `<unk>` | test `<unk>` |
|---|---:|---:|---:|---:|
| E1 source (vi) | 3,616 types | 421,215 | 0.202 % (≈22 tok) | 0.217 % (≈24 tok) |
| target (zh) | 5,392 types | 377,024 | 0.361 % (≈34 tok) | 0.474 % (≈45 tok) |

(Fairseq's token counts include one EOS per sentence: 401,997 + 19,218 = 421,215 ✓.)
The target dictionary is identical for E1 and E3 — the target side is untouched by
augmentation. Target OOV is the harder number: ~0.4 % of gold characters are unproducible
by construction.

#### 1.5.2 E3-custom — source-side knowledge retrieval

Same path, plus `KnowledgeAugmenter` (`knowledge.py`) rewriting the source before export.
This is a **preprocessing-only** intervention: the architecture, the criterion and the
decoding settings are byte-identical to E1's config.

*Index construction* (`knowledge.py:54-71`): for every canonical entry, harvest the
Vietnamese lookup forms from `pronunciation`, `nom_query`, `lookup_query` and
`query_variants`; casefold and whitespace-normalize each; key a dict from the **token
tuple** to the set of Chinese characters it can realize. Result: **6,389 distinct lookup
forms**, indexed by their token length so multi-word forms match as spans.

*Retrieval per sentence* (`rank_candidates`, `knowledge.py:80-109`): slide over every start
position × every known form length, count a hit for each matching span, and accumulate a
score per candidate character. Ranking is fully deterministic —
**match count → training-target frequency → Unicode codepoint order** — with the frequency
table built from the *train* split only (`knowledge.py:73-78`), so no val/test information
enters ranking.

*Rendering for Fairseq* (`augment`, `knowledge.py:111-134`): append a sentinel token
`__knowledge__` and up to `max_candidates: 64` characters to the tokenized source, budgeted
against `max_source_positions` so the result can never overflow the encoder.

*What it did* (`work/e3_.../knowledge_augmentation.json`, archived into the manifest):

| Split | rows | rows augmented | mean candidates | max | mean matched spans |
|---|---:|---:|---:|---:|---:|
| train | 19,218 | 19,218 (100 %) | 61.53 | 64 | 20.79 |
| val | 510 | 510 (100 %) | 62.41 | 64 | 20.19 |
| test | 510 | 510 (100 %) | 61.98 | 64 | 20.53 |

Every single sentence got hints, and nearly all of them saturated the 64-candidate budget
— retrieval is high-recall and barely selective. Mean hint block ≈ 62 characters against a
mean source of ~21 tokens.

Three downstream consequences, all measured, none of them cosmetic:

- **Source length ×3.9.** Train source tokens go 421,215 → **1,622,823**; mean tokens per
  line 20.92 → 83.44 (= 20.92 source + 1 separator + 61.53 hints ✓).
- **Source vocabulary ×4.8.** 3,616 → **17,264 types**, because the retrieved Han
  characters now live in the *source* dictionary. Source-side val/test OOV *rates* fall
  (0.164 % / 0.158 %) but only because the denominator quadrupled; in absolute terms E3
  carries **more** unknown source tokens than E1. Measured against
  `work/e3_.../data-bin/dict.vi.txt`, split by position around the `__knowledge__`
  separator:

  | | tokens in the original source span | tokens in the hint span |
  |---|---:|---:|
  | val `<unk>` | 22 | **49** |
  | test `<unk>` | 24 | **44** |

  The source-span counts are exactly E1's (22 / 24), so augmentation left the Vietnamese
  side untouched and **every extra unknown is a retrieved character that never appeared in a
  training source**. The model sees `<unk>` where those hints should be — a direct, measured
  cost of the 91.85 % coverage ceiling.
- **Model size +10.3 %.** `share_decoder_input_output_embed` shares only the *decoder*
  tables, so the larger source dictionary enlarges the encoder embedding outright:
  **67,652,608 → 74,640,384 parameters** (`train.log`, `num. shared model params`).
  The delta is exactly 6,987,776 = 13,648 new types × 512 dims.

So E1 vs E3-custom is **not** a controlled comparison of a single change: E3 has 10 % more
parameters, 3.9× longer sources, and — see §2.3 — a truncated training schedule. This must
be stated wherever the +0.99 BLEU is reported.

#### 1.5.3 E2 — chat-template encoding with response-only loss

No binarization; examples are tokenized in-process (`_encoded_examples`, `llm_runner.py:65-100`).

1. Build a two-turn chat: a Vietnamese system instruction ("dịch câu tiếng Việt sang Hán
   văn cổ… chỉ trả về bản dịch") plus `"Dịch sang Hán văn cổ:\n{source}"`
   (`configs/e2_qwen3_qlora.yaml:16-18`). The target is the **detokenized** `reference`
   (`llm_runner.py:80`) — E2 learns to emit natural Chinese, not character-spaced text.
2. Render the prompt twice through `apply_chat_template`, with `enable_thinking=False`
   both times: once with `add_generation_prompt=True` (the prefix), once with the assistant
   turn appended (the full sequence).
3. **Assert the full encoding starts with the prefix encoding** (`llm_runner.py:92-95`) and
   refuse to proceed otherwise. Without this, a template change would shift the mask and
   silently train on the prompt. A fully truncated target is also a hard error
   (`llm_runner.py:96-97`).
4. Labels = `[-100] × len(prefix) + full_ids[len(prefix):]` — **loss on the translation
   only**, never on the instruction (`llm_runner.py:98`).
5. Truncate to `max_sequence_length: 768`. This was lowered from 1024 after a memory
   preflight; the longest encoded training example was 481 tokens, so **nothing was
   truncated** (`docs/EXECUTION_STATUS.md:15`).
6. **Pack** complete examples greedily into 768-token windows, after a seeded shuffle
   (`_pack_examples`, `llm_runner.py:103-122`). Packing is on for E2
   (`pack_examples` defaults to `augmenter is None`, `llm_runner.py:203`) and off for
   knowledge backends, because packing without an attention reset would let one example see
   the previous example's hint block (`llm_runner.py:199-202`).

Result: **19,218 examples → 2,903 packed sequences** (`run_manifest.json` →
`train_examples_after_packing`). Whole examples are never split, so response masking stays
valid inside every window.

> The manifest's `packing_enabled` / `train_sequences` / `train_tokens` fields are **absent**
> for this run — they were added to the code afterwards. Absent, not `false`: packing was
> on, and `train_examples_after_packing: 2903` against 19,218 examples is the proof.

The same prompt construction is reused verbatim at inference (`llm_runner.py:296-298`
comments on exactly why), so the fit-time and decode-time prompt shapes cannot drift.

---

## 2. Running the experiments

### 2.1 E1 — Fairseq Transformer baseline

`make e1-train` → `$(FAIRSEQ) train --config configs/e1_fairseq.yaml`.

Launched as `python -m mt_pipeline.fairseq_train`, not the `fairseq-train` console script,
because the console script offers no hook for the determinism switches
(`fairseq_train.py:12-16`): `cudnn.deterministic=True`, `cudnn.benchmark=False`,
`torch.use_deterministic_algorithms(True, warn_only=True)`. The child process also gets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=<seed>`
(`runtime.py:10-18`, applied at `fairseq_runner.py:186`).

| | value |
|---|---|
| Architecture | `transformer`, 6+6 layers, d_model 512, FFN 2048, 8 heads, dropout 0.1, shared decoder input/output embeddings |
| Parameters | 67,652,608 (all trained) |
| Objective | `label_smoothed_cross_entropy`, smoothing 0.1 |
| Optimizer | Adam β=(0.9, 0.998), lr 5e-4, `inverse_sqrt`, 8,000 warmup updates, weight decay 0, no gradient clipping |
| Batch | `--max-tokens 1000` × `--update-freq 4` ≈ **4,000 tokens effective**, 1 GPU, fp16 |
| Stopping | `max_update 50000`, `max_epoch 500`, **`patience 20`** |
| Selection | `--best-checkpoint-metric bleu --maximize-best-checkpoint-metric`, in-training `--eval-bleu` at beam 7, `--eval-bleu-detok space`, `keep_best_checkpoints 1`, `--no-epoch-checkpoints` |
| Seed | 42 |

**What the run did:** early-stopped on patience at **epoch 171 / 24,791 updates**, having
peaked at **epoch 151 (in-training valid BLEU 29.39)** — exactly 20 epochs of no
improvement, so this is a converged run. 3,119.5 s (~52 min). Realized `train_bsz` 132.5
sentences, `train_wpb` 2,600 tokens.

> ⚠️ This checkpoint was **warm-started**: `train.log:268` records
> `Loaded checkpoint … checkpoint_last.pt (epoch 5 @ 579 updates)` from an interrupted first
> attempt. A single clean `make e1-train` will not reproduce BLEU 28.63. The no-resume guard
> that now prevents this (`fairseq_runner.py:102-115`) was added afterwards. Full account in
> `docs/REPRO_AUDIT.md`.

### 2.2 E2 — Qwen3-8B NF4 QLoRA

`make e2-train` → `$(LLM) train --config configs/e2_qwen3_qlora.yaml`. Refuses to start
without CUDA (`llm_runner.py:143-144`) and calls `transformers.set_seed(42)` before
anything touches the model.

| | value |
|---|---|
| Base model | `Qwen/Qwen3-8B`, revision pinned to commit **`b968826d9c46dd6066d109eabc6255188de91218`**, recorded in the manifest and reused verbatim at inference (`llm_runner.py:256`, `:314`) |
| Thinking mode | disabled at both encode and decode (`enable_thinking=False`) |
| Quantization | 4-bit **NF4**, double quantization, fp16 compute (`BitsAndBytesConfig`, `llm_runner.py:25-34`) |
| LoRA | r 16, α 32, dropout 0, no bias, `target_modules: all-linear`, `task_type=CAUSAL_LM`, after `prepare_model_for_kbit_training` with gradient checkpointing |
| Batch | per-device 1 × accumulation 16 = **16 packed windows** per step, up to ~12 k tokens (16 × 768 is the capacity; greedy packing leaves some slack) |
| Optimizer | `paged_adamw_8bit`, lr 2e-4, cosine, warmup ratio 0.05, weight decay 0.01, grad-norm clip 1.0, fp16 |
| Schedule | 3 epochs = **546 steps**; eval + save every 200 steps, `save_total_limit 3` |
| Selection | `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False` |
| Seed | `seed` and `data_seed` both 42 |

**What the run did:** 546 steps / 3.0 epochs in **20,328.3 s (5 h 39 m)**, final
`train_loss` 1.5388, `total_flos` 2.82e17. Because `eval_steps: 200` and the run is only
546 steps long, there were exactly **two** validation points:

| step | epoch | `eval_loss` |
|---:|---:|---:|
| 200 | 1.10 | 1.4444 |
| 400 | **2.20** | **1.4322** ← best |

So the **saved adapter is the step-400 / epoch-2.2 model, not the end-of-epoch-3 model** —
`load_best_model_at_end` restores it before `trainer.save_model`
(`llm_runner.py:227-228`, `:250-252`). `docs/EXECUTION_STATUS.md:15` reads as though the
3-epoch model was kept; the report should say "best of two checkpoints by validation loss
at epoch 2.2". Two evaluation points is a thin selection signal, and worth disclosing as
such.

Outputs: `checkpoint/e2_.../adapter` (adapter + tokenizer),
`work/e2_.../training_history.json` (57 log records), `trainer/trainer_state.json`.

> ⚠️ `Makefile:111` pipes training through `tee`, which truncates the log file *before* the
> in-process guard can refuse the run — a later `make e2-train` attempt therefore erased the
> real 5 h 39 m log; `work/e2_.../train.log` now holds only that guard's traceback. The
> quantitative record survives in `training_history.json`, `trainer_state.json` and
> `run_manifest.train_metrics`. E1/E3 are immune because their guard fires before
> `run_logged` opens the log. Details and the fix in `docs/REPRO_AUDIT.md`.

### 2.3 E3-custom — Fairseq + source-side retrieval

Identical command path to E1 with `backend: fairseq_knowledge`, and a config whose
`model`, `training` and `decoding` blocks are byte-identical to E1's. The only differences
are the `knowledge:` block and the resulting data (§1.5.2).

**What the run did:** ran to **epoch 87, hitting `max_update: 50000`**, in 4,758.7 s
(~79 min), with realized `train_bsz` 33.3 (vs E1's 132.5 — the 3.9× longer sources mean
far fewer sentences fit a 1,000-token batch). Best in-training valid BLEU **31.05 at epoch
77**; the last epoch scored 29.67.

This sharpens the confound in §1.5.2: **E1 converged and early-stopped on patience; E3 was
cut off by the update cap 10 epochs after its best, with `patience 20` never given the
chance to fire.** E1 and E3 share a config budget, not a realized budget:

| | E1 | E3-custom |
|---|---:|---:|
| epochs | 171 | 87 |
| updates | 24,791 | 50,000 (cap) |
| stopped by | patience 20 | `max_update` |
| best epoch | 151 | 77 |
| realized `train_bsz` | 132.5 | 33.3 |
| parameters | 67.7 M | 74.6 M |
| wall clock | 3,119.5 s | 4,758.7 s |

E3's +0.99 test BLEU over E1 should therefore be read as a **lower bound from a truncated,
larger model**, not as a clean measurement of retrieval's effect.

### 2.4 What was not run

- **`e3_fairseq_knowledge` (the official lab `knowledge_v2`)** — `status: blocked`; the lab
  integration code was never supplied. `cli.py:27-29` raises the config's `blocked_reason`
  rather than substituting anything. Nothing may be reported under this ID.
- **E4 (Qwen3 + the same retrieval)** — implemented (`qlora_knowledge` backend,
  `render_hint`, a `hinted < len(records)//2` tripwire at `llm_runner.py:375-384`) and
  cost-measured (`metrics/e4_knowledge_preflight.json` is an **estimate only**), but never
  trained. No `work/e4…`, no checkpoint, no predictions. It is an extension, not the named
  bonus.

### 2.5 What every run records

`base_manifest()` + the backend's own fields → `work/<id>/run_manifest.json`
(`io_utils.py:84-129`, `fairseq_runner.py:191-205`, `llm_runner.py:257-278`), automatically:

- UTC timestamp, hostname, platform, full Python version, `git_revision` (currently `null`
  — there is no repository; see `docs/REPRO_AUDIT.md`);
- `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, complete `conda list --json` and `pip freeze`;
- `CUDA_VISIBLE_DEVICES` and, per device, name/total memory/CUDA version;
- the **entire config dict** plus its SHA-256;
- the dataset fingerprint (6 split hashes + knowledge hash);
- the trained artifact's hash — `checkpoint_sha256` for Fairseq,
  `adapter_config_sha256` + `resolved_model_revision` for QLoRA;
- pinned versions of the packages that matter (`fairseq`/`torch`/`sacrebleu`/`PyYAML`, or
  `torch`/`transformers`/`peft`/`bitsandbytes`/`accelerate`/`datasets`);
- for E3, the full `knowledge_augmentation` report.

### 2.6 Guards against accidental clobbering

| Guard | Location | Refuses to |
|---|---|---|
| `ensure_clean_checkpoint_dir` | `fairseq_runner.py:102-115` | train when `checkpoint_last.pt` exists — Fairseq would silently resume and yield a model the same command cannot reproduce. Opt in with `resume: true`. |
| `ensure_clean_adapter_dir` | `llm_runner.py:125-137` | train over an existing adapter, which would destroy the artifact the frozen selection points at |
| `ensure_selection_frozen` | `freeze.py:66-82` | decode the test split before validation selection is frozen (§3.2) |
| `_refuse_to_discard_annotations` | `error_analysis.py:79-98` | regenerate the annotation sheet when any row already carries manual labels |
| Retrieval tripwire | `llm_runner.py:375-384` | accept a "knowledge run" whose predictions are mostly hint-free (a `backend:` typo would otherwise pass every downstream gate) |

---

## 3. Validation

### 3.1 Model selection on the validation split

Selection is per-experiment, and the two backends use **different criteria**:

| | E1 / E3-custom | E2 |
|---|---|---|
| criterion | in-training **BLEU** on `valid`, maximized | **`eval_loss`**, minimized |
| computed by | Fairseq `--eval-bleu`, beam 7, `--eval-bleu-detok space` | `Trainer` eval loop |
| cadence | every epoch | every 200 steps (**2 points total**) |
| kept | `checkpoint_best.pt`, `keep_best_checkpoints 1`, no per-epoch checkpoints | best adapter via `load_best_model_at_end` |
| chose | E1 epoch 151 (29.39) · E3 epoch 77 (31.05) | step 400, epoch 2.2 (1.4322) |

The in-training BLEU is a *selection* signal and should not be quoted as a result, but it
is not a differently-defined metric. `--eval-bleu-detok space` **disables** detokenization
(`fairseq/tasks/translation.py:241-245`) and `eval_tokenized_bleu` defaults to `False`, so
Fairseq calls `sacrebleu.corpus_bleu(hyps, [refs])` at its default `13a`
(`translation.py:494-497`) — character-spaced text through `tok:13a`, exactly like §3.4.
The residual 0.02 gap (E1 29.39 vs 29.37; E3 31.05 vs 31.04) is therefore a **decoding-path
difference, not a metric-definition difference**: standalone generation disables the
encoder fast path (`fairseq_generate.py:7-15`) and batches at `--batch-size 64`, while
in-training validation keeps the fast path and batches by `--max-tokens`, and under fp16
either changes numerics slightly. Treat the near-agreement as a consistency check on the
scoring path.

Once the best checkpoint exists, `make e*-val` decodes the val split through the ordinary
prediction path and scores it with the same `evaluate` command used for test — so the
reported validation numbers and the reported test numbers are produced by identical code.

### 3.2 The freeze gate — the test set is locked until selection is committed

`freeze_selection` (`freeze.py:29-63`) writes `work/<id>/selection_frozen.json` recording
`experiment_id`, config path + **config SHA-256**, checkpoint path + **checkpoint hash**
(`sha256_file` for a `.pt`, `sha256_tree` for an adapter directory), the validation
prediction and metric paths + hashes, the `decoding_config`, and
`test_generation_authorized: true`. It refuses to freeze from anything that is not 510
validated `val` rows of the right experiment (`freeze.py:38-46`).

`ensure_selection_frozen` (`freeze.py:66-82`) then runs before **every** test decode and
re-checks three things against disk: the config hash, the checkpoint hash, and the decoding
block. Any post-hoc edit to the model, the config, or the beam settings makes test
generation fail rather than quietly produce a number.

Current state: **E1 and E3 verify** (`config_sha256` `b8bed7be…` and `25f9abb90e…` both
match their files today). **E2 does not** — its frozen record holds `ca68e42c…` while the
file hashes to `d4057d53…`, a transient state that was never committed anywhere. What
remains provable for E2, and what does not, is spelled out in `docs/REPRO_AUDIT.md`; the
short version is that the *training* config and the *decoding* config are both recoverable
(from `run_manifest.json` and from all 510 prediction rows), while `prompt` and `model.*`
as they stood at test-generation time are not. **Do not re-run `make e2-freeze`** — it would
hash today's config and stamp authorization dated after the predictions.

### 3.3 Prediction records

Both backends emit the same schema, one JSON object per line, 510 lines per file
(`schema.py:7-19`): `sample_id`, `split`, `source`, `reference`, `prediction`,
`prediction_raw`, `prediction_scored`, `experiment_id`, `model_name`, `checkpoint`,
`decoding_config` (+ `knowledge_hint` on knowledge backends only, dropped when unset so
other experiments' rows stay byte-identical).

`validate_prediction_rows` (`schema.py:76-85`) enforces the count, no duplicate
`sample_id`, `sample_id`/`split` agreement, no empty fields, and **source row order** by
numeric suffix. It runs at freeze, at evaluation, at comparison, and at error-analysis
sampling — a misordered or short prediction file cannot be scored.

Keeping `prediction_raw` next to `prediction` is what makes the post-processing auditable:

- **Fairseq** (`fairseq_runner.py:208-291`): decode via `python -m mt_pipeline.fairseq_generate`
  (which disables an incompatible PyTorch encoder fast path), capture stdout, parse `H-`
  lines, and **assert the hypothesis IDs are exactly `0..509`** (`:264-267`) — a dropped or
  reordered sentence is a hard failure. Beam 7, lenpen 1.0, `max_len_a` 1.2, `max_len_b` 10,
  batch 64. `prediction` = `detokenize_chinese(raw)`.
- **QLoRA** (`llm_runner.py:281-387`): greedy-free beam search, `num_beams 4`,
  `do_sample false`, per-sentence `max_new_tokens = min(512, ceil(2 × source_tokens) + 32)`.
  `clean_llm_generation` (`normalize.py:26-37`) strips only known wrappers —
  `<think>…</think>`, `assistant:`/`答案：`/`译文：`/`翻译：` prefixes, matched outer quotes —
  and never rewrites content. Leaked thinking content (`:354-355`) or an empty prediction
  (`:357-358`) is a hard failure, not a silent blank.

### 3.4 Metric computation — `make e*-val` / `e*-test`

`evaluate_predictions` (`evaluation.py:41-69`) revalidates the rows, additionally
**recomputes `score_form_chinese(prediction)` for all 510 rows and refuses to score if the
stored `prediction_scored` differs** (`evaluation.py:34-37`), then scores
`prediction_scored` against `score_form_chinese(reference)` with SacreBLEU 2.6.0:

- `BLEU(lowercase=False, tokenize="13a", smooth_method="exp", effective_order=False)`
- `CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False)` → chrF2++

Output includes the score, the **signature**, and SacreBLEU's verbose string:

```
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

> **Methodological caveat that must appear in the report.** Both metrics are
> **character-level here, not word-level.** Inputs are pre-spaced one codepoint per token,
> so `tok:13a` has nothing left to segment, and chrF++'s word bigrams (`nw:2`) are in fact
> *character* bigrams. `ref_len = 8,982` on test equals the corpus test target token count
> exactly, confirming it. Character-level BLEU on Chinese runs systematically higher than
> word-level BLEU on Latin-script targets, so these numbers are **not on the same scale** as
> word-level EN/VI scores from other groups. Note also that the spec asks for a unified
> Moses tokenizer, which SacreBLEU 2.6.0 does not expose — `13a` (its `mteval-v13a`
> equivalent) is the closest available, and over character-spaced input the choice is
> effectively inert.

**Results** (510 examples per split; every file recomputes exactly from its predictions,
§3.6):

| Experiment | val BLEU | val chrF++ | test BLEU | test chrF++ |
|---|---:|---:|---:|---:|
| E1 Fairseq | 29.37 | 31.12 | 28.63 | 30.58 |
| E2 Qwen3-8B QLoRA | 41.06 | 41.20 | **40.77** | **40.87** |
| E3-custom Fairseq + retrieval | 31.04 | 32.72 | 29.62 | 31.41 |

Test-set n-gram precisions and brevity penalty, which say more than the single score:

| Experiment | 1-gram | 2-gram | 3-gram | 4-gram | BP | hyp/ref len |
|---|---:|---:|---:|---:|---:|---|
| E1 | 62.2 | 35.5 | 23.1 | 16.6 | 0.943 | 8,486 / 8,982 |
| E2 | 69.9 | 46.6 | 33.6 | 25.3 | **1.000** | 8,979 / 8,982 |
| E3-custom | 62.0 | 35.8 | 23.4 | 16.9 | 0.968 | 8,697 / 8,982 |

Two readings worth putting in the report: E2's advantage grows with n-gram order
(+7.7 unigram → +8.7 4-gram in absolute points, a much larger relative gain), i.e. it is
producing better *sequences*, not just better characters; and both Fairseq systems
**under-generate** (BP 0.943 / 0.968) while E2 matches the reference length almost exactly.
E3's unigram precision is actually a hair *below* E1's — its gain comes from length
calibration and higher-order matches, not from getting more characters right.

### 3.5 Statistical comparison — `make compare`

`compare_predictions` (`evaluation.py:77-136`) runs a **paired bootstrap**: it verifies both
files carry the same ordered `sample_id`s and identical references (`:88-93`), then draws
1,000 resamples with replacement using `random.Random(seed)` and recomputes both corpus
metrics on each resample. Reported per metric: observed scores, observed delta, the
2.5/97.5 percentiles of the delta distribution, a two-sided p-value from the
`(count + 1)/(n + 1)`-smoothed tail, and the metric signature.

**Seed 42, 1,000 samples, test split.** The Makefile passes `--seed $(COMPARE_SEED)`
explicitly because the CLI default is 12345 (`Makefile:52-53`) — omitting it silently
changes every interval.

| Comparison | ΔBLEU | 95 % interval | p | ΔchrF++ | 95 % interval | p |
|---|---:|---|---:|---:|---|---:|
| E2 − E1 | **+12.14** | [10.22, 13.77] | 0.002 | +10.29 | [9.03, 11.79] | 0.002 |
| E3-custom − E1 | +0.99 | **[−0.14, 2.16]** | 0.092 | +0.83 | [−0.06, 1.71] | 0.064 |
| E2 − E3-custom | **+11.15** | [9.26, 12.77] | 0.002 | +9.46 | [8.05, 10.88] | 0.002 |

The LLM's advantage is large and unambiguous. **Retrieval's +0.99 BLEU is not significant
at α = 0.05** — the interval straddles zero on both metrics — and per §2.3 it also comes
from a bigger, truncated model. Report it as inconclusive, with the confound named.

### 3.6 Verification machinery

**`make repro-check`** (`repro_check.py`) — the standing gate over the artifacts on disk. It
does three things, writing nothing outside `/tmp`:

1. **Freeze state.** Re-runs `ensure_selection_frozen` per experiment and compares the
   outcome against `EXPECTED_FREEZE_STATE` (`repro_check.py:29-36`). Missing artifacts count
   as *skipped*, not failed, so a fresh clone passes honestly.
2. **Stored metrics.** Recomputes every `metrics/<id>.{val,test}.json` from its prediction
   file into a scratch directory and asserts **full dict equality including signatures**
   (`:100-112`).
3. **Dataset hashes.** Re-fingerprints the corpus and compares against every manifest's
   recorded split hashes and knowledge hash (`:116-142`).

Run on 2026-08-17: **`ok: true`**. All six metric files recomputed exactly (29.37 / 28.63 /
41.06 / 40.77 / 31.04 / 29.62), all three dataset fingerprints matched, E1 and E3 freeze
records verified, E2 failed as expected, E4 skipped.

> ⚠️ Read the report, not just the exit code: E2's entry is asserted as an **expected
> failure** (`"configs/e2_qwen3_qlora.yaml": False`), so `ok: true` means "the known break
> is still exactly the known break", not "everything verifies". The assertion inverts if E2
> is ever legitimately re-frozen.

**`make repro-smoke`** — the only *determinism* gate. Derives the smoke config into two
isolated trees with `--gpu` (forcing cuda+fp16 so the runs take E1/E3's actual code path,
not a vacuous CPU-fp32 path), trains both, decodes both, and asserts identical weights and
identical translations via `compare-runs`. It covers the Fairseq path at smoke scale only —
never E1/E3 at full scale, and never E2.

**`make test`** — 34 unit tests across 9 files under `.conda/eval`, including a pinned
regression test that re-derives E3's augmented sources and asserts they still hash to the
values in `work/e3_.../knowledge_augmentation.json`. Run on 2026-08-17: **34 passed** in
6.2 s. **`make check`** = tests + a data audit
into `/tmp` + `repro-check`, all without a GPU. **`make status`** (`status.py`) reports which
deliverables exist.

Determinism measures in force, and one honest limit: the switches in `fairseq_train.py:12-16`
use `warn_only=True`, so an op without a deterministic kernel warns instead of failing. It
never fired — **neither Fairseq log contains a single non-deterministic-op warning**, and
`run_logged` captures child stderr into the log (`runtime.py:39-45`), so such a warning would
have been recorded had one occurred. E2's QLoRA path has no equivalent guarantee.

### 3.7 Error analysis — prepared, not yet done

`make error-analysis` → `prepare_annotation_template` (`error_analysis.py:23-66`) samples
**100 test `sample_id`s** with `random.Random(42).sample` and emits one row per
(sample, system) — **300 rows** for E1/E2/E3 — sorted by numeric ID then experiment, each
with `annotation_status: PENDING` and empty `annotator_1_*`, `annotator_2_*`,
`adjudicated_*` fields. All three systems are annotated on the *same* 100 sentences, which
is what makes per-system label counts comparable.

Taxonomy (`error_analysis.py:12-20`): `CORRECT`, `MISTRANSLATION`, `OMISSION`,
`UNSUPPORTED_ADDITION`, `ENTITY_ERROR`, `NUMBER_OR_TIME_ERROR`, `OTHER`. `validate_labels`
rejects unknown labels and refuses `CORRECT` alongside any error label.

`summarize_annotations` (`error_analysis.py:109-146`) is strict by design: it fails unless
**every** row is `ADJUDICATED`, every label is in the taxonomy, each experiment has exactly
100 sample IDs, and all experiments share the same 100 IDs. Only then does it emit label
counts and percentages per system.

**Current state: all 300 rows are `PENDING`, zero labels assigned**, and there is no
`error_analysis/summary.json`. This is human judgment work — the labels have to come from
the group. Note the file is named `manual_sample.jsonl` but is **JSONL**, and that
`summarize-error-analysis` exists in the CLI but has no Makefile target.

### 3.8 Interpretation caveats, in one place

Everything in §1–§3 describes machinery that works. Four facts change how the numbers
should be *read*, and all four belong in the report's Limitations section:

1. **E1's checkpoint was warm-started** from an interrupted run (`train.log:268`); BLEU 28.63
   is real but not reproducible from one clean command.
2. **E2's frozen-selection chain does not verify**; state precisely what is recoverable
   (training config from `run_manifest.json`, decoding config from the frozen record, the
   config, and all 510 prediction rows) and what is not (`prompt`, `model.*` at
   test-generation time).
3. **E1 vs E3-custom is not a controlled comparison** — E3 has 10.3 % more parameters, 3.9×
   longer sources, and was truncated by `max_update` 10 epochs after its best while E1 ran to
   convergence. The +0.99 BLEU is not significant (p = 0.092).
4. **BLEU and chrF++ are character-level here**, so these scores are not comparable to
   word-level scores on Latin-script targets.

Full evidence, severities and remediation options: `docs/REPRO_AUDIT.md`.
