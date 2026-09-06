# Presenter walkthrough — the Group 10 MT pipeline end to end

September 2026: this walkthrough's recorded scores describe historical 13a runs.
New evaluation and notebook result exports use [Moses](MOSES_EVALUATION.md), with
rescored artifacts under `metrics/moses/`; model-selection history is preserved.

A briefing for whoever is presenting this pipeline (~15–20 min slot). It answers, in
order: what is the architecture, what goes in and out of each stage, what were the
low-level decisions, why were they made that way, and what did each one cost.

**This is not `docs/PIPELINE.md`.** That document is the reference — what the code does,
cited to `file:line`, organized to be lifted into the report. This one is organized for
*speaking*: one diagram to talk from, decisions framed as choice → alternative → tradeoff,
and rehearsed answers to the questions you will actually be asked. Every number here was
re-read from the artifacts on disk on 2026-08-22; where you need the citation, follow the
pointer into `PIPELINE.md`.

**Two handling rules.**

- **The results are frozen.** Every "con" below is a *disclosure the talk must make*, never
  a to-do. Do not offer to retrain, re-tune, or re-freeze anything on stage.
- **The corpus is restricted** (`README.md`, `docs/DATASET.md`). Keep this file local — do
  not publish it, do not paste it into a slide deck or anywhere outside this repo. **Unlike
  the rest of this document, §1.3 now contains verbatim corpus text** (three real
  train/test sentences, chosen deliberately per the note at the top of that section) —
  treat this file as more sensitive than the rest of `docs/` for that reason.

---

## §0 — The 60-second version

Say this first, then everything else hangs off it:

1. We translate **Vietnamese → Classical Chinese** on *Đại Việt sử ký toàn thư*, a
   historical corpus: **19,218** training pairs (silver, self-aligned) and **510 / 510**
   validation and test pairs (gold, manually aligned).
2. We built **three systems**: E1 a from-scratch Fairseq Transformer (67.7 M params), E2
   Qwen3-8B fine-tuned with 4-bit QLoRA (8 B params), E3-custom = E1 plus source-side
   character hints retrieved from a knowledge base.
3. All three read the **same rows through the same loader**, diverge for exactly one step
   (how each renders those rows for its trainer), and reconverge onto **one prediction
   schema** and **one character-level scoring path** — which is the only reason a
   6-layer Transformer and an 8 B LLM are comparable at all.
4. **Test is locked** until validation selection is frozen and hashed. Test decoding
   physically fails if the config, the checkpoint, or the beam settings changed after the
   freeze.
5. Headline: **E2 40.77 test BLEU, E3-custom 29.62, E1 28.63.** The LLM's +12.14 is
   unambiguous; retrieval's +0.99 is not significant and is confounded.

---

## §1 — The architecture on one screen

### 1.1 The dataflow

```
                     configs/dataset.yaml          zh-vi/knowledge.json
                  (paths, expected_lines)         (16,000 entries)
                              │                            │
                              ▼                            │
                    ┌───────────────────┐                  │
   GATE ═══════════▶│    read_split()   │  hard-fail: side lengths differ,
   expected_lines   │  data.py:44-63    │  or line count ≠ expected_lines
                    └─────────┬─────────┘
                              │  ParallelRow: sample_id, source,
                              │  reference_tokenized  (+ 2 derived forms)
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
  ┌───────────┐      ┌──────────────────┐   ┌──────────────────┐
  │ E1 render │      │ E3 render        │   │ E2 render        │   ◀── THE ONLY
  │ text/ →   │      │ + KnowledgeAug   │   │ chat template,   │       DIVERGENCE
  │ data-bin  │      │ → text/ data-bin │   │ response-only    │
  └─────┬─────┘      └────────┬─────────┘   │ loss, packing    │
        │                     │             └────────┬─────────┘
        ▼                     ▼                      ▼
  ┌───────────┐        ┌───────────┐          ┌───────────┐
  │ E1 train  │        │ E3 train  │          │ E2 train  │
  │ 67.7 M    │        │ 74.6 M    │          │ QLoRA r16 │
  └─────┬─────┘        └─────┬─────┘          └─────┬─────┘
        │                    │                      │
   GATE ═══════ ensure_clean_checkpoint_dir / ensure_clean_adapter_dir
        │                    │                      │       (no silent warm-start)
        ▼                    ▼                      ▼
        └──────────▶ predict --split val ◀──────────┘
                              │
                              ▼
                        evaluate (val)
                              │
                              ▼
                    ┌───────────────────┐
                    │ freeze-selection  │  writes selection_frozen.json:
                    │  freeze.py:29-63  │  config hash + checkpoint hash +
                    └─────────┬─────────┘  decoding block + val pred/metric hashes
                              │
   GATE ═══════════▶ ensure_selection_frozen  ── re-checks all three against disk
                              │                  BEFORE every test decode
                              ▼
                     predict --split test
                              │
   GATE ═══════════▶ validate_prediction_rows  (510 rows, unique IDs, source order)
                              │
                              ▼
                    ┌───────────────────┐
                    │     evaluate      │  recomputes prediction_scored and REFUSES
                    │ evaluation.py:34  │  to score if the stored value differs
                    └─────────┬─────────┘        ◀── THE RECONVERGENCE POINT
                              │
                              ▼
                   compare (paired bootstrap, seed 42)
                              │
                              ▼
                       metrics/*.json
```

### 1.2 The one sentence that explains the design

**Three systems, one divergence point, one reconvergence point.** Everything upstream of
the render step is shared code (so the systems see identical data), and everything
downstream of `prediction_scored` is shared code (so the systems are scored identically).
The only thing that differs is the middle. That is what makes the comparison a comparison
rather than three unrelated numbers.

If you have time for only one architecture slide, draw this.

### 1.3 The worked example — three real test sentences, traced through all three systems

Tracing sentences through all three renderings is the single most illustrative thing this
talk can show. Doing it with real text (rather than the schematic below) was a deliberate
call, made because the intended audience is the instructor who supplied the corpus — do not
reuse this section, verbatim or adapted, in front of any other audience without making that
same call again.

All three rows below are read directly from disk: source and reference from
`zh-vi/test/…no_punct.test.{vi,cn}`, E1/E3 renderings from
`work/e1_fairseq_vi_zh_v1/text/test.{vi,zh}` and
`work/e3_custom_fairseq_knowledge_vi_zh_v1/text/test.{vi,zh}`, E2's prompt from
`code/mt_pipeline/llm_runner.py:56-91` rendered through the actual cached Qwen3-8B
tokenizer, and every prediction from `predictions/{e1,e2,e3_custom}...test.jsonl`.

**test-000001** — source `tôn mẹ là lê thị làm linh hiển`, reference `尊母黎氏曰靈顯`

| System | Rendered input | Prediction |
|---|---|---|
| E1 | identical to source (whitespace-tokenized): `tôn mẹ là lê thị làm linh hiển` | `尊母黎氏爲靈顯` |
| E3 | source + `__knowledge__` + 64 retrieved hint characters: `tôn mẹ là lê thị làm linh hiển __knowledge__ 黎 是 宗 令 侍 氏 尊 孫 母 視 靈 羅 示 施 顯 市 美 濫 灵 …` (64 total) | `尊母黎氏爲顯靈` |
| E2 | `<\|im_start\|>system`⏎`Bạn là hệ thống dịch văn bản lịch sử. Hãy dịch câu tiếng Việt sang Hán văn cổ. Chỉ trả về bản dịch Hán văn, không giải thích.<\|im_end\|>`⏎`<\|im_start\|>user`⏎`Dịch sang Hán văn cổ:`⏎`tôn mẹ là lê thị làm linh hiển<\|im_end\|>`⏎`<\|im_start\|>assistant`⏎`<think>`⏎`</think>` (70-token prefix; generation continues from here) | `尊母黎氏爲靈顯` |

All three get 6 of 7 characters right and all three substitute 爲 for the reference's 曰 —
a genuine ambiguity in the source, not a system-specific error.

**test-000002** — source `tân khách đầy nhà đại đứng dậy cầm phù tiết đọc tờ chiếu kể tội huy tả hữu trói quặt`,
reference `賓客满座岱起擁節讀詔數徽罪過左右因反`

| System | Rendered input | Prediction |
|---|---|---|
| E1 | identical to source | `新客满室大起持節讀書火詔罪左右就縛` |
| E3 | source + `__knowledge__` + 64 hints: `… quặt __knowledge__ 代 有 大 等 又 家 左 今 詔 節 罪 右 夫 禁 昭 揮 新 計 擒 祐 辛 扶 符 讀 俘 徽 照 詞 浮 …` | `辛客满大起琴興節讀書詔書罪左右縛` |
| E2 | same wrapper; user turn `Dịch sang Hán văn cổ:`⏎`tân khách đầy nhà đại đứng dậy cầm phù tiết đọc tờ chiếu kể tội huy tả hữu trói quặt` | `辛客滿庭大起持節讀詔數其罪左執右縛` |

The hardest of the three: a long, low-frequency sentence, and none of the systems is close.
Worth saying out loud if this one comes up — it is not cherry-picked to make anyone look
good, it is included *because* it is representative of where all three struggle.

**test-000003** — source `vua sai bọn viên ngoại lang phạm đình đem sang`, reference `帝遣員外郎范廷等遣之`

| System | Rendered input | Prediction |
|---|---|---|
| E1 | identical to source | `帝遣員外郎范廷率等徃` |
| E3 | source + `__knowledge__` + 64 hints: `… đem sang __knowledge__ 郎 廊 瀧 郞 外 范 差 廷 君 員 良 犯 庭 貟 停 創 圓 伴 園 垣 亭 範 隕 鎗 …` | `帝遣員外郎范廷等賫書` |
| E2 | same wrapper; user turn `Dịch sang Hán văn cổ:`⏎`vua sai bọn viên ngoại lang phạm đình đem sang` | `帝遣員外郎范廷等賫` |

All three reproduce the first seven characters (`帝遣員外郎范廷`) exactly — a formulaic
"the emperor dispatched official X" pattern that recurs often in the corpus — then diverge
on how to render "sent/dispatched" at the end.

**What this makes visible that a schematic can't:** E1 and E3's inputs really are
near-identical apart from the appended hint block, and that block really is consistently
several times longer than the sentence it annotates (confirms the "barely selective" claim
in §3.6). E2 never sees the hint block — only natural Vietnamese inside the chat template —
and its output is the only one already detokenized, which is why `score_form_chinese` has
to re-space it before any of the three can be compared on the same footing (§3.3).

**Schematic version**, for any context where real corpus text isn't appropriate to show:

```
row              sample_id = test-000042
source           <vi token> <vi token> … <vi token>            (~21 tokens, no punctuation)
reference_tokn.  <hàn char> <hàn char> … <hàn char>            (1 codepoint per token)
reference        <hànchar><hànchar>…                           (whitespace removed)

E1 source line   <vi token> <vi token> … <vi token>
E3 source line   <vi token> … <vi token> __knowledge__ <c1> <c2> … <c62>
E2 encoded       <|im_start|>system  <Vietnamese instruction> <|im_end|>
                 <|im_start|>user    Dịch sang Hán văn cổ:\n<source> <|im_end|>
                 <|im_start|>assistant  <reference>            ◀── labels only here
                 └──────── labels = -100 ────────┘

all three  →  prediction  →  prediction_scored = "<c> <c> <c> …"   → SacreBLEU
```

---

## §2 — Stage contract table

Every stage is `python -m mt_pipeline <subcommand>`. The Makefile's only job is to run each
one under the environment that owns its dependencies.

| Stage | `make` target | Env | Input | Output | Blocked by |
|---|---|---|---|---|---|
| Data audit | `audit` | `eval` | `configs/dataset.yaml`, corpus, `knowledge.json` | `metrics/data_audit.json` | line-count / length mismatch |
| Prepare (E1/E3) | *(implicit in train)* | `fairseq` | corpus rows | `work/<id>/text/*`, `data-bin/`, `preprocess.log` | reuses `data-bin` unless `--force` |
| Train E1 / E3 | `e1-train`, `e3-custom-train` | `fairseq` | `data-bin/` | `checkpoint_best.pt`, `train.log`, `run_manifest.json` | `ensure_clean_checkpoint_dir` |
| Train E2 | `e2-train` | `llm` | corpus rows + HF base model | `checkpoint/<id>/adapter`, `training_history.json`, manifest | `ensure_clean_adapter_dir`; refuses without CUDA |
| Predict val | `e*-val` | backend env | checkpoint + val rows | `predictions/<id>.val.jsonl` (510) | hypothesis IDs must be exactly `0..509` |
| Evaluate | *(same target)* | `eval` | prediction JSONL | `metrics/<id>.val.json` | `prediction_scored` recompute must match |
| Freeze | `e*-freeze` | `eval` | config, checkpoint, val preds + metrics | `work/<id>/selection_frozen.json` | must be 510 validated `val` rows of that experiment |
| Predict test | `e*-test` | backend env | checkpoint + test rows | `predictions/<id>.test.jsonl` (510) | **`ensure_selection_frozen`** |
| Evaluate test | *(same target)* | `eval` | prediction JSONL | `metrics/<id>.test.json` | same recompute check |
| Compare | `compare` | `eval` | two test prediction files | `metrics/e1_vs_e2.test.json` etc. | same ordered `sample_id`s and identical references |
| Error analysis | `error-analysis` | `eval` | three test prediction files | `error_analysis/manual_sample.jsonl` (300 rows) | refuses to overwrite existing manual labels |
| Verify | `repro-check` | `eval` | everything on disk | `/tmp/group10_repro/repro_check.json` | — (writes nothing outside `/tmp`) |

**Why three environments** (`Makefile:15-20`): they are genuinely incompatible.
`.conda/fairseq` is fairseq 0.12.2 on **torch 1.13.1**; `.conda/llm` is transformers 4.57.6
/ peft 0.20.0 / bitsandbytes 0.50.1 on **torch 2.5.1**. They cannot coexist. `.conda/eval`
holds only sacrebleu 2.6.0 + PyYAML + pytest — deliberately the smallest, so **every
reported metric can be recomputed without a GPU or either training stack.**

**Reruns never touch graded artifacts.** `make all RUN_ROOT=runs/rerun-…` rebases exactly
three writable config fields (`work_dir`, `checkpoint_dir`, `prediction_dir`) into a
parallel tree; `dataset_config` and `knowledge.file` stay pointed at the read-only inputs.

---

## §3 — The data path, step by step

### 3.1 One config is the single source of truth

`configs/dataset.yaml` names the six corpus files, the knowledge file, a `quality` label per
split, and an `expected_lines` count per split. Nothing else in the codebase knows a file
path to the corpus.

**Why:** the `expected_lines` field is a tripwire. A silently swapped or truncated corpus
file cannot enter a run — `read_split` raises before a single token is tokenized.

### 3.2 `read_split` — the only reader (`data.py:44-63`)

Per split, in order:

1. Read both sides; **hard-fail if the line counts differ**.
2. **Hard-fail if the count differs from `expected_lines`.**
3. Assign `sample_id = f"{split}-{index:06d}"`, **1-based** (`enumerate(…, 1)`).
4. Apply `normalize_whitespace` to both sides: **Unicode NFC** → collapse whitespace runs →
   strip (`normalize.py:11-12`).

**Nothing else happens.** No lowercasing, no punctuation stripping, no length filtering, no
deduplication. The corpus enters training as supplied.

**Why so minimal:** the files are already `no_punct` and pre-tokenized (Vietnamese
word/syllable-segmented, Chinese **one character per token**). Every transformation we
*don't* apply is one fewer thing that can differ between the three systems, and one fewer
thing to explain when a number looks odd. NFC is the exception because Vietnamese
diacritics have two valid Unicode encodings, and unnormalized text would produce two
different tokens for the same word.

### 3.3 Three forms of the target, and why (`data.py:16-29`)

| Form | Definition | Consumed by |
|---|---|---|
| `reference_tokenized` | corpus line, whitespace-normalized: `如 使 越 人 …` | E1/E3 training and eval target |
| `reference` | `detokenize_chinese` — all whitespace removed: `如使越人…` | human-readable; the **E2 training target** |
| `reference_scored` | `score_form_chinese` — re-spaced, one codepoint per token | **the scoring input**, for all systems |

**This is the crux of the whole design.** E1 trains on character-spaced text; E2 trains on
natural unspaced Chinese, because that is what a pretrained LLM has seen. If we scored them
on their own output surfaces we would be measuring two different things. Instead every
prediction is projected onto `score_form_chinese` before scoring, so both systems are
measured on an identical representation.

`score_form_chinese` is a join-then-respace round trip, so it is lossless only if every
corpus target token is a single codepoint. **This was verified, not assumed:** across all
20,238 rows, `score_form_chinese(reference_tokenized) == reference_tokenized` exactly. The
scored reference *is* the corpus reference, byte for byte.

### 3.4 `make audit` — "we know what our data is" (`data.py:96-179`)

Produces `metrics/data_audit.json`. Four things worth a slide:

**Sizes and shape.** Train 401,997 source tokens / 357,806 target; mean ~20.9 source tokens
per line, max 199 — comfortably under E1's 256-position encoder limit, so **no example is
truncated by `max_source_positions`.**

**Contamination: clean.** All three pairwise overlaps (train×val, train×test, val×test) are
**0 shared pairs, 0 shared sources, 0 shared targets.** Test scores are not inflated by
leakage. This is the single most important audit result — say it explicitly.

**Internal duplication: train only.** 1,596 duplicate pairs, 1,653 duplicate source lines,
1,635 duplicate target lines, and **44 sources with more than one distinct target.** Val and
test are 0 on all four counts. Those 44 are a hard ceiling on what any model can get right
from the source alone.

**Defects: 6 rows.** Six *train-side target* lines contain U+FFFD (`�`) — characters lost
before the corpus reached us. All in train, none in val/test. Surfaced in
`report["issues"]`, **not removed** (see §5).

Plus SHA-256 for all six corpus files, recomputed at every train and re-verified by
`repro-check`.

### 3.5 The knowledge resource

`zh-vi/knowledge.json`, SHA-256 `68dc8623…`, **16,000** character entries, each with all four
top-level fields. Coverage of the corpus: the three splits' target sides use **5,458**
distinct characters, **5,013** have an entry → **91.85 %**. The 445 uncovered are rare
Nôm/variant forms.

**That 8.15 % gap is the honest upper bound on how much retrieval can possibly help**, and
§5 shows it turning into a measured cost.

### 3.6 The fork: three renderings of the same rows

**E1 — Fairseq binarization** (`fairseq_runner.py:28-99`). Write
`work/<id>/text/{train,valid,test}.{vi,zh}` (note `val` → `valid`, which is what Fairseq's
`--validpref` expects), then `fairseq-preprocess --thresholdsrc 1 --thresholdtgt 1`.
Threshold 1 means **no subword model and no vocabulary pruning**: full word vocabulary on
Vietnamese (3,616 types), full character vocabulary on Chinese (5,392). Dictionaries are
built from `--trainpref` only, so val/test OOV is a genuine held-out measurement:
**22 unknown source tokens and 34 unknown target tokens on val** (0.20 % / 0.36 % against
Fairseq's EOS-inclusive counts), 24 / 45 on test (0.22 % / 0.47 %).

**E3-custom — source-side retrieval** (`knowledge.py`). A **preprocessing-only**
intervention: architecture, criterion, and decoding are byte-identical to E1's config.

- *Index*: harvest Vietnamese lookup forms from `pronunciation`, `nom_query`,
  `lookup_query`, `query_variants`; casefold and normalize; key a dict from the **token
  tuple** to the Chinese characters it can realize → **6,389 distinct lookup forms**.
- *Retrieval*: slide over every start position × every known form length, count matching
  spans, accumulate a score per candidate character.
- *Ranking*: **match count → training-target frequency → Unicode codepoint order** — fully
  deterministic, and the frequency table is built from the **train split only**, so no
  val/test information enters ranking.
- *Rendering*: append the sentinel token `__knowledge__` plus up to `max_candidates: 64`
  characters, budgeted against `max_source_positions` so it can never overflow the encoder.

What it actually did: **100 % of rows in all three splits got hints**, mean 61.5–62.4
candidates against a 64 cap, mean ~20.5 matched spans. Retrieval is high-recall and
**barely selective** — nearly every sentence saturates the budget.

**E2 — chat-template encoding with response-only loss** (`llm_runner.py:65-122`). No
binarization; tokenized in-process.

1. Build a two-turn chat: Vietnamese system instruction + `"Dịch sang Hán văn cổ:\n{source}"`.
   The target is the **detokenized** `reference` — E2 learns to emit natural Chinese.
2. Render the prompt twice through `apply_chat_template` with `enable_thinking=False`: once
   with `add_generation_prompt=True` (the prefix), once with the assistant turn appended.
3. **Assert the full encoding starts with the prefix encoding** and refuse otherwise.
   Without this, a template change would shift the mask and silently train on the prompt.
4. `labels = [-100] × len(prefix) + full_ids[len(prefix):]` — **loss on the translation
   only.**
5. Truncate at `max_sequence_length: 768`. Lowered from 1024 after a memory preflight; the
   longest encoded example was 481 tokens, so **nothing was truncated**.
6. **Pack** complete examples greedily into 768-token windows after a seeded shuffle:
   **19,218 examples → 2,903 packed sequences.** Whole examples are never split, so response
   masking stays valid inside every window.

The same prompt construction is reused verbatim at inference (`llm_runner.py:325-326` names
exactly why), so fit-time and decode-time prompt shapes cannot drift.

---

## §4 — Model architectures and what the runs actually did

Distinguish **configured** from **realized** — the gap is where the interesting caveats live.

### E1 — Fairseq Transformer baseline

| | |
|---|---|
| Architecture | `transformer`, 6+6 layers, d_model 512, FFN 2048, 8 heads, dropout 0.1, shared decoder input/output embeddings |
| Parameters | **67,652,608**, all trained |
| Objective | `label_smoothed_cross_entropy`, smoothing 0.1 |
| Optimizer | Adam β=(0.9, 0.998), lr 5e-4, `inverse_sqrt`, 8,000 warmup updates |
| Batch | `--max-tokens 1000` × `--update-freq 4` ≈ **4,000 tokens effective**, 1 GPU, fp16 |
| Stopping | `max_update 50000`, `max_epoch 500`, **`patience 20`** |
| Selection | in-training BLEU at beam 7, `keep_best_checkpoints 1`, `--no-epoch-checkpoints` |
| Decoding | beam 7, lenpen 1.0, `max_len_a` 1.2, `max_len_b` 10, batch 64 |

**Realized:** early-stopped on patience at **epoch 171 / 24,791 updates**, having peaked at
**epoch 151** (in-training valid BLEU 29.39) — exactly 20 epochs of no improvement, so this
is a **converged** run. 3,119.5 s (~52 min). Realized `train_bsz` 132.5 sentences.

Launched as `python -m mt_pipeline.fairseq_train` rather than the `fairseq-train` console
script, purely because the console script offers no hook for the determinism switches
(`cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(warn_only=True)`,
plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED`).

### E2 — Qwen3-8B NF4 QLoRA

| | |
|---|---|
| Base | `Qwen/Qwen3-8B`, revision pinned to commit **`b968826d…`**, recorded in the manifest and reused verbatim at inference |
| Thinking | disabled at both encode and decode |
| Quantization | 4-bit **NF4**, double quantization, fp16 compute |
| LoRA | r 16, α 32, dropout 0, no bias, `target_modules: all-linear`, after `prepare_model_for_kbit_training` with gradient checkpointing |
| Batch | per-device 1 × accumulation 16 = **16 packed windows** per step |
| Optimizer | `paged_adamw_8bit`, lr 2e-4, cosine, warmup ratio 0.05, weight decay 0.01, clip 1.0, fp16 |
| Schedule | 3 epochs = **546 steps**; eval + save every 200 steps |
| Selection | `load_best_model_at_end`, `metric_for_best_model="eval_loss"` |
| Decoding | beam 4, `do_sample false`, per-sentence `max_new_tokens = min(512, ceil(2 × src_tokens) + 32)` |

**Realized:** 546 steps / 3.0 epochs in **20,328.3 s (5 h 39 m)**, final `train_loss` 1.5388.
Because `eval_steps: 200` against a 546-step run, there were exactly **two** validation
points — step 200 (`eval_loss` 1.4444) and step 400 (**1.4322**, best).

> **The saved adapter is the step-400 / epoch-2.2 model, not the end-of-epoch-3 model.**
> `load_best_model_at_end` restores it before `save_model`. `docs/EXECUTION_STATUS.md:15`
> reads as though the 3-epoch model was kept. Say "best of two checkpoints by validation
> loss, at epoch 2.2."

### E3-custom — Fairseq + source-side retrieval

Config `model`, `training`, and `decoding` blocks are **byte-identical to E1's**. The only
difference is the `knowledge:` block and the data that results from it — but that data
difference propagates into three model-level differences:

- **Source length ×3.9.** Train source tokens 421,215 → **1,622,823**; mean tokens per line
  20.92 → 83.44 (= 20.92 source + 1 separator + 61.53 hints ✓).
- **Source vocabulary ×4.8.** 3,616 → **17,264 types**, because retrieved Han characters now
  live in the *source* dictionary.
- **Model size +10.3 %.** `share_decoder_input_output_embed` shares only the *decoder*
  tables, so the larger source dictionary enlarges the encoder embedding outright:
  **67,652,608 → 74,640,384** parameters. The delta is exactly 6,987,776 = 13,648 new types
  × 512 dims.

**Realized:** ran to **epoch 87, hitting the `max_update: 50000` cap** in 4,758.7 s (~79 min).
Best in-training valid BLEU **31.05 at epoch 77**; the final epoch scored 29.67. Realized
`train_bsz` **33.3** vs E1's 132.5 — the same 1,000-token budget holds ~4× fewer sentences
when sources are 3.9× longer.

**And a measured cost of the coverage gap.** Split by position around the `__knowledge__`
separator, against E3's own dictionary:

| | tokens in the original source span | tokens in the hint span |
|---|---:|---:|
| val `<unk>` | 22 | **49** |
| test `<unk>` | 24 | **44** |

The source-span counts are **exactly E1's** — augmentation left the Vietnamese side
untouched — so **every extra unknown is a retrieved character that never appeared in a
training source.** The model sees `<unk>` where those hints should be.

### Not run

- **`e3_fairseq_knowledge`** (the official lab `knowledge_v2`) — `status: blocked`; the lab
  integration code was never supplied. `cli.py:33-35` raises the config's `blocked_reason`
  rather than substituting anything. **Nothing may be reported under this ID.**
- **E4** (Qwen3 + the same retrieval) — fully implemented (`qlora_knowledge` backend,
  `render_hint`, a "mostly hint-free predictions" tripwire) and cost-estimated, but **never
  trained**. It is an extension, not the named bonus.

---

## §5 — The decisions, and what each one cost

Each row is labeled by **where the rationale comes from**: *documented* (cited to a project
doc), *reconstructable* (derivable from artifacts/code comments), or *unrecorded* (a fixed
recipe, never ablated). **If asked about an unrecorded one, say "fixed low-resource recipe,
not ablated" — do not invent a justification.**

| Decision | Alternative not taken | Why / cost | Provenance |
|---|---|---|---|
| `threshold_src/tgt: 1`, no BPE | subword/BPE, vocabulary pruning | Chinese targets are already 1 char/token, so subwording buys nothing on the target; keeping full vocab preserves rare characters, which is the point in a historical corpus. **Cost:** 34 of 8,908 val target characters (~0.4 %) are unproducible `<unk>` by construction | documented — `docs/EXPERIMENTS.md:8` |
| Keep 1,596 duplicate pairs, 44 ambiguous sources, 6 `�` rows | clean them out | Raw files are immutable inputs; cleaning would make the splits non-comparable with anything else built on this corpus, and guessing the lost characters would be fabrication. **Cost:** the 44 ambiguous sources cap achievable accuracy; the duplicates over-weight 1,596 pairs | documented — `docs/DATASET.md` ("exact duplicates remain … for comparability") |
| Score everything character-level, both systems | word-level BLEU, or per-system tokenizers | The **only** way a character-tokenized Fairseq model and an LLM emitting natural Chinese are comparable. **Cost:** these numbers are not on the same scale as other groups' word-level EN/VI scores — §7 | documented — `docs/GROUP10_MT_PLAN.md` step 4 |
| Append hints into the source with a `__knowledge__` sentinel | prompt-level hints (E4's `render_hint`), constrained decoding | Keeps E3 a *preprocessing-only* change so the architecture stays E1's. **Cost:** it isn't preprocessing-only in effect — the source dictionary grows, so the model does too (+10.3 % params) | documented — `docs/EXPERIMENTS.md`; E4 built, never trained |
| `max_candidates: 64` | a smaller, more selective budget | — | **unrecorded**, and measurably unselective: mean 61.5–62.4 of 64 used, 100 % of rows augmented. Hints are ~3× longer than the sentence they annotate |
| Rank by match count → **train** frequency → Unicode | frequency over all splits; a learned scorer | Deterministic and leak-free by construction — the frequency table is built from train only | reconstructable — `knowledge.py:73-78` |
| Packing **on** for E2, **off** for knowledge backends | pack everything | Packing concatenates examples with no attention reset, so a sequence could see the *previous* example's hint block. Harmless at E2's ~107-token mean, corrupting once a hint block exists | reconstructable — `llm_runner.py:225-229` (explicit comment) |
| E2 selected on `eval_loss`; E1/E3 on BLEU | BLEU for all three | **Cost:** not a like-for-like selection criterion across systems, and only **2** eval points existed | **unrecorded** — `llm_runner.py:253-255` is the code doing it, not a rationale. It is `Trainer`'s default metric; no doc says why BLEU wasn't wired up |
| Three conda environments | one environment | fairseq 0.12.2/torch 1.13.1 and torch 2.5.1 cannot coexist; and isolating scoring in the smallest env means metrics are recomputable with no GPU | documented — `Makefile:15-20` |
| Freeze gate before test decoding | trust the process | Makes "we didn't tune on test" **mechanically enforced** rather than asserted. **Cost:** when a config legitimately drifts, the gate blocks you — which is exactly what happened to E2 (§7) | documented — `docs/GROUP10_MT_PLAN.md:28`; mechanism at `freeze.py:66-82` |
| `--seed 42` passed explicitly to `compare` | rely on the CLI default | The CLI default is **12345**; omitting the flag silently changes every confidence interval | documented — `Makefile:53-54` comment |

---

## §6 — Evaluation and the verification machinery

### 6.1 How a number is produced

`evaluate_predictions` (`evaluation.py:41-69`) revalidates the rows, **recomputes
`score_form_chinese(prediction)` for all 510 rows and refuses to score if the stored
`prediction_scored` differs**, then scores with SacreBLEU 2.6.0:

```
BLEU   nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF++ nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

Both signatures are stored in every metrics file. Quote them — they are what make the
numbers checkable by someone else.

### 6.2 Results (re-read from `metrics/*.json`, 2026-08-22)

| System | val BLEU | val chrF++ | test BLEU | test chrF++ |
|---|---:|---:|---:|---:|
| E1 Fairseq | 29.37 | 31.12 | 28.63 | 30.58 |
| E2 Qwen3-8B QLoRA | 41.06 | 41.20 | **40.77** | **40.87** |
| E3-custom Fairseq + retrieval | 31.04 | 32.72 | 29.62 | 31.41 |

Test n-gram precisions and brevity penalty — these say more than the single score:

| System | 1-gram | 2-gram | 3-gram | 4-gram | BP | hyp/ref len |
|---|---:|---:|---:|---:|---:|---|
| E1 | 62.2 | 35.5 | 23.1 | 16.6 | 0.943 | 8,486 / 8,982 |
| E2 | 69.9 | 46.6 | 33.6 | 25.3 | **1.000** | 8,979 / 8,982 |
| E3-custom | 62.0 | 35.8 | 23.4 | 16.9 | 0.968 | 8,697 / 8,982 |

**Two readings worth saying out loud.**

1. **E2's advantage grows with n-gram order in relative terms** — ×1.12 unigram, ×1.31
   bigram, ×1.46 trigram, ×1.52 4-gram. It is producing better *sequences*, not just
   better characters. (Absolute deltas peak at bigram, +11.1; the monotone story is the
   relative one, so phrase it that way.)
2. **Both Fairseq systems under-generate** (BP 0.943 / 0.968) while E2 matches reference
   length almost exactly (BP 1.000, 8,979 vs 8,982). E3's unigram precision is actually a
   hair *below* E1's — **its gain comes from length calibration and higher-order matches,
   not from getting more characters right.**

### 6.3 Statistical comparison

`compare_predictions` verifies both files carry the same ordered `sample_id`s and identical
references, then runs a **paired bootstrap**: 1,000 resamples with replacement, both corpus
metrics recomputed on each. Reports observed delta, 2.5/97.5 percentiles, and a two-sided
p-value from the `(count+1)/(n+1)`-smoothed tail. **Test split, seed 42, n = 1,000:**

| Comparison | ΔBLEU | 95 % interval | p | ΔchrF++ | 95 % interval | p |
|---|---:|---|---:|---:|---|---:|
| E2 − E1 | **+12.14** | [10.22, 13.77] | 0.002 | +10.29 | [9.03, 11.79] | 0.002 |
| E3-custom − E1 | +0.99 | **[−0.14, 2.16]** | 0.092 | +0.83 | [−0.06, 1.71] | 0.064 |
| E2 − E3-custom | **+11.15** | [9.26, 12.77] | 0.002 | +9.46 | [8.05, 10.88] | 0.002 |

### 6.4 The gates, in one list

| Gate | Refuses to |
|---|---|
| `expected_lines` (`data.py:52-54`) | run on a swapped or truncated corpus file |
| `ensure_clean_checkpoint_dir` | train when `checkpoint_last.pt` exists — Fairseq would silently resume |
| `ensure_clean_adapter_dir` | train over an existing adapter, destroying what the freeze points at |
| `ensure_selection_frozen` | decode test before selection is frozen, or after config/checkpoint/decoding changed |
| hypothesis-ID assertion | accept a Fairseq decode whose IDs aren't exactly `0..509` |
| `validate_prediction_rows` | score a file that is short, misordered, or has duplicate IDs |
| `prediction_scored` recompute | score a file whose scoring form was edited |
| thinking / empty-output checks | emit a blank or reasoning-contaminated prediction |
| retrieval tripwire | accept a "knowledge run" whose predictions are mostly hint-free |
| `_refuse_to_discard_annotations` | regenerate the annotation sheet over existing manual labels |

**`make repro-check`** re-runs the freeze checks, recomputes **every** metrics file from its
predictions into a scratch dir and asserts full dict equality *including signatures*, and
re-fingerprints the corpus against every manifest. **`make repro-smoke`** is the only
*determinism* gate: trains the smoke config twice into isolated trees with `--gpu` forced,
and asserts identical weights and identical translations — Fairseq path, smoke scale only.
**`make test`**: 34 tests / 9 files, including a pinned regression test that re-derives E3's
augmented sources and asserts they still hash to the recorded values.

### 6.5 Error analysis — prepared, not done

`make error-analysis` samples **100 test `sample_id`s** with `random.Random(42).sample` and
emits one row per (sample, system) = **300 rows**, all three systems on the *same* 100
sentences, which is what makes per-system counts comparable. Taxonomy: `CORRECT`,
`MISTRANSLATION`, `OMISSION`, `UNSUPPORTED_ADDITION`, `ENTITY_ERROR`,
`NUMBER_OR_TIME_ERROR`, `OTHER`. `summarize_annotations` is strict by design: it fails
unless *every* row is `ADJUDICATED` and all three experiments share the same 100 IDs.

**Current state: all 300 rows are `PENDING`.** If your slot includes error analysis, say
the sheet is generated and the labels are outstanding human work — do not present numbers
that don't exist.

---

## §7 — Q&A prep

Each item: the question → what to say → the evidence behind it. All of these are
**disclosures**, not action items.

### "BLEU 29–41 seems high for a low-resource historical language pair."

> "These are **character-level** BLEU and chrF++, not word-level. Our inputs are pre-spaced
> one codepoint per token, so `tok:13a` has nothing left to segment, and chrF++'s `nw:2`
> word bigrams are in fact character bigrams. Character-level BLEU on Chinese runs
> systematically higher than word-level BLEU on Latin-script targets, so these numbers are
> **not on the same scale** as word-level EN/VI scores from other groups."

Evidence: `ref_len = 8,982` on test equals the corpus test target token count exactly. Also
note the spec asks for a unified Moses tokenizer. New evaluation explicitly applies
Moses before SacreBLEU with `tokenize=none`; 13a is a different historical protocol.
The retained Chinese character segmentation must also be aligned across groups.

### "So retrieval works? +0.99 BLEU."

> "No — we report it as **inconclusive**. The 95 % interval is [−0.14, 2.16], p = 0.092, so
> it straddles zero on BLEU and on chrF++. And it isn't a controlled comparison anyway."

**Four independent confounds** — have these ready:

1. **+10.3 % parameters** — 67.65 M → 74.64 M, because the retrieved characters enlarge the
   source dictionary and only the *decoder* embeddings are shared.
2. **Sources 3.9× longer** — 421 k → 1.62 M train source tokens.
3. **Realized batch size 33.3 vs 132.5 sentences** — same 1,000-token budget, ~4× fewer
   sentences per update. A real difference in optimization dynamics.
4. **Truncated schedule** — E1 early-stopped on `patience 20` at epoch 171 (a converged
   run); E3 hit the `max_update: 50000` cap at epoch 87, **10 epochs after its best**, with
   patience never given the chance to fire.

Optional color on (4), verified from both `train.log` tails: E3 was still moving when the
cap hit — it finished at lr **2.00e-4** against E1's **2.84e-4** on the shared `inverse_sqrt`
schedule. Present that as a *consequence* of the different update counts, not a fifth axis;
under a shared schedule the final LR follows arithmetically from the step count.

**Frame it as:** "+0.99 is a lower bound from a bigger, truncated model — not a clean
measurement of retrieval's effect."

> **Reconcile this with your own slides.** `docs/EXPERIMENTS.md:33` says the custom E3 "uses
> E1's seed, architecture, effective batch, **training budget**, and deterministic beam-7
> decoding." That describes the **configured** budget, which was genuinely identical — same
> `max_update`, same `patience`, same `max_tokens` × `update_freq`. The **realized** budget was
> not: E1 stopped early on patience at 24,791 updates, E3 ran into the 50,000 cap, and the same
> token budget held 4× fewer sentences. Say "identical configuration, different realized
> budget" and both documents are true at once.

### "Why did retrieval help so little, given 91.85 % coverage?"

Two measured reasons, both good answers: retrieval is **barely selective** (100 % of rows
augmented, mean 61.5–62.4 of a 64 cap — the hint block is ~3× longer than the sentence), and
the coverage gap converts directly into noise: **49 of val's `<unk>` tokens and 44 of test's
are in the hint span**, i.e. retrieved characters that never appeared in a training source.
The source-span `<unk>` counts are identical to E1's, which proves the extra unknowns are
all hints.

### "Only two validation checkpoints for the LLM?"

> "Yes — `eval_steps: 200` against a 546-step run gives exactly two evaluation points, at
> steps 200 and 400. That's a thin selection signal and we disclose it as such. The saved
> adapter is the step-400 / epoch-2.2 model, not the end-of-epoch-3 model."

### "Can you reproduce E1's 28.63 with one command?"

> "No, and we say so in Limitations. `train.log:268` shows that checkpoint was
> **warm-started** from an interrupted first attempt at epoch 5 / 579 updates. The result is
> real — the run happened, both logs survive, and the checkpoint is hashed into the manifest
> and the frozen record — but a single clean `make e1-train` will not reproduce it. The
> guard that now prevents silent resumption was added after E1 ran."

Do **not** offer to rerun it. E3 is clean by contrast — no `Loaded checkpoint` line anywhere
in its log.

### "Your `repro-check` says `ok: true` — does everything verify?"

> "Read the report, not the exit code. **E2's frozen-selection chain does not verify**, and
> `repro-check` asserts that as a *known expected failure*. `ok: true` means 'the known break
> is still exactly the known break', not 'everything verifies'."

The detail: E2's frozen record holds config hash `ca68e42c…`; the file hashes to `d4057d53…`
today. The config was edited after training, frozen and used for test generation in that
intermediate state, then edited back. What that means precisely:

- **Recoverable:** the training config (archived field-for-field in `run_manifest.json`,
  identical to today's file); the decoding config (identical across `selection_frozen.json`,
  today's config, and all 510 test prediction rows); the adapter hash.
- **Not recoverable:** `prompt` and `model.*` **as they stood at test-generation time.**
  Those are the fields to disclose as unverifiable — not the configuration as a whole.

And if someone suggests fixing it: re-running `make e2-freeze` would hash today's config and
stamp `test_generation_authorized: true` with a timestamp *after* the test predictions,
converting broken provenance into apparent validity. Disclosure is the only honest remedy.

### "Where is E2's training log?"

A later `make e2-train` attempt truncated it; `work/e2_.../train.log` now holds only the
overwrite guard's traceback. **The quantitative record survives** in `training_history.json`
(57 records), `trainer_state.json`, and `run_manifest.train_metrics` — that is where the
20,328 s, the 546 steps, and both eval losses come from. The mechanism that caused it (an
external `tee` truncating the file before the in-process guard could fire) was fixed on
2026-08-20; training now tees from inside the process, *after* the guards pass.

### "Why is there no official E3?"

The lab's `knowledge_v2` integration code was never supplied. `cli.py:33-35` raises the
config's `blocked_reason` rather than substituting anything, so nothing can accidentally be
reported under that ID. Our custom retrieval experiment carries a **distinct** experiment ID
and must not be presented as the official method.

### "Why 510 test sentences? Isn't that small?"

It's the supplied gold split — manually aligned, and the same 510 for every system. Small
test sets are exactly why we report **paired bootstrap intervals** rather than bare deltas:
the pairing removes sentence-difficulty variance, and the interval tells you which
differences survive it. That's what lets us say +12.14 is real and +0.99 isn't.

### "Why three environments? Isn't that over-engineering?"

They are genuinely incompatible — fairseq 0.12.2 needs torch 1.13.1, the QLoRA stack needs
torch 2.5.1. The deliberate part is the *third* one: scoring lives in an env with
**neither** torch nor fairseq, so any reported metric can be recomputed by anyone with
sacrebleu and the prediction files. No GPU, no training stack.

### "Why the freeze gate — wouldn't you just not look at test?"

Because "we didn't tune on test" should be **mechanically enforced, not asserted.**
`freeze-selection` hashes the config, the checkpoint, the decoding block, and the validation
artifacts; `ensure_selection_frozen` re-checks all of it before every test decode. Any
post-hoc edit to the model, config, or beam settings makes test generation **fail** rather
than quietly produce a number. E2 is the proof that the gate actually bites.

---

## §8 — Numbers to have cold vs numbers to look up

### Memorize these ten

| | |
|---|---|
| Splits | **19,218 / 510 / 510** (silver train, gold val, gold test) |
| Test BLEU | E1 **28.63** · E2 **40.77** · E3-custom **29.62** |
| Key deltas | E2−E1 **+12.14** (p = 0.002) · E3−E1 **+0.99** (p = **0.092**, n.s.) |
| Model sizes | E1 **67.7 M** · E3 **74.6 M** · E2 base **8 B** + LoRA r16 |
| Knowledge coverage | **91.85 %** (5,013 / 5,458 target characters) |
| E2 packing | 19,218 examples → **2,903** windows of 768 tokens |
| Leakage | **0** shared pairs, sources, or targets across all three split pairs |
| Train duplicates | **1,596** duplicate pairs, **44** sources with multiple targets |
| Wall clock | E1 **52 min** · E3 **79 min** · E2 **5 h 39 m**, all on one RTX 3060 12 GB |
| Metrics | SacreBLEU **2.6.0**, `tok:13a`, chrF2++ `nc:6 nw:2` — **character-level** |

### Look these up rather than memorizing

| You need | It lives in |
|---|---|
| Any exact score, signature, or n-gram precision | `metrics/<id>.{val,test}.json` |
| Bootstrap intervals and p-values | `metrics/e1_vs_e2.test.json`, `e1_vs_e3_custom.test.json`, `e3_custom_vs_e2.test.json` |
| Corpus hashes, token/type counts, the six `�` sample IDs | `metrics/data_audit.json` |
| Full env, package versions, hardware, config + its hash | `work/<id>/run_manifest.json` |
| E2's step-by-step losses | `work/e2_.../training_history.json`, `trainer/trainer_state.json` |
| Retrieval statistics per split | `work/e3_.../knowledge_augmentation.json` |
| Any `file:line` citation for a mechanism | `docs/PIPELINE.md` |
| The full integrity story, with severities | `docs/REPRO_AUDIT.md` |
