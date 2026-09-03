# Local E1/E2 translation UI

This application exposes the completed E1 (Fairseq) and E2 (Qwen3-8B QLoRA)
models through a local-only Vietnamese → Classical Chinese translation desk.
React/Vite owns the browser UI; FastAPI owns the model processes.

## Architecture

E1 and E2 **cannot share a Python process**. fairseq 0.12.2 requires numpy<2
and runs on torch 1.13, while transformers 4.57 / peft / bitsandbytes require
torch ≥ 2.2 and numpy 2.x. The Makefile states the same constraint for the
training stages (see its header comment: *"the three environments are not
interchangeable"*). E2 additionally requires CUDA, whereas E1 runs on CPU, so
the two cannot even share one device setting.

The browser therefore talks to a single origin, and the E1 process forwards E2
requests to a sidecar over localhost HTTP:

```
Browser :5173 (dev) or :8000 (ui-serve)
   │
   └─ /api/*  ──►  FastAPI :8000   (.conda/e1-ui)
                   ├─ e1 → E1Translator      (fairseq, in-process, eager)
                   └─ e2 → RemoteTranslator ──httpx──► FastAPI :8001 (.conda/e2-ui)
                                                        └─ LazyTranslator(E2Translator)
                                                             Qwen3-8B 4-bit + LoRA, cuda
```

Leaving `MT_E2_URL` unset keeps the original E1-only behaviour: the gateway
lists one model and never opens a socket.

## Private artifacts

The repository contains neither `checkpoint_best.pt`, the generated `data-bin`,
nor the base Qwen weights. Supply them through environment variables; do not
copy, symlink, or commit them under this repository.

```bash
cp .env.example .env
export MT_CHECKPOINT_PATH=/absolute/path/to/checkpoint_best.pt
export MT_DATA_BIN_PATH=/absolute/path/to/data-bin
export MT_DEVICE=cpu
export MT_E2_URL=http://127.0.0.1:8001      # omit to run E1 only
```

The Makefile already defaults these to the in-repo artifact paths, so the UI
targets run with nothing exported; set them only to point somewhere else. When
you launch the servers by hand instead of through `make`, they must genuinely be
exported: nothing reads the `.env` file automatically — `settings.py` reads
`os.environ` directly.

E1 needs a data-bin containing `dict.vi.txt` and `dict.zh.txt`, and the loaded
network must match the expected E1 parameter count. E2 needs:

- the LoRA adapter at `checkpoint/e2_qwen3_8b_qlora_vi_zh_v1/adapter/` (~182 MB),
- `work/e2_qwen3_8b_qlora_vi_zh_v1/run_manifest.json`,
- the base `Qwen/Qwen3-8B` weights (~16 GB) in the Hugging Face cache,
- a CUDA GPU with roughly 7 GB free (4-bit NF4 + double quantization).

**The base-model revision comes from the run manifest, not from the config.**
`predict_qlora` reads `resolved_model_revision` from `run_manifest.json`, and
the serving path does the same. It deliberately never falls back to
`config["model"]["revision"]`: diverging silently would pair the adapter with a
different base model.

If an artifact is missing, the UI still opens, marks that model unavailable with
the corrective message, and keeps the other model usable.

## Install

Node.js 20.19+ is required by Vite 8.

```bash
conda env create -p .conda/e1-ui -f environments/e1-ui-macos.yml
npm --prefix ui install

make ui-install-e2        # only if you want E2
```

`ui-install-e2` clones `.conda/llm` rather than solving a fresh environment,
because `environments/llm.yml` pins `pytorch=2.5.1` while `.conda/llm` actually
holds torch 2.6.0+cu124 — which is what `run_manifest.json` records as the stack
that produced the adapter. A fresh solve would not reproduce it. `conda create
--clone` hardlinks from the shared package cache, so the real disk cost is far
below the environment's nominal size.

## Running it

```bash
make ui-up         # gateway + sidecar + Vite, backgrounded, waits for readiness
```

Then open <http://127.0.0.1:5173>. The Makefile supplies the `MT_*` defaults, so
nothing has to be exported by hand; any value already in the environment still
wins. Companion targets:

```bash
make ui-status     # per-service pid plus the gateway's model health
make ui-logs       # last 40 lines of each service log (UI_LOG_LINES to change)
make ui-restart    # after editing server code
make ui-down       # stop all three
make ui-up WITH_E2=0   # E1 and the browser UI only
```

Services are started with `nohup` and tracked by pidfiles in `.run/` (gitignored),
one process each — the interpreter and the Vite binary are invoked directly
rather than through `conda run` or `npm run`, both of which fork a child whose
PID would not be the one `ui-down` needs to kill. Re-running `ui-up` is safe: it
reports what is already running instead of starting a second copy. Because the
services are detached, they outlive the shell that started them; `make ui-down`
is how you stop them.

If the `.conda/e2-ui` environment is missing, `ui-up` starts E1 and the UI and
says so rather than failing — E2 then shows as unavailable in the selector.

### Running the services by hand

```bash
make ui-api        # E1 gateway on :8000, foreground
make ui-api-e2     # E2 sidecar on :8001, foreground
make ui-web        # Vite on :5173, foreground
make ui-api MT_E2_URL=      # gateway with E2 disabled entirely
```

Vite proxies `/api` to the gateway. Each model processes one generation at a
time, under its own lock — a slow E2 request never blocks E1.

## Local release build

```bash
make ui-serve      # plus `make ui-api-e2` in another terminal
```

Open <http://127.0.0.1:8000>. FastAPI serves the compiled client and the API from
the same origin, and E2 still works because the gateway forwards it rather than
relying on Vite's proxy. Both processes bind `127.0.0.1`; there is no
authentication between them and this branch implements no public deployment,
accounts, request history, or analytics.

## First request latency

The E2 sidecar starts in about two seconds and reports `not_loaded`. The
Qwen3-8B weights load on the **first** E2 translation, not at startup, so that
VRAM stays free while you use E1. Measured on an RTX 3060 with a warm page
cache: ~16 s for the first request end to end, ~1.6 s afterwards. A genuinely
cold read of the 16 GB base model is slower. `MT_E2_TIMEOUT_SECONDS` (default
900) bounds it; the connect timeout stays at 5 s so a sidecar that is not
running fails fast instead of hanging.

The UI shows an elapsed-seconds notice once a request passes ten seconds.

## Input contract

Shared: input is normalized to Unicode NFC, repeated whitespace is collapsed,
case and punctuation are never silently removed, and the cap is 4,000 characters.

Both models were trained on tokenized, punctuation-light historical Vietnamese
(`configs/dataset.yaml` points at the `*.no_punct.*` files). Modern or
out-of-domain prose may produce unreliable output from either.

**E1** — 256 encoded positions including Fairseq's EOS token. Tokens absent from
the train dictionary map to `<unk>` and are reported in the UI. The browser
counts positions exactly.

**E2** — 512 tokens in the chat-templated prompt, enforced server-side by the
sidecar's own tokenizer. This is *not* the config's `max_sequence_length: 768`:
that is the training window covering prompt **and** target, and `predict_qlora`
never truncates at inference, so the prompt budget leaves at least 256 positions
for generation. The browser cannot run Qwen BPE, so its counter is an estimate
using 3.39 characters per token — measured over the 510 validation sources
(45,346 characters / 13,380 tokens). It is shown with a `~` and blocks only well
past the limit; the server issues the authoritative rejection.

E2 has no out-of-vocabulary concept, so it never reports unknown tokens. Its
prompt, decoding parameters (beam 4, greedy, `max_new_tokens = min(512,
ceil(2·n)+32)`) and post-processing are identical to `predict_qlora`, including
`enable_thinking=False` — which is load-bearing, because the adapter's chat
template injects an empty `<think></think>` block when it is false and training
used the same flag. Consequently UI output reproduces the offline predictions
exactly, modulo the whitespace normalization the UI applies and the offline path
does not.

## API contract

`GET /api/health`

```json
{
  "status": "ready | degraded | model_unavailable",
  "default_model": "e1",
  "models": [
    {
      "key": "e1", "model_id": "…", "label": "E1", "sublabel": "…",
      "status": "ready | not_loaded | unavailable",
      "device": "cpu", "parameter_count": 67652608, "message": null,
      "limits": {
        "max_characters": 4000, "max_units": 256,
        "unit": "position", "unit_label": "vị trí",
        "client_estimate": false, "chars_per_unit": null
      },
      "reports_unknown_tokens": true, "slow_first_request": false
    }
  ]
}
```

`limits.max_units` is always a number, including when the sidecar is
unreachable, because the browser renders it directly.

`POST /api/translate` takes `{"text": "…", "model": "e1"}`; `model` accepts a
short key or a full model id and defaults to `e1`.

| code | status | meaning |
|---|---|---|
| `EMPTY_INPUT` | 422 | blank after normalization |
| `INPUT_TOO_LONG` | 422 | over the character or unit cap |
| `UNKNOWN_MODEL` | 422 | model not served by this process |
| `MODEL_UNAVAILABLE` | 503 | that model failed to load |
| `E2_SIDECAR_UNREACHABLE` | 503 | sidecar not running |
| `E2_TIMEOUT` | 504 | sidecar did not answer in time |
| `E2_UPSTREAM_ERROR` | 502 | sidecar returned an unexpected failure |
| `THINKING_EMITTED` | 500 | E2 returned reasoning instead of a translation |
| `EMPTY_GENERATION` | 500 | E2 generated nothing usable |
| `GPU_OUT_OF_MEMORY` | 503 | CUDA ran out of memory while decoding |
| `TRANSLATION_FAILED` | 500 | anything else |

## Verification

```bash
make ui-check          # source-only backend suite + tsc + vitest + vite build
make ui-check-e2       # sidecar shape checks, under .conda/e2-ui
make ui-private-model  # loads the real E1 checkpoint
make ui-e2-model       # real E2 model; needs a CUDA GPU and the adapter
```

The source-only suites use injected fake translators and `httpx.MockTransport`;
they read no private data and open no sockets.

Stop the sidecar before `make ui-e2-model`: it loads its own copy of the model,
and two copies do not fit in 12 GB of VRAM — the second fails with *"Some modules
are dispatched on the CPU or the disk"* from the bitsandbytes quantizer.

`make ui-e2-model` is the strongest gate: it loads the real model and asserts that rows from
`predictions/e2_qwen3_8b_qlora_vi_zh_v1.val.jsonl` come back character for
character, proving the serving path has not drifted from the offline pipeline.
