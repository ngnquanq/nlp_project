# Local E1 translation UI

This application exposes the completed E1 Fairseq model through a local-only
Vietnamese → Classical Chinese translation desk. React/Vite owns the browser UI;
FastAPI owns the model process because E1 and its dictionaries are Python/Fairseq
artifacts.

## Private artifacts

The repository intentionally contains neither `checkpoint_best.pt` nor the
generated `data-bin`. Supply both through environment variables. Do not copy,
symlink, or commit them under this repository.

```bash
cp .env.example .env
export MT_CHECKPOINT_PATH=/absolute/path/to/checkpoint_best.pt
export MT_DATA_BIN_PATH=/absolute/path/to/data-bin
export MT_DEVICE=cpu
```

The data-bin must contain `dict.vi.txt` and `dict.zh.txt`. At startup the service
also checks that the loaded network has the expected E1 parameter count. If an
artifact is missing or incompatible, the UI still opens and shows the corrective
setup message; translation requests return `MODEL_UNAVAILABLE`.

## Install

Node.js 20.19+ is required by Vite 8. The project was prepared with Node.js 26.

```bash
conda env create -p .conda/e1-ui -f environments/e1-ui-macos.yml
npm --prefix ui install
```

## Development

Run the API and Vite in separate terminals:

```bash
make ui-api
make ui-web
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to the local FastAPI process.
The API loads E1 once and processes one generation at a time.

## Local release build

```bash
make ui-serve
```

Open <http://127.0.0.1:8000>. FastAPI serves the compiled client and API from the
same origin. The server is deliberately bound to `127.0.0.1`; this branch does
not implement public deployment, accounts, request history, or analytics.

## Input contract

- Input is normalized to Unicode NFC and repeated whitespace is collapsed.
- Case and punctuation are never silently removed.
- The 256-position model limit includes Fairseq's EOS token.
- Tokens absent from the train dictionary map to `<unk>` and are reported in the UI.
- E1 was trained on tokenized, punctuation-light historical Vietnamese. Modern or
  out-of-domain prose may produce unreliable output.

## Verification

```bash
make ui-check
make ui-private-model  # only when both private artifact variables are set
```

The source-only suite uses an injected fake translator and does not read private
data. The private test loads the real checkpoint and checks that repeated decoding
of the same sentence is deterministic and non-empty.
