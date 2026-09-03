# Conda environments

- `eval.yml` is the small cross-platform Python 3.10 environment for audits, metrics, and tests.
- `fairseq.yml` is the Linux/NVIDIA environment for E1, pinned to the legacy Fairseq-compatible PyTorch stack.
- `llm.yml` is the Linux/NVIDIA environment for Qwen3 4-bit QLoRA.
  Note the drift: it pins `pytorch=2.5.1`, but the `.conda/llm` environment that
  actually produced the E2 adapter holds torch 2.6.0+cu124, as recorded in
  `work/e2_qwen3_8b_qlora_vi_zh_v1/run_manifest.json`. Re-solving from this file
  would not reproduce that stack, which is why `make ui-install-e2` clones the
  existing environment into `.conda/e2-ui` instead of writing a new yml.
- `fairseq-smoke-macos.yml` exists only to exercise Fairseq preprocessing and a two-update CPU smoke run on Apple Silicon; it is not an experiment environment.

The GPU environments contain CUDA packages and are intended for private Kaggle or another Linux NVIDIA machine. They are not expected to solve on macOS. Each real training manifest records the resolved Conda/pip inventory and accelerator details.
