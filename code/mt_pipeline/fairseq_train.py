"""Run Fairseq training with deterministic kernels selected where PyTorch offers them.

The ``fairseq-train`` console script gives no hook for setting the process-wide
determinism switches, so training is launched through this module instead. Mirrors
the delegation pattern in ``fairseq_generate``.
"""

import torch
from fairseq_cli.train import cli_main


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# warn_only keeps training runnable when an op has no deterministic kernel; the
# warning names the op, which is what repro-check divergence triage needs.
torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    cli_main()
