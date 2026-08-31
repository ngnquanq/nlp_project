"""Run Fairseq generation with its incompatible PyTorch fast path disabled."""

from fairseq_cli.generate import cli_main

from mt_pipeline.serving.translator import _disable_fairseq_encoder_fastpath

_disable_fairseq_encoder_fastpath()


if __name__ == "__main__":
    cli_main()
