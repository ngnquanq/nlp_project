"""Run Fairseq generation with its incompatible PyTorch fast path disabled."""

from fairseq.modules.transformer_layer import TransformerEncoderLayerBase
from fairseq_cli.generate import cli_main


_original_init = TransformerEncoderLayerBase.__init__


def _compatible_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.can_use_fastpath = False


TransformerEncoderLayerBase.__init__ = _compatible_init


if __name__ == "__main__":
    cli_main()
