from ..builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class VMUNetHead(BaseDecodeHead):
    """Decode head that returns backbone logits directly."""

    def __init__(self, **kwargs):
        kwargs.setdefault('in_index', 0)
        kwargs.setdefault('input_transform', None)
        super().__init__(**kwargs)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # logits tensor
        return x
