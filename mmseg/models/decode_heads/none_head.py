from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from ..builder import HEADS


@HEADS.register_module()
class NoneHead(BaseDecodeHead):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs):
        return inputs[-1]