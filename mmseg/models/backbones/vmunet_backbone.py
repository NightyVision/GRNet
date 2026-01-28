import torch
from mmcv.runner import BaseModule
from ..builder import BACKBONES

from .vmunet.vmamba import VSSM


def _strip_prefix(state_dict, prefix='module.'):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


@BACKBONES.register_module()
class VMUNetBackbone(BaseModule):
    """VM-UNet backbone wrapper for MMSeg 0.30.0.

    It outputs FINAL segmentation logits directly and returns a tuple: (logits,).
    """

    def __init__(self,
                 input_channels=3,
                 num_classes=2,
                 patch_size=4,
                 depths=(2, 2, 9, 2),
                 depths_decoder=(2, 9, 2, 2),
                 dims=(96, 192, 384, 768),
                 dims_decoder=(768, 384, 192, 96),
                 d_state=16,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.2,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained='',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.num_classes = int(num_classes)

        # Your VSSM outputs logits [N, num_classes, H, W]
        self.net = VSSM(
            patch_size=patch_size,
            in_chans=input_channels,
            num_classes=self.num_classes,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            dims=list(dims),
            dims_decoder=list(dims_decoder),
            d_state=d_state,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )

    def init_weights(self):
        super().init_weights()

        if not isinstance(self.pretrained, str) or self.pretrained.strip() == '':
            return

        ckpt = torch.load(self.pretrained, map_location='cpu')
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        state = _strip_prefix(state)

        model_dict = self.net.state_dict()

        # 1) encoder direct load (shape-compatible)
        enc = {
            k: v for k, v in state.items()
            if k in model_dict and hasattr(v, 'shape') and v.shape == model_dict[k].shape
        }

        # 2) symmetric init for decoder: layers.i -> layers_up.(3-i)
        dec_map = {}
        for k, v in state.items():
            for i in range(4):  # your code uses 4 stages: layers.0..3
                src = f'layers.{i}'
                dst = f'layers_up.{3 - i}'
                if src in k:
                    dec_map[k.replace(src, dst)] = v

        dec = {
            k: v for k, v in dec_map.items()
            if k in model_dict and hasattr(v, 'shape') and v.shape == model_dict[k].shape
        }

        merged = dict(model_dict)
        merged.update(enc)
        merged.update(dec)

        missing, unexpected = self.net.load_state_dict(merged, strict=False)
        print(f'[VMUNetBackbone] load_from: {self.pretrained}')
        print(f'  encoder matched: {len(enc)} keys')
        print(f'  decoder matched (symmetric): {len(dec)} keys')
        print(f'  missing: {len(missing)}, unexpected: {len(unexpected)}')

    def forward(self, x):
        # handle grayscale input
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        logits = self.net(x)  # [N, num_classes, H, W]
        return (logits,)
