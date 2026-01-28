# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.ops import Upsample
from ..builder import BACKBONES
from ..utils import UpConvBlock

from .modules.deformable_LKA import deformable_LKA_Attention
from .modules.scse import scSE
from .modules.dcsg import DCSG


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


@UPSAMPLE_LAYERS.register_module()
class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 kernel_size=4,
                 scale_factor=2):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


@UPSAMPLE_LAYERS.register_module()
class InterpConv(nn.Module):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsample_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(
                scale_factor=2, mode='bilinear', align_corners=False).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 conv_cfg=None,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()

        self.with_cp = with_cp
        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        upsample = Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out


@BACKBONES.register_module()
class UNet(BaseModule):
    """UNet backbone.

    This backbone is the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondence encoder stage.
            Default: (1, 1, 1, 1, 1).
        enc_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence encoder stage.
            Default: (2, 2, 2, 2, 2).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondence encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_divisible.
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None,
                 frozen_stages=-1, # <--- 【新增 1】添加这个参数，默认 -1 (不冻结)
                 use_dlka=False,
                 dlka_position='bottleneck',
                 # -------------------- 新增：scSE 开关与参数 --------------------
                 use_scse=False,
                 scse_reduction=16,
                 scse_min_channels=4,
                 # -------------------- 新增：DCSG 开关与参数 --------------------
                use_dcsg=False,
                dcsg_reduction=16,
                dcsg_min_channels=4,
                dcsg_inter_channels=None,
                ):
        super(UNet, self).__init__(init_cfg)

        self.pretrained = pretrained
        self.frozen_stages = frozen_stages # <--- 【新增 2】保存参数
        
        self.use_dlka = use_dlka
        self.dlka_position = dlka_position
        
        self.dlka_alpha = nn.Parameter(torch.zeros(1))

        # -------------------- 新增：scSE --------------------
        self.use_scse = use_scse
        self.scse_reduction = scse_reduction
        self.scse_min_channels = scse_min_channels

        # -------------------- 新增：DCSG --------------------
        self.use_dcsg = use_dcsg
        self.dcsg_reduction = dcsg_reduction
        self.dcsg_min_channels = dcsg_min_channels
        self.dcsg_inter_channels = dcsg_inter_channels

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, '\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, '\
            f'while the enc_num_convs is {enc_num_convs}, the length of '\
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(downsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {downsamples}, the length of '\
            f'downsamples is {len(downsamples)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, '\
            f'while the enc_dilations is {enc_dilations}, the length of '\
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        self.use_dlka = use_dlka
        self.dlka_position = dlka_position

        # -------------------- bottleneck：DLKA --------------------
        bottleneck_ch = base_channels * 2**(num_stages - 1)
        if self.use_dlka and self.dlka_position == 'bottleneck':
            self.dlka = deformable_LKA_Attention(d_model=bottleneck_ch)
        else:
            self.dlka = nn.Identity()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # -------------------- 新增：每个 encoder stage 一个 scSE --------------------
        if self.use_scse:
            self.scse_enc = nn.ModuleList([
                scSE(base_channels * 2**i,
                     reduction=self.scse_reduction,
                     min_channels=self.scse_min_channels)
                for i in range(num_stages)
            ])
        else:
            self.scse_enc = nn.ModuleList([nn.Identity() for _ in range(num_stages)])

        # -------------------- build encoder/decoder --------------------
        cur_in = in_channels
        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2**(i - 1),
                        out_channels=base_channels * 2**(i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None))

            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=cur_in,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None))
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            cur_in = base_channels * 2**i


        # ✅ 在这里加 DCSG（循环结束后）
        # -------------------- 新增：每个 skip stage 一个 DCSG（decoder-conditioned） --------------------
        if self.use_dcsg:
            self.dcsg = nn.ModuleList([
                DCSG(
                    skip_channels=base_channels * 2**i,
                    gate_channels=base_channels * 2**(i + 1),
                    reduction=self.dcsg_reduction,
                    min_channels=self.dcsg_min_channels,
                    inter_channels=self.dcsg_inter_channels,
                )
                for i in range(num_stages - 1)
            ])
        else:
            self.dcsg = nn.ModuleList([nn.Identity() for _ in range(num_stages - 1)])


    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            # -------------------- encoder 每层后插入 scSE（你要的“放在每个 stage 输出”） --------------------
            x = self.scse_enc[i](x)
            enc_outs.append(x)
            
        # —— D-LKA 插入点：瓶颈特征增强 ——
        x = x + self.dlka_alpha * self.dlka(x)
        
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            skip = enc_outs[i]
            # -------------------- DCSG：用 decoder 特征 x 门控 skip（concat 之前） --------------------
            # i 取值: 0..num_stages-2；x 的通道数与 gate_channels 设计匹配
            if self.use_dcsg:
                skip = self.dcsg[i](skip, x)

            x = self.decoder[i](skip, x)
            dec_outs.append(x)

        return dec_outs

    # <--- 【新增 3】将此整个函数复制到 UNet 类中
    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            # 循环冻结前 N 个 Stage
            for i in range(self.frozen_stages):
                m = self.encoder[i]
                
                # 1. 切换到 eval 模式 (固定 BN 的均值和方差)
                m.eval() 
                
                # 2. 关闭梯度 (不更新权重)
                for param in m.parameters():
                    param.requires_grad = False

                # 同步冻结 scSE[i]（关键：否则 encoder 冻了，scSE 仍在改变分布）
                if i < len(self.scse_enc):
                    a = self.scse_enc[i]
                    a.eval()
                    for p in a.parameters():
                        p.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNet, self).train(mode)
        
        # <--- 【新增 4】在这里调用冻结函数
        self._freeze_stages()
        
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'
