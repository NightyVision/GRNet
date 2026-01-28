import torch
import torch.nn as nn
import torch.nn.functional as F


class DCSG(nn.Module):
    """
    Decoder-Conditioned Skip Gating (DCSG)
    Residual soft gating:
        skip_out = skip * (1 + beta * sigmoid(g))
    beta is learnable, initialized to 0 -> identity at start.
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        reduction: int = 16,
        min_channels: int = 4,
        inter_channels: int | None = None,
        gate_out_channels: int = 1,      # 1: spatial gate, C: channel-wise gate
        init_gate_bias: float = -2.0,    # optional, sigmoid(-2)=0.12
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(skip_channels // reduction, min_channels)

        # project skip & gate to inter_channels
        self.proj_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        # fuse -> gate logits
        self.fuse = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.gate_conv = nn.Conv2d(inter_channels, gate_out_channels, kernel_size=1, bias=True)

        # learnable residual strength, init 0 => identity
        self.beta = nn.Parameter(torch.zeros(1))

        # (optional) stable init: make initial sigmoid small
        nn.init.constant_(self.gate_conv.bias, init_gate_bias)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        skip: [B, C_s, H, W]
        gate: [B, C_g, h, w] (often lower-res)
        """
        # align spatial size
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        s = self.proj_skip(skip)
        g = self.proj_gate(gate)

        x = self.fuse(s + g)
        gate_logits = self.gate_conv(x)                 # [B,1,H,W] or [B,C,H,W]
        gate_prob = torch.sigmoid(gate_logits)

        # ✅ 改法 A：残差式软门控（关键就在这两行）
        return skip * (1.0 + self.beta * gate_prob)
