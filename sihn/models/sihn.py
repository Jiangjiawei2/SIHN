"""Semantic-Injected Hierarchical Network."""

from __future__ import annotations

import torch
from torch import nn

from sihn.models.blocks import (
    ContextualExtractionUnit,
    ConvBlock,
    DualScaleAttentionModule,
    InterscaleAggregationUnit,
    ProximalBlock,
    SemanticGraphReasoningModule,
    UpConvBlock,
)
from sihn.models.operators import DataConsistencyStep, adjoint_operator
from sihn.utils.complex import channels_to_complex, complex_to_channels


class SIHN(nn.Module):
    """Model-driven MRI reconstruction with semantic hierarchical injection."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        stages: int = 5,
        semantic_nodes: int = 64,
        eta: float = 0.1,
        rho: float = 0.5,
    ):
        super().__init__()
        if stages != 5:
            raise ValueError("The manuscript architecture uses K=5 stages.")
        c1, c2, c3, c4, c5 = [base_channels * m for m in (1, 2, 4, 8, 16)]
        self.stages = stages
        self.dc_steps = nn.ModuleList([DataConsistencyStep(eta=eta, rho=rho) for _ in range(stages)])
        self.z_prox = nn.ModuleList([ProximalBlock(in_channels, base_channels) for _ in range(stages)])
        self.b_prox = nn.ModuleList([ProximalBlock(in_channels, base_channels) for _ in range(stages)])

        self.cb1 = ConvBlock(in_channels, c1)
        self.cb2 = ConvBlock(c1, c2)
        self.cb3 = ConvBlock(c2, c3)
        self.cb4 = ConvBlock(c3, c4)
        self.cb5 = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(2)

        self.dsam1 = DualScaleAttentionModule(c1)
        self.dsam2 = DualScaleAttentionModule(c2)
        self.ceu3 = ContextualExtractionUnit(c3, semantic_nodes)
        self.iau4 = InterscaleAggregationUnit(c3, c4, semantic_nodes)
        self.ceu4 = ContextualExtractionUnit(c4, semantic_nodes)
        self.iau5 = InterscaleAggregationUnit(c4, c5, semantic_nodes)
        self.sgrm = SemanticGraphReasoningModule(c5, semantic_nodes)

        self.up5 = UpConvBlock(c5, c4, c4)
        self.up4 = UpConvBlock(c4, c3, c3)
        self.up3 = UpConvBlock(c3, c2, c2)
        self.up2 = UpConvBlock(c2, c1, c1)
        self.out = nn.Conv2d(c1, out_channels, 1)

    def _admm_unroll(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity: torch.Tensor | None,
    ) -> torch.Tensor:
        z = adjoint_operator(kspace, mask, sensitivity)
        b = z.clone()
        u = torch.zeros_like(z)

        for dc, z_prox, b_prox in zip(self.dc_steps, self.z_prox, self.b_prox):
            z_candidate = dc(z, b, u, kspace, mask, sensitivity)
            z = channels_to_complex(z_prox(complex_to_channels(z_candidate)))
            b_candidate = z + u
            b = channels_to_complex(b_prox(complex_to_channels(b_candidate)))
            u = u + z - b
        return complex_to_channels(b)

    def _semantic_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.cb1(x)
        e2 = self.cb2(self.pool(e1))
        e3 = self.cb3(self.pool(e2))
        e4 = self.cb4(self.pool(e3))
        e5 = self.cb5(self.pool(e4))

        r1 = self.dsam1(e1)
        r2 = self.dsam2(e2)
        s3 = self.ceu3(e3)
        o4 = self.iau4(s3, e4)
        s4 = self.ceu4(o4)
        o5 = self.iau5(s4, e5)
        r5 = self.sgrm(o5)

        d4 = self.up5(r5, o4)
        d3 = self.up4(d4, e3)
        d2 = self.up3(d3, r2)
        d1 = self.up2(d2, r1)
        return x + self.out(d1)

    def forward(
        self,
        kspace_or_image: torch.Tensor,
        mask: torch.Tensor | None = None,
        sensitivity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run SIHN.

        If `kspace_or_image` is complex k-space, `mask` is required and the ADMM
        data-consistency path is used. If a real tensor with shape `[B, 2, H, W]`
        is supplied, it is treated as an already zero-filled image.
        """

        if torch.is_complex(kspace_or_image):
            if mask is None:
                raise ValueError("A sampling mask is required for complex k-space input.")
            image = self._admm_unroll(kspace_or_image, mask, sensitivity)
        else:
            if kspace_or_image.ndim != 4 or kspace_or_image.shape[1] != 2:
                raise ValueError("Real-valued image input must have shape [B, 2, H, W].")
            image = kspace_or_image
        return self._semantic_reconstruct(image)
