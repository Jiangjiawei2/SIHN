"""Network blocks used by SIHN."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProximalBlock(nn.Module):
    def __init__(self, channels: int = 2, hidden_channels: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ContextualExtractionUnit(nn.Module):
    """CEU: gathers global semantic nodes from high-level feature maps."""

    def __init__(self, channels: int, semantic_nodes: int):
        super().__init__()
        self.feature_proj = nn.Conv2d(channels, channels, 1)
        self.attention_proj = nn.Conv2d(channels, semantic_nodes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        feat = self.feature_proj(x).flatten(2)
        att = self.attention_proj(x).flatten(2)
        att = torch.softmax(att, dim=-1)
        semantic = torch.einsum("bch,bnh->bcn", feat, att)
        return semantic / math.sqrt(height * width)


class InterscaleAggregationUnit(nn.Module):
    """IAU: projects semantic nodes back to a higher-resolution feature map."""

    def __init__(self, semantic_channels: int, guide_channels: int, semantic_nodes: int):
        super().__init__()
        self.attention_proj = nn.Conv2d(guide_channels, semantic_nodes, 1)
        self.semantic_to_guide = nn.Conv2d(semantic_channels, guide_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(guide_channels * 2, guide_channels, 1, bias=False),
            nn.BatchNorm2d(guide_channels),
            nn.ReLU(inplace=True),
            ConvBlock(guide_channels, guide_channels),
        )

    def forward(self, semantic: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = guide.shape
        att = self.attention_proj(guide).flatten(2)
        att = torch.softmax(att, dim=1)
        desc = torch.einsum("bcn,bnh->bch", semantic, att)
        desc = desc.view(bsz, semantic.shape[1], height, width)
        desc = self.semantic_to_guide(desc)
        gated = guide * torch.sigmoid(desc) + desc
        return self.refine(torch.cat([guide, gated], dim=1))


class SemanticGraphReasoningModule(nn.Module):
    """SGRM: graph projection, graph convolution, node attention, and reprojection."""

    def __init__(self, channels: int, semantic_nodes: int):
        super().__init__()
        self.channels = channels
        self.semantic_nodes = semantic_nodes
        self.pre = ConvBlock(channels, channels)
        self.anchors = nn.Parameter(torch.randn(semantic_nodes, channels) * 0.02)
        self.adjacency = nn.Parameter(torch.eye(semantic_nodes))
        self.graph_linear = nn.Linear(channels, channels, bias=False)
        self.node_norm = nn.LayerNorm(channels)
        reduction = max(4, channels // 4)
        self.node_attention = nn.Sequential(
            nn.Linear(channels, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, 1),
            nn.Sigmoid(),
        )
        self.post = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        feat = self.pre(x)
        flat = feat.flatten(2).transpose(1, 2)
        anchors = F.normalize(self.anchors, dim=-1)
        logits = torch.einsum("bhc,nc->bhn", flat, anchors) / math.sqrt(channels)

        pixel_to_node = torch.softmax(logits, dim=-1)
        node_gather = torch.softmax(logits, dim=1)
        nodes = torch.einsum("bhn,bhc->bnc", node_gather, flat)

        adjacency = torch.softmax(self.adjacency, dim=-1)
        nodes = torch.einsum("nm,bmc->bnc", adjacency, nodes)
        nodes = self.node_norm(self.graph_linear(nodes))
        nodes = nodes * self.node_attention(nodes)

        reprojected = torch.einsum("bhn,bnc->bhc", pixel_to_node, nodes)
        reprojected = reprojected.transpose(1, 2).view(bsz, channels, height, width)
        return x + self.post(reprojected)


class DualScaleAttentionModule(nn.Module):
    """DSAM: local detail extraction plus compressed global attention."""

    def __init__(self, channels: int, global_grid: int = 4, num_heads: int = 4):
        super().__init__()
        self.global_grid = global_grid
        self.local = DepthwiseSeparableConv(channels, channels)
        heads = max(1, min(num_heads, channels))
        while channels % heads != 0:
            heads -= 1
        self.global_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.global_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        pooled = F.adaptive_avg_pool2d(x, (self.global_grid, self.global_grid))
        bsz, channels, gh, gw = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        global_tokens, _ = self.global_attn(tokens, tokens, tokens, need_weights=False)
        global_map = global_tokens.transpose(1, 2).view(bsz, channels, gh, gw)
        global_map = F.interpolate(global_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        global_map = self.global_proj(global_map)
        gate = self.gate(torch.cat([local, global_map], dim=1))
        return x + gate * local + (1.0 - gate) * global_map


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
