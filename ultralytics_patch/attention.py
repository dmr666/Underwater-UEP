# -*- coding: utf-8 -*-
"""
轻量注意力模块 (创新点 C2) — ultralytics 8.3.x 注册
在 YOLOv8n Neck 的 Concat 之后插入, 增强跨尺度融合的特征判别力。

模块: 通道注意力 (ECA 式, 1D 卷积) + 空间注意力 (7x7 conv 单通道), 轻量无重参。
注册方式: register_attention() 后, yaml 中可用 LCA (Lightweight Channel-Spatial Attention)。
"""
import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules import Conv
    from ultralytics.nn.tasks import parse_model
except ImportError:
    Conv = None
    parse_model = None


class ECA(nn.Module):
    """Efficient Channel Attention: 1D conv on GAP, 参数 ~k (典型 5)"""
    def __init__(self, channels: int, k_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)      # (B, C)
        y = self.conv(y.unsqueeze(1)).squeeze(1)          # (B, C)
        return x * self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)


class LCA(nn.Module):
    """Lightweight Channel-Spatial Attention (C2): ECA 通道注意力 + 7x7 空间注意力"""
    def __init__(self, c1, c2, k_size: int = 5):
        super().__init__()
        self.eca = ECA(c2, k_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.eca(x)
        # 空间注意力: 通道最大池化 + 平均池化 -> 2 通道
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x * att


def register_attention() -> None:
    """注册 LCA 到 ultralytics 模块表, 使 yaml 可用 -1: LCA [c]"""
    from ultralytics.nn.tasks import parse_model
    # 直接注入模块表 (ultralytics 8.3 通过 globals 查找模块名)
    import ultralytics.nn.tasks as tasks
    tasks.C2f  # 确保模块表已加载
    globals_dict = tasks.__dict__
    if "LCA" not in globals_dict:
        globals_dict["LCA"] = LCA
    print("[attn] LCA 模块已注册: ECA + spatial attention")
