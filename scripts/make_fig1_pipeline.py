# make_fig1_pipeline.py — 重画图 1（UEP 流水线示意）
# 修正：降质估计框删除未定义的 "contrast σ"；自适应增强框删除 "CLAHE"（UEP 实际为白平衡 + 自适应饱和度）
# 输出：figures/uep_pipeline.pdf（矢量，供 LaTeX）+ figures/uep_pipeline.png（预览）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = Path(__file__).resolve().parent.parent / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, YELL, GRN, PINK, GREY = "#DCE9F7", "#FDF6D8", "#DDF0DC", "#FBE4E4", "#F2F2F2"
EDGE = {"blue": "#2E4E7E", "yel": "#B8860B", "grn": "#2F7A3A", "pink": "#8B2E4A", "grey": "#7A7A7A"}
TEAL = "#127C86"

fig, ax = plt.subplots(figsize=(10.5, 3.25), dpi=300)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

Y0, Y1 = 0.44, 0.72          # 主行所有方框的上下沿

def box(x0, y0, x1, y1, fill, edge, text, fs=10.5, head=None, dashed=False, round_=0.022, gray_text=False):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle=f"round,pad=0.004,rounding_size={round_}",
                 linewidth=1.3, edgecolor=edge, facecolor=fill,
                 linestyle=(0, (5, 3)) if dashed else "solid"))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    col = "#444444" if gray_text else "#111111"
    if head:
        ax.text(cx, cy + 0.075, head, ha="center", va="center", fontsize=fs, color=col)
        ax.text(cx, cy - 0.062, text, ha="center", va="center", fontsize=fs - 1.2, color=col)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=col, linespacing=1.5)

def arrow(x0, y0, x1, y1, dashed=False, lw=1.3, ms=10):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle=f"-|>,head_width={ms/42},head_length={ms/26}",
                 linewidth=lw, color="black", shrinkA=0, shrinkB=0,
                 linestyle=(0, (4, 3)) if dashed else "solid", mutation_scale=ms))

# UEP 虚线容器 + 标题
box(0.128, 0.30, 0.548, 0.845, "none", TEAL, "", dashed=True, round_=0.012)
ax.text((0.128 + 0.548) / 2, 0.905, "UEP (zero trainable parameters)",
        ha="center", va="center", fontsize=10.5, color=TEAL, fontweight="bold")

# 主行
box(0.012, Y0 + 0.03, 0.108, Y1 - 0.03, BLUE, EDGE["blue"], "Input image\n$I$ (BGR)")
box(0.158, Y0, 0.318, Y1, YELL, EDGE["yel"], r"color cast $c$ $\cdot$ saturation $s$",
    head="Degradation estimation", fs=9.6)
box(0.348, Y0, 0.518, Y1, GRN, EDGE["grn"], r"white balance $\cdot$ adaptive saturation",
    head="Adaptive enhancement", fs=9.6)
box(0.578, Y0 + 0.03, 0.668, Y1 - 0.03, BLUE, EDGE["blue"], "Enhanced image\n$\\hat{I}$")
box(0.698, Y0, 0.836, Y1, PINK, EDGE["pink"], "backbone + neck + head", head="YOLOv8n", fs=10)
box(0.866, Y0, 0.996, Y1, PINK, EDGE["pink"], "holothurian $\\cdot$ echinus\nscallop $\\cdot$ starfish",
    head="Detections", fs=9.6)

# 主行箭头
for xa, xb in [(0.110, 0.156), (0.320, 0.346), (0.520, 0.576), (0.670, 0.696), (0.838, 0.864)]:
    arrow(xa, (Y0 + Y1) / 2, xb, (Y0 + Y1) / 2)

# LCA 支路
box(0.678, 0.115, 0.812, 0.29, GREY, EDGE["grey"], "LCA (309 params)\nstudied $\\cdot$ excluded",
    dashed=True, fs=9.2, gray_text=True)
arrow(0.745, 0.292, 0.745, Y0 - 0.004, dashed=True, lw=1.0, ms=8)
ax.text(0.735, 0.352, "inserted at neck\nP3/P4/P5", ha="right", va="center",
        fontsize=7.6, color="#333333", style="italic", linespacing=1.4)

fig.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)
fig.savefig(OUT / "uep_pipeline.pdf", format="pdf")
fig.savefig(OUT / "uep_pipeline.png", format="png", dpi=200)
print("written:", OUT / "uep_pipeline.pdf")
