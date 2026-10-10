# make_fig1_pipeline.py — 图 1（UEP 流水线示意）
# ① 删掉未定义的 "contrast σ"；② 删掉 "CLAHE"（UEP 实为白平衡 + 自适应饱和度）；
# ③ 设计尺寸 = 单栏成稿尺寸（3.30 in 宽，满栏 \columnwidth 排版），保证渲染后图内文字 ≥6 pt。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = Path(__file__).resolve().parent.parent / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, YELL, GRN, PINK, GREY = "#DCE9F7", "#FDF6D8", "#DDF0DC", "#FBE4E4", "#F2F2F2"
EDGE = {"blue": "#2E4E7E", "yel": "#B8860B", "grn": "#2F7A3A", "pink": "#8B2E4A", "grey": "#7A7A7A"}
TEAL = "#127C86"

fig, ax = plt.subplots(figsize=(3.30, 1.72), dpi=400)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def rbox(x0, y0, x1, y1, fill, edge, dashed=False, round_=0.03, lw=1.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle=f"round,pad=0.006,rounding_size={round_}",
                 linewidth=lw, edgecolor=edge, facecolor=fill,
                 linestyle=(0, (4, 2.4)) if dashed else "solid"))

def txt(x, y, s, fs=7.2, col="#000000", bold=False, italic=False, bg=False, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=col,
            fontweight="bold" if bold else "normal", style="italic" if italic else "normal",
            linespacing=1.35,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.9) if bg else None)

def arrow(x0, y0, x1, y1, dashed=False, ms=6.5, lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle=f"-|>,head_width={ms/28},head_length={ms/18}",
                 linewidth=lw, color="black", shrinkA=0, shrinkB=0,
                 linestyle=(0, (3, 2)) if dashed else "solid", mutation_scale=ms))

# ---- UEP 容器（顶边下移到 0.90，标题置于其上不再被虚线穿过）----
rbox(0.245, 0.235, 0.735, 0.90, "none", TEAL, dashed=True, round_=0.02)
txt(0.49, 0.955, "UEP (zero trainable parameters)", fs=7.2, col=TEAL, bold=True)

# ---- 左：输入 ----
rbox(0.015, 0.50, 0.212, 0.80, BLUE, EDGE["blue"])
txt(0.1135, 0.65, "Input\n$I$ (BGR)", fs=7.6)
# ---- 容器内两个子块 ----
rbox(0.275, 0.615, 0.705, 0.865, YELL, EDGE["yel"])
txt(0.49, 0.795, "Degradation estimation", fs=7.4, bold=True)
txt(0.49, 0.675, "color cast $c$, saturation $s$", fs=6.8)
rbox(0.275, 0.285, 0.705, 0.545, GRN, EDGE["grn"])
txt(0.49, 0.475, "Adaptive enhancement", fs=7.4, bold=True)
txt(0.49, 0.355, "white balance $+$ adaptive saturation", fs=6.8)
# ---- 右：检测器 ----
rbox(0.760, 0.50, 0.998, 0.80, PINK, EDGE["pink"])
txt(0.879, 0.65, "YOLOv8n", fs=7.4)
# ---- 下：LCA（虚线，排除）----
rbox(0.760, 0.03, 0.998, 0.40, GREY, EDGE["grey"], dashed=True)
txt(0.879, 0.315, "LCA (309 p.)", fs=7.0, col="#3A3A3A")
txt(0.879, 0.165, "studied,\nexcluded", fs=7.0, col="#3A3A3A")

arrow(0.214, 0.65, 0.273, 0.74)
arrow(0.49, 0.612, 0.49, 0.548)
arrow(0.707, 0.74, 0.758, 0.74)
arrow(0.879, 0.404, 0.879, 0.485, dashed=True, ms=5.5)
txt(0.755, 0.445, "inserted at neck\nP3/P4/P5", fs=5.8, col="#333333", italic=True, ha="left", bg=True)

fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
fig.savefig(OUT / "uep_pipeline.pdf", format="pdf")
fig.savefig(OUT / "uep_pipeline.png", format="png", dpi=400)
print("written:", OUT / "uep_pipeline.pdf")
