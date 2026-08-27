"""绘制问题三的“设备读数—残骸”关联矩阵热力图。

数据来源：问题三联合求解结果（残骸按音爆时刻由早到晚编号 #1～#4）。
输出：SVG 矢量图与 600 dpi PNG 高清位图。
"""

from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# 7 行对应设备 A～G，4 列对应各设备按到达时间排序的第 1～4 个读数。
# 单元格值是该读数所属的残骸编号。
ASSOCIATION = np.array(
    [
        [1, 3, 4, 2],  # A
        [4, 1, 3, 2],  # B
        [4, 2, 3, 1],  # C
        [2, 3, 4, 1],  # D
        [4, 3, 1, 2],  # E
        [2, 3, 4, 1],  # F
        [3, 1, 2, 4],  # G
    ],
    dtype=int,
)

DEVICES = list("ABCDEFG")
READING_NUMBERS = [1, 2, 3, 4]

# Nature Publishing Group 风格配色，并保持既定语义：
# #1 红、#2 蓝、#3 绿、#4 黄。颜色在白色分隔线和论文打印中保持清晰区分。
COLORS = ["#E64B35", "#4DBBD5", "#57CA87", "#F6C85F"]


def _font(size: float, bold: bool = False) -> FontProperties:
    """优先使用宋体；没有宋体时回退到 Matplotlib 可用的中文字体。"""
    simsun = Path("C:/Windows/Fonts/simsun.ttc")
    if simsun.exists():
        return FontProperties(fname=simsun, size=size, weight="bold" if bold else "normal")
    return FontProperties(family=["SimSun", "Songti SC", "Noto Serif CJK SC"],
                          size=size, weight="bold" if bold else "normal")


def validate_association(matrix: np.ndarray) -> None:
    """确认每台设备的 4 个读数恰好分属 4 个残骸。"""
    if matrix.shape != (7, 4):
        raise ValueError(f"关联矩阵必须为 7×4，实际为 {matrix.shape}")
    expected = [1, 2, 3, 4]
    for device, row in zip(DEVICES, matrix):
        if sorted(row.tolist()) != expected:
            raise ValueError(f"设备 {device} 未形成一对一覆盖：{row.tolist()}")


def draw_heatmap(output_dir: Path) -> tuple[Path, Path]:
    validate_association(ASSOCIATION)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 156 mm = 6.1417 in；略留余量，确保不超过论文约 156 mm 的版心宽度。
    fig = plt.figure(figsize=(6.10, 5.20), facecolor="white")
    ax = fig.add_axes([0.12, 0.18, 0.80, 0.8])

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    ax.imshow(ASSOCIATION, cmap=cmap, norm=norm,
              aspect="auto", interpolation="none")

    ax.set_xticks(np.arange(4), labels=READING_NUMBERS,
                  fontproperties=_font(14))
    ax.set_yticks(np.arange(7), labels=DEVICES, fontproperties=_font(14))
    ax.set_xlabel("读数序号（按到达时间排序）", fontproperties=_font(14), labelpad=10)
    ax.set_ylabel("监测设备", fontproperties=_font(14), labelpad=12)
    ax.tick_params(axis="both", which="major", length=0)

    # 白色分隔线使 28 个归属单元清楚可辨。
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")

    # 在颜色之外直接标注残骸编号，便于黑白打印和快速核读。
    text_colors = {1: "white", 2: "white", 3: "#1F2A16", 4: "#3A3300"}
    for i in range(ASSOCIATION.shape[0]):
        for j in range(ASSOCIATION.shape[1]):
            debris_id = int(ASSOCIATION[i, j])
            ax.text(
                j,
                i,
                f"#{debris_id}",
                ha="center",
                va="center",
                color=text_colors[debris_id],
                fontproperties=_font(16, bold=True),
            )

    legend_handles = [
        Patch(facecolor=color, edgecolor="#666666",
              linewidth=0.6, label=f"残骸 #{idx}")
        for idx, color in enumerate(COLORS, start=1)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        prop=_font(14),
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=0.7,
    )

    png_path = output_dir / "q3_设备读数与残骸关联矩阵.png"
    fig.savefig(png_path, format="png", dpi=600, facecolor="white")
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    output_directory = Path("./output/figs")
    draw_heatmap(output_directory)
    print("绘图已完成")
