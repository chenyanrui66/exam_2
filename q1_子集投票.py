"""绘制问题一的四台子集投票散点图（稳健性检验）。"""

from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Ellipse
from matplotlib import font_manager
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# 问题一 q1_solution_v2.py 的复算结果（经度 °E，纬度 °N）
SUBSET_SOLUTIONS = {
    "ABCE": (110.49860865794430, 27.310108125594336),
    "ABCG": (110.50644801935094, 27.300166564596815),
    "ABEG": (110.49784993203127, 27.314287078054736),
    "ACEG": (110.46577837478698, 27.333150791755780),
    "BCEG": (110.49909192132274, 27.310757968630790),
}
JOINT_SOLUTION = (110.49890224957825, 27.310516790594760)

# 约 156 mm 版心内留出余量。
FIGURE_WIDTH_MM = 154
FIGURE_HEIGHT_MM = 132


def configure_chinese_font():
    """优先使用宋体；缺失时回退到常见中文字体。"""
    candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(
                fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            return font_name
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei",
                                       "SimSun", "DejaVu Sans"]
    return plt.rcParams["font.sans-serif"][0]


def make_figure():
    configure_chinese_font()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "svg.fonttype": "none",  # SVG 保留可编辑文字
            "font.size": 14,
        }
    )

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.145, right=0.965, top=0.950, bottom=0.140)

    regular = [name for name in SUBSET_SOLUTIONS if name != "ACEG"]
    x_regular = [SUBSET_SOLUTIONS[name][0] for name in regular]
    y_regular = [SUBSET_SOLUTIONS[name][1] for name in regular]

    # 4 个主簇子集解。
    ax.scatter(
        x_regular,
        y_regular,
        s=62,
        marker="o",
        facecolor="#3B82C4",
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
        label="四台子集解（主簇）",
    )

    # ACEG 离群票。
    aceg_lon, aceg_lat = SUBSET_SOLUTIONS["ACEG"]
    ax.scatter(
        [aceg_lon],
        [aceg_lat],
        s=72,
        marker="o",
        facecolor="#D95F4B",
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="四台子集解（ACEG 离群）",
    )

    # 五台联合解：中心大星号。
    joint_lon, joint_lat = JOINT_SOLUTION
    ax.scatter(
        [joint_lon],
        [joint_lat],
        # 星号小于普通散点，避免遮挡与联合解近邻的 ABCE、BCEG 两票。
        s=55,
        marker="*",
        facecolor="#F2B134",
        edgecolor="#7A4E00",
        linewidth=1.0,
        zorder=7,
        label="五台联合解（ABCEG）",
    )

    # 主簇范围仅作视觉引导，不表示统计置信区间。
    cluster = Ellipse(
        (110.5014, 27.3074),
        width=0.0175,
        height=0.0240,
        facecolor="#3B82C4",
        edgecolor="#3B82C4",
        linewidth=1.0,
        linestyle="--",
        alpha=0.10,
        zorder=1,
    )
    ax.add_patch(cluster)

    # 标注子集名称，错开 ABCE/BCEG/联合解的密集位置。
    offsets = {
        "ABCE": (-18, -17),
        "ABCG": (-1, -17),
        "ABEG": (-22, 8),
        "ACEG": (4, 8),
        "BCEG": (12, 20),
    }
    for name, (lon, lat) in SUBSET_SOLUTIONS.items():
        ax.annotate(
            name,
            (lon, lat),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=13,
            fontweight="normal",
            color="#8E3026" if name == "ACEG" else "#1F2937",
            ha="left",
            va="center",
            zorder=8,
        )

    ax.annotate(
        "五台联合解",
        (joint_lon, joint_lat),
        xytext=(110.5002, joint_lat),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="normal",
        color="#6B4600",
        zorder=8,
    )

    # 依照论文叙述口径标出 ACEG 相对主簇的约 3.5 km 偏离。
    ax.annotate(
        "偏离约 3.5 km",
        xy=(aceg_lon, aceg_lat),
        xytext=(110.4808, 27.3272),
        arrowprops=dict(
            arrowstyle="-|>", color="#D95F4B", lw=1.1,
            connectionstyle="arc3,rad=-0.10"
        ),
        fontsize=13,
        fontweight="normal",
        color="#A23D30",
        ha="center",
        va="bottom",
        zorder=6,
    )

    ax.set_xlabel("经度（°E）", fontsize=14)
    ax.set_ylabel("纬度（°N）", fontsize=14)
    ax.set_xlim(110.4625, 110.5115)
    ax.set_ylim(27.2970, 27.3370)
    # 使横纵方向的图上距离近似反映当地实际距离。
    ax.set_aspect(111.263 / 97.304, adjustable="box")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", labelsize=12.5)
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.42, color="#94A3B8")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#64748B")
        spine.set_linewidth(0.8)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        fontsize=9,
        markerscale=0.78,
        borderpad=0.42,
        labelspacing=0.38,
        handlelength=1.15,
        handletextpad=0.42,
    )
    legend.get_frame().set_edgecolor("#CBD5E1")
    legend.get_frame().set_linewidth(0.7)

    # 联合解与 ABCE、BCEG 仅相距几十米，主图中会重叠；局部放大保证 5 票均可辨。
    # 放在左侧无数据的空白区域，避免遮挡主簇和离群点。
    inset = ax.inset_axes([0.105, 0.225, 0.425, 0.365])
    close_names = ["ABCE", "BCEG"]
    inset.scatter(
        [SUBSET_SOLUTIONS[name][0] for name in close_names],
        [SUBSET_SOLUTIONS[name][1] for name in close_names],
        s=38,
        marker="o",
        facecolor="#3B82C4",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    inset.scatter(
        [joint_lon], [joint_lat], s=50, marker="*",
        facecolor="#F2B134", edgecolor="#7A4E00", linewidth=0.8, zorder=5
    )
    inset.annotate("ABCE", SUBSET_SOLUTIONS["ABCE"], xytext=(4, 4),
                   textcoords="offset points", ha="left", va="bottom", fontsize=10.5)
    inset.annotate("BCEG", SUBSET_SOLUTIONS["BCEG"], xytext=(-4, -5),
                   textcoords="offset points", ha="right", va="top", fontsize=10.5)
    inset.annotate("联合解", JOINT_SOLUTION, xytext=(10, 0),
                   textcoords="offset points", ha="left", va="center",
                   fontsize=10.5, color="#6B4600")
    inset.set_xlim(110.49848, 110.49938)
    inset.set_ylim(27.30998, 27.31086)
    inset.set_aspect(111.263 / 97.304, adjustable="box")
    inset.set_xticks([110.4985, 110.4990])
    inset.set_yticks([27.3100, 27.3104, 27.3108])
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    inset.tick_params(axis="both", labelsize=8.5, length=2.2, pad=1.5)
    inset.grid(True, linestyle="--", linewidth=0.4,
               alpha=0.35, color="#94A3B8")
    inset.set_title("中心区域局部放大", fontsize=11.5, pad=3.5)
    for spine in inset.spines.values():
        spine.set_color("#64748B")
        spine.set_linewidth(0.65)
    return fig


def main():
    output_dir = Path("./output/figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(
        output_dir / "q1_子集投票散点图.png",
        format="png",
        dpi=600,
        facecolor="white",
    )
    plt.close(fig)
    print("绘图已完成")


if __name__ == "__main__":
    main()
