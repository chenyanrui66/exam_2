"""绘制问题三四个残骸的“水平投影 + 高程剖面”双联图。

数据来源：
1. 题目问题三表格中的 7 个监测站经纬度与高程；
2. 《问题二三联合求解方案.py》复核得到的 4 个残骸定位结果。

输出：PDF、SVG（矢量）和 600 dpi PNG（高清位图）。
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np


# ---------------------------- 可调整参数 ----------------------------
FIGURE_NO = 1  # 插入论文时按全文顺序修改


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = OUTPUT_DIR / "图1_问题三_水平投影与高程剖面"

# 论文版心宽度约 156 mm；图宽严格取 156 mm。
MM_TO_INCH = 1 / 25.4
FIG_WIDTH = 156 * MM_TO_INCH
FIG_HEIGHT = 110 * MM_TO_INCH

# 经纬度到局部距离的题设换算系数
K_LON = 97.304  # km/degree
K_LAT = 111.263  # km/degree


# ---------------------------- 问题三数据 ----------------------------
station_names = np.array(list("ABCDEFG"))
station_lon = np.array([110.241, 110.783, 110.762, 110.251,
                        110.524, 110.467, 110.047])
station_lat = np.array([27.204, 27.456, 27.785, 28.025,
                        27.617, 28.081, 27.521])
station_alt_km = np.array([824, 727, 742, 850, 786, 678, 575]) / 1000

# 联合求解结果，按音爆时刻排序：#1 南、#2 北、#3 西、#4 东。
debris_id = np.array([1, 2, 3, 4])
debris_lon = np.array([110.500001, 110.499999, 110.300000, 110.699999])
debris_lat = np.array([27.309998, 27.949998, 27.650000, 27.650000])
debris_alt_km = np.array([12.514, 11.529, 11.478, 13.468])
debris_tau_s = np.array([11.9999, 13.0014, 14.0000, 15.0000])

# “十字”几何中心；正交剖面均以此为距离原点。
center_lon, center_lat = 110.5, 27.65


def setup_chinese_font():
    """优先使用宋体，保证论文图题与中文标注正确显示。"""
    candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            mpl.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            return family
    mpl.rcParams["font.sans-serif"] = ["SimSun",
                                       "Noto Sans CJK SC", "DejaVu Sans"]
    return "SimSun"


FONT_FAMILY = setup_chinese_font()
mpl.rcParams.update({
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14.5,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def idw_height(query_lon, query_lat, power=2.0):
    """由 7 个站点高程做反距离加权插值，仅作为地形背景示意。"""
    dx = (query_lon[..., None] - station_lon) * K_LON
    dy = (query_lat[..., None] - station_lat) * K_LAT
    dist2 = dx * dx + dy * dy
    weights = 1.0 / np.maximum(dist2, 1e-8) ** (power / 2)
    return np.sum(weights * station_alt_km, axis=-1) / np.sum(weights, axis=-1)


def terrain_at(lon_value, lat_value):
    """返回指定经纬度处的 IDW 地面高程（km）。"""
    return float(idw_height(np.asarray(lon_value), np.asarray(lat_value)))


def make_figure():
    colors = ["#2F6B9A", "#8E5AA9", "#D97706", "#C43D4B"]
    markers = ["o"] * 4

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1, 1],
        left=0.085, right=0.985, top=0.90, bottom=0.17, wspace=0.55
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_prof = fig.add_subplot(gs[0, 1])

    # ---- 左图：水平投影 + 站点高程插值等高线 ----
    lon_min, lon_max = 110.00, 110.83
    lat_min, lat_max = 27.15, 28.13
    gx, gy = np.meshgrid(
        np.linspace(lon_min, lon_max, 240),
        np.linspace(lat_min, lat_max, 260),
    )
    gz = idw_height(gx, gy)
    levels = np.arange(0.56, 0.88, 0.04)
    filled = ax_map.contourf(gx, gy, gz, levels=levels,
                             cmap="terrain", alpha=0.68, extend="both")
    lines = ax_map.contour(gx, gy, gz, levels=levels,
                           colors="#6B665D", linewidths=0.45, alpha=0.65)
    ax_map.clabel(lines, inline=True, fontsize=10.5,
                  fmt=lambda value: f"{value * 1000:.0f} m")

    # 监测站与名称
    ax_map.scatter(station_lon, station_lat, s=42, marker="^",
                   facecolor="white", edgecolor="#303030", linewidth=1.0,
                   zorder=4)
    # 边缘台站的文字向图心收回，避免 B、F 等标签越出边框。
    station_label_style = {
        "A": dict(xytext=(5, -6), ha="left", va="bottom"),
        "B": dict(xytext=(-5, 0), ha="right", va="bottom"),
        "C": dict(xytext=(-4, 0), ha="right", va="bottom"),
        "D": dict(xytext=(-10, 0), ha="right", va="center"),
        "E": dict(xytext=(3, -4), ha="left", va="top"),
        "F": dict(xytext=(8, 5.5), ha="left", va="top"),
        "G": dict(xytext=(2, -0.5), ha="left", va="bottom"),
    }
    for name, x, y in zip(station_names, station_lon, station_lat):
        ax_map.annotate(name, (x, y), textcoords="offset points",
                        fontsize=12, color="#303030", zorder=5,
                        **station_label_style[name])

    # 正交剖面线：南北与东西剖面在中心相交。
    ax_map.plot([center_lon, center_lon], [debris_lat[0], debris_lat[1]],
                color="#255F85", linewidth=1.0, linestyle="--", zorder=3)
    ax_map.plot([debris_lon[2], debris_lon[3]], [center_lat, center_lat],
                color="#B35A22", linewidth=1.0, linestyle="--", zorder=3)
    ax_map.scatter([center_lon], [center_lat], s=17, marker="+",
                   color="#111111", linewidth=1.0, zorder=5)

    # A 图仅保留残骸点位，编号与高程统一放到 B 图中展示。
    for x, y, color, marker in zip(debris_lon, debris_lat, colors, markers):
        ax_map.scatter(x, y, s=72, marker=marker, facecolor=color,
                       edgecolor="white", linewidth=1.0, zorder=6)

    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_aspect(K_LAT / K_LON, adjustable="box")
    ax_map.set_xlabel("经度（°E）")
    ax_map.set_ylabel("纬度（°N）")
    ax_map.set_title("（a）水平投影", pad=7)
    ax_map.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_map.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_map.grid(color="white", linewidth=0.45, alpha=0.50)

    cbar = fig.colorbar(filled, ax=ax_map, orientation="horizontal",
                        fraction=0.06, pad=0.19, aspect=28,
                        ticks=[0.56, 0.64, 0.72, 0.80, 0.84])
    cbar.set_label("站点高程 IDW 插值（km）", fontsize=12)
    cbar.ax.tick_params(labelsize=9.5, length=2)

    legend_items = [
        Line2D([0], [0], marker="^", linestyle="none", markersize=7,
               markerfacecolor="white", markeredgecolor="#303030",
               label="监测站"),
        Line2D([0], [0], color="#255F85", linestyle="--", linewidth=1,
               label="南北剖面"),
        Line2D([0], [0], color="#B35A22", linestyle="--", linewidth=1,
               label="东西剖面"),
    ]
    # 放置于两幅子图上方的中缝，避开 B 图纵坐标轴文字与刻度。
    fig.legend(handles=legend_items, loc="center", bbox_to_anchor=(0.48, 0.78),
               frameon=True, framealpha=0.92, borderpad=0.3,
               handlelength=1.45, labelspacing=0.35, fontsize=9)

    # ---- 右图：以十字中心为零点的正交高程剖面 ----
    ns_distance = (debris_lat[:2] - center_lat) * K_LAT
    ew_distance = (debris_lon[2:] - center_lon) * K_LON

    ns_x = np.linspace(-42, 39, 220)
    ns_lat = center_lat + ns_x / K_LAT
    ns_ground = idw_height(np.full_like(ns_x, center_lon), ns_lat)
    ew_x = np.linspace(-24, 24, 180)
    ew_lon = center_lon + ew_x / K_LON
    ew_ground = idw_height(ew_lon, np.full_like(ew_x, center_lat))

    ax_prof.axhspan(debris_alt_km.min(), debris_alt_km.max(),
                    color="#8FA9BF", alpha=0.13, zorder=0)
    ax_prof.plot(ns_x, ns_ground, color="#4E7D51", linewidth=1.0,
                 label="南北剖面地面")
    ax_prof.plot(ew_x, ew_ground, color="#78945B", linewidth=1.0,
                 linestyle=":", label="东西剖面地面")
    ax_prof.fill_between(ns_x, 0, ns_ground, color="#90A86B", alpha=0.20)

    # 南北剖面：#1（南）与 #2（北）
    for event_index, distance in zip([0, 1], ns_distance):
        ground = terrain_at(center_lon, debris_lat[event_index])
        ax_prof.vlines(distance, ground, debris_alt_km[event_index],
                       colors="#255F85", linewidth=0.75, linestyles="--")
        ax_prof.scatter(distance, debris_alt_km[event_index], s=72,
                        marker=markers[event_index], color=colors[event_index],
                        edgecolor="white", linewidth=0.7, zorder=5)
        # 数值与编号垂直对齐，数值直接排在编号下方。
        label_style = (
            dict(xytext=(10, 10), ha="center", va="bottom")
            if event_index == 0
            else dict(xytext=(10, 10), ha="right", va="bottom")
        )
        ax_prof.annotate(
            f"{debris_alt_km[event_index]:.3f}",
            (distance, debris_alt_km[event_index]), textcoords="offset points",
            fontsize=10.5,
            **label_style,
        )

    # 东西剖面：#3（西）与 #4（东）
    for event_index, distance in zip([2, 3], ew_distance):
        ground = terrain_at(debris_lon[event_index], center_lat)
        ax_prof.vlines(distance, ground, debris_alt_km[event_index],
                       colors="#B35A22", linewidth=0.75, linestyles="--")
        ax_prof.scatter(distance, debris_alt_km[event_index], s=72,
                        marker=markers[event_index], color=colors[event_index],
                        edgecolor="white", linewidth=0.7, zorder=5)
        # #3、#4 同样采用“编号在上、数值在下”的对齐方式。
        label_style = dict(xytext=(0, 10), ha="center", va="bottom")
        ax_prof.annotate(
            f"{debris_alt_km[event_index]:.3f}",
            (distance, debris_alt_km[event_index]), textcoords="offset points",
            fontsize=10.5,
            **label_style,
        )

    ax_prof.axvline(0, color="#777777", linewidth=0.65, linestyle="-")
    ax_prof.set_xlim(-43, 41)
    # 抬高纵轴上限，使彩色点相对下压，为顶部编号与高程预留空间。
    ax_prof.set_ylim(0, 16.5)
    ax_prof.set_xlabel("相对中心距离（km）")
    ax_prof.set_ylabel("高程（km）")
    ax_prof.set_title("（b）正交高程剖面", pad=7)
    ax_prof.grid(axis="both", color="#D9D9D9", linewidth=0.55,
                 linestyle="--", alpha=0.8)
    ax_prof.legend(loc="lower right", frameon=True, framealpha=0.90,
                   borderpad=0.45, handlelength=2.2)

    # 统一边框与图下注：宋体小四（12 pt）加粗、居中。
    for ax in (ax_map, ax_prof):
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("#444444")

    return fig


def main():
    fig = make_figure()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), format="pdf",
                bbox_inches=None, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), format="svg",
                bbox_inches=None, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), format="png", dpi=600,
                bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"已输出：{OUTPUT_STEM}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
