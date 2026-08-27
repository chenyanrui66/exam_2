"""问题二：4台与5台设备组合的PDOP空间热力图。

说明
----
1. 本脚本只讨论问题二的设备数量与几何构型，不使用问题三的到达时刻、
   5 s 时间窗或读数关联结果。
2. 台站坐标取题目给定的 A-G 布局；图中采用论文表8的 ABEF 与 ABDEF
   组合，以便与正文数值口径一致。
3. 在 110-111 E、27-28 N 网格上，固定音爆高程为 12 km，计算含未知
   音爆时刻的 TOA 雅可比矩阵所对应的 PDOP。
4. 输出 600 dpi PNG 和矢量 SVG；图宽为 156 mm，图题置于图下。
   全图文字以 14 pt 为基准，并针对避免遮挡作小幅调整。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


# --------------------------- 可调整参数 ---------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_NO_4 = 1  # 按全文图号连续性修改
FIGURE_NO_5 = 2

LON_MIN, LON_MAX = 110.0, 111.0
LAT_MIN, LAT_MAX = 27.0, 28.0
GRID_SIZE = 241
SOURCE_ALT_KM = 12.0

SOUND_SPEED_KM_S = 0.340
KM_PER_DEG_LON = 97.304
KM_PER_DEG_LAT = 111.263

# 论文表8采用的两组设备
COMBINATION_4 = "ABEF"
COMBINATION_5 = "ABDEF"

# 完全相同的分级色标，保证两幅图可直接对比
PDOP_LEVELS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                        4.0, 5.0, 6.0, 8.0, 10.0])


# --------------------------- 台站数据 ---------------------------
STATION_NAMES = np.array(list("ABCDEFG"))
STATION_LON = np.array([110.241, 110.783, 110.762, 110.251,
                        110.524, 110.467, 110.047])
STATION_LAT = np.array([27.204, 27.456, 27.785, 28.025,
                        27.617, 28.081, 27.521])
STATION_ALT_KM = np.array([0.824, 0.727, 0.742, 0.850,
                           0.786, 0.678, 0.575])


def configure_fonts() -> None:
    """设置中文字体；图题按要求使用宋体。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def station_xyz_km() -> np.ndarray:
    """经纬高转换为以 (110 E, 27 N) 为原点的局部直角坐标，单位 km。"""
    return np.column_stack((
        (STATION_LON - 110.0) * KM_PER_DEG_LON,
        (STATION_LAT - 27.0) * KM_PER_DEG_LAT,
        STATION_ALT_KM,
    ))


def combination_indices(combination: str) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(STATION_NAMES)}
    return np.array([lookup[name] for name in combination], dtype=int)


def pdop_at_point(source_xyz_km: np.ndarray,
                  stations_xyz_km: np.ndarray) -> float:
    """计算含未知音爆时刻 tau 的三维 TOA-PDOP，结果单位为 km/s。

    观测模型为 t_i = tau + ||p-s_i||/c。雅可比矩阵 H 的前三列
    是到达时刻对位置(km)的偏导，最后一列是对 tau(s) 的偏导。
    Q=(H^T H)^(-1)，PDOP=sqrt(trace(Q_position))。
    """
    delta = source_xyz_km - stations_xyz_km
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance < 1e-12):
        return np.nan

    h_position = delta / (SOUND_SPEED_KM_S * distance[:, None])
    h = np.column_stack((h_position, np.ones(len(stations_xyz_km))))

    singular_values = np.linalg.svd(h, compute_uv=False)
    if singular_values[-1] <= 1e-12 * singular_values[0]:
        return np.nan

    normal = h.T @ h
    q = np.linalg.inv(normal)
    return float(np.sqrt(np.trace(q[:3, :3])))


def compute_grid(combination: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在经纬度网格上计算指定设备组合的 PDOP。"""
    lons = np.linspace(LON_MIN, LON_MAX, GRID_SIZE)
    lats = np.linspace(LAT_MIN, LAT_MAX, GRID_SIZE)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    ids = combination_indices(combination)
    stations = station_xyz_km()[ids]
    pdop = np.empty_like(lon_grid)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            source = np.array([
                (lon_grid[row, col] - 110.0) * KM_PER_DEG_LON,
                (lat_grid[row, col] - 27.0) * KM_PER_DEG_LAT,
                SOURCE_ALT_KM,
            ])
            pdop[row, col] = pdop_at_point(source, stations)

    return lon_grid, lat_grid, pdop


def summarize(pdop: np.ndarray) -> dict[str, float]:
    values = pdop[np.isfinite(pdop)]
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "area_le_3": float(np.mean(values <= 3.0) * 100),
        "area_le_5": float(np.mean(values <= 5.0) * 100),
    }


def draw_station(ax: plt.Axes, index: int) -> None:
    """绘制计算区域内的台站；区域外台站由图内注记单独说明。"""
    lon = STATION_LON[index]
    lat = STATION_LAT[index]
    name = STATION_NAMES[index]

    if lat > LAT_MAX or lat < LAT_MIN or lon > LON_MAX or lon < LON_MIN:
        return

    display_lat = lat
    label = name
    va = "bottom"
    offset = (4, 4)

    ax.scatter(lon, display_lat, s=50, marker="^", color="white",
               edgecolor="black", linewidth=0.8, zorder=5, clip_on=True)
    ax.annotate(label, (lon, display_lat), xytext=offset,
                textcoords="offset points", fontsize=14,
                fontweight="bold", color="black", va=va, zorder=6)


def make_figure(combination: str, figure_number: int,
                caption_description: str,
                lon_grid: np.ndarray, lat_grid: np.ndarray,
                pdop: np.ndarray) -> None:
    stats = summarize(pdop)
    redundancy = len(combination) - 4

    # 156 mm = 6.142 in；稍留安全余量，满足“宽度不超过版心”。
    fig, ax = plt.subplots(figsize=(6.10, 5.85), dpi=160)
    fig.subplots_adjust(left=0.16, right=0.85, top=0.93, bottom=0.27)

    contour = ax.contourf(
        lon_grid, lat_grid, pdop,
        levels=PDOP_LEVELS,
        cmap="YlOrRd",
        extend="max",
        antialiased=True,
    )
    # 关键等值线：PDOP=3 与 PDOP=5
    lines = ax.contour(lon_grid, lat_grid, pdop,
                       levels=[3.0, 5.0], colors=["#155724", "#6f1d1b"],
                       linewidths=[1.2, 1.1])

    for idx in combination_indices(combination):
        draw_station(ax, int(idx))

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(KM_PER_DEG_LAT / KM_PER_DEG_LON)
    ax.set_xlabel("经度（°E）", fontsize=16)
    ax.set_ylabel("纬度（°N）", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(color="white", linestyle="--", linewidth=0.55, alpha=0.48)

    ax.text(
        0.02, 0.975,
        f"组合 {combination}｜{len(combination)}台\n"
        f"冗余自由度 {redundancy}｜高程 {SOURCE_ALT_KM:g} km",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#555555", alpha=0.90),
        zorder=20,
    )
    outside = [
        f"{STATION_NAMES[idx]} {STATION_LAT[idx]:.3f}°N"
        for idx in combination_indices(combination)
        if STATION_LAT[idx] > LAT_MAX
    ]
    ax.text(
        0.02, 0.025,
        f"中位PDOP {stats['median']:.2f}｜P95 {stats['p95']:.2f}\n"
        f"PDOP≤3：{stats['area_le_3']:.1f}%｜PDOP≤5：{stats['area_le_5']:.1f}%",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor="#666666", alpha=0.88),
        zorder=20,
    )

    colorbar = fig.colorbar(contour, ax=ax, pad=0.025, fraction=0.052)
    colorbar.set_label("PDOP（km/s，数值越小越好）", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    # 图题位置、格式与字体：图下方居中，宋体加粗，无冒号。
    caption = f"图{figure_number} {caption_description}"
    caption_artist = fig.text(
        0.5, 0.055, caption, ha="center", va="center",
        fontfamily="SimSun", fontsize=14, fontweight="normal",
    )
    # 宋体通常没有独立粗体字形，用细描边模拟 Word 的“宋体加粗”视觉效果。
    caption_artist.set_path_effects([
        path_effects.Stroke(linewidth=0.25, foreground="black"),
        path_effects.Normal(),
    ])
    stem = f"q2_pdop_{len(combination)}stations_{combination}"
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600,
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def print_comparison(stats4: dict[str, float],
                     stats5: dict[str, float]) -> None:
    def reduction(key: str) -> float:
        return (stats4[key] - stats5[key]) / stats4[key] * 100

    print("问题二 PDOP 空间统计（固定高程 12 km）")
    print(f"4台 {COMBINATION_4}: {stats4}")
    print(f"5台 {COMBINATION_5}: {stats5}")
    print(f"中位PDOP降低 {reduction('median'):.1f}%")
    print(f"95%分位PDOP降低 {reduction('p95'):.1f}%")
    print(f"PDOP≤3覆盖率增加 "
          f"{stats5['area_le_3'] - stats4['area_le_3']:.1f} 个百分点")
    print(f"PDOP≤5覆盖率增加 "
          f"{stats5['area_le_5'] - stats4['area_le_5']:.1f} 个百分点")


def main() -> None:
    configure_fonts()

    lon4, lat4, pdop4 = compute_grid(COMBINATION_4)
    lon5, lat5, pdop5 = compute_grid(COMBINATION_5)

    make_figure(
        COMBINATION_4,
        FIGURE_NO_4,
        "问题二四台设备组合ABEF的PDOP空间热力图",
        lon4, lat4, pdop4,
    )
    make_figure(
        COMBINATION_5,
        FIGURE_NO_5,
        "问题二五台设备组合ABDEF的PDOP空间热力图",
        lon5, lat5, pdop5,
    )
    print_comparison(summarize(pdop4), summarize(pdop5))


if __name__ == "__main__":
    main()
