"""问题二：4台与5台设备组合的PDOP空间热力图。

说明
----
1. 本脚本只讨论问题二的设备数量与几何构型，不使用问题三的到达时刻、
   5 s 时间窗或读数关联结果。
2. 台站坐标取题目给定的 A-G 布局；图中采用论文表8的 ABEF 与 ABDEF
   组合，以便与正文数值口径一致。
3. 在 110-111 E、27-28 N 网格上，固定音爆高程为 12 km，计算含未知
   音爆时刻的 TOA 雅可比矩阵所对应的 PDOP。
4. 两张子图合并为一张并排图；区域外台站由图下注记单独说明。
   输出 600 dpi PNG；图宽按双栏排版调整。全图文字以 14 pt 为基准。
"""

from __future__ import annotations
import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# --------------------------- 可调整参数 ---------------------------
OUTPUT_DIR = Path("./output/figs")
FIGURE_NO = 1  # 按全文图号连续性修改

LON_MIN, LON_MAX = 110.0, 111.2
LAT_MIN, LAT_MAX = 27.0, 28.2
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


def get_outside_stations(combination: str) -> list[str]:
    """返回区域外台站的名称与坐标字符串列表。"""
    outside = []
    for idx in combination_indices(combination):
        lon = STATION_LON[idx]
        lat = STATION_LAT[idx]
        name = STATION_NAMES[idx]
        if lat > LAT_MAX or lat < LAT_MIN or lon > LON_MAX or lon < LON_MIN:
            outside.append(
                f"{name} ({lon:.3f} deg E, {lat:.3f} deg N)"
            )
    return outside


def draw_subplot(ax: plt.Axes, combination: str,
                 lon_grid: np.ndarray, lat_grid: np.ndarray,
                 pdop: np.ndarray) -> mpl.contour.QuadContourSet:
    """绘制单个子图（一个设备组合的热力图），返回 contourf 对象供 colorbar 使用。"""
    stats = summarize(pdop)
    redundancy = len(combination) - 4

    contour = ax.contourf(
        lon_grid, lat_grid, pdop,
        levels=PDOP_LEVELS,
        cmap="YlOrRd",
        extend="max",
        antialiased=True,
    )
    # 关键等值线：PDOP=3 与 PDOP=5
    ax.contour(lon_grid, lat_grid, pdop,
               levels=[3.0, 5.0], colors=["#155724", "#6f1d1b"],
               linewidths=[1.2, 1.1])

    for idx in combination_indices(combination):
        draw_station(ax, int(idx))

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(KM_PER_DEG_LAT / KM_PER_DEG_LON)
    ax.set_xlabel("经度（$^\circ$ E）", fontsize=14)
    ax.set_ylabel("纬度（$^\circ$ N）", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(color="white", linestyle="--", linewidth=0.55, alpha=0.48)

    # 左下角统计信息框
    stat_text = (
        f"中位PDOP {stats['median']:.2f} | P95 {stats['p95']:.2f}\n"
        f"PDOP<=3：{stats['area_le_3']:.1f}% | PDOP<=5：{stats['area_le_5']:.1f}%"
    )
    ax.text(
        0.02, 0.025, stat_text,
        transform=ax.transAxes, va="bottom", ha="left", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor="#666666", alpha=0.88),
        zorder=20,
    )

    return contour


def make_combined_figure(lon4, lat4, pdop4, lon5, lat5, pdop5) -> None:
    """将两张PDOP热力图合并为一张并排图，并在图下注记区域外台站。"""
    # 双栏并排：总宽度约 320 mm (12.6 in)，高度约 155 mm (6.1 in)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.1), dpi=160)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.20,
                        wspace=0.28)

    # 左图：4台 ABEF
    contour4 = draw_subplot(axes[0], COMBINATION_4, lon4, lat4, pdop4)
    axes[0].set_title(f"(a) {COMBINATION_4} 组合", fontsize=14, pad=8)

    # 右图：5台 ABDEF
    contour5 = draw_subplot(axes[1], COMBINATION_5, lon5, lat5, pdop5)
    axes[1].set_title(f"(b) {COMBINATION_5} 组合", fontsize=14, pad=8)

    # 各子图独立 colorbar
    cbar4 = fig.colorbar(contour4, ax=axes[0], pad=0.02, fraction=0.046)
    cbar4.set_label("PDOP（km/s）", fontsize=12)
    cbar4.ax.tick_params(labelsize=10)

    cbar5 = fig.colorbar(contour5, ax=axes[1], pad=0.02, fraction=0.046)
    cbar5.set_label("PDOP（km/s）", fontsize=12)
    cbar5.ax.tick_params(labelsize=10)

    # 保存 PNG（600 dpi），路径保持原样
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / "q2_pdop热力图.png", dpi=600,
        bbox_inches="tight", pad_inches=0.03
    )
    plt.close(fig)
    print("合并图已保存")


def main() -> None:
    configure_fonts()

    lon4, lat4, pdop4 = compute_grid(COMBINATION_4)
    lon5, lat5, pdop5 = compute_grid(COMBINATION_5)

    make_combined_figure(lon4, lat4, pdop4, lon5, lat5, pdop5)
    print("绘图已完成")


if __name__ == "__main__":
    main()
