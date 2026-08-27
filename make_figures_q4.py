# -*- coding: utf-8 -*-
"""问题四论文插图：加密台网布局、噪声幅度扫描退化曲线（学术风格）"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.serif'] = ['SimSun', 'NSimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 11

Klon, Klat = 97.304e3, 111.263e3


def to_ll(x, y):
    return 110 + x / Klon, 27 + y / Klat


# ---------- 台站坐标（与 问题4_误差修正与加密台网方案.py 一致） ----------
lon = np.array([110.241, 110.783, 110.762, 110.251, 110.524, 110.467, 110.047])
lat = np.array([27.204, 27.456, 27.785, 28.025, 27.617, 28.081, 27.521])
center_xy = np.array([(110.5 - 110) * Klon, (27.65 - 27) * Klat])


def ring(n, r_km, az0=0.0):
    ang = np.deg2rad(az0 + np.arange(n) * 360.0 / n)
    return np.column_stack((center_xy[0] + r_km * 1000 * np.cos(ang),
                            center_xy[1] + r_km * 1000 * np.sin(ang)))


inner = ring(4, 15, 45)
outer = ring(8, 45, 22.5)

debris = np.array([(110.500001, 27.309998), (110.499999, 27.949998),
                   (110.300000, 27.650000), (110.699999, 27.650000)])

# ---------- 图1：加密台网布局 ----------
fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=300)
ax.scatter(lon, lat, marker='^', s=70, facecolors='none', edgecolors='k',
           linewidths=1.2, zorder=4, label='原7台')
for k, x, y in zip('ABCDEFG', lon, lat):
    ax.text(x + 0.008, y + 0.010, k, fontsize=10)

c_lon, c_lat = to_ll(*center_xy)
ax.scatter(c_lon, c_lat, marker='P', s=90, c='k', zorder=5, label='中心站')
ax.text(c_lon + 0.010, c_lat - 0.028, 'N01', fontsize=10)

in_lon, in_lat = to_ll(inner[:, 0], inner[:, 1])
ax.scatter(in_lon, in_lat, marker='s', s=45, facecolors='none',
           edgecolors='k', linewidths=1.1, zorder=4, label='内环4台（15 km）')

ou_lon, ou_lat = to_ll(outer[:, 0], outer[:, 1])
ax.scatter(ou_lon, ou_lat, marker='o', s=45, facecolors='none',
           edgecolors='k', linewidths=1.1, zorder=4, label='外环8台（45 km）')

ax.scatter(debris[:, 0], debris[:, 1], marker='*', s=200, c='k', zorder=5,
           label='音爆点（地面投影）')
for j, (x, y) in enumerate(debris):
    ax.text(x + 0.012, y + 0.012, f'#{j+1}', fontsize=10)

th = np.linspace(0, 2 * np.pi, 200)
for r_km in (15, 45):
    cx, cy = to_ll(center_xy[0] + r_km * 1000 * np.cos(th),
                   center_xy[1] + r_km * 1000 * np.sin(th))
    ax.plot(cx, cy, color='0.55', lw=0.9, ls='--', zorder=2)

ax.set_xlabel('经度 (°E)')
ax.set_ylabel('纬度 (°N)')
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.set_aspect(1.0 / np.cos(np.deg2rad(27.65)))
fig.tight_layout()
fig.savefig('fig_q4_network.pdf')
plt.close(fig)

# ---------- 图2：噪声幅度扫描退化曲线（数据来自压力测试实际运行） ----------
delta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
p95 = np.array([253, 523, 833, 1010, 1233, 1495, 1892, 2089, 2305, 2447])
succ = np.array([100, 100, 100, 100, 100, 100, 100, 100, 92.5, 82.5])

fig, ax1 = plt.subplots(figsize=(6.2, 4.2), dpi=300)
ax1.plot(delta, p95 / 1000, 'o-', color='k', ms=4.5, lw=1.2, label='3D误差95%分位')
ax1.axhline(1.0, color='0.45', ls='--', lw=1.0)
ax1.text(0.105, 1.04, '1 km 精度线', fontsize=9.5, color='0.30')
ax1.axvline(0.4, color='0.70', ls=':', lw=1.0)
ax1.text(0.405, 0.12, '精度崩溃点 $\\Delta\\approx0.4$ s', fontsize=9.5,
         color='0.30', rotation=90, va='bottom')
ax1.set_xlabel('噪声界 $\\Delta$ (s)')
ax1.set_ylabel('3D误差95%分位 (km)')
ax1.set_xlim(0.05, 1.05)
ax1.set_ylim(0, 2.6)

ax2 = ax1.twinx()
ax2.plot(delta, succ, 's--', color='0.40', ms=4, lw=1.0, label='关联成功率')
ax2.set_ylabel('关联成功率 (%)')
ax2.set_ylim(60, 105)
ax2.tick_params(axis='y', direction='in')
ax2.axvline(0.9, color='0.85', ls=':', lw=1.0)
ax2.text(0.89, 63, '关联失败首现 $\\Delta=0.9$ s', fontsize=9.5,
         color='0.30', ha='right')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9.5, frameon=False)
fig.tight_layout()
fig.savefig('fig_q4_sweep.pdf')
plt.close(fig)

print('已生成 fig_q4_network.pdf 与 fig_q4_sweep.pdf')
