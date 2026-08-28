
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from csv_writer import read_sections

_here = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_here, "output", "q4_results.csv")
FIG_DIR = os.path.join(_here, "output", "figs")


def _zh_font():
    keys = ("CJK", "Hei", "Song", "WenQuanYi", "SimSun", "SimHei",
            "Microsoft YaHei", "PingFang", "Source Han")
    for f in font_manager.fontManager.ttflist:
        if any(k in f.name for k in keys):
            return font_manager.FontProperties(fname=f.fname)
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
              "/System/Library/Fonts/PingFang.ttc"):
        if os.path.exists(p):
            return font_manager.FontProperties(fname=p)
    return None


ZH = _zh_font()


def to_xy_km(lon_deg, lat_deg):

    return (lon_deg - 110) * 97304.0 / 1000.0, (lat_deg - 27) * 111263.0 / 1000.0


def main():
    sec = read_sections(CSV_PATH)
    sta = sec["台网台站"]
    deb = sec["残骸音爆点真值"]
    d7 = sec["误差明细_原7台"]
    d20 = sec["误差明细_20台加密"]

    sx, sy = to_xy_km(sta["经度(°E)"].to_numpy(), sta["纬度(°N)"].to_numpy())
    dx, dy = to_xy_km(deb["经度(°E)"].to_numpy(), deb["纬度(°N)"].to_numpy())
    grp = sta["组别"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    ax = axes[0]
    m_old = grp == "原台站"
    ax.scatter(sx[m_old], sy[m_old], marker='^', s=90, c='#1f77b4',
               label='原7台设备', zorder=3)
    for i in np.where(m_old)[0]:
        ax.annotate(sta["台站"].iloc[i], (sx[i], sy[i]),
                    textcoords="offset points", xytext=(6, 5), fontsize=9)
    m_new = ~m_old
    ax.scatter(sx[m_new], sy[m_new], marker='s', s=55, c='#2ca02c',
               label='新增13台（中心1+内环4+外环8）', zorder=3)
    ax.scatter(dx, dy, marker='*', s=260, c='#d62728',
               label='4个残骸音爆点（地面投影）', zorder=4)
    for j in range(len(dx)):
        ax.annotate(deb["残骸"].iloc[j], (dx[j], dy[j]),
                    textcoords="offset points", xytext=(8, -3),
                    fontsize=10, color='#d62728')

    cx, cy = sx[grp == "中心"][0], sy[grp == "中心"][0]
    th_c = np.deg2rad(np.arange(0, 361, 5))
    for gname in ("内环", "外环"):
        gx, gy = sx[grp == gname], sy[grp == gname]
        r_km = float(np.mean(np.hypot(gx - cx, gy - cy)))
        ax.plot(cx + r_km * np.cos(th_c), cy + r_km * np.sin(th_c),
                '--', c='#2ca02c', lw=0.8, alpha=0.6)
    ax.set_xlabel('东向 x (km，110°E为原点)', fontproperties=ZH)
    ax.set_ylabel('北向 y (km，27°N为原点)', fontproperties=ZH)
    ax.set_title('加密台网布局：原7台 + 新增13台 = 20台', fontproperties=ZH)
    ax.legend(loc='lower right', fontsize=9, prop=ZH)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[1]
    for df, lab, c in [(d7, '原7台（修正后）', '#1f77b4'),
                       (d20, '20台加密', '#d62728')]:
        v = np.sort(df["3D误差(m)"].to_numpy())
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v / 1000, cdf, label=lab, lw=2, color=c)
    ax.axvline(1.0, color='k', ls='--', lw=1.2, label='1 km 精度要求')
    ax.set_xlabel('3D定位误差 (km)', fontproperties=ZH)
    ax.set_ylabel('经验累积概率', fontproperties=ZH)
    ax.set_title('蒙特卡洛300次：3D误差分布对比', fontproperties=ZH)
    ax.legend(fontsize=9, prop=ZH)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 2.1)
    p95_7 = np.percentile(d7["3D误差(m)"], 95) / 1000
    p95_20 = np.percentile(d20["3D误差(m)"], 95) / 1000
    ax.annotate(f'95%分位: {p95_7:.2f} km', xy=(p95_7, 0.95), xytext=(1.25, 0.72),
                arrowprops=dict(arrowstyle='->', color='#1f77b4'),
                color='#1f77b4', fontsize=9, fontproperties=ZH)
    ax.annotate(f'95%分位: {p95_20:.2f} km', xy=(p95_20, 0.95), xytext=(0.62, 0.55),
                arrowprops=dict(arrowstyle='->', color='#d62728'),
                color='#d62728', fontsize=9, fontproperties=ZH)

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, '问题4_台网布局与误差对比.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
