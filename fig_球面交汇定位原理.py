import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm

# ---------- 字体自动检测（优先宋体/衬线体）----------


def setup_chinese_font():
    candidates = [
        'SimSun', 'STSong', 'AR PL UMing CN', 'Noto Serif CJK SC',
        'Noto Serif CJK TC', 'Noto Serif CJK JP', 'Noto Serif CJK',
        'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Microsoft YaHei', 'PingFang SC'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    for font in fm.fontManager.ttflist:
        lname = font.name.lower()
        if any(k in lname for k in ['song', 'simsun', 'serif cjk', 'uming', 'yahei', 'pingfang']):
            return font.name
    return 'sans-serif'


# ---------- 全局样式（基准字号 14）----------
BASE_SIZE = 16.5
COLORS = {
    'bg': '#FFFFFF', 'bg_panel': '#F5F5F0', 'grid': '#E8E8E0',
    'text': '#000000', 'text_light': '#7F8C8D',
    'sphere1': '#81B29A', 'sphere2': '#6B8E9F', 'sphere3': '#D4A373', 'sphere4': '#C9ADA7',
    'device': '#264653', 'solution': '#E07A5F', 'line_light': '#BDC3C7',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['bg_panel'],
    'axes.edgecolor': COLORS['text_light'], 'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'], 'xtick.color': COLORS['text_light'], 'ytick.color': COLORS['text_light'],
    'grid.color': COLORS['grid'], 'grid.alpha': 0.5, 'axes.grid': True,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.facecolor': COLORS['bg'],
    'font.family': [setup_chinese_font(), 'sans-serif'], 'axes.unicode_minus': False,
    'axes.titlesize': BASE_SIZE + 2, 'axes.labelsize': BASE_SIZE,
    'xtick.labelsize': BASE_SIZE, 'ytick.labelsize': BASE_SIZE,
    'legend.fontsize': BASE_SIZE, 'mathtext.fontset': 'stix',
})

# ---------- 数据 ----------
devices = {'A': np.array([0, 0]), 'B': np.array(
    [6, 2]), 'C': np.array([3, 5.5]), 'D': np.array([-2, 3])}
source = np.array([2.5, 2.5])
radii = {name: np.linalg.norm(source - pos) for name, pos in devices.items()}

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(11, 8.5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg_panel'])

sphere_colors = [COLORS['sphere1'], COLORS['sphere2'],
                 COLORS['sphere3'], COLORS['sphere4']]
for i, (name, pos) in enumerate(devices.items()):
    r = radii[name]
    color = sphere_colors[i]
    ax.add_patch(Circle(pos, r, fill=True, facecolor=color,
                 edgecolor='none', alpha=0.10, zorder=1))
    ax.add_patch(Circle(pos, r, fill=False, edgecolor=color,
                 linewidth=1.8, linestyle='--', alpha=0.75, zorder=2))
    angle = np.arctan2(source[1] - pos[1], source[0] - pos[0])
    mid_r = pos + 0.55 * r * np.array([np.cos(angle), np.sin(angle)])
    ax.annotate(f'$r_{name}=c(t_{name}-t_0)$', xy=mid_r, fontsize=BASE_SIZE-1,
                color=color, fontweight=400, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color, alpha=0.9, linewidth=0.8))

for name, pos in devices.items():
    ax.plot(pos[0], pos[1], 'o', markersize=13, color=COLORS['device'],
            markeredgecolor='white', markeredgewidth=2.5, zorder=5)
    ax.annotate(f'$S_{name}$', xy=pos, xytext=(pos[0]+0.45, pos[1]+0.45),
                fontsize=BASE_SIZE+1, fontweight=400, color=COLORS['device'], ha='left', va='bottom')

ax.plot(source[0], source[1], '*', markersize=22, color=COLORS['solution'],
        markeredgecolor='white', markeredgewidth=2.5, zorder=6)
ax.annotate('音爆点 $P(x,y,z)$\\n（球面交汇）', xy=source, xytext=(source[0]+1.5, source[1]+1.8),
            fontsize=BASE_SIZE+1, fontweight=400, color=COLORS['solution'], ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color=COLORS['solution'], lw=1.8),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF5F2', edgecolor=COLORS['solution'], alpha=0.95, linewidth=1.2))

for pos in devices.values():
    ax.annotate('', xy=source, xytext=pos,
                arrowprops=dict(arrowstyle='->', color=COLORS['line_light'], lw=1.0, ls='-', alpha=0.4))

ax.add_patch(Circle(source, 0.7, fill=True,
             facecolor=COLORS['solution'], alpha=0.06, zorder=0))

legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=COLORS['device'], markersize=10, label='监测设备'),
    Line2D([0], [0], marker='*', color='w',
           markerfacecolor=COLORS['solution'], markersize=14, label='音爆点'),
    Line2D([0], [0], color=COLORS['sphere1'],
           lw=2, ls='--', alpha=0.75, label='球面'),
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white',
          edgecolor=COLORS['grid'], fancybox=True)

ax.set_title('图1  球面交汇定位原理（TOA截面示意）', fontsize=BASE_SIZE + 2,
             fontweight=400, pad=18, color=COLORS['text'])


props = dict(boxstyle='round,pad=0.6', facecolor='white',
             edgecolor=COLORS['grid'], alpha=0.95, linewidth=1)
ax.text(0.85, 0.1, r'$\|\mathbf{p}-\mathbf{s}_1\|^2 = c^2(t_1-t_0)^2$', transform=ax.transAxes, fontsize=BASE_SIZE,
        verticalalignment='top', horizontalalignment='center', bbox=props, color=COLORS['text'], linespacing=1.6)

ax.set_xlabel('东向距离 (km)', fontsize=BASE_SIZE)
ax.set_ylabel('北向距离 (km)', fontsize=BASE_SIZE)
ax.set_xlim(-4.5, 9.5)
ax.set_ylim(-2, 8.5)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('./output/figs/球面交汇定位原理.png',
            bbox_inches='tight', facecolor=COLORS['bg'])
