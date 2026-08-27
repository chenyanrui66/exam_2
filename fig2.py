import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import font_manager as fm

# ---------- 字体自动检测 ----------


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


# ---------- 全局样式 ----------
BASE_SIZE = 14
COLORS = {
    'bg': '#FAFAFA', 'bg_panel': '#F5F5F0', 'grid': '#E8E8E0',
    'text': '#2C3E50', 'text_light': '#7F8C8D',
    'accent': '#E07A5F', 'accent2': '#3D9970',
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
    'xtick.labelsize': BASE_SIZE - 2, 'ytick.labelsize': BASE_SIZE - 2,
    'legend.fontsize': BASE_SIZE - 2, 'mathtext.fontset': 'stix',
})

# ---------- 数据 ----------
S1, S2, S3, S4 = np.array([0, 0]), np.array(
    [5, 1]), np.array([2, 4.5]), np.array([-1.5, 3])
P_true = np.array([2.2, 2.3])
r1, r2, r3, r4 = [np.linalg.norm(P_true - s) for s in [S1, S2, S3, S4]]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor(COLORS['bg'])

# ===== 子图1：两球面相交 → 根平面 =====
ax1 = axes[0]
ax1.set_facecolor(COLORS['bg_panel'])
for s, r, c in [(S1, r1, COLORS['sphere1']), (S2, r2, COLORS['sphere2'])]:
    ax1.add_patch(Circle(s, r, fill=True, facecolor=c,
                  edgecolor='none', alpha=0.12, zorder=1))
    ax1.add_patch(Circle(s, r, fill=False, edgecolor=c,
                  linewidth=2, linestyle='--', alpha=0.8, zorder=2))

ax1.plot(*S1, 'o', markersize=11,
         color=COLORS['device'], markeredgecolor='white', markeredgewidth=2, zorder=5)
ax1.plot(*S2, 'o', markersize=11,
         color=COLORS['device'], markeredgecolor='white', markeredgewidth=2, zorder=5)
ax1.annotate('$S_1$', xy=S1, xytext=(
    S1[0]-0.6, S1[1]-0.5), fontsize=BASE_SIZE-0.5, color=COLORS['device'], fontweight='bold')
ax1.annotate('$S_2$', xy=S2, xytext=(
    S2[0]+0.3, S2[1]-0.5), fontsize=BASE_SIZE-0.5, color=COLORS['device'], fontweight='bold')

d = np.linalg.norm(S2 - S1)
a = (r1**2 - r2**2 + d**2) / (2 * d)
h = np.sqrt(max(0, r1**2 - a**2))
P2 = S1 + a * (S2 - S1) / d
perp = np.array([-(S2[1]-S1[1]), S2[0]-S1[0]]) / d
i1, i2 = P2 + h * perp, P2 - h * perp
ax1.plot([i1[0], i2[0]], [i1[1], i2[1]], '-',
         color=COLORS['accent2'], linewidth=3, alpha=0.85, zorder=4)
ax1.plot(*P_true, '*', markersize=16,
         color=COLORS['solution'], markeredgecolor='white', markeredgewidth=2, zorder=6)

mid = (i1 + i2) / 2
ax1.annotate('根平面交线\n（两球面交集）', xy=mid, xytext=(mid[0]+1.5, mid[1]+1.2),
             fontsize=BASE_SIZE-1.5, color=COLORS['accent2'], fontweight='bold', ha='left', va='bottom',
             arrowprops=dict(arrowstyle='->', color=COLORS['accent2'], lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=COLORS['accent2'], alpha=0.9, linewidth=1))

ax1.set_title('(a) 两球面相交 → 根平面（圆）', fontsize=BASE_SIZE,
              fontweight='bold', pad=10, color=COLORS['text'])
ax1.set_xlim(-2, 7.5)
ax1.set_ylim(-1.5, 6)
ax1.set_aspect('equal')
ax1.set_xlabel('东向距离 (km)', fontsize=BASE_SIZE-0.5)
ax1.set_ylabel('北向距离 (km)', fontsize=BASE_SIZE-0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(0.5, -0.14,
         r'$\|\mathbf{p}-\mathbf{s}_1\|^2 - \|\mathbf{p}-\mathbf{s}_2\|^2 = c^2(t_1^2-t_2^2)$',
         transform=ax1.transAxes, fontsize=BASE_SIZE-3, ha='center', va='top',
         color=COLORS['text_light'])
ax1.text(0.5, -0.20,
         '二次项抵消 → 线性方程（根平面）',
         transform=ax1.transAxes, fontsize=BASE_SIZE-3, ha='center', va='top',
         color=COLORS['text_light'])

# ===== 子图2：三个根平面 → 交汇直线 =====
ax2 = axes[1]
ax2.set_facecolor(COLORS['bg_panel'])
for s, r, c in [(S1, r1, COLORS['sphere1']), (S2, r2, COLORS['sphere2']), (S3, r3, COLORS['sphere3'])]:
    ax2.add_patch(Circle(s, r, fill=True, facecolor=c,
                  edgecolor='none', alpha=0.08, zorder=1))
    ax2.add_patch(Circle(s, r, fill=False, edgecolor=c,
                  linewidth=1.5, linestyle='--', alpha=0.5, zorder=1))

for s, label in [(S1, '$S_1$'), (S2, '$S_2$'), (S3, '$S_3$')]:
    ax2.plot(*s, 'o', markersize=9,
             color=COLORS['device'], markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax2.annotate(label, xy=s, xytext=(
        s[0]+0.25, s[1]+0.25), fontsize=BASE_SIZE-1.5, color=COLORS['device'])

ax2.plot([i1[0], i2[0]], [i1[1], i2[1]], '-',
         color=COLORS['sphere1'], linewidth=2, alpha=0.5, zorder=2)

d13 = np.linalg.norm(S3 - S1)
a13 = (r1**2 - r3**2 + d13**2) / (2 * d13)
h13 = np.sqrt(max(0, r1**2 - a13**2))
P13 = S1 + a13 * (S3 - S1) / d13
perp13 = np.array([-(S3[1]-S1[1]), S3[0]-S1[0]]) / d13
ax2.plot([*(P13 + h13 * perp13)], [*(P13 - h13 * perp13)], '-',
         color=COLORS['sphere3'], linewidth=2, alpha=0.5, zorder=2)

d23 = np.linalg.norm(S3 - S2)
a23 = (r2**2 - r3**2 + d23**2) / (2 * d23)
h23 = np.sqrt(max(0, r2**2 - a23**2))
P23 = S2 + a23 * (S3 - S2) / d23
perp23 = np.array([-(S3[1]-S2[1]), S3[0]-S2[0]]) / d23
ax2.plot([*(P23 + h23 * perp23)], [*(P23 - h23 * perp23)], '-',
         color=COLORS['sphere2'], linewidth=2, alpha=0.5, zorder=2)

ax2.plot(*P_true, '*', markersize=18,
         color=COLORS['solution'], markeredgecolor='white', markeredgewidth=2, zorder=6)
ax2.annotate('三个根平面\n交于一条直线', xy=P_true, xytext=(P_true[0]+1.8, P_true[1]+1.5),
             fontsize=BASE_SIZE-1.5, color=COLORS['accent'], fontweight='bold', ha='left', va='bottom',
             arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor=COLORS['accent'], alpha=0.9, linewidth=1))

ax2.set_title('(b) 三个根平面 → 交汇直线', fontsize=BASE_SIZE,
              fontweight='bold', pad=10, color=COLORS['text'])
ax2.set_xlim(-2, 7.5)
ax2.set_ylim(-1.5, 6)
ax2.set_aspect('equal')
ax2.set_xlabel('东向距离 (km)', fontsize=BASE_SIZE-0.5)
ax2.set_ylabel('北向距离 (km)', fontsize=BASE_SIZE-0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(0.5, -0.14,
         r'$\mathbf{p}(\tau) = \mathbf{p}_0 + \tau \mathbf{p}_1$',
         transform=ax2.transAxes, fontsize=BASE_SIZE-3, ha='center', va='top',
         color=COLORS['text_light'])
ax2.text(0.5, -0.20,
         '位置表示为时刻的线性函数',
         transform=ax2.transAxes, fontsize=BASE_SIZE-3, ha='center', va='top',
         color=COLORS['text_light'])


fig.suptitle('四站差分法几何分解',
             fontsize=BASE_SIZE + 2, fontweight='bold', y=0.95, color=COLORS['text'])

plt.tight_layout()
plt.savefig('fig2_radical_plane.png',
            bbox_inches='tight', facecolor=COLORS['bg'])
