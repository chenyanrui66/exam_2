import numpy as np
from itertools import combinations
from scipy.optimize import least_squares

# ================= 数据（问题 1） =================
devs = {
    'A': (110.241, 27.204, 824, 100.767),
    'B': (110.780, 27.456, 727, 112.220),
    'C': (110.712, 27.785, 742, 188.020),
    'D': (110.251, 27.825, 850, 258.985),
    'E': (110.524, 27.617, 786, 118.443),
    'F': (110.467, 27.921, 678, 266.871),
    'G': (110.047, 27.121, 575, 163.024),
}
c = 340.0                        # 声速 m/s
lon0, lat0 = 110.241, 27.204     # 局部坐标原点（设备 A）
Klon, Klat = 97.304e3, 111.263e3  # 每度折合米数

# 经纬度 -> 局部东-北-天直角坐标 (m)
pos = {k: np.array([(v[0]-lon0)*Klon, (v[1]-lat0)*Klat, float(v[2])])
       for k, v in devs.items()}
tobs = {k: v[3] for k, v in devs.items()}

# ================= 核心函数 =================


def residual(v, P, T):
    """残差 r_i = 几何距离 ||S-P_i|| - 声波距离 c*(t_i-t0)，单位米。
       v = [x, y, z, t0]；P: n×3 设备坐标；T: n 个到达时刻。"""
    return np.linalg.norm(P - v[:3], axis=1) - c * (T - v[3])


def solve_toa(keys):
    """TOA 定位：线性化求初值 + 多初值非线性精炼，返回 (RMS残差, 解向量)"""
    P = np.array([pos[k] for k in keys])
    T = np.array([tobs[k] for k in keys])
    # --- 第一步：平方后两两相减，得关于 (x,y,z,t0) 的线性方程组 ---
    A, b = [], []
    for i in range(1, len(keys)):
        A.append(np.concatenate([2*(P[i]-P[0]), [-2*c**2*(T[i]-T[0])]]))
        b.append(P[i] @ P[i] - P[0] @ P[0] - c**2*(T[i]**2 - T[0]**2))
    X, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    t0_init = X[3] if np.isfinite(X[3]) else 15.0
    # --- 第二步：多初值 LM 精炼，防伪解分支 ---
    inits = [X[:3]] + [np.array([(110.5-lon0)*Klon, (27.31-lat0)*Klat, z])
                       for z in (1000.0, 5000.0, 12000.0)]
    best = None
    for x0 in inits:
        r = least_squares(residual, np.concatenate(
            [x0, [t0_init]]), args=(P, T))
        rms = np.sqrt(np.mean(r.fun**2))
        if best is None or rms < best[0]:
            best = (rms, r.x)
    return best


def lonlat(v):
    """局部坐标解 -> (经度, 纬度, 高程, t0)"""
    return lon0 + v[0]/Klon, lat0 + v[1]/Klat, v[2], v[3]


# ================= 第 1 步：4 台子集枚举，发现异常 =================
print("== 4 台组合枚举（仅列物理合理解）==")
for combo in combinations(devs, 4):
    rms, v = solve_toa(list(combo))
    lo, la, z, t0 = lonlat(v)
    if rms < 500 and 0 < z < 20000 and -60 < t0 < 300:
        print(f"{''.join(combo)}: RMS={rms:6.1f}m  ({lo:.4f},{la:.4f}) "
              f"z={z:6.0f}m t0={t0:6.2f}s")

# ================= 第 2 步：残差检验，确认 D、F 异常 =================
rms5, v5 = solve_toa(list('ABCEG'))
print("\n== ABCEG 五台联合解 ==")
print("经度 %.5f  纬度 %.5f  高程 %.0f m  t0 %.3f s  RMS %.1f m"
      % (*lonlat(v5), rms5))
P_all = np.array([pos[k] for k in devs])
T_all = np.array([tobs[k] for k in devs])
for k, r in zip(devs, residual(v5, P_all, T_all)):
    print(f"  设备{k}: 残差 {r:+9.1f} m ({r/c:+6.2f} s)")

# ================= 第 3 步：子集投票 =================
print("\n== {A,B,C,E,G} 内 4 台子集投票 ==")
for combo in combinations('ABCEG', 4):
    rms, v = solve_toa(list(combo))
    lo, la, z, t0 = lonlat(v)
    print(f"{''.join(combo)}: ({lo:.4f},{la:.4f}) z={z:5.0f}m t0={t0:5.2f}s")

# ================= 第 4 步：蒙特卡洛噪声散布评估 =================
rng = np.random.default_rng(0)
keys5 = list('ABCEG')
P5 = np.array([pos[k] for k in keys5])
T5 = np.array([tobs[k] for k in keys5])


def monte_carlo(coord_noise_deg, time_noise_s, n=300):
    outs = []
    for _ in range(n):
        dP = P5 + rng.uniform(-coord_noise_deg, coord_noise_deg, P5.shape) \
            * np.array([Klon, Klat, 0.0])   # 高程视为精确
        dT = T5 + rng.uniform(-time_noise_s, time_noise_s, T5.shape)
        r = least_squares(residual, v5.copy(), args=(dP, dT))
        outs.append(r.x)
    outs = np.array(outs)
    dh = np.hypot(outs[:, 0]-v5[0], outs[:, 1]-v5[1])
    return np.percentile(dh, 95), outs[:, 2].std()


r95a, zsa = monte_carlo(0.0005, 0.0005)   # 坐标量化 + 时刻舍入
r95b, zsb = monte_carlo(0.0,    0.5)      # ±0.5 s 时间误差（问题4水平）
print(f"\n坐标量化噪声下解的 95% 散布半径 ≈ {r95a:.0f} m（高程 ±{zsa:.0f} m）")
print(f"±0.5s 时间噪声下解的 95% 散布半径 ≈ {r95b:.0f} m（高程 ±{zsb:.0f} m）")
