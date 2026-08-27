# -*- coding: utf-8 -*-
"""
问题4：±0.5 s 计时误差下的模型修正、误差分析与加密台网方案
================================================================

三部分内容：
  Part A  修正模型：把问题2/3的"枚举—验真—覆盖"管道从仪器级精度（RMSE阈值0.02 s）
          修正为适应±0.5 s随机误差的统计阈值版本，并加入容错机制
          （分级阈值 + 剩余读数补全第4残骸）。
  Part B  蒙特卡洛算例：对问题3数据叠加 U(-0.5, +0.5) s 噪声，重复300次，
          统计修正模型的定位误差与关联可靠性。
  Part C  误差无法降低时的解决方案：在原7台基础上加密至20台
          （中心1台 + 内环4台 + 外环8台），用问题3的定位结果正向模拟
          新台网的音爆抵达时刻，再用两阶段修正模型反演验证（3D误差<1 km）。

输出约定：
  - 全部q4_results分节写入 ./output/问题4_q4_results.csv（相对本脚本所在目录）；
  - 控制台不打印中间过程，结束时仅打印"结果已保存"字样。

依赖：numpy、pandas、csv_writer（同目录）。
运行：python 问题4_误差修正与加密台网方案.py
"""
import itertools
import os
import time

import numpy as np
import pandas as pd

from csv_writer import CsvCollector

C = 340.0            # 声速 m/s
DELTA = 0.5          # 计时误差界：U(-0.5, +0.5) s
NAMES = np.array(list("ABCDEFG"))
ALL7 = np.arange(7)

# ==================== 1. 原始数据与真值 ====================
# 问题3的7台设备坐标
lon = np.array([110.241, 110.783, 110.762, 110.251, 110.524, 110.467, 110.047])
lat = np.array([27.204, 27.456, 27.785, 28.025, 27.617, 28.081, 27.521])
alt = np.array([824.0, 727.0, 742.0, 850.0, 786.0, 678.0, 575.0])
# 问题3的4组音爆抵达时刻
T = np.array([
    [100.767, 164.229, 214.850, 270.065],
    [92.453, 112.220, 169.362, 196.583],
    [75.560, 110.696, 156.936, 188.020],
    [94.653, 141.409, 196.517, 258.985],
    [78.600, 86.216, 118.443, 126.669],
    [67.274, 166.270, 175.482, 266.871],
    [103.738, 163.024, 206.789, 210.306],
])
# 局部东-北-天直角坐标系（原点 110°E, 27°N），单位 m
S = np.column_stack(((lon - 110) * 97304.0, (lat - 27) * 111263.0, alt))


def to_local(lon_d, lat_d, z_m):
    """经纬度+高程 → 局部直角坐标 (m)"""
    return np.array([(lon_d - 110) * 97304.0, (lat_d - 27) * 111263.0, z_m])


# 问题3解出的4个残骸真值（正向复现题目数据偏差<0.5 ms，确认为出题真值）
truth = np.array([
    [*to_local(110.500001, 27.309998, 12514.0), 11.9999],   # 残骸1
    [*to_local(110.499999, 27.949998, 11529.0), 13.0014],   # 残骸2
    [*to_local(110.300000, 27.650000, 11478.0), 14.0000],   # 残骸3
    [*to_local(110.699999, 27.650000, 13468.0), 15.0000],   # 残骸4
])
TRUTH_LL = [  # 真值的经纬度表示（供CSV输出）
    (110.500001, 27.309998, 12514.0, 11.9999),
    (110.499999, 27.949998, 11529.0, 13.0014),
    (110.300000, 27.650000, 11478.0, 14.0000),
    (110.699999, 27.650000, 13468.0, 15.0000),
]

# 解空间边界（x, y, z, tau）
LOW = np.array([-50000.0, -50000.0, 0.0, -400.0])
HIGH = np.array([150000.0, 220000.0, 120000.0, 0.0])

# 预计算全部 4^7 = 16384 个"单事件组合"的读数索引
combo_idx = np.array(list(itertools.product(range(4), repeat=7)))

# 选条件数最低的4组"4站种子组合"（多组种子站：抗噪声下解析种子退化）
_combos4 = []
for _ids in itertools.combinations(range(7), 4):
    _ii = np.asarray(_ids)
    _combos4.append((np.linalg.cond(2 * (S[_ii[1:]] - S[_ii[0]])), _ii))
_combos4.sort(key=lambda x: x[0])
SEED_GROUPS = [ii for _, ii in _combos4[:4]]
SEED7 = SEED_GROUPS[0]   # 条件数最好的一组，用于补全机制


# ==================== 2. 修正模型核心 ====================
def batch_seeds(Tobs, seed_groups, prescreen=3.0):
    """批量解析种子：对全部16384个组合×多组种子站，向量化求解。

    数学：四站TOA平方方程两两相减消去二次项，得 p(τ)=p0+p1·τ；
    代回任一站方程得一元二次方程，至多两个解析根。
    初筛：因果律 + 空间边界 + 种子在全站上的RMSE < prescreen。
    初筛阈值放得很宽（3 s），只起省算力作用，真正的鉴别靠精化后的统计阈值。
    """
    obs_all = Tobs[ALL7[None, :], combo_idx]      # (16384, 7)
    out = {}
    for sg in seed_groups:
        ss = S[sg]                                # (4, 3)
        tt = obs_all[:, sg]                       # (16384, 4)
        Ainv = np.linalg.inv(2 * (ss[1:] - ss[0]))
        b0 = (np.sum(ss[1:] ** 2, axis=1) - ss[0] @ ss[0]
              - C * C * (tt[:, 1:] ** 2 - tt[:, :1] ** 2))
        b1 = 2 * C * C * (tt[:, 1:] - tt[:, :1])
        p0 = b0 @ Ainv.T
        p1 = b1 @ Ainv.T
        v = p0 - ss[0]
        qa = np.sum(p1 ** 2, axis=1) - C * C
        qb = 2 * (np.sum(p1 * v, axis=1) + C * C * tt[:, 0])
        qc = np.sum(v ** 2, axis=1) - C * C * tt[:, 0] ** 2
        disc = qb * qb - 4 * qa * qc
        ok = disc >= 0
        sq = np.sqrt(np.maximum(disc, 0))
        for tau in ((-qb + sq) / (2 * qa), (-qb - sq) / (2 * qa)):
            tau = np.where(ok & np.isfinite(tau), tau, np.nan)
            p = p0 + p1 * tau[:, None]
            valid = (ok & np.all(p >= LOW[:3], axis=1)
                     & np.all(p <= HIGH[:3], axis=1)
                     & (tau <= obs_all.min(axis=1)))
            d = p[:, None, :] - S[None, :, :]
            pred = tau[:, None] + np.linalg.norm(d, axis=2) / C
            r0 = np.sqrt(np.mean((obs_all - pred) ** 2, axis=1))
            for n in np.where(valid & (r0 < prescreen))[0]:
                out.setdefault(n, []).append(np.r_[p[n], tau[n]])
    return out


def full_res(theta, obs):
    """7站TOA残差：实测 - 预测到达时刻 (s)"""
    return obs - theta[3] - np.linalg.norm(S - theta[:3], axis=1) / C


def refine(theta, obs, iters=80):
    """7站有界 Levenberg-Marquardt 精化（最小二乘，高斯意义下的最大似然）"""
    high = HIGH.copy()
    high[3] = obs.min()                    # 因果律：音爆先于一切观测
    th = np.clip(theta, LOW, high)
    lam = 1e-4
    for _ in range(iters):
        r = full_res(th, obs)
        d = th[:3] - S
        dd = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        J = np.column_stack((d / (C * dd[:, None]), np.ones(7)))
        try:
            step = np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ r)
        except np.linalg.LinAlgError:
            break
        thn = np.clip(th + step, LOW, high)
        if full_res(thn, obs) @ full_res(thn, obs) < r @ r:
            th = thn
            lam = max(lam / 3, 1e-12)
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            lam = min(lam * 10, 1e12)
    r = full_res(th, obs)
    return th, float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


def _enum_candidates(Tobs, rmse_thr, maxabs_thr):
    """枚举候选事件：批量种子 → 逐组合LM精化 → 统计阈值验真"""
    cand = []
    for n, seeds in batch_seeds(Tobs, SEED_GROUPS).items():
        obs = Tobs[ALL7, combo_idx[n]]
        th, rmse, mx = min((refine(sd, obs)
                           for sd in seeds), key=lambda x: x[1])
        if rmse < rmse_thr and mx < maxabs_thr:
            cand.append(
                dict(choice=combo_idx[n], theta=th, rmse=rmse, maxabs=mx))
    cand.sort(key=lambda q: q["rmse"])
    return cand


def _try_cover(cand):
    """精确覆盖：选4个候选使每站4个读数恰好各用一次，最小化联合RMSE²"""
    if len(cand) < 4:
        return None
    best = None
    for ids in itertools.combinations(range(len(cand)), 4):
        g = [cand[i] for i in ids]
        mat = np.array([x["choice"] for x in g])
        if not all(sorted(mat[:, i].tolist()) == [0, 1, 2, 3] for i in range(7)):
            continue
        score = sum(x["rmse"] ** 2 for x in g)
        if best is None or score < best[0]:
            best = (score, g)
    return best


def _analytic_seeds_single(obs):
    """单组合解析种子（SEED7一组种子站），供补全机制使用"""
    ss, tt = S[SEED7], obs[SEED7]
    A = 2 * (ss[1:] - ss[0])
    b0 = np.sum(ss[1:] ** 2, axis=1) - ss[0] @ ss[0] - \
        C * C * (tt[1:] ** 2 - tt[0] ** 2)
    b1 = 2 * C * C * (tt[1:] - tt[0])
    try:
        p0 = np.linalg.solve(A, b0)
        p1 = np.linalg.solve(A, b1)
    except np.linalg.LinAlgError:
        return []
    v = p0 - ss[0]
    qa = p1 @ p1 - C * C
    qb = 2 * (p1 @ v + C * C * tt[0])
    qc = v @ v - C * C * tt[0] ** 2
    out = []
    for r in np.roots([qa, qb, qc]):
        if abs(r.imag) < 1e-6:
            out.append(np.r_[p0 + p1 * r.real, r.real])
    return out


def solve_noisy(Tobs, rmse_thr=0.55, maxabs_thr=0.65):
    """修正模型主入口（容错版）。

    阈值依据（蒙特卡洛标定，U(-0.5,0.5) s 噪声、7站、4参数拟合）：
      正确关联：精化RMSE 99%分位≈0.30 s，max|残差| 99%分位≈0.54 s
      错误关联：最优RMSE 最小≈0.43 s，5%分位≈0.54 s，中位≈0.74 s
    鉴别间隙窄 → 严阈值优先（rmse<0.55 且 maxabs<0.65）；
    若候选不足，用『剩余读数补全』兜底：3个已确认候选锁定21个读数后，
    每站恰好剩1个读数，其组合必为第4残骸，用宽阈值验证即可。
    """
    # 第一级：严阈值
    cand = _enum_candidates(Tobs, rmse_thr, maxabs_thr)
    best = _try_cover(cand)
    if best is not None:
        return best, "strict"
    # 第二级：宽松阈值收候选 + 剩余读数补全
    cand = _enum_candidates(Tobs, 0.75, 0.85)
    for ids in itertools.combinations(range(min(len(cand), 8)), 3):
        g = [cand[i] for i in ids]
        mat = np.array([x["choice"] for x in g])
        if not all(len(set(mat[:, i])) == 3 for i in range(7)):
            continue                       # 同一站的读数被重复占用，跳过
        rem = []
        for i in range(7):
            leftover = [k for k in range(4) if k not in set(mat[:, i])]
            if len(leftover) != 1:
                break
            rem.append(leftover[0])
        else:
            obs_r = Tobs[ALL7, np.array(rem)]
            seeds_r = [sd for sd in _analytic_seeds_single(obs_r)
                       if np.all(sd[:3] >= LOW[:3]) and np.all(sd[:3] <= HIGH[:3])
                       and sd[3] <= obs_r.min()]
            if not seeds_r:
                continue
            th, rmse, mx = min((refine(sd, obs_r)
                               for sd in seeds_r), key=lambda x: x[1])
            if rmse < 0.9 and mx < 1.0:    # 宽阈值验证（物理上它必是第4残骸）
                g4 = g + [dict(choice=np.array(rem), theta=th,
                               rmse=rmse, maxabs=mx)]
                return (sum(x["rmse"] ** 2 for x in g4), g4), "completed"
    return None, "failed"


# ==================== Part B：蒙特卡洛算例 ====================
def add_noise7(rng):
    """对问题3数据叠加 U(-0.5, +0.5) s 噪声；设备按带噪时刻先后重新编号读数"""
    return np.sort(T + rng.uniform(-DELTA, DELTA, size=T.shape), axis=1)


def evaluate(solutions):
    """把解出的4个事件按最近真值匹配，返回 [4残骸 × (水平,高程,3D,Δτ)] 误差"""
    errs = np.full((4, 4), np.nan)
    used = set()
    for th in solutions:
        d3 = np.linalg.norm(truth[:, :3] - th[:3], axis=1)
        j = int(np.argmin(d3))
        if j in used:
            return None                        # 两个事件抢同一真值 → 关联混淆
        used.add(j)
        errs[j] = [np.linalg.norm(truth[j, :2] - th[:2]),
                   abs(th[2] - truth[j, 2]), d3[j], abs(th[3] - truth[j, 3])]
    return errs


def part_b(nmc=300, seed=555):
    """7台现状蒙特卡洛。返回 (误差数组, 成功数, 失败数, 模式统计dict)。"""
    rng = np.random.default_rng(seed)
    E = np.full((nmc, 4, 4), np.nan)
    fail, modes = 0, {}
    for n in range(nmc):
        best, mode = solve_noisy(add_noise7(rng))
        modes[mode] = modes.get(mode, 0) + 1
        if best is None:
            fail += 1
            continue
        e = evaluate([x["theta"] for x in best[1]])
        if e is None:
            fail += 1
        else:
            E[n] = e
    ok = ~np.isnan(E[:, 0, 0])
    return E[ok], int(ok.sum()), fail, modes


# ==================== Part C：加密台网方案 ====================
center = to_local(110.5, 27.65, 800.0)     # 落区中心（问题3四残骸的几何中心）


def ring(n, r_km, az0=0.0, z=750.0):
    """以落区中心为圆心的环形台阵（方位角自正东起逆时针）"""
    ang = np.deg2rad(az0 + np.arange(n) * 360.0 / n)
    return np.column_stack((center[0] + r_km * 1000 * np.cos(ang),
                            center[1] + r_km * 1000 * np.sin(ang),
                            np.full(n, z)))


# 推荐台网：原7台 + 中心1台 + 内环4台(15 km) + 外环8台(45 km) = 20台
S_NEW = np.vstack([S, center[None, :], ring(4, 15, 45), ring(8, 45, 22.5)])
M = len(S_NEW)
# 正向模拟：真值 → 20台无噪声到达时刻
T20_TRUE = np.array([[truth[j, 3] + np.linalg.norm(S_NEW[i] - truth[j, :3]) / C
                      for j in range(4)] for i in range(M)])


def pdop_at(theta, S_arr):
    """几何精度因子 PDOP (km/s)：σ_3D ≈ PDOP × σ_t"""
    d = theta[:3] - S_arr
    dd = np.linalg.norm(d, axis=1)
    H = np.column_stack((d / (C * dd[:, None]), np.ones(len(S_arr))))
    H[:, :3] *= 1000.0
    Q = np.linalg.pinv(H.T @ H)
    return float(np.sqrt(np.trace(Q[:3, :3])))


def refine_M(theta, obs, S_arr, iters=80):
    """M站有界LM精化（obs 为关联到本残骸的读数，S_arr 为对应站坐标）"""
    m = len(obs)
    high = HIGH.copy()
    high[3] = obs.min()
    th = np.clip(theta, LOW, high)
    lam = 1e-4
    for _ in range(iters):
        r = obs - th[3] - np.linalg.norm(S_arr - th[:3], axis=1) / C
        d = th[:3] - S_arr
        dd = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        J = np.column_stack((d / (C * dd[:, None]), np.ones(m)))
        try:
            step = np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ r)
        except np.linalg.LinAlgError:
            break
        thn = np.clip(th + step, LOW, high)
        rn = obs - thn[3] - np.linalg.norm(S_arr - thn[:3], axis=1) / C
        if rn @ rn < r @ r:
            th = thn
            lam = max(lam / 3, 1e-12)
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            lam = min(lam * 10, 1e12)
    return th, float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


def solve_dense(Tobs20, w1=6.0, w2=1.5):
    """两阶段修正模型（适配加密台网，避免 4^M 枚举爆炸）：
    阶段1：原7台子网走『枚举—验真—覆盖』管道 → 4个粗解（关联判别力最强）；
    阶段2：粗解预测全台网到达时刻 → 宽窗口最近邻关联 → 全站联合精化
           → 紧窗口重关联 → 再精化（迭代最近点思想，消除个别站的错配）。
    """
    best, _ = solve_noisy(Tobs20[:7])
    if best is None:
        return None
    thetas = [x["theta"].copy() for x in sorted(
        best[1], key=lambda q: q["theta"][3])]
    for w in (w1, w2):
        new_thetas = []
        for th in thetas:
            pred = th[3] + np.linalg.norm(S_NEW - th[:3], axis=1) / C
            obs_l, sta_l = [], []
            for i in range(M):
                d = np.abs(Tobs20[i] - pred[i])
                k = int(np.argmin(d))
                if d[k] < w:
                    obs_l.append(Tobs20[i, k])
                    sta_l.append(i)
            th2, _, _ = refine_M(th, np.array(obs_l), S_NEW[np.array(sta_l)])
            new_thetas.append(th2)
        thetas = new_thetas
    return thetas


def add_noise20(rng):
    return np.sort(T20_TRUE + rng.uniform(-DELTA, DELTA, size=T20_TRUE.shape), axis=1)


def part_c(nmc=300, seed=2026):
    """20台加密台网蒙特卡洛。返回 (误差数组, 成功数, 失败数)。"""
    rng = np.random.default_rng(seed)
    E = np.full((nmc, 4, 4), np.nan)
    fail = 0
    for n in range(nmc):
        thetas = solve_dense(add_noise20(rng))
        if thetas is None:
            fail += 1
            continue
        e = evaluate(thetas)
        if e is None:
            fail += 1
        else:
            E[n] = e
    ok = ~np.isnan(E[:, 0, 0])
    return E[ok], int(ok.sum()), fail


# ==================== 3. 数据汇总与CSV输出 ====================
ERR_COLS = ["水平误差(m)", "高程误差(m)", "3D误差(m)", "时刻误差(ms)"]


def detail_df(V, nmc):
    """误差数组 (n成功,4,4) → 长表DataFrame（每次试验×每残骸一行）"""
    recs = []
    for t in range(V.shape[0]):
        for j in range(4):
            recs.append({
                "试验序号": t + 1, "残骸": f"#{j + 1}",
                "水平误差(m)": V[t, j, 0], "高程误差(m)": V[t, j, 1],
                "3D误差(m)": V[t, j, 2], "时刻误差(ms)": V[t, j, 3] * 1000.0,
            })
    return pd.DataFrame(recs)


def stats_df(V):
    """误差数组 → 统计汇总DataFrame（4个指标 × 均值/中位/95%分位/最大）"""
    rows = []
    for k, lab in enumerate(ERR_COLS):
        v = V[:, :, k].ravel() * (1000.0 if k == 3 else 1.0)
        rows.append({"指标": lab, "均值": v.mean(), "中位": np.median(v),
                     "95%分位": np.percentile(v, 95), "最大": v.max()})
    return pd.DataFrame(rows)


def demo_case_df(thetas):
    """单次算例：解出的事件列表 → DataFrame（按音爆时刻排序，含相对真值误差）"""
    rows = []
    for th in sorted(thetas, key=lambda q: q[3]):
        d3 = np.linalg.norm(truth[:, :3] - th[:3], axis=1)
        j = int(np.argmin(d3))
        rows.append({
            "残骸": f"#{j + 1}",
            "经度(°E)": 110 + th[0] / 97304, "纬度(°N)": 27 + th[1] / 111263,
            "高程(km)": th[2] / 1000, "音爆时刻(s)": th[3],
            "3D误差(m)": d3[j], "时刻误差(ms)": abs(th[3] - truth[j, 3]) * 1000.0,
        })
    return pd.DataFrame(rows)


def build_csv(path, V7, ok7, fail7, modes7, V20, ok20, fail20, demo7, demo20):
    """汇总全部q4_results，分节写入单个CSV"""
    cc = CsvCollector()
    sig_t = DELTA / np.sqrt(3)

    # 节1：试验概况
    cc.add("试验概况", pd.DataFrame([
        {"台网": "原7台", "蒙特卡洛次数": ok7 + fail7, "成功": ok7, "失败": fail7,
         "3D误差>1km比例(%)": (V7[:, :, 2].ravel() > 1000).mean() * 100,
         "严阈值成功(strict)": modes7.get("strict", 0),
         "补全兜底(completed)": modes7.get("completed", 0)},
        {"台网": "20台加密", "蒙特卡洛次数": ok20 + fail20, "成功": ok20, "失败": fail20,
         "3D误差>1km比例(%)": (V20[:, :, 2].ravel() > 1000).mean() * 100,
         "严阈值成功(strict)": "-", "补全兜底(completed)": "-"},
    ]))

    # 节2/3：误差统计汇总
    cc.add("误差统计_原7台", stats_df(V7))
    cc.add("误差统计_20台加密", stats_df(V20))

    # 节4/5：逐次试验误差明细（绘图CDF数据源）
    cc.add("误差明细_原7台", detail_df(V7, ok7))
    cc.add("误差明细_20台加密", detail_df(V20, ok20))

    # 节6：PDOP精度预算
    layouts = {
        "原7台": S,
        "7+中心1+外环8 (16台)": np.vstack([S, center[None, :], ring(8, 45, 22.5)]),
        "7+中心1+内环4+外环8 (20台, 推荐)": S_NEW,
        "7+内环6+外环10 (23台)": np.vstack([S, ring(6, 18, 0), ring(10, 50, 18)]),
    }
    cc.add("PDOP精度预算", pd.DataFrame([
        {"布局": name, "台数": len(Sa),
         "最坏PDOP(km/s)": max(pdop_at(truth[j], Sa) for j in range(4)),
         "预估σ_3D(m)": max(pdop_at(truth[j], Sa) for j in range(4)) * sig_t * 1000}
        for name, Sa in layouts.items()
    ]))

    # 节7：台网台站坐标（绘图布局数据源）
    groups = (["原台站"] * 7 + ["中心"] + ["内环"] * 4 + ["外环"] * 8)
    sta_names = list("ABCDEFG") + [f"N{k:02d}" for k in range(1, 14)]
    cc.add("台网台站", pd.DataFrame([
        {"台站": sta_names[i], "组别": groups[i],
         "经度(°E)": 110 + S_NEW[i, 0] / 97304,
         "纬度(°N)": 27 + S_NEW[i, 1] / 111263,
         "高程(m)": S_NEW[i, 2]}
        for i in range(M)
    ]))

    # 节8：残骸音爆点真值
    cc.add("残骸音爆点真值", pd.DataFrame([
        {"残骸": f"#{j + 1}", "经度(°E)": lo, "纬度(°N)": la,
         "高程(m)": z, "音爆时刻(s)": tau}
        for j, (lo, la, z, tau) in enumerate(TRUTH_LL)
    ]))

    # 节9/10：单次算例（固定随机种子，可复现）
    cc.add("单次算例_原7台", demo7)
    cc.add("单次算例_20台加密", demo20)

    cc.save(path)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    t0 = time.time()

    # Part B：7台现状蒙特卡洛
    V7, ok7, fail7, modes7 = part_b(nmc=300)
    # Part C：20台加密台网蒙特卡洛
    V20, ok20, fail20 = part_c(nmc=300)

    # 单次算例（固定种子，与报告一致）
    best_d, _ = solve_noisy(add_noise7(np.random.default_rng(20260827)))
    demo7 = demo_case_df([x["theta"] for x in best_d[1]])
    demo20 = demo_case_df(solve_dense(
        add_noise20(np.random.default_rng(20260827))))

    # 输出CSV到 ./output/
    out_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "q4_results.csv")
    build_csv(csv_path, V7, ok7, fail7, modes7,
              V20, ok20, fail20, demo7, demo20)

    print(f"结果已保存: {csv_path}")
