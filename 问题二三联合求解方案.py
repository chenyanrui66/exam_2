"""
问题二/三联合求解方案（新版）

核心设计：
1. 对问题2：建立多残骸TOA关联模型，通过解析种子+全站联合残差验真
   解决"哪个读数属于哪个残骸"的组合爆炸问题。
2. 对问题3：用上述模型求解具体数据，做7站残差诊断验证数据质量，
   并验证加回5秒限制后结果不变。

不依赖：两站传播时差门限、5秒时间窗（问题3验证时加回比对）。
保留：TOA方程、一对一覆盖、因果性、空间边界、全站联合残差。
"""
import itertools
import json
import numpy as np

C = 340.0  # 声速，m/s
NAMES = np.array(list('ABCDEFG'))

# ========== 问题3原始数据（与题目docx核对一致） ==========
lon = np.array([110.241, 110.783, 110.762, 110.251, 110.524, 110.467, 110.047])
lat = np.array([27.204, 27.456, 27.785, 28.025, 27.617, 28.081, 27.521])
alt = np.array([824.0, 727.0, 742.0, 850.0, 786.0, 678.0, 575.0])
T = np.array([
    [100.767, 164.229, 214.850, 270.065],   # A
    [92.453, 112.220, 169.362, 196.583],     # B
    [75.560, 110.696, 156.936, 188.020],     # C
    [94.653, 141.409, 196.517, 258.985],     # D
    [78.600, 86.216, 118.443, 126.669],      # E
    [67.274, 166.270, 175.482, 266.871],     # F
    [103.738, 163.024, 206.789, 210.306],    # G
])

# 坐标转换：统一为米制
S = np.column_stack(((lon - 110) * 97304, (lat - 27) * 111263, alt))

ALL = np.arange(7)
LOW = np.array([-50000., -50000., 0., -400.])
HIGH_BASE = np.array([150000., 220000., 120000., 0.])


def choose_seed_stations():
    """选条件数最低的4台作为解析种子站（仅用于生成初值，非最终定位）。"""
    best = None
    for ids in itertools.combinations(range(7), 4):
        ii = np.asarray(ids)
        A = 2 * (S[ii[1:]] - S[ii[0]])
        k = np.linalg.cond(A)
        if best is None or k < best[0]:
            best = (k, ii)
    return best


SEED_COND, SEED = choose_seed_stations()


def analytic_seeds(obs):
    """四站平方TOA方程相减消去二次项，得到 p = p0 + p1 * tau。
    代回原始方程得一元二次，至多两个解析种子。
    几何：三个根平面交于一条直线，音爆点参数化于时刻 tau。
    """
    ss = S[SEED]
    tt = obs[SEED]
    A = 2 * (ss[1:] - ss[0])
    b0 = np.sum(ss[1:]**2, axis=1) - np.sum(ss[0]**2) - \
        C*C * (tt[1:]**2 - tt[0]**2)
    b1 = 2 * C*C * (tt[1:] - tt[0])
    try:
        p0 = np.linalg.solve(A, b0)
        p1 = np.linalg.solve(A, b1)
    except np.linalg.LinAlgError:
        return []
    v = p0 - ss[0]
    qa = p1 @ p1 - C*C
    qb = 2 * (p1 @ v + C*C * tt[0])
    qc = v @ v - C*C * tt[0] * tt[0]
    roots = np.roots([qa, qb, qc])
    ans = []
    for r in roots:
        if abs(r.imag) > 1e-7:
            continue
        tau = float(r.real)
        ans.append(np.r_[p0 + p1 * tau, tau])
    return ans


def res(theta, obs):
    """全部7站的TOA残差向量：实测 - 预测到达时刻。"""
    return obs - theta[3] - np.linalg.norm(S - theta[:3], axis=1) / C


def refine(theta, obs):
    """全部7站有界Levenberg-Marquardt精化。
    不使用5秒窗口，也不使用两站传播时差门限。
    """
    high = HIGH_BASE.copy()
    high[3] = obs.min()  # 因果性：音爆时刻不能晚于任何观测
    scale = np.array([50000., 80000., 30000., 100.])
    u = np.clip(theta, LOW, high) / scale
    lam = 1e-4
    for _ in range(100):
        th = u * scale
        r = res(th, obs)
        d = th[:3] - S
        dd = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        J = np.column_stack((d / (C * dd[:, None]), np.ones(7))) * scale
        step = np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ r)
        un = np.clip(th + step * scale, LOW, high) / scale
        rn = res(un * scale, obs)
        if rn @ rn < r @ r:
            u = un
            lam = max(lam / 3, 1e-12)
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam = min(lam * 10, 1e12)
    th = u * scale
    r = res(th, obs)
    return th, float(np.sqrt(np.mean(r * r))), float(np.max(abs(r)))


def candidate_events(threshold=0.02):
    """枚举 4^7 = 16384 种单事件组合，无两站时差门限或5秒判断。
    单事件：假设存在一个残骸，从每台设备各取一个读数。
    """
    keep = []
    root_valid = 0
    for choice in itertools.product(range(4), repeat=7):
        choice = np.asarray(choice, dtype=int)
        obs = T[ALL, choice]
        for seed in analytic_seeds(obs):
            # 仅因果性与空间边界
            if not (np.all(seed[:3] >= LOW[:3]) and
                    np.all(seed[:3] <= HIGH_BASE[:3]) and
                    seed[3] <= obs.min()):
                continue
            root_valid += 1
            th, rmse, mx = refine(seed, obs)
            if rmse < threshold and mx < 2 * threshold:
                keep.append(
                    dict(choice=choice, theta=th, rmse=rmse, maxabs=mx))
                break
    keep.sort(key=lambda q: q["rmse"])
    return keep, root_valid


def exact_cover(candidates):
    """从候选集中选出4个单事件，使每台设备的4个读数被恰好各用一次。
    不使用5秒约束，只实施每站一对一覆盖。
    """
    best = None
    ncover = 0
    for ids in itertools.combinations(range(len(candidates)), 4):
        g = [candidates[i] for i in ids]
        mat = np.array([x["choice"] for x in g])
        if not all(sorted(mat[:, i].tolist()) == [0, 1, 2, 3] for i in range(7)):
            continue
        ncover += 1
        score = sum(x["rmse"]**2 for x in g)
        if best is None or score < best[0]:
            best = (score, g)
    return best, ncover


def pdop_analysis(theta, ids):
    """计算给定设备子集的几何精度因子(PDOP)。
    位置项乘1000使单位为 km/s。
    """
    d = theta[:3] - S[ids]
    dd = np.linalg.norm(d, axis=1)
    H = np.column_stack((d / (C * dd[:, None]), np.ones(len(ids))))
    H[:, :3] *= 1000
    sv = np.linalg.svd(H, compute_uv=False)
    rank = int(np.linalg.matrix_rank(H))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else np.inf
    Q = np.linalg.pinv(H.T @ H)
    pdop = float(np.sqrt(np.trace(Q[:3, :3])))
    return rank, cond, pdop


def solve():
    """主求解流程。"""
    print("=" * 70)
    print("问题二/三：多残骸音爆定位联合求解")
    print("=" * 70)

    # 步骤1：生成单事件候选
    cand, roots = candidate_events()
    seed_name = "".join(NAMES[SEED])
    print()
    print(f"解析种子站: {seed_name} (条件数={SEED_COND:.1f})")
    print(f"枚举空间: 4^7 = {4**7}")
    print(f"物理边界内解析根: {roots}")
    print(f"全站TOA候选(精化后): {len(cand)}")

    # 步骤2：精确覆盖
    best, ncover = exact_cover(cand)
    print(f"精确覆盖组合数: {ncover}")
    print(f"最优联合残差平方和: {best[0]:.6e}")

    if best is None:
        raise RuntimeError("没有精确覆盖；可提高阈值检查。")

    # 步骤3：输出结果与残差诊断
    print()
    print("-" * 70)
    print("最终解（按音爆时刻排序）")
    print("-" * 70)

    out = []
    for j, x in enumerate(sorted(best[1], key=lambda q: q["theta"][3]), 1):
        th = x["theta"]
        lon_deg = 110 + th[0] / 97304
        lat_deg = 27 + th[1] / 111263
        z_km = th[2] / 1000
        tau = th[3]
        rmse = x["rmse"]

        # 逐站残差（ms）
        obs = T[ALL, x["choice"]]
        residuals = (obs - tau - np.linalg.norm(S - th[:3], axis=1) / C) * 1000
        maxabs = x["maxabs"]

        choice_parts = []
        for i in ALL:
            choice_parts.append(f"{NAMES[i]}第{x['choice'][i]+1}")
        choice_str = ", ".join(choice_parts)
        print()
        print(f"残骸{j}: {choice_str}")
        print(
            f"  经度={lon_deg:.6f}°, 纬度={lat_deg:.6f}°, 高程={z_km:.3f}km, 音爆时刻={tau:.4f}s")
        print(f"  RMSE={rmse:.6f}s, max|残差|={maxabs:.6f}s")
        res_parts = []
        for i in ALL:
            res_parts.append(f"{NAMES[i]}={residuals[i]:+.2f}")
        res_str = ", ".join(res_parts)
        print(f"  逐站残差(ms): {res_str}")

        out.append({
            "id": j,
            "choice": x["choice"].tolist(),
            "lon": lon_deg,
            "lat": lat_deg,
            "z_km": z_km,
            "tau": tau,
            "rmse": rmse,
            "maxabs": maxabs,
            "residuals_ms": residuals.tolist()
        })

    # 步骤4：5秒限制验证
    taus = sorted([x["theta"][3] for x in best[1]])
    span = max(taus) - min(taus)
    status = "通过" if span <= 5 else "未通过"
    print()
    print("=" * 70)
    print(f"5秒时间窗验证: 跨度={span:.4f}s ({span*1000:.1f}ms) <= 5s: {status}")

    # 步骤5：PDOP分析
    print()
    print("=" * 70)
    print("PDOP分析（最坏值取4个残骸中的最大）")
    print("=" * 70)

    from itertools import combinations
    for combo in ["ABEF", "ABDEF", "ABCDEF", "ABCDEFG", "ABCG", "ABEG", "ABCEG"]:
        ids = np.array([list("ABCDEFG").index(c) for c in combo])
        pdops = []
        for x in best[1]:
            _, _, pdop = pdop_analysis(x["theta"], ids)
            pdops.append(pdop)
        worst = max(pdops)
        avg = np.mean(pdops)
        print(f"  {combo}: 最坏PDOP={worst:.3f} km/s, 平均PDOP={avg:.3f} km/s")

    # 保存JSON
    result = {
        "seed_stations": seed_name,
        "seed_cond": SEED_COND,
        "candidate_count": len(cand),
        "root_valid": roots,
        "covers": ncover,
        "time_span_s": span,
        "answer": out
    }
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print()
    print("结果已保存至 result.json")


if __name__ == "__main__":
    solve()
