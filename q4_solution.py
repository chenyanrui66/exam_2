
import itertools
import os
import time

import numpy as np
import pandas as pd

from csv_writer import CsvCollector

C = 340.0
DELTA = 0.5
NAMES = np.array(list("ABCDEFG"))
ALL7 = np.arange(7)

lon = np.array([110.241, 110.783, 110.762, 110.251, 110.524, 110.467, 110.047])
lat = np.array([27.204, 27.456, 27.785, 28.025, 27.617, 28.081, 27.521])
alt = np.array([824.0, 727.0, 742.0, 850.0, 786.0, 678.0, 575.0])

T = np.array([
    [100.767, 164.229, 214.850, 270.065],
    [92.453, 112.220, 169.362, 196.583],
    [75.560, 110.696, 156.936, 188.020],
    [94.653, 141.409, 196.517, 258.985],
    [78.600, 86.216, 118.443, 126.669],
    [67.274, 166.270, 175.482, 266.871],
    [103.738, 163.024, 206.789, 210.306],
])

S = np.column_stack(((lon - 110) * 97304.0, (lat - 27) * 111263.0, alt))


def to_local(lon_d, lat_d, z_m):

    return np.array([(lon_d - 110) * 97304.0, (lat_d - 27) * 111263.0, z_m])


truth = np.array([
    [*to_local(110.500001, 27.309998, 12514.0), 11.9999],
    [*to_local(110.499999, 27.949998, 11529.0), 13.0014],
    [*to_local(110.300000, 27.650000, 11478.0), 14.0000],
    [*to_local(110.699999, 27.650000, 13468.0), 15.0000],
])
TRUTH_LL = [
    (110.500001, 27.309998, 12514.0, 11.9999),
    (110.499999, 27.949998, 11529.0, 13.0014),
    (110.300000, 27.650000, 11478.0, 14.0000),
    (110.699999, 27.650000, 13468.0, 15.0000),
]


LOW = np.array([-50000.0, -50000.0, 0.0, -400.0])
HIGH = np.array([150000.0, 220000.0, 120000.0, 0.0])

combo_idx = np.array(list(itertools.product(range(4), repeat=7)))


_combos4 = []
for _ids in itertools.combinations(range(7), 4):
    _ii = np.asarray(_ids)
    _combos4.append((np.linalg.cond(2 * (S[_ii[1:]] - S[_ii[0]])), _ii))
_combos4.sort(key=lambda x: x[0])
SEED_GROUPS = [ii for _, ii in _combos4[:4]]
SEED7 = SEED_GROUPS[0]


def batch_seeds(Tobs, seed_groups, prescreen=3.0):

    obs_all = Tobs[ALL7[None, :], combo_idx]
    out = {}
    for sg in seed_groups:
        ss = S[sg]
        tt = obs_all[:, sg]
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

    return obs - theta[3] - np.linalg.norm(S - theta[:3], axis=1) / C


def refine(theta, obs, iters=80):

    high = HIGH.copy()
    high[3] = obs.min()
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

    cand = _enum_candidates(Tobs, rmse_thr, maxabs_thr)
    best = _try_cover(cand)
    if best is not None:
        return best, "strict"

    cand = _enum_candidates(Tobs, 0.75, 0.85)
    for ids in itertools.combinations(range(min(len(cand), 8)), 3):
        g = [cand[i] for i in ids]
        mat = np.array([x["choice"] for x in g])
        if not all(len(set(mat[:, i])) == 3 for i in range(7)):
            continue
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
            if rmse < 0.9 and mx < 1.0:
                g4 = g + [dict(choice=np.array(rem), theta=th,
                               rmse=rmse, maxabs=mx)]
                return (sum(x["rmse"] ** 2 for x in g4), g4), "completed"
    return None, "failed"


# ==================== Part B：蒙特卡洛算例 ====================
def add_noise7(rng):

    return np.sort(T + rng.uniform(-DELTA, DELTA, size=T.shape), axis=1)


def evaluate(solutions):
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


center = to_local(110.5, 27.65, 800.0)


def ring(n, r_km, az0=0.0, z=750.0):

    ang = np.deg2rad(az0 + np.arange(n) * 360.0 / n)
    return np.column_stack((center[0] + r_km * 1000 * np.cos(ang),
                            center[1] + r_km * 1000 * np.sin(ang),
                            np.full(n, z)))


S_NEW = np.vstack([S, center[None, :], ring(4, 15, 45), ring(8, 45, 22.5)])
M = len(S_NEW)

T20_TRUE = np.array([[truth[j, 3] + np.linalg.norm(S_NEW[i] - truth[j, :3]) / C
                      for j in range(4)] for i in range(M)])


def pdop_at(theta, S_arr):

    d = theta[:3] - S_arr
    dd = np.linalg.norm(d, axis=1)
    H = np.column_stack((d / (C * dd[:, None]), np.ones(len(S_arr))))
    H[:, :3] *= 1000.0
    Q = np.linalg.pinv(H.T @ H)
    return float(np.sqrt(np.trace(Q[:3, :3])))


def refine_M(theta, obs, S_arr, iters=80):

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


ERR_COLS = ["水平误差(m)", "高程误差(m)", "3D误差(m)", "时刻误差(ms)"]


def detail_df(V, nmc):

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

    rows = []
    for k, lab in enumerate(ERR_COLS):
        v = V[:, :, k].ravel() * (1000.0 if k == 3 else 1.0)
        rows.append({"指标": lab, "均值": v.mean(), "中位": np.median(v),
                     "95%分位": np.percentile(v, 95), "最大": v.max()})
    return pd.DataFrame(rows)


def demo_case_df(thetas):
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
    cc = CsvCollector()
    sig_t = DELTA / np.sqrt(3)

    cc.add("试验概况", pd.DataFrame([
        {"台网": "原7台", "蒙特卡洛次数": ok7 + fail7, "成功": ok7, "失败": fail7,
         "3D误差>1km比例(%)": (V7[:, :, 2].ravel() > 1000).mean() * 100,
         "严阈值成功(strict)": modes7.get("strict", 0),
         "补全兜底(completed)": modes7.get("completed", 0)},
        {"台网": "20台加密", "蒙特卡洛次数": ok20 + fail20, "成功": ok20, "失败": fail20,
         "3D误差>1km比例(%)": (V20[:, :, 2].ravel() > 1000).mean() * 100,
         "严阈值成功(strict)": "-", "补全兜底(completed)": "-"},
    ]))

    cc.add("误差统计_原7台", stats_df(V7))
    cc.add("误差统计_20台加密", stats_df(V20))

    cc.add("误差明细_原7台", detail_df(V7, ok7))
    cc.add("误差明细_20台加密", detail_df(V20, ok20))

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

    groups = (["原台站"] * 7 + ["中心"] + ["内环"] * 4 + ["外环"] * 8)
    sta_names = list("ABCDEFG") + [f"N{k:02d}" for k in range(1, 14)]
    cc.add("台网台站", pd.DataFrame([
        {"台站": sta_names[i], "组别": groups[i],
         "经度(°E)": 110 + S_NEW[i, 0] / 97304,
         "纬度(°N)": 27 + S_NEW[i, 1] / 111263,
         "高程(m)": S_NEW[i, 2]}
        for i in range(M)
    ]))

    cc.add("残骸音爆点真值", pd.DataFrame([
        {"残骸": f"#{j + 1}", "经度(°E)": lo, "纬度(°N)": la,
         "高程(m)": z, "音爆时刻(s)": tau}
        for j, (lo, la, z, tau) in enumerate(TRUTH_LL)
    ]))

    cc.add("单次算例_原7台", demo7)
    cc.add("单次算例_20台加密", demo20)

    cc.save(path)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    t0 = time.time()

    V7, ok7, fail7, modes7 = part_b(nmc=300)

    V20, ok20, fail20 = part_c(nmc=300)

    best_d, _ = solve_noisy(add_noise7(np.random.default_rng(20260827)))
    demo7 = demo_case_df([x["theta"] for x in best_d[1]])
    demo20 = demo_case_df(solve_dense(
        add_noise20(np.random.default_rng(20260827))))

    out_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "q4_results.csv")
    build_csv(csv_path, V7, ok7, fail7, modes7,
              V20, ok20, fail20, demo7, demo20)

    print(f"结果已保存: {csv_path}")
