
import itertools
import json
import os
import numpy as np
import pandas as pd

C = 340.0
NAMES = np.array(list('ABCDEFG'))

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


S = np.column_stack(((lon - 110) * 97304, (lat - 27) * 111263, alt))

ALL = np.arange(7)
LOW = np.array([-50000., -50000., 0., -400.])
HIGH_BASE = np.array([150000., 220000., 120000., 0.])


class CsvCollector:

    def __init__(self):
        self._blocks = []

    def add(self, section, df, index=False):
        self._blocks.append((section, df, bool(index)))
        return df

    def save(self, path, encoding='utf-8-sig'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding=encoding, newline='') as f:
            for i, (section, df, index) in enumerate(self._blocks):
                if i:
                    f.write('\n')
                f.write(f'#  {section} \n')
                df.to_csv(f, index=index)


def choose_seed_stations():
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

    return obs - theta[3] - np.linalg.norm(S - theta[:3], axis=1) / C


def refine(theta, obs):

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

    keep = []
    root_valid = 0
    for choice in itertools.product(range(4), repeat=7):
        choice = np.asarray(choice, dtype=int)
        obs = T[ALL, choice]
        for seed in analytic_seeds(obs):

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

    collector = CsvCollector()
    seed_name = "".join(NAMES[SEED])

    cand, roots = candidate_events()

    best, ncover = exact_cover(cand)

    if best is None:
        raise RuntimeError("没有精确覆盖；可提高阈值检查。")

    out = []
    for j, x in enumerate(sorted(best[1], key=lambda q: q["theta"][3]), 1):
        th = x["theta"]
        lon_deg = 110 + th[0] / 97304
        lat_deg = 27 + th[1] / 111263
        z_km = th[2] / 1000
        tau = th[3]
        rmse = x["rmse"]

        obs = T[ALL, x["choice"]]
        residuals = (obs - tau - np.linalg.norm(S - th[:3], axis=1) / C) * 1000
        maxabs = x["maxabs"]

        choice_parts = []
        for i in ALL:
            choice_parts.append(f"{NAMES[i]}第{x['choice'][i]+1}")
        choice_str = ", ".join(choice_parts)

        out.append({
            "残骸编号": j,
            "读数分配": choice_str,
            "经度(°)": lon_deg,
            "纬度(°)": lat_deg,
            "高程(km)": z_km,
            "音爆时刻(s)": tau,
            "RMSE(s)": rmse,
            "最大残差(s)": maxabs,
            "A残差(ms)": residuals[0],
            "B残差(ms)": residuals[1],
            "C残差(ms)": residuals[2],
            "D残差(ms)": residuals[3],
            "E残差(ms)": residuals[4],
            "F残差(ms)": residuals[5],
            "G残差(ms)": residuals[6],
        })

    df_results = pd.DataFrame(out)
    collector.add("残骸定位结果", df_results)

    # 步骤4：5秒限制验证
    taus = sorted([x["theta"][3] for x in best[1]])
    span = max(taus) - min(taus)
    status = "通过" if span <= 5 else "未通过"

    df_verify = pd.DataFrame([{
        "音爆时刻跨度(s)": span,
        "5秒限制验证": status,
    }])
    collector.add("5秒时间窗验证", df_verify)

    # 步骤5：PDOP分析
    pdop_rows = []
    from itertools import combinations
    for combo in ["ABEF", "ABDEF", "ABCDEF", "ABCDEFG", "ABCG", "ABEG", "ABCEG"]:
        ids = np.array([list("ABCDEFG").index(c) for c in combo])
        pdops = []
        for x in best[1]:
            _, _, pdop = pdop_analysis(x["theta"], ids)
            pdops.append(pdop)
        worst = max(pdops)
        avg = np.mean(pdops)
        pdop_rows.append({
            "设备组合": combo,
            "最坏PDOP(km/s)": worst,
            "平均PDOP(km/s)": avg,
        })

    df_pdop = pd.DataFrame(pdop_rows)
    collector.add("PDOP分析", df_pdop)

    output_path = "./output/q2q3_results.csv"
    collector.save(output_path)

    result = {
        "seed_stations": seed_name,
        "seed_cond": SEED_COND,
        "candidate_count": len(cand),
        "root_valid": roots,
        "covers": ncover,
        "time_span_s": span,
        "answer": out
    }
    os.makedirs("./output", exist_ok=True)
    with open("./output/q2q3_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("结果已保存")


if __name__ == "__main__":
    solve()
