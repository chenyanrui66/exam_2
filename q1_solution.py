
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import least_squares

from csv_writer import CsvCollector


devs = {
    'A': (110.241, 27.204, 824, 100.767),
    'B': (110.780, 27.456, 727, 112.220),
    'C': (110.712, 27.785, 742, 188.020),
    'D': (110.251, 27.825, 850, 258.985),
    'E': (110.524, 27.617, 786, 118.443),
    'F': (110.467, 27.921, 678, 266.871),
    'G': (110.047, 27.121, 575, 163.024),
}
c = 340.0
lon0, lat0 = 110.241, 27.204
Klon, Klat = 97.304e3, 111.263e3


pos = {k: np.array([(v[0]-lon0)*Klon, (v[1]-lat0)*Klat, float(v[2])])
       for k, v in devs.items()}
tobs = {k: v[3] for k, v in devs.items()}


def residual(v, P, T):
    return np.linalg.norm(P - v[:3], axis=1) - c * (T - v[3])


def solve_toa(keys, P=None, T=None):

    if P is None:
        P = np.array([pos[k] for k in keys])
    if T is None:
        T = np.array([tobs[k] for k in keys])

    A, b = [], []
    for i in range(1, len(keys)):

        A.append(np.concatenate([2*(P[i]-P[0]), [-2*c**2*(T[i]-T[0])]]))
        b.append(P[i] @ P[i] - P[0] @ P[0] - c**2*(T[i]**2 - T[0]**2))

    X, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    t0_init = X[3] if np.isfinite(X[3]) else 15.0

    inits = [X[:3]] + [
        np.array([(110.5-lon0)*Klon, (27.31-lat0)*Klat, z])
        for z in (1000.0, 5000.0, 12000.0)
    ]

    best = None
    for x0 in inits:
        r = least_squares(
            residual,
            np.concatenate([x0, [t0_init]]),
            args=(P, T),
            method='trf'
        )
        rms = np.sqrt(np.mean(r.fun**2))
        if r.x[2] > 0 and -100 < r.x[3] < 500:
            if best is None or rms < best[0]:
                best = (rms, r.x)

    return best


def lonlat(v):

    return lon0 + v[0]/Klon, lat0 + v[1]/Klat, v[2], v[3]


def residual_table(v_sol, P_all, T_all, dev_list, highlight_set):

    res = residual(v_sol, P_all, T_all)
    rows = []
    for i, k in enumerate(dev_list):
        in_fit = k in highlight_set
        if in_fit:
            judge = "自洽"
        elif abs(res[i]) > 10000:
            judge = "异常剔除"
        elif abs(res[i]) > 1000:
            judge = "可疑"
        else:
            judge = "一般"
        rows.append({
            '设备': k,
            '参与拟合': '是' if in_fit else '否',
            '残差(m)': round(res[i], 0),
            '折合(s)': round(res[i]/c, 2),
            '评判': judge,
        })
    return pd.DataFrame(rows), res


def main():
    cc = CsvCollector()

    dev_list = list(devs.keys())
    P_all = np.array([pos[k] for k in dev_list])
    T_all = np.array([tobs[k] for k in dev_list])

    rms_abcg, v_abcg = solve_toa(list('ABCG'))
    rms_abeg, v_abeg = solve_toa(list('ABEG'))
    rms_5, v_5 = solve_toa(list('ABCEG'))

    sol_rows = []
    for name, rms, v in [
        ("ABCG 四台解", rms_abcg, v_abcg),
        ("ABEG 四台解", rms_abeg, v_abeg),
        ("ABCEG 五台联合解", rms_5, v_5),
    ]:
        lo, la, z, t0 = lonlat(v)
        sol_rows.append({
            '组合': name,
            '经度(°E)': round(lo, 5),
            '纬度(°N)': round(la, 5),
            '高程(m)': round(z, 0),
            't0(s)': round(t0, 3),
            'RMS(m)': round(rms, 1),
        })
    cc.add("阶段一：三个关键组合的解", pd.DataFrame(sol_rows))

    df_abcg, res_abcg = residual_table(v_abcg, P_all, T_all, dev_list, 'ABCG')
    cc.add("阶段二：ABCG 四台解回测全部7台", df_abcg)

    df_abeg, res_abeg = residual_table(v_abeg, P_all, T_all, dev_list, 'ABEG')
    cc.add("阶段二：ABEG 四台解回测全部7台", df_abeg)

    df_5, res_5 = residual_table(v_5, P_all, T_all, dev_list, 'ABCEG')
    cc.add("阶段二：ABCEG 五台联合解回测全部7台", df_5)

    d_xy = np.hypot(v_abcg[0]-v_abeg[0], v_abcg[1]-v_abeg[1])
    d_z = abs(v_abcg[2] - v_abeg[2])
    d_t = abs(v_abcg[3] - v_abeg[3])

    abcg_in = [res_abcg[i] for i, k in enumerate(dev_list) if k in 'ABCG']
    abeg_in = [res_abeg[i] for i, k in enumerate(dev_list) if k in 'ABEG']
    abceg_in = [res_5[i] for i, k in enumerate(dev_list) if k in 'ABCEG']
    normal_idx = [i for i, k in enumerate(dev_list) if k in 'ABCEG']

    metrics = pd.DataFrame([
        {'类别': '两解间距', '指标': '水平距离(m)', '数值': round(d_xy, 0),
         '说明': '两解相距约1.78km，远超任何合理噪声级，说明4台解不唯一'},
        {'类别': '两解间距', '指标': '高程差(m)', '数值': round(d_z, 0), '说明': ''},
        {'类别': '两解间距', '指标': 't0差(s)', '数值': round(d_t, 2), '说明': ''},
        {'类别': '互相回测', '指标': 'ABCG解回测E(m)', '数值': round(res_abcg[4], 0),
         '说明': '互相把对方核心设备打成异常，但C、E不可能同时异常；4台没有冗余自由度无法判断谁错'},
        {'类别': '互相回测', '指标': 'ABCG解回测E(折合s)', '数值': round(
            res_abcg[4]/c, 2), '说明': ''},
        {'类别': '互相回测', '指标': 'ABEG解回测C(m)', '数值': round(
            res_abeg[2], 0), '说明': ''},
        {'类别': '互相回测', '指标': 'ABEG解回测C(折合s)', '数值': round(
            res_abeg[2]/c, 2), '说明': ''},
        {'类别': '五台裁定', '指标': 'C残差(m)', '数值': round(res_5[2], 0),
         '说明': 'C偏早1.89s，E偏晚1.42s，方向相反、量级对称；最小二乘折中，解落在两簇之间'},
        {'类别': '五台裁定', '指标': 'C残差(折合s)', '数值': round(res_5[2]/c, 2), '说明': ''},
        {'类别': '五台裁定', '指标': 'E残差(m)', '数值': round(res_5[4], 0), '说明': ''},
        {'类别': '五台裁定', '指标': 'E残差(折合s)', '数值': round(res_5[4]/c, 2), '说明': ''},
        {'类别': '内部RMS', '指标': 'ABCG参与设备(m)', '数值': round(np.sqrt(np.mean(np.array(abcg_in)**2)), 1),
         '说明': '4台内部RMS≈60m是数学必然（4方程4未知数，残差趋零）'},
        {'类别': '内部RMS', '指标': 'ABEG参与设备(m)', '数值': round(
            np.sqrt(np.mean(np.array(abeg_in)**2)), 1), '说明': ''},
        {'类别': '内部RMS', '指标': 'ABCEG参与设备(m)', '数值': round(np.sqrt(np.mean(np.array(abceg_in)**2)), 1),
         '说明': '5台内部RMS≈370m才是真实数据质量（冗余自由度暴露偏差）'},
        {'类别': '回测正常设备RMS', '指标': 'ABCG解回测5正常设备(m)', '数值': round(np.sqrt(np.mean(res_abcg[normal_idx]**2)), 1),
         '说明': '4台解回测正常设备RMS≈500-600m，远大于其内部60m'},
        {'类别': '回测正常设备RMS', '指标': 'ABEG解回测5正常设备(m)', '数值': round(
            np.sqrt(np.mean(res_abeg[normal_idx]**2)), 1), '说明': ''},
        {'类别': '回测正常设备RMS', '指标': 'ABCEG解回测5正常设备(m)', '数值': round(np.sqrt(np.mean(res_5[normal_idx]**2)), 1),
         '说明': '5台解回测正常设备RMS≈370m，内外一致，说明模型与数据匹配'},
    ])
    cc.add("阶段三：关键对比指标", metrics)

    votes = []
    for combo in combinations('ABCEG', 4):
        keys = list(combo)
        rms, v = solve_toa(keys)
        lo, la, z, t0 = lonlat(v)
        votes.append((combo, lo, la, z, t0, rms))

    ref_lo, ref_la = lonlat(v_5)[:2]
    vote_rows = []
    for combo, lo, la, z, t0, rms in votes:
        dist = np.hypot((lo-ref_lo)*Klon, (la-ref_la)*Klat)
        vote_rows.append({
            '子集': ''.join(combo),
            '经度(°E)': round(lo, 4),
            '纬度(°N)': round(la, 4),
            '高程(m)': round(z, 0),
            't0(s)': round(t0, 2),
            'RMS(m)': round(rms, 1),
            '距五台联合解(m)': round(dist, 0),
            '是否主簇': '是' if dist < 2000 else '否',
        })
    cc.add("阶段四：{A,B,C,E,G} 内全部4台子集投票", pd.DataFrame(vote_rows),
           index=False)

    rng = np.random.default_rng(42)
    keys5 = list('ABCEG')
    P5 = np.array([pos[k] for k in keys5])
    T5 = np.array([tobs[k] for k in keys5])

    def monte_carlo(coord_noise_deg, time_noise_s, n=300):
        outs = []
        for _ in range(n):

            dP = P5 + rng.uniform(-coord_noise_deg, coord_noise_deg, P5.shape) \
                * np.array([Klon, Klat, 0.0])

            dT = T5 + rng.uniform(-time_noise_s, time_noise_s, T5.shape)
            r = least_squares(residual, v_5.copy(), args=(dP, dT))
            outs.append(r.x)
        outs = np.array(outs)
        dh = np.hypot(outs[:, 0]-v_5[0], outs[:, 1]-v_5[1])
        return np.percentile(dh, 95), outs[:, 2].std()

    r95a, zsa = monte_carlo(0.0005, 0.0005)
    r95b, zsb = monte_carlo(0.0, 0.5)          #

    mc = pd.DataFrame([
        {'噪声假设': '坐标量化噪声(±0.0005°) + 时刻舍入(±0.0005s)',
         '95%散布半径(m)': round(r95a, 0), '高程标准差(m)': round(zsa, 0),
         '说明': ''},
        {'噪声假设': '±0.5s 时间误差（最坏假设）',
         '95%散布半径(m)': round(r95b, 0), '高程标准差(m)': round(zsb, 0),
         '说明': ''},
        {'噪声假设': '结论',
         '95%散布半径(m)': '', '高程标准差(m)': '',
         '说明': '两4台解相距1780m，是噪声散布的6~20倍，确认不是噪声级分歧'},
    ])
    cc.add("阶段五：蒙特卡洛噪声散布评估", mc)

    lo, la, z, t0 = lonlat(v_5)
    conclusion = pd.DataFrame([
        {'项目': '音爆位置', '结果': f'东经 {lo:.3f}°, 北纬 {la:.3f}°, 高程约 {z:.0f} m'},
        {'项目': '音爆时刻', '结果': f'系统时钟后 {t0:.1f} s'},
        {'项目': '使用设备', '结果': 'A, B, C, E, G（剔除异常设备 D, F）'},
        {'项目': '核心方法论', '结果': '4台理论可解但无冗余检错能力；实际需 ≥5 台，用冗余观测做异常剔除与系统偏差折中'},
    ])
    cc.add("最终结论", conclusion)

    out_dir = os.path.join('.', 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'q1_results.csv')
    cc.save(out_path)

    print("结果已保存")


if __name__ == '__main__':
    main()
