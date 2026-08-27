# -*- coding: utf-8 -*-
"""
问题4 稳健性/灵敏度压力测试（独立测试代码）
================================================================
承接《问题4_误差修正与加密台网方案.py》，在不改动原模型的前提下做 6 项压力测试：

  测试一  全体同号偏差：所有读数 +0.5 s / -0.5 s（时间原点平移）
  测试二  边界两点分布：每读数独立随机取 ±0.5 s（7 台与 20 台，各 300 次蒙特卡洛）
  测试三  台站系统偏差：每台设备所有读数同号 ±0.5 s，7 台穷举全部 2^7=128 种模式，
          20 台随机抽 60 种模式
  测试四  噪声幅度扫描（灵敏度/崩溃点分析）：噪声界 Δ 从 0.1 s 递增至 1.0 s，
          统计关联成功率与 3D 误差 95% 分位，定位模型失效的临界噪声水平
  测试五  单站粗差（野值鲁棒性）：背景噪声 U(-0.5,+0.5) s 上，随机一台设备的随机一个
          读数叠加 +1.0 s / +2.0 s 粗差（模拟设备故障或记录串行）
  测试六  20 台台网对照：Δ=1.0 s 噪声、+2.0 s 单站粗差两种极端情形下加密台网的表现

用法：把本文件与《问题4_误差修正与加密台网方案.py》放在同一目录，然后
    python 问题4_稳健性压力测试.py
全程约 15 分钟（可调小 NMC 加速）。仅依赖 numpy。
"""
import importlib.util
import itertools as it
import os
import time

import numpy as np

# ==================== 加载被测模型 ====================
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "问题4_误差修正与加密台网方案.py")
if not os.path.exists(_SRC):          # 兼容直接在上传目录运行的情形
    _SRC = "/mnt/agents/upload/问题4_误差修正与加密台网方案.py"
_spec = importlib.util.spec_from_file_location("q4", _SRC)
q4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q4)

NAMES4 = ["水平误差(m)", "高程误差(m)", "3D误差(m)", "时刻误差(ms)"]


def _report(tag, E, nmc, fail, extra=""):
    """统一输出误差统计表"""
    ok = ~np.isnan(E[:, 0, 0])
    print(f"[{tag}] 成功 {ok.sum()}/{nmc}，失败/混淆 {fail} {extra}")
    if not ok.any():
        return None
    V = E[ok]
    print(f"  {'指标':<12}{'均值':>9}{'中位':>9}{'95%分位':>9}{'最大':>9}")
    for k, lab in enumerate(NAMES4):
        v = V[:, :, k].ravel() * (1000.0 if k == 3 else 1.0)
        print(f"  {lab:<12}{v.mean():>9.0f}{np.median(v):>9.0f}"
              f"{np.percentile(v, 95):>9.0f}{v.max():>9.0f}")
    v3 = V[:, :, 2].ravel()
    print(f"  3D误差>1 km 比例: {(v3 > 1000).mean() * 100:.2f}%")
    return v3


# ==================== 测试一：全体同号偏差 ====================
def test1_common_shift():
    print("\n" + "=" * 72)
    print("测试一  全体同号偏差（+0.5 s / -0.5 s）：预期被音爆时刻参数整体吸收")
    print("=" * 72)
    for s in (+0.5, -0.5):
        Tn = np.sort(q4.T + s, axis=1)
        best, mode = q4.solve_noisy(Tn)
        assert best is not None, "关联失败！"
        e = q4.evaluate([x["theta"] for x in best[1]])
        print(f"  全{s:+.1f}s ({mode}): 3D误差(m) {np.round(e[:, 2], 1)}，"
              f"时刻误差(ms) {np.round(e[:, 3] * 1000, 0)}")


# ==================== 测试二：边界两点分布 ====================
def test2_boundary_sign(nmc=300, seed=777):
    print("\n" + "=" * 72)
    print("测试二  每读数独立随机取 ±0.5 s（两点分布，σ=0.5 s，比均匀分布恶劣73%）")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    E = np.full((nmc, 4, 4), np.nan); fail = 0; modes = {}
    for n in range(nmc):
        Tn = np.sort(q4.T + rng.choice([-0.5, 0.5], size=q4.T.shape), axis=1)
        best, mode = q4.solve_noisy(Tn)
        modes[mode] = modes.get(mode, 0) + 1
        if best is None:
            fail += 1; continue
        e = q4.evaluate([x["theta"] for x in best[1]])
        if e is None: fail += 1
        else: E[n] = e
    _report("7台 ±0.5两点", E, nmc, fail, f"模式{modes}")

    rng = np.random.default_rng(seed + 111)
    E = np.full((nmc, 4, 4), np.nan); fail = 0
    for n in range(nmc):
        Tn = np.sort(q4.T20_TRUE + rng.choice([-0.5, 0.5], size=q4.T20_TRUE.shape), axis=1)
        thetas = q4.solve_dense(Tn)
        if thetas is None:
            fail += 1; continue
        e = q4.evaluate(thetas)
        if e is None: fail += 1
        else: E[n] = e
    _report("20台 ±0.5两点", E, nmc, fail)


# ==================== 测试三：台站系统偏差 ====================
def test3_systematic_bias():
    print("\n" + "=" * 72)
    print("测试三  台站系统偏差（每站所有读数同号 ±0.5 s）：7台穷举128种，20台抽60种")
    print("=" * 72)
    res, fail = [], 0
    for bits in it.product([0, 1], repeat=7):
        signs = np.array([0.5 if b else -0.5 for b in bits])
        Tn = np.sort(q4.T + signs[:, None], axis=1)
        best, mode = q4.solve_noisy(Tn)
        if best is None:
            fail += 1; continue
        e = q4.evaluate([x["theta"] for x in best[1]])
        if e is None: fail += 1
        else: res.append((bits, mode, e[:, 2].max()))
    arr = np.array([r[2] for r in res])
    print(f"  [7台穷举] 失败/混淆 {fail}/128；单残骸最大3D误差 "
          f"中位 {np.median(arr):.0f} m，95%分位 {np.percentile(arr, 95):.0f} m，"
          f"最大 {arr.max():.0f} m；超1 km占比 {(arr > 1000).mean() * 100:.1f}%")
    res.sort(key=lambda x: -x[2])
    print("  最恶劣3种符号模式:",
          [("".join("+" if b else "-" for b in r[0]), r[1], round(r[2])) for r in res[:3]])

    rng = np.random.default_rng(99); res20, fail20 = [], 0
    for _ in range(60):
        signs = rng.choice([-0.5, 0.5], size=q4.M)
        Tn = np.sort(q4.T20_TRUE + signs[:, None], axis=1)
        thetas = q4.solve_dense(Tn)
        if thetas is None:
            fail20 += 1; continue
        e = q4.evaluate(thetas)
        if e is None: fail20 += 1
        else: res20.append(e[:, 2].max())
    a = np.array(res20)
    print(f"  [20台抽样60种] 失败 {fail20}/60；单残骸最大3D误差 "
          f"中位 {np.median(a):.0f} m，95%分位 {np.percentile(a, 95):.0f} m，"
          f"最大 {a.max():.0f} m；超1 km占比 {(a > 1000).mean() * 100:.1f}%")


# ==================== 测试四：噪声幅度扫描（崩溃点分析） ====================
def test4_delta_sweep():
    print("\n" + "=" * 72)
    print("测试四  噪声幅度扫描：U(-Δ,+Δ)，Δ=0.1→1.0 s，定位模型的失效临界点")
    print("=" * 72)
    print(f"  {'Δ(s)':>5}{'关联成功':>10}{'95%分位(m)':>12}{'最大(m)':>10}{'>1km占比':>10}  模式")
    rng = np.random.default_rng(2024)
    for d, nmc in [(0.1, 100), (0.2, 100), (0.3, 100), (0.4, 100), (0.5, 100),
                   (0.6, 60), (0.7, 60), (0.8, 40), (0.9, 40), (1.0, 40)]:
        fail = 0; modes = {}; E = np.full((nmc, 4, 4), np.nan)
        for n in range(nmc):
            Tn = np.sort(q4.T + rng.uniform(-d, d, size=q4.T.shape), axis=1)
            best, mode = q4.solve_noisy(Tn)
            modes[mode] = modes.get(mode, 0) + 1
            if best is None:
                fail += 1; continue
            e = q4.evaluate([x["theta"] for x in best[1]])
            if e is None: fail += 1
            else: E[n] = e
        ok = ~np.isnan(E[:, 0, 0])
        v3 = E[ok][:, :, 2].ravel() if ok.any() else np.array([np.nan])
        print(f"  {d:>5.1f}{nmc - fail:>7}/{nmc}{np.nanpercentile(v3, 95):>12.0f}"
              f"{np.nanmax(v3):>10.0f}{(v3 > 1000).mean() * 100:>9.1f}%  {modes}", flush=True)
    # 20台在崩溃点 Δ=1.0 s 的对照
    rng = np.random.default_rng(41); fail = 0; E = np.full((60, 4, 4), np.nan)
    for n in range(60):
        Tn = np.sort(q4.T20_TRUE + rng.uniform(-1.0, 1.0, size=q4.T20_TRUE.shape), axis=1)
        thetas = q4.solve_dense(Tn)
        if thetas is None:
            fail += 1; continue
        e = q4.evaluate(thetas)
        if e is None: fail += 1
        else: E[n] = e
    _report("20台 Δ=1.0s对照", E, 60, fail)


# ==================== 测试五：单站粗差 ====================
def test5_gross_outlier(nmc=100, seed=51):
    print("\n" + "=" * 72)
    print("测试五  单站粗差：背景噪声 U(-0.5,+0.5) s + 随机一台设备一个读数叠加粗差")
    print("=" * 72)
    for g in (1.0, 2.0):
        rng = np.random.default_rng(seed)
        fail = 0; modes = {}; E = np.full((nmc, 4, 4), np.nan)
        for n in range(nmc):
            Tn = q4.T + rng.uniform(-0.5, 0.5, size=q4.T.shape)
            Tn[n % 7, int(rng.integers(0, 4))] += g   # 轮流污染每台设备
            Tn = np.sort(Tn, axis=1)
            best, mode = q4.solve_noisy(Tn)
            modes[mode] = modes.get(mode, 0) + 1
            if best is None:
                fail += 1; continue
            e = q4.evaluate([x["theta"] for x in best[1]])
            if e is None: fail += 1
            else: E[n] = e
        _report(f"7台 粗差+{g}s", E, nmc, fail, f"模式{modes}")


# ==================== 测试六：20台粗差对照 ====================
def test6_gross_outlier_20(nmc=60, seed=61):
    print("\n" + "=" * 72)
    print("测试六  20台台网 +2.0 s 单站粗差对照")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    fail = 0; E = np.full((nmc, 4, 4), np.nan)
    for n in range(nmc):
        Tn = q4.T20_TRUE + rng.uniform(-0.5, 0.5, size=q4.T20_TRUE.shape)
        Tn[int(rng.integers(0, q4.M)), int(rng.integers(0, 4))] += 2.0
        Tn = np.sort(Tn, axis=1)
        thetas = q4.solve_dense(Tn)
        if thetas is None:
            fail += 1; continue
        e = q4.evaluate(thetas)
        if e is None: fail += 1
        else: E[n] = e
    _report("20台 粗差+2.0s", E, nmc, fail)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    t0 = time.time()
    test1_common_shift()
    test2_boundary_sign()
    test3_systematic_bias()
    test4_delta_sweep()
    test5_gross_outlier()
    test6_gross_outlier_20()
    print(f"\n全部测试完成，总耗时 {(time.time() - t0) / 60:.1f} 分钟")
