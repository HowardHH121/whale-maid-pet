#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桌宠数学模块的实验与配图。

运行：python experiments/run_experiments.py
输出：F:\\人工智能数学基础\\figs\\*.png 与 metrics.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mathcore as mc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parents[1] / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"

METRICS: dict = {}


def analytic_underdamped(t, k, c, m, x0, v0):
    w0 = math.sqrt(k / m)
    z = c / (2.0 * math.sqrt(k * m))
    if z >= 1.0:
        raise ValueError("only underdamped")
    wd = w0 * math.sqrt(1.0 - z * z)
    return np.exp(-z * w0 * t) * (
        x0 * np.cos(wd * t) + (v0 + z * w0 * x0) / wd * np.sin(wd * t)
    )


def exp_oscillator():
    k, c, m = 90.0, 0.0, 1.0
    x0, v0 = 0.2, 0.0
    h, T = 1.0 / 60.0, 2.0
    n = int(T / h)
    t = np.arange(n + 1) * h
    ref = analytic_underdamped(t, k, c, m, x0, v0)
    e0 = 0.5 * k * x0 * x0

    schemes = {"euler": "显式欧拉", "semi": "半隐式欧拉", "rk4": "RK4"}
    curves, energy_drift, amp_err = {}, {}, {}
    for key in schemes:
        o = mc.DampedOscillator(k=k, c=c, m=m, x=x0, v=v0, scheme=key)
        xs, es = [o.x], [o.energy()]
        for _ in range(n):
            o.step(h)
            xs.append(o.x)
            es.append(o.energy())
        xs = np.array(xs)
        es = np.array(es)
        curves[key] = xs
        energy_drift[key] = float((es[-1] - e0) / e0)
        amp_err[key] = float(np.max(np.abs(xs - ref)))

    k2, c2 = 90.0, 9.0
    damped = {}
    for key in schemes:
        o = mc.DampedOscillator(k=k2, c=c2, m=m, x=0.2, v=0, scheme=key)
        seq = []
        for _ in range(n):
            o.step(h)
            seq.append(o.x)
        damped[key] = np.array(seq)
    damped_dev = {key: float(np.max(np.abs(damped[key] - damped["rk4"])))
                  for key in schemes}

    METRICS["oscillator"] = {
        "omega0": math.sqrt(k / m),
        "zeta": c / (2 * math.sqrt(k * m)),
        "steps": n,
        "dt": h,
        "energy_drift_2s": energy_drift,
        "max_abs_error_vs_analytic": amp_err,
        "damped_max_dev_vs_rk4": damped_dev,
        "app_zeta": c2 / (2 * math.sqrt(k2 * m)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(t, ref, "k--", lw=1.2, label="解析解")
    for key, lab in schemes.items():
        ax.plot(t, curves[key], lw=1.1, label=lab)
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("位移 x")
    ax.set_title("(a) 无阻尼振子：位移曲线")
    ax.legend(fontsize=8)
    ax.set_ylim(-1.2, 1.2)

    ax = axes[1]
    for key, lab in schemes.items():
        o = mc.DampedOscillator(k=k, c=c, m=m, x=x0, v=v0, scheme=key)
        es = [o.energy()]
        for _ in range(n):
            o.step(h)
            es.append(o.energy())
        ax.plot(t, np.abs((np.array(es) - e0) / e0), lw=1.2, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("相对能量漂移 |E−E₀|/E₀")
    ax.set_title("(b) 能量误差（对数坐标）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_energy.png")
    plt.close(fig)


def exp_markov():
    P = mc.BEHAVIOR_P
    pi = mc.stationary_distribution(P)
    residual = float(np.max(np.abs(pi @ P - pi)))

    rng = np.random.default_rng(20261102)
    chain = mc.BehaviorChain(seed=20261102)
    dt = 1.0 / 30.0
    T = 20000.0
    steps = int(T / dt)
    time_in = {s: 0.0 for s in chain.states}
    visits = {s: 0 for s in chain.states}
    visits[chain.current] += 1
    switches = 0
    durations = []
    cur_dur = 0.0
    for _ in range(steps):
        new = chain.update(dt, affinity=0.5)
        time_in[chain.current] += dt
        cur_dur += dt
        if new is not None:
            switches += 1
            visits[new] += 1
            durations.append(cur_dur)
            cur_dur = 0.0
    total_t = sum(time_in.values())
    total_v = sum(visits.values())
    empirical_t = np.array([time_in[s] / total_t for s in chain.states])
    empirical_v = np.array([visits[s] / total_v for s in chain.states])
    l1_visit = float(np.abs(empirical_v - pi).sum())

    mean_dur = np.array([np.mean(mc.BEHAVIOR_DURATION[s]) for s in chain.states])
    q = pi * mean_dur / float((pi * mean_dur).sum())
    l1_time = float(np.abs(empirical_t - q).sum())

    ch2 = mc.BehaviorChain(seed=99)
    v2 = np.zeros(4)
    for _ in range(200000):
        ch2.next_state(0.5)
        v2[ch2.states.index(ch2.current)] += 1
    freq2 = v2 / v2.sum()
    l1_visit = float(np.abs(freq2 - pi).sum())

    # 对照：原方案的均匀随机跳（固定 7~18 s 一次），动作种类只有"跳/不跳"
    rng2 = np.random.default_rng(7)
    uni_dur = rng2.uniform(7.0, 18.0, size=2000)

    # 亲密度：200 次交互，回应概率等于当前 θ 的真值
    theta_true = 0.72
    aff = mc.AffinityBeta(2.0, 2.0)
    means, alphas, betas = [aff.mean], [aff.alpha], [aff.beta]
    rng3 = np.random.default_rng(11)
    for _ in range(200):
        if rng3.random() < theta_true:
            aff.poke()
        else:
            aff.neglect()
        means.append(aff.mean)
        alphas.append(aff.alpha)
        betas.append(aff.beta)

    METRICS["markov"] = {
        "P": P.tolist(),
        "stationary": pi.tolist(),
        "residual_piP_pi": residual,
        "simulated_seconds": T,
        "visit_frequency": freq2.tolist(),
        "l1_visit_vs_pi": l1_visit,
        "time_fraction": empirical_t.tolist(),
        "theoretical_time_fraction": q.tolist(),
        "l1_time_vs_theory": l1_time,
        "mean_duration": mean_dur.tolist(),
        "switches_per_minute": switches / (T / 60.0),
        "mean_dwell": float(np.mean(durations)),
        "uniform_mean_interval": float(uni_dur.mean()),
        "uniform_interval_std": float(uni_dur.std()),
        "affinity_final_alpha": aff.alpha,
        "affinity_final_beta": aff.beta,
        "affinity_final_mean": aff.mean,
        "affinity_true_theta": theta_true,
        "affinity_final_sd": math.sqrt(aff.variance),
    }

    fig = plt.figure(figsize=(11, 3.4))
    ax = fig.add_subplot(1, 3, 1)
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=0.6)
    ax.set_xticks(range(4), chain.states)
    ax.set_yticks(range(4), chain.states)
    ax.set_xlabel("下一状态")
    ax.set_ylabel("当前状态")
    ax.set_title("(a) 转移矩阵 $P$")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                    color="white" if P[i, j] > 0.35 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(1, 3, 2)
    x = np.arange(4)
    ax.bar(x - 0.18, pi, 0.36, label="平稳分布 $\\pi$")
    ax.bar(x + 0.18, freq2, 0.36, label="20 万次转移仿真")
    ax.set_xticks(x, chain.states)
    ax.set_ylabel("状态访问占比")
    ax.set_ylim(0, 0.6)
    ax.set_title(f"(b) 访问频率与 $\\pi$ 对比（L1={l1_visit:.3f}）")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    idx = np.arange(len(means))
    ax.plot(idx, means, lw=1.8, label="后验均值 $E[\\theta]$")
    ax.fill_between(idx, np.array(means) - 2 * np.sqrt(
        np.array(alphas) * np.array(betas) /
        ((np.array(alphas) + np.array(betas)) ** 2 *
         (np.array(alphas) + np.array(betas) + 1))),
        np.array(means) + 2 * np.sqrt(
        np.array(alphas) * np.array(betas) /
        ((np.array(alphas) + np.array(betas)) ** 2 *
         (np.array(alphas) + np.array(betas) + 1))),
        alpha=0.2, label="±2 标准差")
    ax.axhline(theta_true, color="k", ls="--", lw=1, label="真实 θ")
    ax.set_xlabel("交互次数")
    ax.set_ylabel("亲密度")
    ax.set_title("(c) Beta 后验收敛")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_markov.png")
    plt.close(fig)


def exp_svd():
    img = np.asarray(Image.open(ROOT / "pet.png").convert("RGBA"), dtype=np.float64) / 255.0
    h, w, _ = img.shape
    alpha = img[:, :, 3]
    fg = alpha > 0.5

    ranks = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    recons = {}
    psnrs, ratios, mious, times = [], [], [], []
    t_total0 = time.perf_counter()
    svds = [np.linalg.svd(img[:, :, ch], full_matrices=False) for ch in range(4)]
    for k in ranks:
        t0 = time.perf_counter()
        out = np.empty_like(img)
        for ch, (U, s, Vt) in enumerate(svds):
            kk = min(k, s.shape[0])
            out[:, :, ch] = np.clip((U[:, :kk] * s[:kk]) @ Vt[:kk, :], 0, 1)
        dt = time.perf_counter() - t0
        times.append(dt)
        recons[k] = out
        rgb = out[:, :, :3]
        m = fg
        mse = float(np.mean((rgb[m] - img[:, :, :3][m]) ** 2))
        psnrs.append(10 * math.log10(1.0 / mse) if mse > 0 else 99.0)
        am = out[:, :, 3] > 0.5
        mious.append(float((am & fg).sum() / max(1, (am | fg).sum())))
        ratios.append(mc.storage_ratio(h, w, k))
    t_total = time.perf_counter() - t_total0

    METRICS["svd"] = {
        "image_size": [h, w],
        "ranks": ranks,
        "psnr_db": [round(v, 3) for v in psnrs],
        "storage_ratio": [round(v, 4) for v in ratios],
        "alpha_miou": [round(v, 4) for v in mious],
        "recon_time_s": [round(v, 3) for v in times],
    }

    icon = np.asarray(
        Image.fromarray((img * 255).astype(np.uint8), "RGBA").resize((64, 64), Image.LANCZOS),
        dtype=np.float64) / 255.0
    icon_ranks = [6, 8, 12, 16, 24, 32]
    icon_psnr, icon_ratio = [], []
    icon_svds = [np.linalg.svd(icon[:, :, ch], full_matrices=False) for ch in range(4)]
    for k in icon_ranks:
        out = np.empty_like(icon)
        for ch, (U, s, Vt) in enumerate(icon_svds):
            kk = min(k, s.shape[0])
            out[:, :, ch] = np.clip((U[:, :kk] * s[:kk]) @ Vt[:kk, :], 0, 1)
        msk = icon[:, :, 3] > 0.5
        mse = float(np.mean((out[:, :, :3][msk] - icon[:, :, :3][msk]) ** 2))
        icon_psnr.append(10 * math.log10(1 / mse) if mse > 0 else 99.0)
        icon_ratio.append(mc.storage_ratio(64, 64, k))
    METRICS["svd"]["icon_ranks"] = icon_ranks
    METRICS["svd"]["icon_psnr_db"] = [round(v, 3) for v in icon_psnr]
    METRICS["svd"]["icon_storage_ratio"] = [round(v, 4) for v in icon_ratio]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(ranks, psnrs, "o-", color="#1f77b4", label="PSNR / dB")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("保留秩 k")
    ax.set_ylabel("PSNR / dB", color="#1f77b4")
    ax.axhline(35, color="gray", ls=":", lw=1)
    ax.text(ranks[0], 35.6, "35 dB 参考线", fontsize=8, color="gray")
    ax2 = ax.twinx()
    ax2.plot(ranks, [r * 100 for r in ratios], "s--", color="#d62728")
    ax2.set_ylabel("参数量占比 / %", color="#d62728")
    ax.set_title("(a) 秩 k 与重建质量、存储量")

    show = [48, 192, 384]
    fig2, axs = plt.subplots(1, 4, figsize=(11, 4.2))
    disp = img.copy()
    axs[0].imshow(disp)
    axs[0].set_title(f"原图（{w}×{h}）", fontsize=10)
    for i, k in enumerate(show):
        axs[i + 1].imshow(recons[k])
        axs[i + 1].set_title(f"k={k}，PSNR={psnrs[ranks.index(k)]:.1f} dB", fontsize=10)
    for a in axs:
        a.axis("off")
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_svd.png")
    plt.close(fig2)
    plt.close(fig)
    METRICS["svd"]["total_time_s"] = round(t_total, 2)


def exp_affine():
    cx, cy = 407.0, 455.0
    theta = math.radians(1.2)
    sx, sy = 1.0 - 0.0072, 1.0 + 0.010

    def T(tx, ty):
        return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)

    def R(a):
        return np.array([[math.cos(a), -math.sin(a), 0],
                         [math.sin(a), math.cos(a), 0],
                         [0, 0, 1]], dtype=float)

    def S(a, b):
        return np.array([[a, 0, 0], [0, b, 0], [0, 0, 1]], dtype=float)

    pts = np.array([[0, 0, 1], [814, 0, 1], [0, 910, 1], [814, 910, 1]],
                   dtype=float).T
    A1 = T(cx, cy) @ R(theta) @ S(sx, sy) @ T(-cx, -cy)
    A2 = T(cx, cy) @ S(sx, sy) @ R(theta) @ T(-cx, -cy)
    d = np.linalg.norm(A1 @ pts - A2 @ pts, axis=0)

    METRICS["affine"] = {
        "theta_deg": 1.2,
        "sx": sx,
        "sy": sy,
        "det": float(np.linalg.det(A1[:2, :2])),
        "max_order_diff_px": float(d.max()),
        "corner_diff_px": [round(float(v), 3) for v in d],
        "recovered_theta_deg": math.degrees(math.atan2(A1[1, 0], A1[0, 0])),
    }


def exp_least_squares():
    g = 2400.0
    rng = np.random.default_rng(2026)
    n_trial = 800
    err_ls, err_naive, err_3p = [], [], []
    r2s = []
    sample = None
    for _ in range(n_trial):
        x0 = rng.uniform(-200, 200)
        h0 = rng.uniform(60, 260)
        vx = rng.uniform(-500, 500)
        v0 = rng.uniform(100, 600)
        t_hit = (v0 + math.sqrt(v0 * v0 + 2 * g * h0)) / g
        x_true = x0 + vx * t_hit
        ts = np.arange(0, 0.10, 1 / 120.0)
        xs = x0 + vx * ts + rng.normal(0, 1.2, ts.shape)
        hs = h0 + v0 * ts - 0.5 * g * ts ** 2 + rng.normal(0, 1.2, ts.shape)
        th, x_pred = mc.predict_landing(ts, xs, hs, 0.0, g)
        err_ls.append(abs(x_pred - x_true))
        _, x_3p = mc.predict_landing(ts[-3:], xs[-3:], hs[-3:], 0.0, g)
        err_3p.append(abs(x_3p - x_true))
        vh_last = (hs[-1] - hs[-2]) / (ts[-1] - ts[-2])
        vx_last = (xs[-1] - xs[-2]) / (ts[-1] - ts[-2])
        t_naive = (0.0 - hs[-1]) / vh_last if abs(vh_last) > 1e-6 else 0.0
        err_naive.append(abs(xs[-1] + vx_last * t_naive - x_true))
        if sample is None:
            a, b, c = mc.fit_quadratic(ts, hs)
            pred = a * ts ** 2 + b * ts + c
            ss_res = float(np.sum((hs - pred) ** 2))
            ss_tot = float(np.sum((hs - hs.mean()) ** 2))
            r2s.append(1 - ss_res / max(1e-12, ss_tot))
            sample = (ts, xs, hs, x_true, th)

    METRICS["least_squares"] = {
        "trials": n_trial,
        "noise_sigma_px": 1.2,
        "samples_per_track": 12,
        "mae_lstsq_px": float(np.mean(err_ls)),
        "mae_3pt_px": float(np.mean(err_3p)),
        "mae_naive_px": float(np.mean(err_naive)),
        "p90_lstsq_px": float(np.percentile(err_ls, 90)),
        "p90_3pt_px": float(np.percentile(err_3p, 90)),
        "p90_naive_px": float(np.percentile(err_naive, 90)),
        "r2_example": float(r2s[0]),
    }
    ts, xs, hs, x_true, th = sample
    a, b, c = mc.fit_quadratic(ts, hs)
    u, x0f = mc.fit_linear(ts, xs)
    th_abs = float(ts[-1]) + th
    tt = np.linspace(0, th_abs, 100)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.scatter(ts * 1000, hs, s=14, label="采样点", zorder=3)
    ax.plot(tt * 1000, a * tt ** 2 + b * tt + c, lw=1.4, label="二次最小二乘拟合")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(th_abs * 1000, color="gray", ls=":", lw=0.9,
               label=f"预测落地 {th_abs * 1000:.0f} ms")
    ax.set_xlabel("时间 / ms")
    ax.set_ylabel("离地高度 h / px")
    ax.set_title("(a) 拖拽轨迹拟合")
    ax.legend(fontsize=8)

    ax = axes[1]
    bp = ax.boxplot([err_ls, err_3p, err_naive],
                    tick_labels=["12 点最小二乘", "3 点最小二乘", "末速外推"],
                    widths=0.45, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#1f77b4", "#2ca02c", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("落点预测绝对误差 / px")
    ax.set_title("(b) 800 次随机轨迹的预测误差")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fit.png")
    plt.close(fig)


def main():
    exp_oscillator()
    print("[1/4] oscillator ok", flush=True)
    exp_markov()
    print("[2/4] markov ok", flush=True)
    exp_svd()
    print("[3/5] svd ok", flush=True)
    exp_affine()
    print("[4/5] affine ok", flush=True)
    exp_least_squares()
    print("[5/5] least squares ok", flush=True)
    (OUT / "metrics.json").write_text(
        json.dumps(METRICS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(METRICS, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
