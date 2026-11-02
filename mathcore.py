#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桌宠的数学核心
================

这个文件把桌宠里"凭感觉写"的几处逻辑换成了可以推导、可以量化的数学模型：

1. DampedOscillator  阻尼谐振子 m·x'' + c·x' + k·x = 0 的数值积分
                     （显式欧拉 / 半隐式欧拉 / 经典四阶 Runge–Kutta）
2. BehaviorChain     四状态马尔可夫链，负责决定桌宠"下一步做什么"
3. AffinityBeta      Beta–二项共轭先验，用交互次数估计"亲密度"
4. 低秩近似          SVD 截断重建立绘，用于低内存档位
5. 最小二乘          拟合拖拽轨迹，预测松手后的落点

所有函数都不依赖 Qt，可以单独跑实验（见 experiments/run_experiments.py）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

class DampedOscillator:
    """m·x'' + c·x' + k·x = 0

    状态量只有 (x, v)。解析解是 x(t)=A·exp(-ζω₀t)·cos(ω_d t+φ)，
    但桌宠每帧的 dt 是变化的（20 ms 计时器 + 系统抖动），
    用解析解要反复重算相位，不如直接数值积分。

    scheme 取值：
        "euler"        显式欧拉（原版代码用的就是这个）
        "semi"         半隐式欧拉（先更新 v 再更新 x）
        "rk4"          经典四阶 Runge–Kutta
    """

    def __init__(self, k: float = 90.0, c: float = 9.0, m: float = 1.0,
                 x: float = 0.0, v: float = 0.0, scheme: str = "rk4"):
        self.k = float(k)
        self.c = float(c)
        self.m = float(m)
        self.x = float(x)
        self.v = float(v)
        self.scheme = scheme

    @property
    def omega0(self) -> float:
        """固有角频率 ω₀ = √(k/m)。"""
        return math.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        """阻尼比 ζ = c / (2√(km))。ζ<1 欠阻尼，会回弹；ζ≥1 不回弹。"""
        return self.c / (2.0 * math.sqrt(self.k * self.m))

    def energy(self) -> float:
        """机械能 E = ½mv² + ½kx²。无阻尼时应当守恒，可用来检验积分器。"""
        return 0.5 * self.m * self.v * self.v + 0.5 * self.k * self.x * self.x

    def _accel(self, x: float, v: float) -> float:
        return (-self.k * x - self.c * v) / self.m

    def _f(self, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], (-self.k * y[0] - self.c * y[1]) / self.m])

    def step(self, dt: float) -> None:
        """推进一步。dt 由调用方给（秒）。"""
        if dt <= 0.0:
            return
        if self.scheme == "euler":
            a = self._accel(self.x, self.v)
            self.x += self.v * dt
            self.v += a * dt
        elif self.scheme == "semi":
            a = self._accel(self.x, self.v)
            self.v += a * dt
            self.x += self.v * dt
        elif self.scheme == "rk4":
            y = np.array([self.x, self.v], dtype=float)
            k1 = self._f(y)
            k2 = self._f(y + 0.5 * dt * k1)
            k3 = self._f(y + 0.5 * dt * k2)
            k4 = self._f(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            self.x, self.v = float(y[0]), float(y[1])
        else:
            raise ValueError(f"未知积分方案：{self.scheme}")

    def advance(self, dt: float, substeps: int = 1) -> None:
        """把一个帧间隔拆成 substeps 步，减少大步长带来的误差。"""
        h = dt / max(1, substeps)
        for _ in range(max(1, substeps)):
            self.step(h)


BEHAVIOR_STATES = ("idle", "look", "hop", "stretch")

# 行 = 当前状态，列 = 下一状态。每行和为 1。
# 设计原则：动作时长越长，留在原状态的概率越高（惯性）。
BEHAVIOR_P = np.array([
    #  idle  look   hop  stretch
    [0.55, 0.20, 0.15, 0.10],   # idle
    [0.35, 0.40, 0.05, 0.20],   # look
    [0.45, 0.10, 0.30, 0.15],   # hop
    [0.50, 0.15, 0.10, 0.25],   # stretch
], dtype=float)

# 每个状态的持续时长（秒），用均匀分布采样
BEHAVIOR_DURATION = {
    "idle":    (4.0, 10.0),
    "look":    (1.5, 4.0),
    "hop":     (0.6, 1.2),
    "stretch": (1.2, 2.5),
}


def stationary_distribution(P: np.ndarray, iters: int = 4000) -> np.ndarray:
    """幂迭代求平稳分布 π，满足 πP = π 且 Σπ=1。

    从均匀分布出发反复右乘 P，因为 P 不可约且非周期，所以收敛到 π。
    """
    n = P.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(iters):
        pi = pi @ P
    return pi / pi.sum()


class BehaviorChain:
    """马尔可夫行为链。

    与原来的 `hop_cd = random.uniform(7, 18)` 相比，链模型显式刻画了
    "上一个动作影响下一个动作"这一事实：跳跃之后更容易回到 idle，
    发呆久了更容易四处张望。平稳分布还能反推长期各动作的占比。
    """

    def __init__(self, P: np.ndarray = BEHAVIOR_P, seed: int | None = None,
                 hop_gain: float = 1.0):
        self.P = np.asarray(P, dtype=float)
        self.states = BEHAVIOR_STATES
        self.rng = np.random.default_rng(seed)
        self.hop_gain = hop_gain
        self.current = "idle"
        self.duration = 0.0
        self.timer = 0.0

    def _row(self, state: str, affinity: float) -> np.ndarray:
        """按亲密度调整转移概率：越亲越爱跳，再归一化。"""
        i = self.states.index(state)
        row = self.P[i].copy()
        j = self.states.index("hop")
        row[j] *= 0.5 + float(affinity) * self.hop_gain
        s = row.sum()
        return row / s

    def next_state(self, affinity: float = 0.5) -> str:
        row = self._row(self.current, affinity)
        idx = int(self.rng.choice(len(self.states), p=row))
        self.current = self.states[idx]
        lo, hi = BEHAVIOR_DURATION[self.current]
        self.duration = float(self.rng.uniform(lo, hi))
        self.timer = 0.0
        return self.current

    def update(self, dt: float, affinity: float = 0.5) -> str | None:
        """推进计时器；当前状态到期时返回新状态，否则返回 None。"""
        self.timer += dt
        if self.timer >= self.duration:
            return self.next_state(affinity)
        return None


@dataclass
class AffinityBeta:
    """用 Beta(α, β) 描述"亲密度"这一未知比例。

    θ 的物理含义：下一次交互被好好回应的概率。
    先验 Beta(2,2)（均值 0.5，偏保守）；
    每被戳一次 α+1，每被冷落（长时间无交互）一次 β+1。

    Beta 是二项分布的共轭先验，所以后验仍然是 Beta，
    只需要两个计数就能完成全部更新，不用保存历史。
    """

    alpha: float = 2.0
    beta: float = 2.0
    _hist: list = field(default_factory=list)

    def poke(self, n: int = 1) -> None:
        self.alpha += n

    def neglect(self, n: int = 1) -> None:
        self.beta += n

    @property
    def mean(self) -> float:
        """后验均值 E[θ] = α/(α+β)。"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """后验方差 Var[θ] = αβ / [(α+β)²(α+β+1)]。"""
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1.0))

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.beta(self.alpha, self.beta))


def svd_low_rank(img: np.ndarray, k: int) -> np.ndarray:
    """对 H×W×C 图像逐通道做截断 SVD，返回秩不超过 k 的重建结果。

    X ≈ X_k = Σ_{i=1}^{k} σ_i u_i v_iᵀ
    存储量从 m·n 降到 k·(m+n+1)。
    """
    out = np.empty_like(img)
    c = 1 if img.ndim == 2 else img.shape[2]
    for ch in range(c):
        X = img[:, :, ch] if img.ndim == 3 else img
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        kk = min(k, s.shape[0])
        out[:, :, ch] = (U[:, :kk] * s[:kk]) @ Vt[:kk, :]
    if img.ndim == 3:
        out = np.clip(out, 0.0, 1.0)
    return out


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    """峰值信噪比 PSNR = 10·log₁₀(MAX² / MSE)。"""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(peak * peak / mse)


def relative_fro_error(a: np.ndarray, b: np.ndarray) -> float:
    """相对 Frobenius 误差 ‖a-b‖_F / ‖a‖_F。"""
    return float(np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(a)))


def storage_ratio(m: int, n: int, k: int) -> float:
    """秩 k 近似的参数量占原矩阵的比例 k(m+n+1)/(mn)。"""
    return k * (m + n + 1.0) / (m * n)


def fit_quadratic(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """拟合 y(t) = a t² + b t + c，返回 [a, b, c]。

    写成超定方程 A·c = y，A = [t², t, 1]，
    最小二乘解由正规方程 AᵀA·c = Aᵀy 给出。
    这里直接调 lstsq（SVD 求解），比显式求逆数值稳定。
    """
    A = np.column_stack([t * t, t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def fit_linear(t: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """拟合 x(t) = u·t + x₀，返回 (u, x₀)。"""
    A = np.column_stack([t, np.ones_like(t)])
    (u, x0), *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(u), float(x0)


def predict_landing(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                    floor_y: float, g: float) -> tuple[float, float]:
    """由一段拖拽轨迹预测松手后的落点。

    竖直方向用二次多项式（自由落体的解析形式），水平方向用一次多项式
    （忽略空气阻力，水平速度守恒）。解 y(t) = floor_y 得到落地时刻，
    再代回 x(t)。

    返回 (t_hit, x_hit)，若轨迹本身已经在地面则返回 (0, 最后位置)。
    """
    if len(ts) < 3:
        return 0.0, float(xs[-1])
    a, b, c = fit_quadratic(ts, ys)
    disc = b * b - 4.0 * a * (c - floor_y)
    th_abs = None
    if abs(a) > 1e-9 and disc >= 0.0:
        r1 = (-b + math.sqrt(disc)) / (2.0 * a)
        r2 = (-b - math.sqrt(disc)) / (2.0 * a)
        for r in sorted([r1, r2]):
            if r > 1e-6:
                th_abs = r
                break
    if th_abs is None:
        if len(ts) >= 2:
            vy = (ys[-1] - ys[-2]) / max(1e-6, ts[-1] - ts[-2])
            th_abs = float(ts[-1]) + ((floor_y - ys[-1]) / vy if abs(vy) > 1e-6 else 0.0)
        else:
            th_abs = float(ts[-1])
    u, x0 = fit_linear(ts, xs)
    return max(0.0, th_abs - float(ts[-1])), float(u * th_abs + x0)


def projectile_landing(x0: float, h0: float, vx: float, vy: float,
                       g: float) -> tuple[float, float]:
    """抛体落点：h(t) = h₀ − vy·t − ½g·t²，令 h=0 得
    t = (−vy + √(vy² + 2g·h₀)) / g，落点 x = x₀ + vx·t。

    约定 y 轴向下为正：h₀ 为离地高度，vy 为向下速度，vx 为水平速度。
    """
    disc = vy * vy + 2.0 * g * max(0.0, h0)
    t = max(0.0, (-vy + math.sqrt(max(0.0, disc))) / g)
    return t, x0 + vx * t
