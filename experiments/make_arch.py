#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成报告用的系统架构图。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200

OUT = Path(__file__).resolve().parents[1] / "figs" / "fig_arch.png"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2, edgecolor="#3a3a3a", facecolor=color))
    ax.text(x, y + h * 0.22, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1a1a1a")
    ax.text(x, y - h * 0.20, body, ha="center", va="center",
            fontsize=8.6, color="#333333", linespacing=1.5)


def arrow(ax, a, b, style="-|>", ls="-", color="#555555"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12,
                                 linewidth=1.1, linestyle=ls, color=color,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.36, 0.88, 0.40, 0.13, "用户事件",
        "点击 / 拖拽 / 双击 / 右键菜单 / 托盘", "#eef2f8")
    box(ax, 0.36, 0.65, 0.40, 0.15, "决策层 behavior_step",
        "四状态马尔可夫链 P　+　Beta(α, β) 亲密度", "#e6f0e6")
    box(ax, 0.36, 0.42, 0.40, 0.15, "物理层 physics",
        "阻尼振子 RK4 积分　+　抛体重力与碰撞", "#fdf0e3")
    box(ax, 0.36, 0.19, 0.40, 0.15, "表现层 paintEvent",
        "3×3 齐次仿射变换（缩放·旋转·平移）", "#f6e9f2")

    box(ax, 0.82, 0.53, 0.28, 0.22, "数学内核 mathcore.py",
        "无 Qt 依赖\nODE 积分器 / 马尔可夫链\nBeta 后验 / SVD / 最小二乘", "#ffffff")
    box(ax, 0.82, 0.16, 0.28, 0.11, "离线实验 experiments/",
        "复现实验、出图、误差度量", "#ffffff")

    arrow(ax, (0.36, 0.815), (0.36, 0.725))
    arrow(ax, (0.36, 0.575), (0.36, 0.495))
    arrow(ax, (0.36, 0.345), (0.36, 0.265))

    for y in (0.65, 0.42, 0.19):
        arrow(ax, (0.56, y), (0.68, 0.53), style="-|>", ls=(0, (3, 2)),
              color="#7a7a7a")
    arrow(ax, (0.82, 0.42), (0.82, 0.215), style="-|>", ls=(0, (3, 2)),
          color="#7a7a7a")

    ax.text(0.36, 0.045, "数据流：事件 → 决策 → 物理 → 渲染；虚线为调用依赖",
            ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
