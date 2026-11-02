#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大肥鱼桌宠 · 鲸鱼女仆娘（AI 立绘版 + 数学内核）
================================================
立绘：pet.png（AI 生成 + 程序抠图）
数学：mathcore.py（RK4 阻尼振子 / 马尔可夫行为链 / Beta 亲密度 / 最小二乘）

为什么用立绘而不是程序画
------------------------
手绘那种柔和渐变、发丝细节程序画不出来。代价是眼睛不能追鼠标，
所以用「呼吸 + 摇摆 + 弹跳 + 挤压拉伸」把"活着"的感觉补回来。

操作
----
拖动    按住拖走，甩出去会带着速度飞出去并在落点显示标记
拖文件  把文件/文件夹拖到它身上 = 移入回收站，并播放"吃"的动画（可右键关闭）
单击    被戳：压扁回弹 + 冒泡泡
双击    鼓成球
右键    大小 / 置顶 / 落地模式 / 亲密度 / 隐藏 / 退出
托盘    左键唤回，右键同菜单
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QElapsedTimer, QFileInfo, QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QGuiApplication,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileIconProvider,
    QMenu,
    QSystemTrayIcon,
    QWidget,
)

import mathcore as mc

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).parent
    ASSET_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    APP_DIR = Path(__file__).parent
    ASSET_DIR = APP_DIR

SPRITE = ASSET_DIR / "pet.png"
CONFIG = APP_DIR / "pet_config.json"

SPRITE_H = 910
BASE_H = 300
MARGIN_X = 24
MARGIN_TOP = 48
GRAVITY = 2400.0
MAX_THROW = 2400.0
NEGLECT_SECONDS = 120.0


def load_cfg() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cfg(cfg: dict) -> None:
    try:
        CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def recycle_paths(paths: list[str]) -> bool:
    """把文件/文件夹移入 Windows 回收站（可还原），不弹系统确认框。

    调用 shell32.SHFileOperationW，FOF_ALLOWUNDO 是"进回收站"的关键位；
    去掉它就会变成不可恢复的永久删除。
    """
    if sys.platform != "win32" or not paths:
        return False
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_ushort),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE = 3
    FOF_SILENT = 0x0004
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040
    FOF_NOERRORUI = 0x0400

    src = "\0".join(str(p) for p in paths) + "\0\0"
    buf = ctypes.create_unicode_buffer(len(src) + 1)
    buf[:len(src)] = src
    op = SHFILEOPSTRUCTW(
        None, FO_DELETE, ctypes.cast(buf, wintypes.LPCWSTR), None,
        FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI,
        False, None, None)
    res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return res == 0 and not op.fAnyOperationsAborted


class WhalePet(QWidget):
    def __init__(self):
        super().__init__()
        cfg = load_cfg()
        self.scale_f = float(cfg.get("scale", 1.0))
        self.always_on_top = bool(cfg.get("top", True))
        self.gravity_on = bool(cfg.get("gravity", True))

        self.src = QPixmap(str(SPRITE))
        if self.src.isNull():
            raise SystemExit(f"找不到立绘：{SPRITE}")

        self._apply_flags()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

        self.osc = mc.DampedOscillator(
            k=90.0, c=9.0, m=1.0, scheme=cfg.get("integrator", "rk4"))
        self.chain = mc.BehaviorChain(hop_gain=float(cfg.get("hop_gain", 1.0)))
        self.affinity = mc.AffinityBeta(
            float(cfg.get("alpha", 2.0)), float(cfg.get("beta", 2.0)))
        self.behavior = self.chain.current
        self.idle_time = 0.0
        self.lean = 0.0
        self.stretch = 0.0
        self.bubble = 0.0

        self.eat_enabled = bool(cfg.get("eat_files", True))
        self.hover_eat = False
        self.foods: list[dict] = []
        self.eat_t = 0.0
        self.eat_label = ""
        self.eat_alpha = 0.0
        self.chew = 0.0
        self.belly = 0.0
        self.setAcceptDrops(True)

        self.t = 0.0
        self.poke = 0.0
        self.puff = 0.0
        self.bubbles: list[dict] = []

        self.vy = 0.0
        self.vx = 0.0
        self.on_ground = False
        self.dragging = False
        self.trace: list[tuple[float, float, float]] = []
        self.landing: tuple[float, float] | None = None
        self._drag_off = QPointF()
        self._press_pos = None
        self._moved = 0.0
        self._press_ms = 0

        self._rebuild()
        self._restore_pos(cfg)

        self.tray: QSystemTrayIcon | None = None
        self.clock = QElapsedTimer()
        self.clock.start()
        self._last_ms = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)

    # -------------------------------------------------- 数学状态

    @property
    def squash(self) -> float:
        return self.osc.x

    @squash.setter
    def squash(self, v: float) -> None:
        self.osc.x = float(v)
        self.osc.v = 0.0

    def affinity_label(self) -> str:
        a = self.affinity.mean
        mood = "很黏你" if a > 0.72 else ("还行" if a > 0.5 else "有点冷淡")
        return f"亲密度 {a:.2f}（{mood}）"

    # -------------------------------------------------- 尺寸 / 窗口

    def _apply_flags(self):
        flags = Qt.FramelessWindowHint | Qt.Tool
        if self.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def kill_dwm_border(self):
        """Windows 11 会给窗口画一圈 1px 圆角描边，透明窗口上很脏，这里关掉。"""
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            val = ctypes.c_int(0xFFFFFFFE)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 34, ctypes.byref(val), 4)
            pref = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 33, ctypes.byref(pref), 4)
        except Exception as exc:                    # noqa: BLE001
            print("dwm border skip:", exc)

    def _rebuild(self):
        h = int(BASE_H * self.scale_f)
        w = int(h * self.src.width() / self.src.height())
        self.pix = self.src.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setFixedSize(self.pix.width() + MARGIN_X * 2, self.pix.height() + MARGIN_TOP)

    def _restore_pos(self, cfg: dict):
        sw = QGuiApplication.primaryScreen().availableGeometry()
        x = cfg.get("x", sw.right() - self.width() - 40)
        y = cfg.get("y", sw.bottom() - self.height() + 1)
        if not sw.adjusted(-300, -300, 60, 60).contains(int(x), int(y)):
            x = sw.right() - self.width() - 40
            y = sw.bottom() - self.height() + 1
        self.move(int(x), int(y))

    def _save_cfg(self):
        save_cfg({
            "x": self.x(), "y": self.y(),
            "scale": self.scale_f,
            "top": self.always_on_top, "gravity": self.gravity_on,
            "alpha": round(self.affinity.alpha, 3),
            "beta": round(self.affinity.beta, 3),
            "integrator": self.osc.scheme,
            "eat_files": self.eat_enabled,
        })

    def _floor_y(self) -> int:
        scr = self.screen() or QGuiApplication.primaryScreen()
        return scr.availableGeometry().bottom() - self.height() + 1

    # -------------------------------------------------- 主循环

    def tick(self):
        now = self.clock.elapsed()
        dt = min(0.05, (now - self._last_ms) / 1000.0) if self._last_ms else 0.02
        self._last_ms = now
        self.anim(dt)
        self.physics(dt)
        self.update()

    def anim(self, dt: float):
        self.t += dt
        self.poke = max(0.0, self.poke - dt * 2.2)
        self.puff = max(0.0, self.puff - dt * 1.6)

        self.osc.advance(dt, substeps=2)
        self.osc.x = max(-0.35, min(0.20, self.osc.x))

        self.behavior_step(dt)

        if self.behavior == "look" and not self.dragging:
            want = 1.0 if QCursor.pos().x() > self.x() + self.width() / 2 else -1.0
        else:
            want = 0.0
        self.lean += (want - self.lean) * min(1.0, 5.0 * dt)

        want_stretch = 1.0 if self.behavior == "stretch" else 0.0
        self.stretch += (want_stretch - self.stretch) * min(1.0, 4.0 * dt)

        show_bubble = self.behavior in ("look", "stretch") and not self.dragging
        self.bubble += ((1.0 if show_bubble else 0.0) - self.bubble) * min(1.0, 6.0 * dt)

        self._eat_tick(dt)

        for b in list(self.bubbles):
            b["y"] -= b["vy"] * dt
            b["x"] += math.sin(self.t * 3.0 + b["ph"]) * 0.5
            b["life"] -= dt * 0.85
            if b["life"] <= 0:
                self.bubbles.remove(b)

        if self.landing:
            x, life = self.landing
            if life - dt <= 0.0:
                self.landing = None
            else:
                self.landing = (x, life - dt)

    def _eat_tick(self, dt: float):
        self.chew = 0.0
        self.belly = 0.0
        if self.hover_eat:
            self.eat_label = "给我吃？"
            self.eat_alpha = min(1.0, self.eat_alpha + dt * 8.0)
            return
        if not self.foods:
            self.eat_alpha = max(0.0, self.eat_alpha - dt * 2.2)
            return

        self.eat_t += dt
        fly = 0.55
        chew_end = fly + 0.22 + 0.05 * len(self.foods)
        if self.eat_t < fly:
            self.eat_label = "啊呜——"
            self.eat_alpha = min(1.0, self.eat_alpha + dt * 10.0)
        elif self.eat_t < chew_end:
            self.chew = 0.055 * math.sin((self.eat_t - fly) * 2.0 * math.pi * 4.5)
            self.eat_alpha = min(1.0, self.eat_alpha + dt * 10.0)
        else:
            k = max(0.0, 1.0 - (self.eat_t - chew_end) / 0.55)
            self.belly = 0.075 * k
            self.eat_label = "嗝～"
            self.eat_alpha = k
        if self.eat_t > chew_end + 0.55:
            self.foods.clear()
            self.eat_t = 0.0

    def behavior_step(self, dt: float):
        if self.dragging:
            return
        self.idle_time += dt
        if self.idle_time > NEGLECT_SECONDS:
            self.affinity.neglect()
            self.idle_time = 0.0

        new = self.chain.update(dt, self.affinity.mean)
        if new:
            self.behavior = new
            self.enter_behavior(new)

    def enter_behavior(self, name: str):
        if name == "hop" and self.gravity_on and self.on_ground:
            self.vy = -(360.0 + 200.0 * self.affinity.mean)
            self.on_ground = False
            self.squash = 0.14
        elif name == "stretch":
            self.puff = max(self.puff, 0.75)

    def physics(self, dt: float):
        if not self.gravity_on or self.dragging:
            return
        floor = self._floor_y()
        x, y = self.x(), self.y()

        if self.on_ground and y >= floor:
            y = floor
            self.vx *= max(0.0, 1.0 - 6.0 * dt)
            if abs(self.vx) < 4.0:
                self.vx = 0.0
        else:
            self.vy += GRAVITY * dt
            y += self.vy * dt
            scr0 = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
            if y < scr0.top():
                y = scr0.top()
                self.vy = abs(self.vy) * 0.45
            if y >= floor:
                y = floor
                speed = self.vy
                if speed > 220.0:
                    self.squash = -min(0.32, speed / 2800.0)
                    self.vy = -speed * 0.42
                    self.on_ground = False
                    self.spawn_bubbles(3)
                    if speed > 900.0:
                        self.poke = max(self.poke, 0.5)
                else:
                    self.vy = 0.0
                    self.on_ground = True

        if abs(self.vx) > 0.5:
            x += self.vx * dt
            scr = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
            left, right = scr.left(), scr.right() - self.width()
            if x < left:
                x, self.vx = left, -self.vx * 0.55
            elif x > right:
                x, self.vx = right, -self.vx * 0.55

        self.move(int(x), int(y))

    def spawn_bubbles(self, n: int = 3):
        ph = random.uniform(0, 6.28)
        for i in range(n):
            self.bubbles.append({
                "x": self.width() * 0.5 + random.uniform(-26, 26),
                "y": self.height() * 0.32 + random.uniform(-10, 10),
                "r": random.uniform(2.2, 5.2),
                "vy": random.uniform(26, 50),
                "life": random.uniform(0.8, 1.5),
                "ph": ph + i,
            })

    # -------------------------------------------------- 绘制

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self.gravity_on:
            dist = max(0.0, self._floor_y() - self.y())
            k = max(0.0, 1.0 - dist / 260.0)
            if k > 0.02:
                rx = self.pix.width() * 0.32 * (0.7 + 0.3 * k)
                grad = QRadialGradient(QPointF(0, 0), rx)
                grad.setColorAt(0.0, QColor(0, 0, 0, int(80 * k)))
                grad.setColorAt(0.6, QColor(0, 0, 0, int(34 * k)))
                grad.setColorAt(1.0, QColor(0, 0, 0, 0))
                p.save()
                p.translate(self.width() / 2.0, self.height() - 3)
                p.scale(1.0, 0.30)
                p.setPen(Qt.NoPen)
                p.setBrush(grad)
                p.drawEllipse(QPointF(0, 0), rx, rx)
                p.restore()

        if self.landing:
            xh, life = self.landing
            alpha = int(170 * max(0.0, min(1.0, life / 0.9)))
            p.setPen(QPen(QColor(255, 120, 180, alpha), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(xh - self.x(), self.height() - 6), 18, 7)

        breathe = math.sin(self.t * 1.6) * 0.010
        rot = math.sin(self.t * 0.7) * 1.2 + self.lean * 9.0
        bob = math.sin(self.t * 1.6 + 0.6) * 1.4

        sy = ((1.0 + self.squash + breathe + self.chew) * (1.0 + 0.05 * self.puff)
              * (1.0 + 0.12 * self.stretch + self.belly))
        sx = ((1.0 - self.squash * 0.6 - breathe * 0.7) * (1.0 + 0.05 * self.puff)
              * (1.0 - 0.055 * self.stretch))

        p.save()
        p.translate(self.width() / 2.0 + self.lean * 10.0, self.height() + bob)
        p.rotate(rot)
        p.scale(sx, sy)
        p.translate(-self.pix.width() / 2.0, -self.pix.height())
        p.drawPixmap(0, 0, self.pix)

        for b in self.bubbles:
            a = int(210 * max(0.0, min(1.0, b["life"])))
            p.setPen(QColor(200, 225, 245, a))
            p.setBrush(QColor(255, 255, 255, 30))
            r = b["r"]
            p.drawEllipse(QPointF(b["x"] - self.width() / 2.0,
                                  b["y"] - self.height()), r, r)
        p.restore()

        if self.foods:
            mouth = QPointF(self.width() / 2.0 + self.lean * 10.0,
                            self.height() - self.pix.height() * 0.74)
            for f in self.foods:
                t = self.eat_t - f["t0"]
                if t <= 0.0:
                    continue
                prog = min(1.0, t / 0.55)
                u = 1.0 - prog
                a, b = f["from"], f["ctrl"]
                x = u * u * a.x() + 2 * u * prog * b.x() + prog * prog * mouth.x()
                y = u * u * a.y() + 2 * u * prog * b.y() + prog * prog * mouth.y()
                size = 46.0 * (1.0 - 0.82 * prog)
                p.setOpacity(1.0 - 0.55 * prog)
                p.drawPixmap(QRectF(x - size / 2, y - size / 2, size, size),
                             f["pm"], QRectF(f["pm"].rect()))
                p.setOpacity(1.0)

        self._draw_bubble(p)

    def _draw_bubble(self, p):
        if self.eat_alpha > self.bubble and self.eat_label:
            text = self.eat_label
            love = min(1.0, self.eat_alpha)
        elif self.bubble > 0.02:
            text = "？" if self.behavior == "look" else "～"
            love = min(1.0, self.bubble)
        else:
            return
        alpha = int(235 * love)
        p.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        fm = p.fontMetrics()
        w = max(30.0, fm.horizontalAdvance(text) + 22.0)
        h = fm.height() + 8.0
        cx = self.width() / 2.0 + self.lean * 10.0
        rect = QRectF(cx - w / 2.0, MARGIN_TOP * 0.15, w, h)
        p.setPen(QPen(QColor(205, 214, 230, alpha), 1.0))
        p.setBrush(QColor(255, 255, 255, int(alpha * 0.93)))
        p.drawRoundedRect(rect, h / 2.0, h / 2.0)
        p.setPen(QColor(86, 100, 126, alpha))
        p.drawText(rect, Qt.AlignCenter, text)

    # -------------------------------------------------- 交互

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press_pos = e.globalPosition()
            self._press_ms = self.clock.elapsed()
            self._moved = 0.0
            self._drag_off = e.globalPosition() - QPointF(self.geometry().topLeft())
            self.trace = [(float(self._press_ms), e.globalPosition().x(),
                           e.globalPosition().y())]
            self.dragging = True
            self.setCursor(Qt.ClosedHandCursor)
            self.squash = 0.09

    def mouseMoveEvent(self, e):
        if self.dragging and self._press_pos is not None:
            self._moved = max(self._moved,
                              (e.globalPosition() - self._press_pos).manhattanLength())
            self.trace.append((float(self.clock.elapsed()),
                               e.globalPosition().x(), e.globalPosition().y()))
            if len(self.trace) > 24:
                self.trace.pop(0)
            tl = e.globalPosition() - self._drag_off
            scr = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
            x = min(max(tl.x(), scr.left()), scr.right() - self.width())
            y = min(max(tl.y(), scr.top()), scr.bottom() - self.height() + 1)
            self.move(int(x), int(y))

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        self.dragging = False
        self.setCursor(Qt.OpenHandCursor)
        dt = self.clock.elapsed() - self._press_ms
        if self._moved < 6 and dt < 350:
            self.poke_now()
        else:
            self.throw_from_trace()
        self.on_ground = self.gravity_on and self.y() >= self._floor_y()
        self._save_cfg()

    def throw_from_trace(self):
        if not self.gravity_on or len(self.trace) < 3:
            self.vx = 0.0
            self.vy = 0.0
            return
        arr = np.array(self.trace[-16:])
        ts = (arr[:, 0] - arr[0, 0]) / 1000.0
        if ts[-1] <= 0.004:
            return
        vx, _ = mc.fit_linear(ts, arr[:, 1])
        vy, _ = mc.fit_linear(ts, arr[:, 2])
        self.vx = float(np.clip(vx, -MAX_THROW, MAX_THROW))
        self.vy = float(np.clip(vy, -MAX_THROW, MAX_THROW))
        self.on_ground = False
        h0 = max(0.0, self._floor_y() - self.y())
        _, x_hit = mc.projectile_landing(self.x(), h0, self.vx, self.vy, GRAVITY)
        self.landing = (x_hit, 0.9)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.puff = 1.0
            self.poke = 1.0
            self.spawn_bubbles(6)

    def contextMenuEvent(self, e):
        m = QMenu(self)
        m.addAction("戳一下").triggered.connect(self.poke_now)
        info = m.addAction(f"{self.affinity_label()}　状态：{self.behavior}")
        info.setEnabled(False)
        dine = m.addAction("吃掉拖进来的文件（移入回收站）")
        dine.setCheckable(True)
        dine.setChecked(self.eat_enabled)
        dine.triggered.connect(self.toggle_eat)
        m.addAction("重置亲密度").triggered.connect(self.reset_affinity)

        sm = m.addMenu("大小")
        for label, f in (("小", 0.7), ("中", 1.0), ("大", 1.3), ("超大", 1.7)):
            a = sm.addAction(label)
            a.setCheckable(True)
            a.setChecked(abs(self.scale_f - f) < 0.01)
            a.triggered.connect(lambda _c=False, v=f: self.set_scale(v))

        m.addSeparator()
        a = m.addAction("窗口置顶")
        a.setCheckable(True)
        a.setChecked(self.always_on_top)
        a.triggered.connect(self.toggle_top)

        a = m.addAction("落地模式（关掉 = 悬浮）")
        a.setCheckable(True)
        a.setChecked(self.gravity_on)
        a.triggered.connect(self.toggle_gravity)

        m.addSeparator()
        m.addAction("隐藏到托盘").triggered.connect(self.hide_pet)
        m.addAction("退出").triggered.connect(self.quit_app)
        m.exec(e.globalPos())

    def poke_now(self):
        self.poke = 1.0
        self.squash = -0.20
        self.spawn_bubbles(4)
        self.affinity.poke()
        self.idle_time = 0.0
        self._save_cfg()

    def toggle_eat(self):
        self.eat_enabled = not self.eat_enabled
        self.hover_eat = False
        self._save_cfg()

    def _deletable(self, path: str) -> bool:
        try:
            p = Path(path)
            if len(str(p)) <= 3:
                return False
            if APP_DIR == p or APP_DIR in p.parents:
                return False
            if not p.exists():
                return False
            anchor = p.anchor
            if not anchor or anchor.startswith("\\"):
                return False
            import ctypes
            if ctypes.windll.kernel32.GetDriveTypeW(anchor) == 4:
                return False
            return True
        except Exception:
            return False

    def _start_eat(self, paths: list[str], pos):
        prov = QFileIconProvider()
        self.foods = []
        mouth = QPointF(self.width() / 2.0 + self.lean * 10.0,
                        self.height() - self.pix.height() * 0.74)
        for i, path in enumerate(paths[:8]):
            pm = prov.icon(QFileInfo(str(path))).pixmap(64, 64)
            start = QPointF(pos)
            ctrl = QPointF((start.x() + mouth.x()) / 2.0,
                           min(start.y(), mouth.y()) - 90.0)
            self.foods.append({"from": start, "ctrl": ctrl, "pm": pm, "t0": i * 0.07})
        self.eat_t = 0.0
        self.eat_alpha = 1.0
        self.eat_label = "啊呜——"
        self.spawn_bubbles(3)

    def dragEnterEvent(self, e):
        if self.eat_enabled and e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.hover_eat = True

    def dragMoveEvent(self, e):
        if self.eat_enabled and e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.hover_eat = False

    def dropEvent(self, e):
        self.hover_eat = False
        if not self.eat_enabled:
            return
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if self._deletable(p)]
        if not paths:
            return
        if recycle_paths(paths):
            self._start_eat(paths, e.position())
            self.affinity.poke(len(paths))
            self.idle_time = 0.0
            self._save_cfg()
        else:
            self.eat_label = "吃不掉…"
            self.eat_alpha = 1.0
        e.acceptProposedAction()

    def reset_affinity(self):
        self.affinity = mc.AffinityBeta(2.0, 2.0)
        self._save_cfg()

    def set_scale(self, f: float):
        self.scale_f = f
        self._rebuild()
        self._save_cfg()

    def toggle_top(self):
        self.always_on_top = not self.always_on_top
        self._apply_flags()
        self.show()
        self._save_cfg()

    def toggle_gravity(self):
        self.gravity_on = not self.gravity_on
        self.on_ground = False
        self._save_cfg()

    def hide_pet(self):
        self.timer.stop()
        self.hide()

    def show_pet(self):
        self.show()
        self.raise_()
        self._last_ms = 0
        self.timer.start(20)

    def quit_app(self):
        self._save_cfg()
        if self.tray:
            self.tray.hide()
        QApplication.quit()

    def closeEvent(self, e):
        self._save_cfg()
        QApplication.quit()

    # -------------------------------------------------- 托盘

    def setup_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        icon = self.src.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.tray = QSystemTrayIcon(QIcon(icon), self)
        self.tray.setToolTip("鲸鱼女仆娘")
        tm = QMenu()
        tm.addAction("显示 / 唤醒").triggered.connect(self.show_pet)
        tm.addAction("戳一下").triggered.connect(self.poke_now)
        tm.addSeparator()
        tm.addAction("退出").triggered.connect(self.quit_app)
        self.tray.setContextMenu(tm)
        self.tray.activated.connect(
            lambda r: self.show_pet() if r == QSystemTrayIcon.Trigger else None
        )
        self.tray.show()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("WhalePet")
    w = WhalePet()
    w.show()
    w.kill_dwm_border()
    w.raise_()
    w.setup_tray()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
