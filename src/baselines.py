# -*- coding: utf-8 -*-
"""
baselines.py —— 物理基线模型

**这个文件是整个项目的灵魂。**

一个只有 CS 背景的人做这个项目，基线通常是「拿风速做线性回归」。
但风电行业的标准做法是**功率曲线（power curve）**，因为：

    风功率  P = ½ · ρ · A · v³ · Cp

出力与风速是**三次方**关系，不是线性关系。用线性回归去拟合，
在低风速段会高估、高风速段会低估，而且**完全抓不到切入/切出**。

本项目提供两条基线：
  ① PowerCurve   —— 经验功率曲线（物理正确）
  ② LinearWS     —— 风速线性回归（物理错误，作为对照组）

两条基线都跑，README 里的对比表就能说明：
  「我的基线比常见的线性基线好多少，我的模型又比我的基线好多少」
—— 这才叫有说服力的结果。
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class PowerCurveBaseline:
    """经验功率曲线基线。

    做法（风电行业标准）：
      1. 把训练集按风速分箱（比如每 0.5 m/s 一箱）
      2. 每箱取**中位数**出力（取中位数而不是均值，因为出力分布右偏、且有异常值）
      3. 预测时对曲线做线性插值

    为什么不拟合一条解析曲线：
      真实风机有切入、额定、切出三段折点，还有尾流、地形、
      机组的分散性，解析式反而更差。分箱中位数是行业里最常用、
      最鲁棒的做法。
    """

    def __init__(self, bin_width: float = 0.5, ws_col: str = "ws100",
                 v_cutout: float | None = None):
        self.bin_width = bin_width
        self.ws_col = ws_col
        self.v_cutout = v_cutout          # 切出风速，超过则认为停机（出力 0）
        self.curve = None                 # pd.Series: index=风速中心, value=出力
        self.name = "物理基线·功率曲线"

    def fit(self, df: pd.DataFrame, target: str = "power"):
        d = df[[self.ws_col, target]].dropna()
        bins = np.arange(0, np.ceil(d[self.ws_col].max() / self.bin_width) * self.bin_width
                         + self.bin_width, self.bin_width)
        centers = bins[:-1] + self.bin_width / 2
        idx = np.digitize(d[self.ws_col], bins) - 1
        med = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            sel = d[target].values[idx == i]
            if len(sel) >= 5:             # 样本太少的箱子不要，避免噪声
                med[i] = np.median(sel)
        # 用已知箱填补空洞（线性插值 + 两端外推）
        s = pd.Series(med, index=centers)
        self.curve = s.interpolate(limit_direction="both").clip(0, 1)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ws = df[self.ws_col].values
        p = np.interp(ws, self.curve.index.values, self.curve.values)
        if self.v_cutout is not None:
            p = np.where(ws >= self.v_cutout, 0.0, p)   # 切出停机
        return np.clip(p, 0, 1)


class LinearWSBaseline:
    """风速线性回归基线 —— 故意做错的对照组。

    它假设 出力 = a · 风速 + b。这在物理上是错的（应为 v³），
    但很多入门项目就是这么做的。放在这里是为了在 README 里展示差距。
    """

    def __init__(self, ws_col: str = "ws100"):
        self.ws_col = ws_col
        self.a = 0.0
        self.b = 0.0
        self.name = "对照组·风速线性回归"

    def fit(self, df: pd.DataFrame, target: str = "power"):
        d = df[[self.ws_col, target]].dropna()
        x, y = d[self.ws_col].values, d[target].values
        self.a, self.b = np.polyfit(x, y, 1)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.a * df[self.ws_col].values + self.b, 0, 1)


class CubicWSBaseline:
    """纯立方律基线（不加分段）。

    出力 ∝ v³，但需要对额定功率归一化。做法是用训练集拟合
        P = min(1, k · v³)
    比线性对得多，但因为没有切入/额定段，仍然不是最优。
    用来展示「知道 v³ 但不知道分段」是什么水平。
    """

    def __init__(self, ws_col: str = "ws100"):
        self.ws_col = ws_col
        self.k = 0.0
        self.name = "物理基线·纯立方律"

    def fit(self, df: pd.DataFrame, target: str = "power"):
        d = df[[self.ws_col, target]].dropna()
        v3 = d[self.ws_col].values ** 3
        y = d[target].values
        # 只用未饱和样本（出力 < 0.95）拟合斜率，否则会严重低估
        m = y < 0.95
        self.k = float(np.sum(v3[m] * y[m]) / np.sum(v3[m] ** 2))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.k * df[self.ws_col].values ** 3, 0, 1)
