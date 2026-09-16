# -*- coding: utf-8 -*-
"""
features.py —— 特征工程

这个文件是**整个项目最能体现『懂大气』的地方**。
别人写这个项目，通常是直接把 U10/V10/U100/V100 四个列喂给模型；
这里要做的是先把它们还原成物理量，再喂给模型。

四个核心物理量：
  1. 风速   ws = √(U² + V²)                      ← 矢量合成，不是取模！
  2. 风向   wd = atan2(U, V)                      ← 气象风向定义
  3. 风切变 α = ln(ws100/ws10) / ln(100/10)       ← 大气边界层幂律
  4. 时间   小时 / 年积日 —— 太阳辐射与热力环流的日变化和年变化

为什么不能直接用 U/V：
  U 和 V 是**分量**，单独看没有物理意义。同一个 U 值配不同 V，
  实际风速差很多。而且风速与出力是**三次方**关系，
  对风速的误差极其敏感 —— 这一步做错，后面全废。
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------------------------------------------ 物理量还原

def wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    """由 U/V 分量合成风速（m/s）。

    坑：很多人第一反应是取绝对值或只看 U。错误。
    风速是矢量的模：ws = √(U² + V²)
    """
    return np.sqrt(u ** 2 + v ** 2)


def wind_direction(u: pd.Series, v: pd.Series) -> pd.Series:
    """由 U/V 分量算风向（度，0=北，顺时针）。

    气象约定：U 是**向东**分量，V 是**向北**分量。
    风向定义为『风从哪里来』，所以用 atan2(-U, -V) 再转成 0~360。
    """
    wd = np.degrees(np.arctan2(-u, -v))
    return wd % 360


def shear_exponent(ws_hi: pd.Series, ws_lo: pd.Series,
                   z_hi: float = 100.0, z_lo: float = 10.0) -> pd.Series:
    """风切变指数 α（大气边界层幂律指数）。

        幂律： ws(z) = ws(z_ref) · (z / z_ref)^α
      反解： α = ln(ws_hi / ws_lo) / ln(z_hi / z_lo)

    物理含义：
      α ≈ 0.14  开阔海面 / 中性层结（1/7 律）
      α ≈ 0.25  草原、农田
      α ≈ 0.3+  城市、森林，或**夜间稳定边界层**

    α 随稳定度变化，白天小、夜间大 —— 这是模型能学到的真实物理信号。
    这也是本科大气物理才会教的东西，纯 CS 背景的人不会想到加这个特征。
    """
    ratio = ws_hi / ws_lo.replace(0, np.nan)          # 避免除零
    return np.log(ratio) / np.log(z_hi / z_lo)


def air_density_correction(ws: pd.Series, temperature_c: pd.Series = None) -> pd.Series:
    """空气密度修正后的『等效风速』（可选）。

    风机捕获的功率 ∝ ρ · v³。空气密度随温度和气压变化，
    冬季冷空气密度大，同样风速下出力更高。
    GEFCom 数据没有气压，这里只用温度做一阶近似：
        ρ ≈ ρ0 · (273.15 / (273.15 + T))
    等效风速 = ws · (ρ/ρ0)^(1/3)
    —— 用立方根，因为功率是 v³，把密度折进风速里。
    """
    if temperature_c is None:
        return ws
    rho_ratio = 273.15 / (273.15 + temperature_c)
    return ws * (rho_ratio ** (1.0 / 3.0))


# ------------------------------------------------------------ 时间特征

def time_features(ts: pd.Series) -> pd.DataFrame:
    """时间特征。

    对风电最要紧的两个：
      - hour：日变化。夜间边界层稳定、风切变大、轮毂高度风速可能更高。
      - dayofyear：年变化。冬春季风大（我国尤其明显）。
    """
    return pd.DataFrame({
        "hour": ts.dt.hour,
        "dayofyear": ts.dt.dayofyear,
        "month": ts.dt.month,
        # 三角函数编码：让 0 点和 23 点在数值上相邻，否则模型以为它们差 23
        "hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24),
        "doy_sin": np.sin(2 * np.pi * ts.dt.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * ts.dt.dayofyear / 365),
    })


# ------------------------------------------------------------ 总入口

def build_features(df: pd.DataFrame, add_lag: bool = True) -> pd.DataFrame:
    """把原始数据变成模型能用的特征表。

    注意：**滞后特征必须按 zone 分组做**。
    十个风场的时间序列是并排的，不分组的话，zone1 的最后一行会跑去
    给 zone2 的第一行当滞后值 —— 这是新手最常犯的泄漏错误。
    """
    d = df.copy()

    # 1) 物理量
    d["ws10"] = wind_speed(d.u10, d.v10)
    d["ws100"] = wind_speed(d.u100, d.v100)
    d["wd10"] = wind_direction(d.u10, d.v10)
    d["wd100"] = wind_direction(d.u100, d.v100)
    d["shear"] = shear_exponent(d.ws100, d.ws10)

    # 2) 时间
    d = pd.concat([d, time_features(d.timestamp)], axis=1)

    # 3) 滞后 / 滚动特征（历史出力）
    if add_lag:
        d = d.sort_values(["zone", "timestamp"]).reset_index(drop=True)
        g = d.groupby("zone")["power"]
        d["power_lag1"] = g.shift(1)
        d["power_lag2"] = g.shift(2)
        d["power_lag24"] = g.shift(24)          # 前一天同一时刻，捕捉日周期
        d["power_roll3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
        d["power_roll24"] = g.transform(lambda s: s.shift(1).rolling(24).mean())

    return d


# 模型用的特征列（集中在这里定义，避免训练和推理时列不一致）
FEATURE_COLS = [
    # 原始 NWP 分量
    "u10", "v10", "u100", "v100",
    # 物理量
    "ws10", "ws100", "wd10", "wd100", "shear",
    # 时间
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_COLS = ["power_lag1", "power_lag2", "power_lag24", "power_roll3", "power_roll24"]
