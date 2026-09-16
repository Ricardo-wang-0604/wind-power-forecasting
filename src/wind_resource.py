# -*- coding: utf-8 -*-
"""
wind_resource.py —— 风资源评估模块

【为什么有这个文件】
    原来的项目只做「功率预测」，这对应的是**功率预测算法岗**，而那个岗位多半要硕士。
    本科能进的是「**风资源工程师**」，它的核心交付物是**发电量评估**，不是功率预测。

    这个模块把风资源评估的核心计算补齐，让同一个项目能同时支撑两类岗位。

【覆盖的风资源标准流程】
    1. 测风数据分析   → 韦布尔分布拟合（风速的概率分布）
    2. 风功率密度     → 风能资源的综合指标，用于分等级
    3. 风切变外推     → 由观测高度外推到风机轮毂高度
    4. 风资源分级     → 按 GB/T 18710 判定资源好坏
    5. 发电量估算     → AEP（年发电量），给出 P50
    6. 不确定性       → 由 P50 推出 P75 / P90

【行业背景】
    · 韦布尔分布、风功率密度分级、风切变外推 —— GB/T 18710-2002
      《风电场风能资源评估方法》里的标准内容
    · P50 / P75 / P90 是风电场融资与投资决策的实际依据：
        P50 = 有一半概率能达到的发电量（中位期望）
        P75 = 有 75% 概率能达到（银行常用的保守值）
        P90 = 有 90% 概率能达到（最保守，用于还贷能力评估）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from scipy.special import gamma as _gamma
    from scipy.optimize import brentq
    _HAS_SCIPY = True
except Exception:                                  # 允许无 scipy 环境降级
    _HAS_SCIPY = False
    def _gamma(x):
        # Lanczos 近似，够用
        g = 7
        C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
             771.32342877765313, -176.61502916214059, 12.507343278686905,
             -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        if x < 0.5:
            return np.pi / (np.sin(np.pi * x) * _gamma(1 - x))
        x -= 1
        a = C[0]
        t = x + g + 0.5
        for i in range(1, g + 2):
            a += C[i] / (x + i)
        return np.sqrt(2 * np.pi) * t ** (x + 0.5) * np.exp(-t) * a


# ==================================================================== 1. 韦布尔
def weibull_fit_moments(v: np.ndarray):
    """矩估计法拟合韦布尔分布，返回 (k, c)。

        经验式： k ≈ (σ/μ)^(-1.086)
        然后    c = μ / Γ(1 + 1/k)

    优点：不用迭代，永远有解；缺点：精度略低于最大似然。
    工程上常用，因为它稳。
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    v = v[v > 0]                      # 韦布尔定义在 v>0
    if len(v) < 10:
        return np.nan, np.nan
    mu = v.mean()
    sd = v.std(ddof=1)
    if mu <= 0 or sd <= 0:
        return np.nan, np.nan
    k = (sd / mu) ** (-1.086)
    k = float(np.clip(k, 0.5, 10.0))
    c = float(mu / _gamma(1.0 + 1.0 / k))
    return k, c


def weibull_fit_mle(v: np.ndarray):
    """最大似然法拟合韦布尔分布，返回 (k, c)。

    MLE 的 k 满足：
        Σ(vᵢ^k ln vᵢ) / Σ(vᵢ^k) − 1/k = (1/n) Σ ln vᵢ
    左边关于 k 单调递减，用二分求解。
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    v = v[v > 0]
    if len(v) < 10 or not _HAS_SCIPY:
        return weibull_fit_moments(v)
    lnv = np.log(v)
    target = lnv.mean()

    def f(k):
        vk = v ** k
        return np.sum(vk * lnv) / np.sum(vk) - 1.0 / k - target

    try:
        k = brentq(f, 0.2, 20.0, xtol=1e-6)
    except Exception:
        return weibull_fit_moments(v)
    c = float(np.mean(v ** k) ** (1.0 / k))
    return float(k), c


def weibull_pdf(v, k, c):
    """韦布尔概率密度 f(v)。"""
    v = np.asarray(v, dtype=float)
    return (k / c) * (v / c) ** (k - 1) * np.exp(-(v / c) ** k)


def weibull_cdf(v, k, c):
    """韦布尔累积分布 F(v)。"""
    v = np.asarray(v, dtype=float)
    return 1.0 - np.exp(-(v / c) ** k)


def weibull_mean(k, c):
    """韦布尔分布的均值 = c·Γ(1+1/k)。"""
    return c * _gamma(1.0 + 1.0 / k)


# ================================================================ 2. 风功率密度
def air_density(temperature_c: float = 15.0, pressure_hpa: float = 1013.25):
    """空气密度 ρ = P/(R·T)。默认标准大气：15 ℃、1013.25 hPa → 1.225 kg/m³。"""
    T = temperature_c + 273.15
    return (pressure_hpa * 100.0) / (287.05 * T)


def wind_power_density_observed(v, rho: float = 1.225):
    """实测法：WPD = ½·ρ·⟨v³⟩  （W/m²）

    注意是 ⟨v³⟩（三次方的平均），不是 (⟨v⟩)³ —— 两者差很多，
    因为风速分布右偏，三次方会把高风速放大。
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    return float(0.5 * rho * np.mean(v ** 3))


def wind_power_density_weibull(k, c, rho: float = 1.225):
    """解析法：WPD = ½·ρ·c³·Γ(1+3/k)  （W/m²）

    这是韦布尔分布下 ⟨v³⟩ 的解析解，很漂亮也很常用。
    """
    return float(0.5 * rho * c ** 3 * _gamma(1.0 + 3.0 / k))


# ==================================================================== 3. 风切变
def extrapolate_shear(v_ref, z_ref: float, z_target: float, alpha: float):
    """风切变幂律外推：v(z) = v(z_ref)·(z/z_ref)^α

    风资源评估里最常见的用法：
        测风塔在 70m 或 100m，但风机轮毂在 100m~160m，
        必须靠 α 把风速外推上去。
    """
    v_ref = np.asarray(v_ref, dtype=float)
    return v_ref * (z_target / z_ref) ** alpha


def shear_from_two_levels(v_hi, v_lo, z_hi: float, z_lo: float):
    """由两个高度的风速反算风切变指数 α。"""
    v_hi = np.asarray(v_hi, dtype=float)
    v_lo = np.asarray(v_lo, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.log(v_hi / v_lo) / np.log(z_hi / z_lo)
    return a


# ==================================================================== 4. 分级
# GB/T 18710-2002 风功率密度等级表（10m 高度）
# ⚠️ 数值为通行版本；实际项目请以标准原文核对后再使用。
WPD_GRADES = [
    (1,     0, 100),
    (2,   100, 150),
    (3,   150, 200),
    (4,   200, 250),
    (5,   250, 300),
    (6,   300, 400),
    (7,   400, float('inf')),
]


def grade_wind_resource(wpd: float):
    """按风功率密度给风资源分级，返回 (等级, 描述)。

    等级越高资源越好。工程上一般认为 3 级以上具备开发价值，
    但还要结合电价、并网条件、地形、造价综合判断。
    """
    for g, lo, hi in WPD_GRADES:
        if lo <= wpd < hi:
            desc = {1: '资源较差', 2: '资源一般', 3: '资源尚可，可开发',
                    4: '资源较好', 5: '资源好', 6: '资源很好', 7: '资源极好'}[g]
            return g, desc
    return None, '无法判定'


# ==================================================================== 5. 发电量
def aep_from_weibull(k, c, power_curve_v, power_curve_p, capacity_mw: float = 1.0,
                     hours: float = 8760.0, v_max: float = 25.0, dv: float = 0.2):
    """由韦布尔分布 + 功率曲线积分求年发电量 AEP。

        AEP = hours × 装机容量 × Σ[ P_norm(v) · f(v) · Δv ]
                                     └────── 容量因子 CF ──────┘

    注意 CF 是**无量纲的 0~1 比值**，不能把装机容量乘进去，
    否则 AEP 会被重复乘一次容量（这是一个很容易犯的错）。
    P(v) 用归一化功率曲线（0~1），乘以装机容量得到实际功率。

    返回 (AEP_MWh, CF, 各风速段明细 DataFrame)
    """
    v = np.arange(dv / 2, v_max, dv)
    f = weibull_pdf(v, k, c)
    p_norm = np.interp(v, power_curve_v, power_curve_p, left=0.0, right=0.0)
    freq = f * dv                              # 每个风速段的出现概率
    cf = float(np.sum(p_norm * freq))          # 容量因子（无量纲，0~1）
    aep = hours * capacity_mw * cf             # MWh

    detail = pd.DataFrame({
        'v_mid(m/s)': v,
        'frequency': freq,
        'P_norm': p_norm,
        'P_MW': p_norm * capacity_mw,
        'energy_MWh': hours * capacity_mw * p_norm * freq,
    })
    return float(aep), cf, detail


def p_values(aep_p50: float, sigma_total: float):
    """由 P50 与综合不确定性推出 P75 / P90。

    风资源行业惯例（发电量近似服从正态分布）：
        P75 = P50 × (1 − 0.6745·σ)
        P90 = P50 × (1 − 1.2816·σ)

    其中 0.6745 = Φ⁻¹(0.75)，1.2816 = Φ⁻¹(0.90)。

    这三个数是风电场融资的实际依据，不是学术指标。
    """
    z75, z90 = 0.6745, 1.2816
    return {
        'P50':  aep_p50,
        'P75':  aep_p50 * (1 - z75 * sigma_total),
        'P90':  aep_p50 * (1 - z90 * sigma_total),
        'sigma': sigma_total,
    }


# 不确定性分项（典型量级，实际项目要按现场资料重新评估）
UNCERTAINTY_ITEMS = [
    ('风速年际变化', 0.06),
    ('测风数据代表性', 0.05),
    ('长年代订正(MCP)', 0.04),
    ('水平/垂直外推', 0.04),
    ('功率曲线', 0.05),
    ('尾流模型', 0.04),
    ('可利用率', 0.03),
    ('电气与线损', 0.02),
]


def uncertainty_budget():
    """按平方和开根合成各项不确定性，返回 (总不确定性, 分项表)。

    各项视为独立，故：σ_total = √(Σσᵢ²)
    ⚠️ 表中为行业典型量级，仅作演示；实际项目必须按现场资料重新评估。
    """
    df = pd.DataFrame(UNCERTAINTY_ITEMS, columns=['项目', '标准差'])
    total = float(np.sqrt(np.sum(df['标准差'] ** 2)))
    df['方差占比%'] = (df['标准差'] ** 2) / (total ** 2) * 100
    return total, df


# ==================================================================== 7. 报告
def wind_resource_report(ws: pd.Series, hub_height: float = 100.0,
                         z_obs: float = 100.0, capacity_mw: float = 1.0,
                         power_curve=None):
    """一站式风资源评估，返回字典报告。

    参数
    ----
    ws          风速序列（m/s）
    hub_height  风机轮毂高度（m）
    z_obs       风速观测高度（m）
    capacity_mw 装机容量（MW）
    power_curve (v_array, p_norm_array) 归一化功率曲线；不传则用内置典型曲线
    """
    v = np.asarray(ws, dtype=float)
    v = v[np.isfinite(v)]

    k, c = weibull_fit_mle(v)
    k_m, c_m = weibull_fit_moments(v)
    mean_v = float(np.mean(v))

    rho = air_density()
    wpd = wind_power_density_observed(v, rho)
    wpd_w = wind_power_density_weibull(k, c, rho)
    grade, gdesc = grade_wind_resource(wpd)

    # 外推到轮毂高度
    alpha = 0.143                       # 中性层结 1/7 律的默认值
    scale = (hub_height / z_obs) ** alpha
    k_hub, c_hub = k, c * scale
    wpd_hub = wind_power_density_weibull(k_hub, c_hub, rho)

    if power_curve is None:
        pc_v, pc_p = typical_power_curve()
    else:
        pc_v, pc_p = power_curve

    aep, cf, detail = aep_from_weibull(k_hub, c_hub, pc_v, pc_p, capacity_mw)
    sigma, ub = uncertainty_budget()
    pv = p_values(aep, sigma)

    return {
        '样本数': len(v),
        '平均风速(m/s)': round(mean_v, 3),
        '韦布尔_k(MLE)': round(k, 3),
        '韦布尔_c(MLE)': round(c, 3),
        '韦布尔_k(矩估计)': round(k_m, 3),
        '韦布尔_c(矩估计)': round(c_m, 3),
        '风功率密度(W/m2)': round(wpd, 1),
        '风功率密度_韦布尔解析': round(wpd_w, 1),
        '资源等级': f'{grade} 级（{gdesc}）',
        '空气密度(kg/m3)': round(rho, 4),
        '轮毂高度(m)': hub_height,
        '轮毂高度_c': round(c_hub, 3),
        '轮毂高度风功率密度': round(wpd_hub, 1),
        '容量因子': round(cf, 4),
        'AEP_P50(MWh)': round(pv['P50'], 1),
        'AEP_P75(MWh)': round(pv['P75'], 1),
        'AEP_P90(MWh)': round(pv['P90'], 1),
        '综合不确定性': f'{sigma * 100:.1f}%',
        '_分项不确定性': ub,
        '_AEP明细': detail,
        '_功率曲线': (pc_v, pc_p),
    }


def typical_power_curve(v_rated: float = 12.0, v_in: float = 3.0,
                        v_out: float = 25.0, n: int = 200):
    """内置一条典型风机归一化功率曲线。

    三段：
        v < v_in            出力 0
        v_in ≤ v ≤ v_rated  按 v³ 律上升（归一化到额定值 1）
        v_rated < v < v_out 出力 = 1（变桨控制）
        v ≥ v_out           停机，出力 0
    """
    v = np.linspace(0, 30, n)
    p = np.zeros_like(v)
    m = (v >= v_in) & (v <= v_rated)
    p[m] = ((v[m] ** 3 - v_in ** 3) / (v_rated ** 3 - v_in ** 3))
    p[(v > v_rated) & (v < v_out)] = 1.0
    p[v >= v_out] = 0.0
    return v, p
