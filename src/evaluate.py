# -*- coding: utf-8 -*-
"""
evaluate.py —— 评估指标

**为什么要单独写这个文件：因为评估口径本身就是专业知识。**

新人做项目，通常只报 RMSE 和 MSE。但风电功率预测这行有**国家标准口径**，
面试官一看你报的指标就知道你是不是真的了解这个行业。

国标口径（《风电场功率预测系统技术要求》一类规范）：
    准确率 = (1 - RMSE / 装机容量) × 100%
    合格率 = 单点误差 ≤ 25% 装机容量 的样本占比

因为 GEFCom 的 TARGETVAR 已经是归一化出力（0~1），
所以这里的「装机容量」= 1，指标可以直接算。

另外，功率预测的误差是**异方差**的：
    低风速时出力接近 0，误差很小；
    额定风速附近出力饱和，误差也小；
    中间那段（爬坡段）误差最大。
所以除了总体指标，还要**按风速分档报误差** —— 这才看得出模型弱在哪。
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    """均方根误差。"""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    """平均绝对误差。比 RMSE 更抗异常值。"""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    """决定系数。注意：在功率预测里 R² 会显得很好看（因为很多时刻出力为 0），
    所以它只能当参考，不能当主指标。"""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def accuracy_rate(y_true, y_pred, capacity: float = 1.0) -> float:
    """国标『准确率』 = (1 - RMSE/装机容量) × 100%

    这是电网调度最常看的指标。短期预测一般要求 ≥ 80~85%。
    """
    return float((1 - rmse(y_true, y_pred) / capacity) * 100)


def qualified_rate(y_true, y_pred, capacity: float = 1.0, tol: float = 0.25) -> float:
    """国标『合格率』 = 单点误差在 ±tol×装机容量 以内的样本占比。

    新能源场站并网考核用的就是这个，月度合格率不达标要罚款。
    """
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.mean(err <= tol * capacity) * 100)


def evaluate_all(y_true, y_pred, capacity: float = 1.0, name: str = "") -> dict:
    """一次算全部指标。"""
    return {
        "模型": name,
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "R2": round(r2(y_true, y_pred), 4),
        "准确率%": round(accuracy_rate(y_true, y_pred, capacity), 2),
        "合格率%": round(qualified_rate(y_true, y_pred, capacity), 2),
    }


def error_by_wind_bin(df: pd.DataFrame, ws_col: str = "ws100",
                      bins=(0, 3, 6, 9, 12, 25)) -> pd.DataFrame:
    """按风速分档统计误差 —— 定位模型弱点用。

    预期会看到：中间档（6~12 m/s，爬坡段）误差最大。
    这是物理决定的，不是模型不行。
    """
    d = df.copy()
    d["bin"] = pd.cut(d[ws_col], bins=list(bins))
    rows = []
    for b, g in d.groupby("bin", observed=True):
        if len(g) < 10:
            continue
        rows.append({
            "风速区间(m/s)": str(b),
            "样本数": len(g),
            "RMSE": round(rmse(g.power, g.pred), 4),
            "平均出力": round(float(g.power.mean()), 4),
        })
    return pd.DataFrame(rows)


def summary_table(results: list) -> pd.DataFrame:
    """把多个模型的评估结果拼成对比表（README 里直接贴这个）。"""
    return pd.DataFrame(results)
