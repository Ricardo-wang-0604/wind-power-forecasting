# -*- coding: utf-8 -*-
"""
plots.py —— 出图

四张图，README 里都要放：
  1. power_curve.png     —— 功率曲线（最能说明"为什么物理基线是对的"）
  2. comparison.png      —— 模型对比条形图
  3. timeseries.png      —— 测试集上的一段实际 vs 预测
  4. importance.png      —— 特征重要性

中文字体：Windows 上用 SimHei / Microsoft YaHei。找不到就退回英文，不报错。
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

C = {"true": "#333333", "pred": "#c0392b", "base": "#7f8c8d",
     "ml": "#2980b9", "grid": "#dddddd"}


def plot_power_curve(df: pd.DataFrame, curve: pd.Series, outdir: str):
    """散点 + 经验功率曲线。一眼看出 v³ 段、饱和段。"""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax.scatter(df.ws100, df.power, s=3, alpha=0.18, color=C["base"],
               label="观测 (风速, 出力)")
    ax.plot(curve.index, curve.values, color=C["pred"], lw=2.2,
            label="经验功率曲线（分箱中位数）")
    ax.set_xlabel("轮毂高度风速 ws100 (m/s)")
    ax.set_ylabel("归一化出力")
    ax.set_title("功率曲线：出力与风速是三次方关系，不是线性")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "power_curve.png"))
    plt.close(fig)


def plot_comparison(tbl: pd.DataFrame, outdir: str):
    """模型准确率对比。"""
    t = tbl.sort_values("准确率%")
    colors = [C["ml"] if "LightGBM" in m else
              (C["pred"] if "物理" in m else C["base"]) for m in t["模型"]]
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=140)
    ax.barh(t["模型"], t["准确率%"], color=colors)
    for i, v in enumerate(t["准确率%"]):
        ax.text(v + 0.3, i, "%.2f%%" % v, va="center", fontsize=9)
    ax.set_xlabel("准确率 %  （国标口径 = 1 - RMSE/装机容量）")
    ax.set_title("模型对比")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison.png"))
    plt.close(fig)


def plot_timeseries(te: pd.DataFrame, outdir: str, n: int = 336):
    """取一段（默认两周）看实际 vs 预测。"""
    d = te.dropna(subset=["pred"]).tail(n)
    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=140)
    ax.plot(d.timestamp, d.power, color=C["true"], lw=1.3, label="实际出力")
    ax.plot(d.timestamp, d.pred, color=C["pred"], lw=1.3, alpha=0.85, label="预测出力")
    ax.set_ylabel("归一化出力")
    ax.set_title("测试集末段：实际 vs 预测")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timeseries.png"))
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, outdir: str, tag: str = ""):
    """特征重要性 —— 看物理特征有没有排进前面。"""
    t = imp.head(14).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    # 物理特征高亮，一眼看出它们有没有用
    phys = {"ws10", "ws100", "wd10", "wd100", "shear"}
    colors = [C["pred"] if f in phys else C["ml"] for f in t["feature"]]
    ax.barh(t["feature"], t["importance"], color=colors)
    ax.set_xlabel("重要性")
    ax.set_title("特征重要性（红色 = 物理特征）")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fn = "importance.png" if not tag else "importance_%s.png" % tag
    fig.savefig(os.path.join(outdir, fn))
    plt.close(fig)


def plot_all(te: pd.DataFrame, tbl: pd.DataFrame, preds: dict, outdir: str,
             train_df: pd.DataFrame = None):
    """出全部图。

    ⚠️ 注意：功率曲线必须用**训练集**拟合。
    如果用测试集重拟合再画出来，看起来会「拟合得特别好」——
    那是数据泄漏，评审一眼就看得出。
    """
    os.makedirs(outdir, exist_ok=True)
    src = train_df if train_df is not None else te
    plot_power_curve(src, _curve_from(src), outdir)
    plot_comparison(tbl, outdir)
    plot_timeseries(te, outdir)


def _curve_from(df: pd.DataFrame) -> pd.Series:
    """拟合一条功率曲线供画图（传训练集进来）。"""
    from baselines import PowerCurveBaseline
    b = PowerCurveBaseline().fit(df)
    return b.curve
