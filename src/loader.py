# -*- coding: utf-8 -*-
"""
loader.py —— 数据读取与时间解析

GEFCom2014 风电数据的原始格式：
    ZONEID,TIMESTAMP,TARGETVAR,U10,V10,U100,V100
    1,20120101 1:00,0,2.1246,-2.6820,2.8643,-3.6661

几个坑（说明书里有详细解释）：
1. TIMESTAMP 是 "20120101 1:00" 这种**不规则格式**（小时没有补零），
   直接用 pd.to_datetime 会解析失败或解析成错的时间，必须自己拼。
2. 数据没有年份跨年问题，但**有缺测**：GEFCom 原始数据会缺几天，
   不能假设时间连续。
3. TARGETVAR 是**归一化出力**（0~1），不是 MW，所以评估指标不能直接换算成电量。
"""

from __future__ import annotations
import os
import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------- 路径
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")


def _parse_timestamp(s: pd.Series) -> pd.Series:
    """把 '20120101 1:00' 解析成真正的 datetime。

    为什么不用 pd.to_datetime(s)：
      原始串里小时是不补零的（'1:00' 而不是 '01:00'），
      pandas 在部分版本上会推断失败，或把 '1:00' 当成时区偏移。
      这里手工拆，最稳。
    """
    s = s.astype(str).str.strip()
    # 抓出 YYYYMMDD 和 H
    ymd = s.str.slice(0, 8)
    hm = s.str.slice(9)                       # 'H:00'
    hour = hm.str.split(":").str[0].astype(int)
    return pd.to_datetime(ymd, format="%Y%m%d") + pd.to_timedelta(hour, unit="h")


def load_zone(zone: int, task: int = 1) -> pd.DataFrame:
    """读单个风电场的原始数据（默认 Task 1）。

    返回列：timestamp, zone, power, u10, v10, u100, v100
    """
    fn = os.path.join(RAW, "Task%d_W_Zone1_10" % task, "Task%d_W_Zone%d.csv" % (task, zone))
    if not os.path.exists(fn):
        raise FileNotFoundError(
            "找不到 %s\n请先运行：python data/download.py" % fn
        )
    df = pd.read_csv(fn)
    df.columns = [c.strip().upper() for c in df.columns]
    out = pd.DataFrame({
        "timestamp": _parse_timestamp(df["TIMESTAMP"]),
        "zone": df["ZONEID"].astype(int),
        "power": df["TARGETVAR"].astype(float),      # 归一化出力 0~1
        "u10": df["U10"].astype(float),
        "v10": df["V10"].astype(float),
        "u100": df["U100"].astype(float),
        "v100": df["V100"].astype(float),
    })
    return out.sort_values("timestamp").reset_index(drop=True)


def load_zones(zones=None, task: int = 1) -> pd.DataFrame:
    """读多个风电场并纵向拼接（多风场数据一起用，样本量更大）。"""
    if zones is None:
        zones = range(1, 11)
    parts = []
    for z in zones:
        try:
            parts.append(load_zone(z, task=task))
        except FileNotFoundError:
            continue
    if not parts:
        raise FileNotFoundError("一个风场都没读到，请先运行 data/download.py")
    return pd.concat(parts, ignore_index=True)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """缺测统计 —— 写进 README 的『踩过的坑』一节很有用。"""
    rep = df.isna().sum().to_frame("缺失数")
    rep["缺失率%"] = (rep["缺失数"] / len(df) * 100).round(3)
    return rep


if __name__ == "__main__":
    d = load_zone(1)
    print("读入 %d 行，时间 %s → %s" % (len(d), d.timestamp.min(), d.timestamp.max()))
    print(d.head())
    print()
    print(missing_report(d))
