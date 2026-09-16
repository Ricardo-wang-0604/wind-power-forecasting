# -*- coding: utf-8 -*-
"""
download.py —— 下载 GEFCom2014 风电数据集

============================================================================
 数据来源与合规说明（请务必读一遍）
============================================================================

【数据集来源】
    GEFCom2014（Global Energy Forecasting Competition 2014）
    由 IEEE Power & Energy Society 主办，风电赛道含 10 个风电场。

    引用：
      Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A.,
      & Hyndman, R. J. (2016). Probabilistic energy forecasting:
      Global Energy Forecasting Competition 2014 and beyond.
      International Journal of Forecasting, 32(3), 896-913.

【权利归属】
    数据集的知识产权归**原始竞赛主办方**所有，不属于本项目。

    本项目**不重新分发**该数据集：
      · data/raw/ 已在 .gitignore 中排除
      · 数据由本脚本在运行时下载

    ⚠️ 如果你要发布这个项目，请勿把 data/raw/ 提交到仓库。

【获取途径】
    本脚本按以下顺序尝试：
      1. 社区 GitHub 镜像（下面 MIRRORS 里列出的几个）
      2. 如果全部失败，可从官方渠道手动获取后放进 data/raw/

    ⚠️ 关于镜像仓库的许可状态（已核查）：
      · greenlytics/gefcom2014-wind  —— **未声明开源协议**
      · andland/GEFCOM2014           —— GPL-2.0（但其覆盖的是作者的 R 代码）

      本项目**没有复制任何镜像仓库的代码**，源码为独立编写（MIT）。
      把镜像作为下载途径，不对其内容主张任何权利。

    其他可用渠道：
      · Kaggle 上的 GEFCom2014 镜像数据集
      · 竞赛官方历史存档

============================================================================
 数据内容
============================================================================
    data/Task N/TaskN_W_Zone1_10/TaskN_W_ZoneZ.csv
      - 10 个风电场（Zone 1~10）
      - 列：ZONEID, TIMESTAMP, TARGETVAR, U10, V10, U100, V100
      - TARGETVAR 是归一化出力（0~1）
      - U/V 是 10m 和 100m 高度的风矢量分量（NWP 预报）

============================================================================
 用法
============================================================================
    python data/download.py                 # 下载 Task 1 的 10 个风场
    python data/download.py --tasks 1 2 3   # 下载多个 Task
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import urllib.request
import urllib.parse

REPO = "greenlytics/gefcom2014-wind"
BRANCH = "master"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# 多镜像兜底：国内直连 raw.githubusercontent.com 经常被重置，
# 这几个镜像实测可用。
MIRRORS = [
    "https://fastly.jsdelivr.net/gh/{repo}@{branch}/{path}",
    "https://gcore.jsdelivr.net/gh/{repo}@{branch}/{path}",
    "https://gh-proxy.com/https://raw.githubusercontent.com/{repo}/{branch}/{path}",
    "https://ghfast.top/https://raw.githubusercontent.com/{repo}/{branch}/{path}",
    "https://raw.githubusercontent.com/{repo}/{branch}/{path}",
]


def fetch(path: str, dest: str, timeout: int = 120) -> bool:
    """从多个镜像尝试下载；成功返回 True。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print("      已存在，跳过")
        return True
    enc = urllib.parse.quote(path)
    for i, m in enumerate(MIRRORS, 1):
        url = m.format(repo=REPO, branch=BRANCH, path=enc)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 1000:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print("      镜像 %d 失败：%s" % (i, str(e)[:50]))
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", type=int, default=[1],
                    help="要下载的 Task 编号，默认 1")
    ap.add_argument("--zones", nargs="*", type=int, default=list(range(1, 11)),
                    help="要下载的风场编号，默认 1-10")
    args = ap.parse_args()

    print("下载 GEFCom2014 风电数据 → %s" % RAW)
    ok = fail = 0
    for task in args.tasks:
        print("\n Task %d" % task)
        for z in args.zones:
            path = "data/Task %d/Task%d_W_Zone1_10/Task%d_W_Zone%d.csv" % (task, task, task, z)
            dest = os.path.join(RAW, "Task%d_W_Zone1_10" % task, "Task%d_W_Zone%d.csv" % (task, z))
            print("   Zone %-2d ... " % z, end="")
            if fetch(path, dest):
                kb = os.path.getsize(dest) / 1024
                print("OK  %.0f KB" % kb)
                ok += 1
            else:
                print("失败")
                fail += 1

    print("\n完成：成功 %d，失败 %d" % (ok, fail))
    if fail:
        print("如果有失败，可以稍后重跑（已下载的会自动跳过），")
        print("或从 https://github.com/%s 手动下载后放进 %s" % (REPO, RAW))
        sys.exit(1)


if __name__ == "__main__":
    import urllib.parse   # noqa
    main()
