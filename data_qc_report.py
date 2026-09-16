# -*- coding: utf-8 -*-
"""
data_qc_report.py —— 多场站数据独立性检验（含泄漏影响的实测）

三部分：
    1. 独立性检验报告     —— 谁和谁共享数据、有效样本量是多少
    2. 单站数据质量校验   —— 范围、冻结、突变、缺测
    3. 泄漏影响实测       —— 留一场交叉验证：按场站划分 vs 按组划分

用法：
    python data_qc_report.py
    python data_qc_report.py --tol 0.5      # 用容差找近似重复
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from loader import load_zones                       # noqa: E402
from features import wind_speed, build_features, FEATURE_COLS   # noqa: E402
import data_qc as qc                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=0.0,
                    help='判定重复的容差，0 表示要求完全相同')
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()

    print('=' * 64)
    print(' 多场站数据质量与独立性检验')
    print('=' * 64)

    df = load_zones()
    df['ws10'] = wind_speed(df.u10, df.v10)
    df['ws100'] = wind_speed(df.u100, df.v100)
    print('\n  数据：%d 行，%d 个场站，%s → %s\n'
          % (len(df), df.zone.nunique(), df.timestamp.min(), df.timestamp.max()))

    # ---------------- 第一部分 ----------------
    print('=' * 64)
    print(' 【第一部分】独立性检验')
    print('=' * 64)
    rep = qc.independence_report(df, station_col='zone',
                                 value_cols=['u100', 'v100'],
                                 corr_col='power', corr_threshold=0.9,
                                 tol=args.tol)
    qc.print_report(rep)

    # ---------------- 第二部分 ----------------
    print('=' * 64)
    print(' 【第二部分】单站数据质量校验')
    print('=' * 64)
    rows = []
    for s in sorted(df.zone.unique()):
        sub = df[df.zone == s]
        r = {'场站': 'Zone %d' % s}
        r.update(qc.check_range(sub.ws100, 0, 60))
        r.pop('检验', None); r.pop('区间', None)
        fr = qc.check_frozen(sub.ws100, min_run=6)
        r['冻结片段'] = fr['冻结片段数']
        r['最长冻结'] = fr['最长片段']
        jp = qc.check_jumps(sub.ws100, max_delta=6.0)
        r['突变点数'] = jp['突变点数']
        r['最大变化'] = jp['最大变化']
        ms = qc.missing_summary(sub.ws100)
        r['缺测率%'] = ms['缺测率%']
        rows.append(r)
    qc_tbl = pd.DataFrame(rows)
    print('  （校验对象：轮毂以下可用高度 ws100）')
    print('  ' + qc_tbl.to_string(index=False).replace('\n  ', '\n  '))

    # ---------------- 第三部分 ----------------
    print()
    print('=' * 64)
    print(' 【第三部分】泄漏影响实测（控制变量设计）')
    print('=' * 64)
    print('  问题：按场站划分交叉验证时，共享输入数据的场站会互相泄漏。')
    print()
    print('  ⚠️ 不能简单对比「留一场」与「留一组」—— 那样训练集场站数也变了，')
    print('     指标变差可能只是数据变少。所以用控制变量：')
    print('       A 含双胞胎（9 站）')
    print('       B 去双胞胎（8 站）')
    print('       C 去一个无关场站（8 站）  ← 与 B 场站数相同，唯一差别是去掉谁')
    print('     若 B 明显差于 C，说明双胞胎在训练集里提供了不该有的信息。')
    print()

    from leakage_test import controlled_leakage_test
    d_feat = build_features(df.copy(), add_lag=False).dropna(
        subset=FEATURE_COLS + ['power'])
    leak_tbl = controlled_leakage_test(d_feat, FEATURE_COLS, verbose=True)
    leak_tbl.to_csv(os.path.join(args.outdir, 'leakage_controlled_test.csv'),
                    index=False, encoding='utf-8-sig')

    # ---------------- 存盘 ----------------
    os.makedirs(args.outdir, exist_ok=True)
    qc_tbl.to_csv(os.path.join(args.outdir, 'data_qc_by_station.csv'),
                  index=False, encoding='utf-8-sig')
    rep['_相关矩阵'].to_csv(os.path.join(args.outdir, 'station_correlation.csv'),
                            encoding='utf-8-sig')

    print('\n  结果已存：')
    print('    results/data_qc_by_station.csv')
    print('    results/station_correlation.csv')
    print('    results/leakage_controlled_test.csv')

    # ---------------- 出图 ----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for f in ['Microsoft YaHei', 'SimHei']:
            try:
                matplotlib.rcParams['font.sans-serif'] = [f]
                break
            except Exception:
                pass
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=140)

        # 左：相关矩阵热图
        cm = rep['_相关矩阵']
        im = ax[0].imshow(cm.values, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax[0].set_xticks(range(len(cm.columns)))
        ax[0].set_xticklabels(['Z%s' % c for c in cm.columns], fontsize=9)
        ax[0].set_yticks(range(len(cm.index)))
        ax[0].set_yticklabels(['Z%s' % c for c in cm.index], fontsize=9)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax[0].text(j, i, '%.2f' % cm.values[i, j], ha='center',
                           va='center', fontsize=7,
                           color='white' if cm.values[i, j] > 0.6 else 'black')
        ax[0].set_title('场站出力相关系数矩阵')
        plt.colorbar(im, ax=ax[0], fraction=0.046)

        # 右：控制变量实验（两组的 A/B/C 并列）
        n = len(leak_tbl)
        x = np.arange(n)
        w = 0.26
        ax[1].bar(x - w, leak_tbl['A_含双胞胎(9站)'], width=w,
                  color='#7f8c8d', label='A 含双胞胎（9 站）')
        ax[1].bar(x, leak_tbl['B_去双胞胎(8站)'], width=w,
                  color='#c0392b', label='B 去双胞胎（8 站）')
        ax[1].bar(x + w, leak_tbl['C_去无关站(8站)均值'], width=w,
                  color='#2980b9',
                  yerr=leak_tbl['C_标准差'], capsize=4,
                  label='C 去无关场站（8 站）')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(leak_tbl['测试场站'], fontsize=9)
        ax[1].set_ylabel('留出该场站的 RMSE')
        ax[1].set_title('泄漏控制变量实验')
        ax[1].legend(fontsize=8)
        ax[1].grid(axis='y', alpha=0.3, linestyle=':')

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'data_qc.png'))
        plt.close(fig)
        print('    results/data_qc.png')
    except Exception as e:
        print('  绘图跳过：%s' % str(e)[:70])

    print('\n完成。')


if __name__ == '__main__':
    main()
