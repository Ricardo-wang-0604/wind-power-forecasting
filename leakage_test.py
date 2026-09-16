# -*- coding: utf-8 -*-
"""
leakage_test.py —— 交叉验证泄漏的控制变量实验

【要回答的问题】
    如果两个场站共享同一份 NWP 预报，那么「按场站划分交叉验证」时，
    把其中一个留在训练集里，会不会让对另一个的预测**虚假变好**？

【为什么不能简单对比】
    最直觉的做法是比「留一场」与「留一组」：
        留一场：训练集 9 个场站，其中含该场站的双胞胎
        留一组：训练集 8 个场站，不含双胞胎
    但这样有两个变量同时变了 —— 既去掉了双胞胎，又少了一个场站的数据。
    指标变差到底是「去掉双胞胎」还是「数据变少」造成的，分不清。

【控制变量设计】
    固定测试场站 T，对比以下条件（**训练集场站数完全相同**）：

        条件 A（有双胞胎）：训练 = 全部 − {T}
        条件 B（去双胞胎）：训练 = 全部 − {T, 双胞胎}
        条件 C（去随机场站）：训练 = 全部 − {T, 某个非双胞胎 k}

    A vs C：场站数不同（9 vs 8），用于看「多一个场站」的普遍收益
    **B vs C：场站数相同（都是 8），唯一差别是去掉的是双胞胎还是无关场站**
            → 若 B 明显差于 C，说明双胞胎在训练集里提供了**不该有的信息**
               （模型见过完全相同的输入行），即存在泄漏。

    为了降低单次随机波动，条件 C 对多个 k 重复取平均。
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def controlled_leakage_test(df: pd.DataFrame, feature_cols, target: str = 'power',
                            station_col: str = 'zone',
                            dup_groups: dict = None,
                            n_estimators: int = 200,
                            verbose: bool = True):
    """对每个「重复组」做控制变量实验，返回结果表。

    参数
    ----
    df            已含特征的数据（需已 build_features）
    feature_cols  特征列
    dup_groups    重复组，形如 {代表站: [成员站]}；None 则自动只测已知的 (4,5)、(7,8)

    返回 DataFrame，每组一行，列出三种条件的 RMSE。
    """
    import lightgbm as lgb

    params = dict(n_estimators=n_estimators, learning_rate=0.08,
                  num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    stations = sorted(df[station_col].unique())
    all_text = df

    if dup_groups is None:
        dup_groups = {4: [4, 5], 7: [7, 8]}

    rows = []
    for target_st in sorted(dup_groups):
        members = dup_groups[target_st]
        twin = [m for m in members if m != target_st]
        if not twin:
            continue
        twin = twin[0]
        controls = [s for s in stations if s not in members]

        te = all_text[all_text[station_col] == target_st]

        def rmse_for(exclude):
            tr = all_text[~all_text[station_col].isin(exclude)]
            m = lgb.LGBMRegressor(**params)
            m.fit(tr[feature_cols], tr[target])
            p = np.clip(m.predict(te[feature_cols]), 0, 1)
            return float(np.sqrt(np.mean((te[target].values - p) ** 2)))

        # 条件 A：训练集含双胞胎（9 个场站）
        a = rmse_for([target_st])
        # 条件 B：去掉双胞胎（8 个场站）
        b = rmse_for([target_st, twin])
        # 条件 C：去掉一个无关场站（8 个场站），多个 k 取平均
        cs = []
        for k in controls:
            cs.append(rmse_for([target_st, k]))
        c_mean = float(np.mean(cs))
        c_std = float(np.std(cs))

        rows.append({
            '测试场站': 'Zone %d' % target_st,
            '其双胞胎': 'Zone %d' % twin,
            'A_含双胞胎(9站)': round(a, 4),
            'B_去双胞胎(8站)': round(b, 4),
            'C_去无关站(8站)均值': round(c_mean, 4),
            'C_标准差': round(c_std, 4),
            'B−C(泄漏量)': round(b - c_mean, 4),
            'A−C(多一站收益)': round(a - c_mean, 4),
            '_C各次': [round(x, 4) for x in cs],
        })

    tbl = pd.DataFrame(rows)

    if verbose and len(tbl):
        print('  控制变量实验（训练集场站数已对齐）')
        print('  ' + tbl.drop(columns=['_C各次']).to_string(index=False)
              .replace('\n  ', '\n  '))
        print()
        for _, r in tbl.iterrows():
            print('  【解读】%s（双胞胎 %s）' % (r['测试场站'], r['其双胞胎']))
            leak = r['B−C(泄漏量)']
            gain = r['A−C(多一站收益)']
            print('      去掉双胞胎 vs 去掉无关场站（都是 8 站）：RMSE 差 %+.4f' % leak)
            print('      含双胞胎 vs 去掉无关场站（9 站 vs 8 站）：RMSE 差 %+.4f' % gain)
            if leak > 2 * r['C_标准差'] and leak > 0:
                print('      → 去掉双胞胎的损失明显大于去掉无关场站，'
                      '**构成泄漏证据**')
            elif leak > 0:
                print('      → 方向上支持泄漏，但差异在噪声范围内，证据不足')
            else:
                print('      → 未观察到泄漏')
            print()

    return tbl


def main():
    import os, sys, argparse
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, 'src'))
    from loader import load_zones
    from features import build_features, FEATURE_COLS

    ap = argparse.ArgumentParser()
    ap.add_argument('--trees', type=int, default=200)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()

    print('=' * 64)
    print(' 交叉验证泄漏 · 控制变量实验')
    print('=' * 64)
    df = load_zones()
    d = build_features(df, add_lag=False).dropna(subset=FEATURE_COLS + ['power'])
    print('  样本 %d 行，%d 个场站\n' % (len(d), d.zone.nunique()))

    tbl = controlled_leakage_test(d, FEATURE_COLS, n_estimators=args.trees)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, 'leakage_controlled_test.csv')
    tbl.to_csv(out, index=False, encoding='utf-8-sig')
    print('  结果已存：%s' % out)


if __name__ == '__main__':
    main()
