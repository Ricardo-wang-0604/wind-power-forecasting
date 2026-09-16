# -*- coding: utf-8 -*-
"""
data_qc.py —— 多场站数据质量与独立性检验

【这个模块解决什么问题】

    做多场站实验时有一个很隐蔽的陷阱：
    **你以为你有 N 个独立的场站，实际上可能只有 M 个（M < N）。**

    情况包括：
      · 几个场站共用同一份气象预报（本项目在 GEFCom2014 上就发现了）
      · 几个测风塔挨得太近，风速高度相关
      · 数据供应商把同一份数据重复卖给了你
      · 传感器故障导致数据被"复制"填充

【为什么这很危险】

    1. 样本量被高估 —— 你以为有 10 份独立信息，实际只有 8 份
    2. 交叉验证泄漏 —— 按场站划分时，共享组训练/测试之间互相泄漏，指标虚高
    3. 相关分析失真 —— 共享组内会得到虚高的相关系数

【本模块的能力】

    A. 独立性检验
       exact_duplicate_groups   逐时刻完全相同的场站分组（哈希法，快）
       near_duplicate_pairs     数值接近的场站对（容差法）
       correlation_matrix       场站间相关矩阵
       correlation_clusters     按相关性聚类
       independence_report      一站式报告

    B. 影响量化
       effective_sample_size    有效独立样本量
       cv_impact                对交叉验证的影响评估

    C. 单站质量校验（基础版）
       check_range              合理范围检验
       check_frozen             冻结数据识别（传感器卡死）
       check_jumps              相邻时刻突变检验
       missing_summary          缺测统计

【通用性】

    不绑定 GEFCom。任何「多站点 × 时间」的数据都能用：
    测风塔、气象站、光伏电站、风电场出力均适用。
"""

from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


# ==================================================================== 工具
_SENTINEL = -1e308          # 用哨兵值替换 NaN，保证哈希稳定


def _align(df: pd.DataFrame, station_col: str, value_cols):
    """把长表转成按场站对齐的数组。

    返回 (sigs, time_index, stations)
        sigs[station] = ndarray, shape = (n_times, n_value_cols)
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    pivots = []
    for c in value_cols:
        p = df.pivot(index='timestamp', columns=station_col, values=c)
        pivots.append(p)

    idx = pivots[0].index
    stations = list(pivots[0].columns)

    sigs = {}
    for s in stations:
        sigs[s] = np.column_stack([p[s].to_numpy(dtype=float) for p in pivots])
    return sigs, idx, stations


def _hash_array(a: np.ndarray) -> str:
    """对数组做稳定哈希（NaN 统一替换成哨兵，避免位模式差异）。"""
    x = np.where(np.isnan(a), _SENTINEL, a)
    x = np.ascontiguousarray(np.round(x, 9))
    return hashlib.sha1(x.tobytes()).hexdigest()


# ============================================================== A. 独立性检验
def exact_duplicate_groups(df: pd.DataFrame, station_col: str = 'station',
                           value_cols=('u100', 'v100'), tol: float = 0.0):
    """找出【逐时刻完全相同】的场站组（哈希法）。

    参数
    ----
    df           长表：timestamp / <station_col> / <value_cols>
    station_col  场站标识列名
    value_cols   用于比较的数值列（可以是单个字符串或列表）
    tol          容差；0 表示要求完全一致（走哈希快路径）

    返回
    ----
    dict: {代表站: [该组所有站]}
          只返回成员数 >= 2 的组

    【为什么用哈希】
        两两比对是 O(n²)，场站多了会慢。
        哈希把每个场站压成一个指纹，比对指纹是 O(n)。
        本项目 10 个场站两种方法都很快，但上百个测风塔时差距明显。
    """
    sigs, idx, stations = _align(df, station_col, value_cols)

    if tol and tol > 0:
        return _near_duplicate_groups(sigs, tol)

    buckets = {}
    for s, arr in sigs.items():
        buckets.setdefault(_hash_array(arr), []).append(s)
    return {v[0]: sorted(v) for v in buckets.values() if len(v) > 1}


def _near_duplicate_groups(sigs: dict, tol: float):
    """容差法找近似重复组（并查集）。"""
    stations = sorted(sigs)
    parent = {s: s for s in stations}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(stations):
        for b in stations[i + 1:]:
            d = np.nanmax(np.abs(sigs[a] - sigs[b]))
            if np.isfinite(d) and d <= tol:
                union(a, b)

    groups = {}
    for s in stations:
        groups.setdefault(find(s), []).append(s)
    return {v[0]: sorted(v) for v in groups.values() if len(v) > 1}


def near_duplicate_pairs(df: pd.DataFrame, station_col: str = 'station',
                         value_cols=('u100', 'v100'), tol: float = 0.5):
    """列出数值接近的场站对，附带最大绝对差（不要求完全相同）。"""
    sigs, idx, stations = _align(df, station_col, value_cols)
    rows = []
    for i, a in enumerate(sorted(stations)):
        for b in sorted(stations)[i + 1:]:
            diff = np.abs(sigs[a] - sigs[b])
            dmax = float(np.nanmax(diff))
            dmean = float(np.nanmean(diff))
            if dmax <= tol:
                rows.append({'场站A': a, '场站B': b,
                             '最大绝对差': round(dmax, 4),
                             '平均绝对差': round(dmean, 4)})
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, station_col: str = 'station',
                       value_col: str = 'power'):
    """场站间的相关系数矩阵。"""
    p = df.pivot(index='timestamp', columns=station_col, values=value_col)
    return p.corr()


def correlation_clusters(corr: pd.DataFrame, threshold: float = 0.9):
    """按相关系数阈值分组（并查集）。

    用于找出「虽然不是完全相同，但高度相关」的场站簇 ——
    它们可能是同一片风区的邻居，做交叉验证时同样会互相影响。
    """
    stations = list(corr.columns)
    parent = {s: s for s in stations}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(stations):
        for b in stations[i + 1:]:
            v = corr.loc[a, b]
            if np.isfinite(v) and v >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    groups = {}
    for s in stations:
        groups.setdefault(find(s), []).append(s)
    return {v[0]: sorted(v) for v in groups.values()}


# ============================================================== B. 影响量化
def effective_sample_size(n_stations: int, n_times: int,
                          duplicate_groups: dict):
    """计算有效独立样本量。

    逻辑：n 个场站里若有一组 k 个场站共享同一份数据，
    它们对「独立信息量」的贡献是 1 而不是 k。

    返回 dict：
        名义场站数 / 有效独立场站数 / 名义样本量 / 有效样本量 / 虚高倍数
    """
    n_dup_extra = sum(len(v) - 1 for v in duplicate_groups.values())
    n_eff = n_stations - n_dup_extra
    return {
        '名义场站数': n_stations,
        '重复组数': len(duplicate_groups),
        '因重复而多算的场站数': n_dup_extra,
        '有效独立场站数': n_eff,
        '名义样本量': n_stations * n_times,
        '有效样本量': n_eff * n_times,
        '样本量虚高倍数': round(n_stations / n_eff, 3) if n_eff else np.nan,
    }


def cv_impact(n_stations: int, duplicate_groups: dict, n_splits: int = None):
    """评估「按场站划分交叉验证」的泄漏风险。

    【为什么这是个问题】
        留一场交叉验证（leave-one-station-out）默认各场站独立。
        若两个场站共享同一份输入数据，则：
          · 训练集里有 A，测试集是 B
          · B 的输入数据与 A 完全相同 → 模型"见过"测试输入
          · 指标会虚高

    返回 dict，含风险等级与建议。
    """
    if n_splits is None:
        n_splits = n_stations
    n_eff = n_stations - sum(len(v) - 1 for v in duplicate_groups.values())

    leaking = []
    for rep, members in duplicate_groups.items():
        if len(members) > 1:
            leaking.append(' — '.join(str(m) for m in members))

    ratio = n_eff / n_stations if n_stations else np.nan
    if ratio >= 0.95:
        level, advice = '低', '场站基本独立，常规交叉验证可用'
    elif ratio >= 0.8:
        level, advice = '中', ('存在共享组。建议按【组】划分交叉验证，'
                               '即同组成员始终一起进训练集或一起进测试集')
    else:
        level, advice = '高', ('重复严重。必须按组划分交叉验证，'
                               '否则指标会明显虚高')

    return {
        '名义折数': n_splits,
        '建议折数（按组）': len(duplicate_groups) + (n_stations -
                              sum(len(v) for v in duplicate_groups.values())),
        '有效独立场站数': n_eff,
        '泄漏风险': level,
        '建议': advice,
        '会互相泄漏的组': leaking if leaking else ['（无）'],
    }


# ============================================================== C. 单站质量校验
def check_range(s: pd.Series, lo: float, hi: float):
    """合理范围检验：超出物理可能范围的点。"""
    v = pd.to_numeric(s, errors='coerce')
    bad = v[(v < lo) | (v > hi)]
    return {
        '检验': '范围检验',
        '区间': '[%g, %g]' % (lo, hi),
        '越界点数': int(bad.shape[0]),
        '越界占比%': round(bad.shape[0] / max(len(v.dropna()), 1) * 100, 3),
    }


def check_frozen(s: pd.Series, min_run: int = 6):
    """冻结数据识别：连续 min_run 个时刻数值完全相同。

    真实的风速不可能连续 6 小时一模一样 —— 这是传感器卡死的典型特征。
    （注意：风电场出力在"满发"或"停机"时确实会长时间恒定，
      所以这个方法更适合用在风速/气象数据上，用在出力上要调大 min_run。）
    """
    v = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
    runs = []
    i, n = 0, len(v)
    while i < n:
        if not np.isfinite(v[i]):
            i += 1
            continue
        j = i + 1
        while j < n and np.isfinite(v[j]) and v[j] == v[i]:
            j += 1
        if j - i >= min_run:
            runs.append({'起点': i, '长度': j - i, '数值': float(v[i])})
        i = j
    return {
        '检验': '冻结数据检验',
        '阈值(连续点数)': min_run,
        '冻结片段数': len(runs),
        '最长片段': max([r['长度'] for r in runs], default=0),
        '_明细': runs,
    }


def check_jumps(s: pd.Series, max_delta: float = 6.0):
    """相邻时刻突变检验。

    相邻两小时风速变化超过 max_delta m/s 通常是异常
    （除非是极端天气过程，需要结合其他站判断）。
    """
    v = pd.to_numeric(s, errors='coerce')
    d = v.diff().abs()
    bad = d[d > max_delta]
    return {
        '检验': '突变检验',
        '阈值(m/s)': max_delta,
        '突变点数': int(bad.shape[0]),
        '突变占比%': round(bad.shape[0] / max(len(d.dropna()), 1) * 100, 3),
        '最大变化': round(float(d.max()), 3) if len(d.dropna()) else np.nan,
    }


def missing_summary(s: pd.Series):
    v = pd.to_numeric(s, errors='coerce')
    n = len(v)
    miss = int(v.isna().sum())
    return {'检验': '缺测统计', '总点数': n, '缺测数': miss,
            '缺测率%': round(miss / n * 100, 3) if n else np.nan}


# ============================================================== 一站式报告
def independence_report(df: pd.DataFrame, station_col: str = 'station',
                        value_cols=('u100', 'v100'),
                        corr_col: str = 'power',
                        corr_threshold: float = 0.9,
                        tol: float = 0.0):
    """多场站数据独立性检验的一站式报告。

    返回一个 dict，里面每一项都可以直接打印或落盘。
    """
    sigs, idx, stations = _align(df, station_col, value_cols)
    n_st, n_t = len(stations), len(idx)

    dup = exact_duplicate_groups(df, station_col, value_cols, tol=tol)
    eff = effective_sample_size(n_st, n_t, dup)

    corr = correlation_matrix(df, station_col, corr_col)
    clusters = correlation_clusters(corr, corr_threshold)
    cv = cv_impact(n_st, dup)

    # 组内最大相关系数（用于说明泄漏程度）
    within = []
    for rep, members in dup.items():
        if len(members) < 2:
            continue
        vals = [corr.loc[a, b] for i, a in enumerate(members)
                for b in members[i + 1:]]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            within.append({'组': ' — '.join(str(m) for m in members),
                           '组内最大相关': round(float(np.max(vals)), 4),
                           '组内平均相关': round(float(np.mean(vals)), 4)})

    return {
        '场站数': n_st,
        '时间点数': n_t,
        '完全重复组': dup,
        '完全重复组数': len(dup),
        '高相关簇(阈值%.2f)' % corr_threshold: clusters,
        '有效样本量': eff,
        '交叉验证影响': cv,
        '组内相关系数': pd.DataFrame(within),
        '_相关矩阵': corr,
        '_场站': stations,
    }


def print_report(rep: dict):
    """把报告打印成可读形式。"""
    print('=' * 64)
    print(' 多场站数据独立性检验报告')
    print('=' * 64)
    print('  场站数：%d　时间点数：%d　名义样本量：%s'
          % (rep['场站数'], rep['时间点数'], f"{rep['有效样本量']['名义样本量']:,}"))

    print('\n  【一】逐时刻完全相同的场站组')
    if rep['完全重复组']:
        for rep_st, members in rep['完全重复组'].items():
            print('      ⚠️  %s' % ' — '.join(str(m) for m in members))
            print('          这些场站的输入数据完全相同，只有目标变量不同')
    else:
        print('      ✅ 未发现完全重复')

    print('\n  【二】有效样本量')
    e = rep['有效样本量']
    print('      名义场站数　　　　%d' % e['名义场站数'])
    print('      重复组数　　　　　%d' % e['重复组数'])
    print('      因重复多算的场站　%d' % e['因重复而多算的场站数'])
    print('      有效独立场站数　　%d' % e['有效独立场站数'])
    print('      名义样本量　　　　%s' % f"{e['名义样本量']:,}")
    print('      有效样本量　　　　%s' % f"{e['有效样本量']:,}")
    print('      样本量虚高倍数　　%.3f' % e['样本量虚高倍数'])

    print('\n  【三】交叉验证影响')
    cv = rep['交叉验证影响']
    print('      泄漏风险　　　　%s' % cv['泄漏风险'])
    print('      名义折数　　　　%d' % cv['名义折数'])
    print('      建议折数（按组）%d' % cv['建议折数（按组）'])
    print('      建议：%s' % cv['建议'])
    if cv['会互相泄漏的组'] and cv['会互相泄漏的组'][0] != '（无）':
        print('      会互相泄漏的组：')
        for g in cv['会互相泄漏的组']:
            print('        · %s' % g)

    print('\n  【四】组内相关系数（说明泄漏程度）')
    w = rep['组内相关系数']
    if len(w):
        print('      ' + w.to_string(index=False).replace('\n', '\n      '))
    else:
        print('      （无重复组）')

    k = [k for k in rep if k.startswith('高相关簇')][0]
    print('\n  【五】高相关簇（阈值见标题）')
    cl = rep[k]
    if cl:
        for rep_st, members in cl.items():
            print('      · %s' % ' — '.join(str(m) for m in members))
    else:
        print('      （无）')
    print()
