# -*- coding: utf-8 -*-
"""
wind_resource_report.py —— 风资源评估演示

用 GEFCom2014 数据里的 U10/V10 与 U100/V100 两个高度做完整的风资源评估：
    风速合成 → 韦布尔分布拟合 → 风功率密度 → 资源分级
    → 风切变外推至轮毂高度 → 发电量 AEP → P50/P75/P90

用法：
    python wind_resource_report.py
    python wind_resource_report.py --zones 1 2 3 --hub 120
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from loader import load_zones                                   # noqa: E402
from features import wind_speed                                 # noqa: E402
import wind_resource as wr                                      # noqa: E402
# 注意：shear_from_two_levels 定义在 wind_resource 里（与 features.shear_exponent 等价，
# 但风资源模块自带一份，避免风资源评估依赖预测流程。此处统一从 wind_resource 引入）
shear_from_two_levels = wr.shear_from_two_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="*", type=int, default=None)
    ap.add_argument("--hub", type=float, default=100.0, help="轮毂高度 m")
    ap.add_argument("--capacity", type=float, default=50.0, help="单场装机容量 MW")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    print("=" * 66)
    print(" 风资源评估（数据来源：GEFCom2014 风电赛道）")
    print("=" * 66)

    df = load_zones(args.zones)
    df["ws10"] = wind_speed(df.u10, df.v10)
    df["ws100"] = wind_speed(df.u100, df.v100)
    # 用两个实测高度反算真实风切变，而不是套默认值
    df["shear"] = shear_from_two_levels(df.ws100, df.ws10, 100, 10)

    zones = sorted(df.zone.unique())
    print(f"\n  风场数 {len(zones)}　样本 {len(df)} 条")
    print(f"  观测高度 10 m 与 100 m　轮毂高度 {args.hub:.0f} m")
    print(f"  单场装机 {args.capacity:.0f} MW\n")

    # 全局风切变（各场均值）
    alpha_mean = float(np.nanmean(df.shear.values))
    print(f"  【风切变】全样本平均 α = {alpha_mean:.3f}")
    print("      参考值：开阔海面≈0.14，草原≈0.25，城市或夜间稳定层结>0.30\n")

    rows = []
    for z in zones:
        sub = df[df.zone == z]
        # 用实测 α 外推，而不是套默认值 0.143
        alpha = float(np.nanmean(sub.shear.values))
        ws_hub = wr.extrapolate_shear(sub.ws100.values, 100.0, args.hub, alpha)

        # ⚠️ 分级必须在【10 m 高度】做：GB/T 18710 的风功率密度等级表是按 10 m / 50 m 给的，
        #    拿轮毂高度的风功率密度去套 10 m 的表会得出错误等级。
        rep10 = wr.wind_resource_report(pd.Series(sub.ws10.values),
                                        hub_height=10.0, z_obs=10.0,
                                        capacity_mw=args.capacity)
        rep = wr.wind_resource_report(pd.Series(ws_hub), hub_height=args.hub,
                                      z_obs=args.hub, capacity_mw=args.capacity)
        rows.append({
            '风场': f'Zone {z}',
            '10m风速': round(float(np.nanmean(sub.ws10.values)), 2),
            '轮毂风速': rep['平均风速(m/s)'],
            'k': rep['韦布尔_k(MLE)'],
            'c': rep['韦布尔_c(MLE)'],
            '切变α': round(alpha, 3),
            '10m风功率密度': round(rep10['风功率密度(W/m2)'], 0),
            '资源等级': rep10['资源等级'],
            '轮毂风功率密度': round(rep['风功率密度(W/m2)'], 0),
            '容量因子': round(rep['容量因子'], 3),
            'P50(MWh)': round(rep['AEP_P50(MWh)'], 0),
            'P75(MWh)': round(rep['AEP_P75(MWh)'], 0),
            'P90(MWh)': round(rep['AEP_P90(MWh)'], 0),
        })

    tbl = pd.DataFrame(rows)
    print("  【逐场风资源评估】（资源等级按 10 m 高度判定）")
    print(tbl.to_string(index=False).replace('\n', '\n  '))

    print("\n  【总体】")
    print(f"      10m 平均风速 {tbl['10m风速'].mean():.2f} m/s"
          f"　轮毂 {args.hub:.0f}m 平均风速 {tbl['轮毂风速'].mean():.2f} m/s")
    print(f"      10m 平均风功率密度 {tbl['10m风功率密度'].mean():.0f} W/m²"
          f"　轮毂平均 {tbl['轮毂风功率密度'].mean():.0f} W/m²")
    print(f"      平均容量因子 {tbl['容量因子'].mean():.3f}")
    print(f"      合计 P50 {tbl['P50(MWh)'].sum():,.0f} MWh/年"
          f"（{len(zones)} 场 × {args.capacity:.0f} MW）")

    # 不确定性预算
    sigma, ub = wr.uncertainty_budget()
    print(f"\n  【不确定性预算】综合 {sigma * 100:.1f}%")
    print(ub.to_string(index=False).replace('\n', '\n      '))

    # 单场样例的详细分布
    z0 = zones[0]
    sub = df[df.zone == z0]
    alpha = float(np.nanmean(sub.shear.values))
    ws_hub = wr.extrapolate_shear(sub.ws100.values, 100.0, args.hub, alpha)
    rep = wr.wind_resource_report(pd.Series(ws_hub), hub_height=args.hub,
                                  z_obs=args.hub, capacity_mw=args.capacity)

    print(f"\n  【Zone {z0} 的韦布尔分布参数】")
    print(f"      MLE 法  ：k = {rep['韦布尔_k(MLE)']}　c = {rep['韦布尔_c(MLE)']}")
    print(f"      矩估计法：k = {rep['韦布尔_k(矩估计)']}　c = {rep['韦布尔_c(矩估计)']}")
    print(f"      （两法接近说明拟合可靠；k 通常 1.8~2.5）")

    print(f"\n  【Zone {z0} 的发电量构成（前 8 个风速段）】")
    det = rep['_AEP明细'].copy()
    det = det.sort_values('energy_MWh', ascending=False).head(8)
    print(det[['v_mid(m/s)', 'frequency', 'P_norm', 'energy_MWh']]
          .to_string(index=False).replace('\n', '\n      '))

    # 存盘
    os.makedirs(args.outdir, exist_ok=True)
    tbl.to_csv(os.path.join(args.outdir, 'wind_resource_by_zone.csv'),
               index=False, encoding='utf-8-sig')
    ub.to_csv(os.path.join(args.outdir, 'uncertainty_budget.csv'),
              index=False, encoding='utf-8-sig')
    print(f"\n  结果已存：results/wind_resource_by_zone.csv")
    print(f"           results/uncertainty_budget.csv")

    # 出图
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

        fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=140)

        v = np.linspace(0.05, 30, 300)
        ax[0].hist(ws_hub, bins=60, density=True, alpha=0.45,
                   color='#7f8c8d', label='观测频率')
        ax[0].plot(v, wr.weibull_pdf(v, rep['韦布尔_k(MLE)'], rep['韦布尔_c(MLE)']),
                   color='#c0392b', lw=2,
                   label=f"韦布尔拟合 k={rep['韦布尔_k(MLE)']:.2f} c={rep['韦布尔_c(MLE)']:.2f}")
        ax[0].set_xlabel('轮毂高度风速 (m/s)')
        ax[0].set_ylabel('概率密度')
        ax[0].set_title(f'Zone {z0} 风速分布与韦布尔拟合')
        ax[0].legend(fontsize=9)
        ax[0].grid(alpha=0.3, linestyle=':')

        pc_v, pc_p = rep['_功率曲线']
        ax[1].plot(pc_v, pc_p, color='#2980b9', lw=2, label='归一化功率曲线')
        d2 = rep['_AEP明细']
        ax[1].fill_between(d2['v_mid(m/s)'], 0, d2['energy_MWh'] / d2['energy_MWh'].max(),
                           color='#c9a227', alpha=0.35, label='各风速段发电量贡献（归一化）')
        ax[1].set_xlabel('风速 (m/s)')
        ax[1].set_ylabel('归一化')
        ax[1].set_title('功率曲线与发电量构成')
        ax[1].legend(fontsize=9)
        ax[1].grid(alpha=0.3, linestyle=':')

        fig.tight_layout()
        out_png = os.path.join(args.outdir, 'wind_resource.png')
        fig.savefig(out_png)
        plt.close(fig)
        print(f"            {out_png}")
    except Exception as e:
        print(f"  绘图跳过：{str(e)[:70]}")

    print("\n完成。")


if __name__ == '__main__':
    main()
