# -*- coding: utf-8 -*-
"""风电功率预测项目的核心模块。

模块职责：
    loader     —— 读数据、解析时间戳
    features   —— 物理量还原与特征工程
    baselines  —— 物理基线（功率曲线 / 立方律 / 线性）
    models     —— 机器学习模型（持续性 / Ridge / LightGBM）
    evaluate   —— 国标口径评估指标
    plots      —— 出图
"""
