# -*- coding: utf-8 -*-
"""
models.py —— 机器学习模型

除了物理基线，还要有一个**持续性基线（persistence）**：
    预测 t+1 时刻的出力 = t 时刻的实际出力
它是所有时间序列预测的**最低门槛**。如果你的机器学习模型打不过持续性，
说明模型根本没学到东西 —— 这是新人最容易忽略的一道检验。

本项目模型：
  1. Persistence   —— 持续性基线（最低门槛）
  2. LightGBM      —— 主力模型
  3. Ridge         —— 线性模型，用来验证「特征是否已经够好」
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class Persistence:
    """持续性预测：直接用上一时刻的实测出力。

    在超前 1 小时（t+1）的预测里，它其实很强 —— 因为风是连续的。
    很多论文号称模型比基线好 30%，但没跟持续性比过，
    结果实际上连持续性都不如。
    """

    name = "基线·持续性"

    def __init__(self, lag_col: str = "power_lag1"):
        self.lag_col = lag_col

    def fit(self, df, target="power"):
        return self

    def predict(self, df):
        return np.clip(df[self.lag_col].values, 0, 1)


class LightGBMModel:
    """LightGBM 回归。

    为什么选 LightGBM 而不是深度学习：
      - 表格数据上树模型通常更强
      - 训练快、可解释（能出特征重要性）
      - 对缺失值天然鲁棒
      - **面试时能讲清楚为什么选它，比硬上一个 LSTM 更有说服力**
    """

    def __init__(self, **params):
        self.params = dict(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.params.update(params)
        self.model = None
        self.feature_names = None
        self.name = "LightGBM"

    def fit(self, df, feature_cols, target="power"):
        import lightgbm as lgb
        d = df.dropna(subset=feature_cols + [target])
        self.feature_names = list(feature_cols)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(d[feature_cols], d[target])
        return self

    def predict(self, df):
        return np.clip(self.model.predict(df[self.feature_names]), 0, 1)

    def importance(self) -> pd.DataFrame:
        """特征重要性 —— README 里放这张图，能看出「物理特征有没有起作用」。"""
        return (pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True))


class RidgeModel:
    """岭回归。需要一个标准化 + 缺失填充的管道。

    它的作用是当「下限检验」：如果 LightGBM 比 Ridge 好很多，
    说明特征里有非线性关系（比如 v³、分段功率曲线）——
    这正好证明物理特征是有用的。
    """

    name = "Ridge回归"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pipe = None
        self.feature_names = None

    def fit(self, df, feature_cols, target="power"):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        d = df.dropna(subset=list(feature_cols) + [target])
        self.feature_names = list(feature_cols)
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self.pipe.fit(d[self.feature_names], d[target])
        return self

    def predict(self, df):
        x = df[self.feature_names].fillna(df[self.feature_names].median())
        return np.clip(self.pipe.predict(x), 0, 1)
