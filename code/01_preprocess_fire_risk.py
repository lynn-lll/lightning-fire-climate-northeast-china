# path: pipelines/stepwise_fire_pipeline.py
# -*- coding: utf-8 -*-
"""
Stepwise Fire Modeling Preprocess (改动版)
改动要点：
变更点：
1) 完全取消去极值（winsorize）。Step3 作为空操作仅做“承上启下”的占位。
仍保留你之前的改动：
2) Step2: 缺失值填补后立即做“特征标准化”（支持 standard/minmax/robust）。
3) 切分比例：train/val/test = 50%/30%/20%（val_within_train=0.375）。
4) 类别不平衡：先下采样多数类，再 SMOTE，最终得到正负样本=2000:2000。

Stepwise Fire Modeling Preprocess:

  step1: 原始→清洗→可视化
惰性读 CSV（scan_csv），经纬度过滤、去重、派生时间字段（year/month/doy/week/quarter）。
写出 step1_stage.parquet。
质量检查：缺失率表、特征描述统计。
可视化：按配置抽样，绘制直方图/箱线图与缺失率条形图。
输出：step1_stage.parquet、step1_missing_report.csv、step1_describe_features.csv、figures/step1_*.

  step2: 缺失值填补 + 标准化
仅对特征列填补：全局中位数（或可选“分月中位数”）。
仅对特征列做缩放：默认 StandardScaler，可选 MinMaxScaler（0–1）或 RobustScaler（分位稳健）。
保存拟合后的 scaler。
输出：step2_imputed_std.parquet、scaler_step2_<scaler>.joblib、step2_missing_report.csv.

  step3: 不去极值（No-op）
不做 IQR/Winsorize，直接把 Step2 的结果原样写为下一步输入名。
输出：step3_winsor.parquet（名称保留，内容等同 Step2 结果）。

  step4: 再次可视化（确认填补与标准化效果）
同 Step1：缺失率、描述统计、分布图（基于 step3_winsor.parquet）。
输出：step4_missing_report.csv、step4_describe_features.csv、figures/step4_*.

  step5: 时间切分（50% / 30% / 20%）
按 valid_time 排序切分：Train 50%，Val 30%，Test 20%（通过 test_size=0.20 与 val_size_within_train=0.375 实现）。
导出分类数据集（特征 + fire_presence）与回归数据集（特征 + burned_area_ha_sum、duration_days_sum）。
输出：
分类：step5_cls_train.parquet / step5_cls_val.parquet / step5_cls_test.parquet
回归：step5_reg_train.parquet / step5_reg_val.parquet / step5_reg_test.parquet
以及类别分布图表（train/val/test）。

  step6: 类别不平衡处理（仅训练集 → 2000:2000）
先随机下采样多数类到目标上限；再对少数类做 SMOTE 合成（无 imblearn 时回退为有放回采样），使正负各 2000（若某类不足 2000，会按可达上限对齐并告警）。
输出：step6_cls_train_balanced.parquet + 类别分布图表。

  step7: 汇总元信息
汇总元信息到 step7_meta.json（包含步骤产物路径、特征/目标列、缩放器路径等）。

"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Iterable
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
from polars import col, lit, when
from polars import DataType

import matplotlib
matplotlib.use("Agg")  # 后端无GUI环境
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import joblib
from dataclasses import dataclass, asdict

# optional imblearn（有则用SMOTE，否则仅下采样）
try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except Exception:
    RandomUnderSampler = None
    SMOTE = None
    HAS_IMBLEARN = False

# =========================
# 配置 & 列名
# =========================

ALL_COLUMNS: List[str] = [
    "valid_time","grid_id","lon","lat","avg_ie","avg_pevr","avg_tprate","slhf","sro","sshf","ssr","str","tp",
    "cbh","i10fg","blh","cape","cvh","cvl","d2m","lai_hv","lai_lv","si10","skt","swvl1","t2m","tcc","z",
    "K_index","SPEI_1","SPEI_3","VHI","fire_count","fire_presence","burned_area_ha_sum","burned_area_ha_mean",
    "duration_days_sum","events"
]
# 目标
CLS_TARGET = "fire_presence"
REG_TARGETS = ["burned_area_ha_sum", "duration_days_sum"]
# ✅ 明确：这6列均为“火灾结果”，不作为特征参与建模
EXCLUDE_FROM_FEATURES = [
    "fire_count",
    "fire_presence",
    "burned_area_ha_sum",
    "burned_area_ha_mean",
    "duration_days_sum",
    "events",
]

ID_COLS = ["grid_id", "lon", "lat"]
TIME_COL = "valid_time"

# def infer_feature_cols(all_cols: List[str]) -> List[str]:
#     # 仅保留真正气象/环境特征；剔除目标、时间、ID、以及6项火灾结果列
#     skip = set([TIME_COL] + ID_COLS + EXCLUDE_FROM_FEATURES)
#     return [c for c in all_cols if c not in skip]

# 供识别连续特征用：原来就有的气象/环境特征集合（不含 mon_*）
def infer_continuous_feature_cols(all_cols: list[str]) -> list[str]:
    skip = set([TIME_COL] + ID_COLS + EXCLUDE_FROM_FEATURES)
    return [c for c in all_cols if c not in skip]

# 生成 5-8 月份 One-Hot 列（默认使用 cfg.keep_months）
def add_month_dummies(df: pl.DataFrame, months=(5,6,7,8), prefix: str = "mon_") -> tuple[pl.DataFrame, list[str]]:
    mons = sorted(set(int(m) for m in months))
    dummy_cols = []
    out = df
    for m in mons:
        col_name = f"{prefix}{m:02d}"
        dummy_cols.append(col_name)
        out = out.with_columns(
            (pl.when(pl.col("month") == m).then(1).otherwise(0)).cast(pl.Int8).alias(col_name)
        )
    return out, dummy_cols

# 连续特征 + 哑变量 合并得到最终特征列清单
def list_final_feature_cols(df: pl.DataFrame) -> list[str]:
    cont = infer_continuous_feature_cols(ALL_COLUMNS)
    # mon_* 一律视为特征
    mon_cols = [c for c in df.columns if c.startswith("mon_")]
    return cont + mon_cols

DTYPES = {
    "valid_time": pl.Datetime,
    "grid_id": pl.Int32,
    "lon": pl.Float32,
    "lat": pl.Float32,
    "avg_ie": pl.Float32,
    "avg_pevr": pl.Float32,
    "avg_tprate": pl.Float32,
    "slhf": pl.Float32,
    "sro": pl.Float32,
    "sshf": pl.Float32,
    "ssr": pl.Float32,
    "str": pl.Float32,
    "tp": pl.Float32,
    "cbh": pl.Float32,
    "i10fg": pl.Float32,
    "blh": pl.Float32,
    "cape": pl.Float32,
    "cvh": pl.Float32,
    "cvl": pl.Float32,
    "d2m": pl.Float32,
    "lai_hv": pl.Float32,
    "lai_lv": pl.Float32,
    "si10": pl.Float32,
    "skt": pl.Float32,
    "swvl1": pl.Float32,
    "t2m": pl.Float32,
    "tcc": pl.Float32,
    "z": pl.Float32,
    "K_index": pl.Float32,
    "SPEI_1": pl.Float32,
    "SPEI_3": pl.Float32,
    "VHI": pl.Float32,
}

@dataclass
class CFG:
    input_csv: str
    out_dir: str = "artifacts"
    fig_dir: str = "figures"
    sample_for_plots: int = 200_000
    random_state: int = 42
    use_monthwise_impute: bool = False
    test_size: float = 0.20
    val_size_within_train: float = 0.375
    # target_ratio_after_undersample: float = 20.0
    # target_ratio_after_smote: float = 4.0
    plot_k: int = 28
    plot_list: Optional[List[str]] = None
    scaler: str = "minmax"  # 新增：standard / minmax / robust
    final_samples: int = 2000  # 新增：Step6 最终正/负样本数
    # 新增：建模月份
    keep_months: tuple = (5, 6, 7, 8)

# =========================
# 通用工具
# =========================

# 统一的安全 groupby 封装（兼容老/新版本）
def _gb(df: pl.DataFrame, by):
    try:
        return df.group_by(by)   # 新 API
    except AttributeError:
        return df.groupby(by)    # 旧 API（若可用）

def ensure_dirs(cfg: CFG) -> Tuple[str, str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    fdir = os.path.join(cfg.out_dir, cfg.fig_dir)
    os.makedirs(fdir, exist_ok=True)
    return cfg.out_dir, fdir

def read_csv_lazy(path: str) -> pl.LazyFrame:
    lf = pl.scan_csv(
        path,
        has_header=True,
        try_parse_dates=True,
        ignore_errors=True,
        schema_overrides=DTYPES,   # 代替 dtypes=
    )
    # 获取可用列名（不触发昂贵的 schema 解析警告）
    sch = lf.collect_schema()
    # 兼容不同版本：Schema 可能有 .names()；没有就用 list(sch)
    try:
        available = set(sch.names())
    except AttributeError:
        available = set(list(sch))
    # 仅选择文件里真的存在的列，避免 KeyError
    keep = [c for c in ALL_COLUMNS if c in available]
    return lf.select([pl.col(c) for c in keep])

# 在 basic_clean 里过滤月份（5–8）
def basic_clean(lf: pl.LazyFrame, keep_months=(5,6,7,8)) -> pl.LazyFrame:
    return (
        lf.filter((col("lon") >= -180) & (col("lon") <= 180) & (col("lat") >= -90) & (col("lat") <= 90))
          .unique(subset=ALL_COLUMNS)
          .with_columns([
              col("valid_time").dt.cast_time_unit("us").alias("valid_time"),
              col("valid_time").dt.truncate("1d").alias("valid_date"),
          ])
          .with_columns([
              col("valid_date").dt.year().alias("year"),
              col("valid_date").dt.month().alias("month"),
              col("valid_date").dt.ordinal_day().alias("doy"),
              col("valid_date").dt.week().alias("week"),
              col("valid_date").dt.quarter().alias("quarter"),
          ])
          # ★ 仅保留 5–8 月
          .filter(col("month").is_in(list(keep_months)))
    )

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    # 若 events 为文本可在此编码；本版作为“结果列”已被排除，不参与特征
    return df

def compute_missing_report(df: pl.DataFrame) -> pl.DataFrame:
    """兼容当前 Polars：null_count() 返回 1 行 DataFrame。"""
    n = df.height
    miss_df = df.null_count()          # 1 行，每列=对应列的缺失数
    cols = miss_df.columns
    counts = miss_df.row(0)            # -> tuple 对应 cols 顺序
    if n == 0:
        rates = [0.0] * len(cols)
    else:
        rates = [float(c) / n for c in counts]
    return (
        pl.DataFrame({
            "column": cols,
            "missing_count": [int(c) for c in counts],
            "missing_rate": rates,
        })
        .sort("missing_rate", descending=True)
    )


def save_csv(df: pl.DataFrame, path: str):
    df.write_csv(path)

def sample_df(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    if df.height == 0:
        return df
    if df.height <= n:
        return df
    frac = n / df.height
    try:
        # 新版本
        return df.sample(fraction=frac, with_replacement=False, seed=seed)
    except TypeError:
        # 旧版本回退：按行数采样
        m = max(1, int(n))
        return df.sample(n=m, with_replacement=False, seed=seed)

def plot_missing_bar(miss_df: pl.DataFrame, save_path: str):
    pdf = miss_df.to_pandas()
    plt.figure(figsize=(12, 6))
    plt.bar(pdf["column"], pdf["missing_rate"])
    plt.xticks(rotation=90); plt.ylabel("Missing Rate"); plt.title("Missing Rate by Column")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def select_plot_columns(all_cols: List[str], k: int = 28) -> List[str]:
    # pri = ["t2m","tp","si10","i10fg","d2m","skt","tcc","cape","VHI","SPEI_1","SPEI_3","avg_tprate","sshf","slhf","ssr","str"]
    pri = ["avg_ie", "avg_pevr", "avg_tprate", "slhf", "sro", "sshf", "ssr",
           "str", "tp", "cbh", "i10fg", "blh", "cape", "cvh",
           "cvl", "d2m", "lai_hv", "lai_lv", "si10", "skt", "swvl1",
           "t2m", "tcc", "z", "K_index", "SPEI_1", "SPEI_3", "VHI"]
    pri = [c for c in pri if c in all_cols]
    # 也排除6个结果列
    extras = [c for c in all_cols if c not in pri and c not in [TIME_COL] + ID_COLS + EXCLUDE_FROM_FEATURES]
    return (pri + extras)[:k]

def plot_hists_and_boxes(df: pl.DataFrame, cols: List[str], out_dir: str, prefix: str):
    pdf = df.select(cols).to_pandas()
    for c in cols:
        s = pdf[c].dropna()
        if s.empty: continue
        plt.figure(figsize=(6,4)); plt.hist(s, bins=50)
        plt.title(f"{c} - Histogram"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}hist_{c}.png")); plt.close()

        plt.figure(figsize=(4,6)); plt.boxplot(s, vert=True, showfliers=True)
        plt.title(f"{c} - Boxplot"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}box_{c}.png")); plt.close()

# =========================
# 缺失/标准化/去极值
# =========================
def impute_missing(df: pl.DataFrame, feat_cols: List[str], monthwise: bool) -> pl.DataFrame:
    # 仅给特征列填补；目标列保持原样
    if not monthwise:
        meds = df.select([col(c).median().alias(c) for c in feat_cols]).row(0)
        mmap = {feat_cols[i]: meds[i] for i in range(len(feat_cols))}
        return df.with_columns([when(col(c).is_null()).then(lit(mmap[c])).otherwise(col(c)).alias(c) for c in feat_cols])
    joined = df.join(
        df.groupby("month").median().select(["month"] + feat_cols).rename({c: f"{c}_med" for c in feat_cols}),
        on="month", how="left"
    )
    for c in feat_cols:
        joined = joined.with_columns(when(col(c).is_null()).then(col(f"{c}_med")).otherwise(col(c)).alias(c)).drop(f"{c}_med")
    return joined

def build_scaler(name: str):
    if name == "minmax":
        return MinMaxScaler()         # 0-1 标准化（Max-Min）
    if name == "robust":
        return RobustScaler()         # 对重尾/异常更稳
    return StandardScaler()           # 默认：均值0方差1

def standardize_inplace(df: pl.DataFrame, feat_cols: List[str], scaler) -> Tuple[pl.DataFrame, object]:
    X = df.select(feat_cols).to_numpy()
    scaler.fit(X)
    Xs = scaler.transform(X)
    df_s = df.with_columns([pl.Series(name=feat_cols[i], values=Xs[:, i]) for i in range(len(feat_cols))])
    return df_s, scaler

# def winsorize_iqr(df: pl.DataFrame, feat_cols: List[str], k: float) -> pl.DataFrame:
#     # 仅裁剪特征列
#     q1s = df.select([col(c).quantile(0.25).alias(f"{c}_q1") for c in feat_cols]).row(0)
#     q3s = df.select([col(c).quantile(0.75).alias(f"{c}_q3") for c in feat_cols]).row(0)
#     out = df
#     for i, c in enumerate(feat_cols):
#         q1, q3 = q1s[i], q3s[i]
#         if q1 is None or q3 is None: continue
#         iqr = q3 - q1; lo, hi = q1 - k*iqr, q3 + k*iqr
#         out = out.with_columns(when(col(c) < lo).then(lit(lo)).when(col(c) > hi).then(lit(hi)).otherwise(col(c)).alias(c))
#     return out

# =========================
# 切分 & 采样
# =========================
def time_split(df: pl.DataFrame, test_size: float, val_within: float) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = df.sort(TIME_COL)
    n = df.height
    te_start = int(n * (1 - test_size))      # 20% test
    trainval = df.slice(0, te_start)         # 80%
    test = df.slice(te_start)
    val_size = int(trainval.height * val_within)  # 80% * 0.375 = 30%
    train = trainval.slice(0, trainval.height - val_size)  # 50%
    val = trainval.slice(trainval.height - val_size)       # 30%
    return train, val, test

def class_balance_plot(df: pl.DataFrame, title: str, out_dir: str, prefix: str):
    counts = _gb(df, CLS_TARGET).agg(pl.len().alias("count")).sort(CLS_TARGET)
    counts.write_csv(os.path.join(out_dir, f"{prefix}class_balance_{title}.csv"))
    pdf = counts.to_pandas()
    plt.figure(figsize=(5,4))
    plt.bar(pdf[CLS_TARGET].astype(str), pdf["count"])
    plt.title(f"Class Balance - {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}class_balance_{title}.png"))
    plt.close()

def resample_to_2000_2000(train_df: pl.DataFrame, feat_cols: List[str], final_n: int, rs: int) -> pl.DataFrame:
    """先下采样多数类，再对少数类 SMOTE/或下采样，使最终正负各 final_n。"""
    X = train_df.select(feat_cols).to_numpy()
    y = train_df.select(CLS_TARGET).to_numpy().reshape(-1)

    # 先把多数类下到 final_n
    # 负类=0，正类=1（按你的定义）
    rng = np.random.default_rng(rs)
    neg_idx = np.where(y == 0)[0]
    pos_idx = np.where(y == 1)[0]

    if len(neg_idx) >= final_n:
        sel_neg = rng.choice(neg_idx, size=final_n, replace=False)
    else:
        # 负类不够 final_n，就全部保留（后续正类对齐到 available）
        sel_neg = neg_idx
        final_n = min(final_n, len(neg_idx))  # 统一最终目标
        print(f"[WARN] 负类不足 {final_n}，将以负类数量为准：{final_n}")

    X_r = np.concatenate([X[sel_neg], X[pos_idx]], axis=0)
    y_r = np.concatenate([np.zeros(len(sel_neg), dtype=y.dtype), np.ones(len(pos_idx), dtype=y.dtype)], axis=0)

    # 然后把正类调到 final_n：若多于 final_n → 下采样；若少于 final_n → SMOTE 上采样
    pos_mask = (y_r == 1)
    X_pos, X_neg = X_r[pos_mask], X_r[~pos_mask]
    P = X_pos.shape[0]
    N = X_neg.shape[0]

    if P > final_n:
        # 下采样正类
        sel = rng.choice(P, size=final_n, replace=False)
        X_pos2 = X_pos[sel]
    elif P < final_n:
        if HAS_IMBLEARN and P >= 2:
            # 用 SMOTE 合成到 final_n
            sm = SMOTE(sampling_strategy={1: final_n}, random_state=rs)
            X_tmp = np.vstack([X_pos, X_neg])           # 先合并再分标签映射
            y_tmp = np.concatenate([np.ones(P), np.zeros(N)]).astype(int)
            X_syn, y_syn = sm.fit_resample(X_tmp, y_tmp)
            # 取出合成后的正类 exact=final_n；负类保持 N
            X_pos2 = X_syn[y_syn == 1]
            if X_pos2.shape[0] != final_n:
                raise RuntimeError("SMOTE 未能达到目标正类数量")
        else:
            # 无 imblearn 或正样本太少：随机有放回采样到 final_n
            rep = final_n - P
            idx = rng.choice(P, size=rep, replace=True)
            X_pos2 = np.vstack([X_pos, X_pos[idx]])
    else:
        X_pos2 = X_pos

    # 负类确保为 final_n（若 N>final_n 已在开头截断；若 N<final_n 已前移 final_n）
    if N > final_n:
        seln = rng.choice(N, size=final_n, replace=False)
        X_neg2 = X_neg[seln]
    else:
        X_neg2 = X_neg

    X_b = np.vstack([X_neg2, X_pos2])
    y_b = np.concatenate([np.zeros(X_neg2.shape[0], dtype=np.int8),
                          np.ones(X_pos2.shape[0], dtype=np.int8)])

    # 回到 Polars
    feat_df = pl.DataFrame({feat_cols[i]: X_b[:, i] for i in range(len(feat_cols))})
    y_df = pl.DataFrame({CLS_TARGET: y_b})
    return pl.concat([feat_df, y_df], how="horizontal")

# =========================
# 各 Step
# =========================

def step1_visualize_raw(cfg: CFG) -> str:
    out_dir, fig_dir = ensure_dirs(cfg)
    lf = read_csv_lazy(cfg.input_csv)
    lf = basic_clean(lf, keep_months=cfg.keep_months)

    stage_path = os.path.join(out_dir, "step1_stage.parquet")
    lf.sink_parquet(stage_path, compression="zstd", statistics=True)

    df = pl.read_parquet(stage_path)
    df = encode_categoricals(df)

    feat_cols = infer_continuous_feature_cols(ALL_COLUMNS)
    plot_cols = select_plot_columns(feat_cols, k=cfg.plot_k)

    miss = compute_missing_report(df.select(ALL_COLUMNS))
    save_csv(miss, os.path.join(out_dir, "step1_missing_report.csv"))
    plot_missing_bar(miss, os.path.join(fig_dir, "step1_missing_rate.png"))

    desc = df.select(feat_cols).describe()
    save_csv(desc, os.path.join(out_dir, "step1_describe_features.csv"))

    sample = sample_df(df.select(plot_cols + [CLS_TARGET] + REG_TARGETS), cfg.sample_for_plots, cfg.random_state)
    plot_hists_and_boxes(sample, plot_cols, fig_dir, prefix="step1_")
    return stage_path

def step2_impute_and_standardize(cfg: CFG) -> str:
    """改动：Step2 同时做填补 + 标准化，并保存 scaler。"""
    out_dir, _ = ensure_dirs(cfg)
    in_path = os.path.join(out_dir, "step1_stage.parquet")
    df = pl.read_parquet(in_path)

    # 1) 连续特征（气象/环境）列
    cont_feats = infer_continuous_feature_cols(ALL_COLUMNS)

    # 2) 缺失值填补（仅连续特征）
    df2 = impute_missing(df, cont_feats, cfg.use_monthwise_impute)

    # 3) 加入月份 One-Hot（只针对 5-8 月，已在 Step1 过滤）
    df2, mon_cols = add_month_dummies(df2, months=cfg.keep_months, prefix="mon_")

    # 4) 标准化（仅连续特征；哑变量保持 0/1）
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    scaler = build_scaler(cfg.scaler)
    X = df2.select(cont_feats).to_numpy()
    scaler.fit(X)
    Xs = scaler.transform(X)
    df_scaled = df2.with_columns([pl.Series(name=cont_feats[i], values=Xs[:, i]) for i in range(len(cont_feats))])

    # 5) 写出
    out_path = os.path.join(out_dir, "step2_imputed_std.parquet")
    df_scaled.write_parquet(out_path, compression="zstd", statistics=True)
    joblib.dump(scaler, os.path.join(out_dir, f"scaler_step2_{cfg.scaler}.joblib"))

    # 6) 质量报告（含 mon_*）
    miss = compute_missing_report(df_scaled.select(ALL_COLUMNS + mon_cols))
    save_csv(miss, os.path.join(out_dir, "step2_missing_report.csv"))
    return out_path

def step4_visualize_clean(cfg: CFG) -> None:
    out_dir, fig_dir = ensure_dirs(cfg)
    print(f"[info] out_dir={Path(out_dir).resolve()}")
    print(f"[info] fig_dir={Path(fig_dir).resolve()}")
    in_path = os.path.join(out_dir, "step2_imputed_std.parquet")
    if not Path(in_path).exists():
        raise FileNotFoundError(f"未找到输入文件: {in_path}")

    df = pl.read_parquet(in_path)
    print(f"[info] df shape = {df.shape}")

    # 只选择连续特征用于可视化
    cont_feats = infer_continuous_feature_cols(ALL_COLUMNS)
    plot_cols = select_plot_columns(cont_feats, k=getattr(cfg, "plot_k", None))
    plot_cols = [c for c in plot_cols if c in df.columns]
    if not plot_cols:
        print("[warn] step4: 无可用的作图列（与 df.columns 无交集），已跳过绘图。")
        return
    print(f"[info] plot columns = {len(plot_cols)} -> {plot_cols[:10]}{'...' if len(plot_cols) > 10 else ''}")
    # 2) 缺失率（包含 mon_*）
    mon_cols = [c for c in df.columns if c.startswith("mon_")]
    miss_cols = [c for c in ALL_COLUMNS if c in df.columns] + mon_cols
    miss = compute_missing_report(df.select(miss_cols))
    # miss = compute_missing_report(df.select(ALL_COLUMNS + [c for c in df.columns if c.startswith("mon_")]))
    save_csv(miss, os.path.join(out_dir, "step4_missing_report.csv"))
    plot_missing_bar(miss, os.path.join(fig_dir, "step4_missing_rate.png"))
    # 3) 描述统计（只对存在的连续列）
    desc = df.select(cont_feats).describe()
    save_csv(desc, os.path.join(out_dir, "step4_describe_features.csv"))
    # 4) 采样（目标列缺失时不强行 select）
    target_cols = [c for c in [CLS_TARGET] + REG_TARGETS if c in df.columns]
    keep_cols = list(dict.fromkeys(plot_cols + target_cols))
    sample = sample_df(df.select(plot_cols + [CLS_TARGET] + REG_TARGETS), cfg.sample_for_plots, cfg.random_state)
    if sample.is_empty():
        print("[warn] step4: 采样为空，已跳过绘图。")
        return
    # 5) 逐列绘图并保存
    plot_hists_and_boxes(sample, plot_cols, fig_dir, prefix="step4_")
    print("[OK] step4 完成。图像输出目录：", Path(fig_dir).resolve())


def step5_split(cfg: CFG) -> Tuple[str, str, str, str, str, str]:
    """改动：仅做时间切分；不再标准化。"""
    out_dir, _ = ensure_dirs(cfg)
    in_path = os.path.join(out_dir, "step2_imputed_std.parquet")
    df = pl.read_parquet(in_path)

    # 最终特征 = 连续特征 + mon_*
    feat_cols = list_final_feature_cols(df)
    train, val, test = time_split(df, cfg.test_size, cfg.val_size_within_train)

        # 分类集（不再带 _std）
    cls_cols = feat_cols + [CLS_TARGET]
    train_cls = train.select(cls_cols)
    val_cls = val.select(cls_cols)
    test_cls = test.select(cls_cols)

    train_cls_path = os.path.join(out_dir, "step5_cls_train.parquet")
    val_cls_path = os.path.join(out_dir, "step5_cls_val.parquet")
    test_cls_path = os.path.join(out_dir, "step5_cls_test.parquet")
    train_cls.write_parquet(train_cls_path, compression="zstd", statistics=True)
    val_cls.write_parquet(val_cls_path, compression="zstd", statistics=True)
    test_cls.write_parquet(test_cls_path, compression="zstd", statistics=True)

    # 回归集
    reg_cols = feat_cols + REG_TARGETS
    train_reg = train.select(reg_cols)
    val_reg = val.select(reg_cols)
    test_reg = test.select(reg_cols)

    train_reg_path = os.path.join(out_dir, "step5_reg_train.parquet")
    val_reg_path = os.path.join(out_dir, "step5_reg_val.parquet")
    test_reg_path = os.path.join(out_dir, "step5_reg_test.parquet")
    train_reg.write_parquet(train_reg_path, compression="zstd", statistics=True)
    val_reg.write_parquet(val_reg_path, compression="zstd", statistics=True)
    test_reg.write_parquet(test_reg_path, compression="zstd", statistics=True)

    class_balance_plot(train_cls, "train_before_resample", out_dir, prefix="step5_")
    class_balance_plot(val_cls, "val", out_dir, prefix="step5_")
    class_balance_plot(test_cls, "test", out_dir, prefix="step5_")

    return (train_cls_path, val_cls_path, test_cls_path, train_reg_path, val_reg_path, test_reg_path)


def step6_resample(cfg: CFG) -> str:
    out_dir, _ = ensure_dirs(cfg)
    train_cls_path = os.path.join(out_dir, "step5_cls_train.parquet")
    train_cls = pl.read_parquet(train_cls_path)
    feat_cols = list_final_feature_cols(train_cls)  # ★ 包含 mon_*
    train_bal = resample_to_2000_2000(train_cls, feat_cols, cfg.final_samples, cfg.random_state)


    out_path = os.path.join(out_dir, "step6_cls_train_balanced.parquet")
    train_bal.write_parquet(out_path, compression="zstd", statistics=True)
    class_balance_plot(train_bal, "train_after_resample", out_dir, prefix="step6_")
    return out_path

def step7_meta(cfg: CFG) -> str:
    out_dir, _ = ensure_dirs(cfg)
    df2 = pl.read_parquet(os.path.join(out_dir, "step2_imputed_std.parquet"))
    feat_cols = list_final_feature_cols(df2)  # ★
    meta = {
        "config": asdict(cfg),
        "classification_target": CLS_TARGET,
        "regression_targets": REG_TARGETS,
        "feature_columns": feat_cols,  # ★ 连续+mon_*
        "excluded_from_features": EXCLUDE_FROM_FEATURES,
        "id_columns": ID_COLS,
        "time_column": TIME_COL,
        "artifacts": {
            "step1_stage": os.path.join(out_dir, "step1_stage.parquet"),
            "step2_imputed_std": os.path.join(out_dir, "step2_imputed_std.parquet"),
            "scaler": os.path.join(out_dir, f"scaler_step2_{cfg.scaler}.joblib"),
            "cls_train": os.path.join(out_dir, "step5_cls_train.parquet"),
            "cls_val": os.path.join(out_dir, "step5_cls_val.parquet"),
            "cls_test": os.path.join(out_dir, "step5_cls_test.parquet"),
            "reg_train": os.path.join(out_dir, "step5_reg_train.parquet"),
            "reg_val": os.path.join(out_dir, "step5_reg_val.parquet"),
            "reg_test": os.path.join(out_dir, "step5_reg_test.parquet"),
            "cls_train_balanced": os.path.join(out_dir, "step6_cls_train_balanced.parquet"),
            "figures_dir": os.path.join(out_dir, cfg.fig_dir),
        }
    }
    meta_path = os.path.join(out_dir, "step7_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


# =========================
# CLI
# =========================

def run_to_step(cfg: CFG, step: int):
    assert 1 <= step <= 7, "step 必须在 1..7"
    print("==> Step 1"); step1_visualize_raw(cfg)
    if step == 1: return
    print("==> Step 2"); step2_impute_and_standardize(cfg)  # 改：Step2 包含标准化
    if step == 2: return
    print("==> Step 4"); step4_visualize_clean(cfg)
    if step == 4: return
    print("==> Step 5"); step5_split(cfg)                   # 改：仅切分
    if step == 5: return
    print("==> Step 6"); step6_resample(cfg)
    if step == 6: return
    print("==> Step 7"); step7_meta(cfg)

def parse_args():
    ap = argparse.ArgumentParser(description="Stepwise preprocessing pipeline (NO winsorize)")
    ap.add_argument("--input", default="/home/dell/Daxinganling2025/output/model_monthly_5km.csv", help="输入CSV路径")
    ap.add_argument("--out", default="/home/dell/Daxinganling2025/output/Redata/step7_1112_1", help="输出目录")
    ap.add_argument("--step", type=int, default=7, help="运行到第几步(1..7)")
    ap.add_argument("--sample", type=int, default=200_000, help="绘图抽样行数")
    ap.add_argument("--monthwise_impute", action="store_true", help="Step2：按月中位数填补缺失")
    ap.add_argument("--scaler", type=str, choices=["standard", "minmax", "robust"], default="minmax",
                    help="Step2 标准化方法：standard=均值0方差1；minmax=0-1；robust=分位稳健")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_within", type=float, default=0.375)    # 50/30/20
    ap.add_argument("--rs", type=int, default=42)
    # ap.add_argument("--undersample_ratio", type=float, default=20.0)
    # ap.add_argument("--smote_ratio", type=float, default=1.0)
    ap.add_argument("--final_samples", type=int, default=2000, help="Step6 最终正负样本数量（各）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = CFG(
        input_csv=args.input,
        out_dir=args.out,
        sample_for_plots=args.sample,
        random_state=args.rs,
        use_monthwise_impute=args.monthwise_impute,
        test_size=args.test_size,
        val_size_within_train=args.val_within,
        plot_k=28,
        scaler=args.scaler,
        keep_months=(5, 6, 7, 8),
        final_samples=args.final_samples,
    )
    run_to_step(cfg, args.step)