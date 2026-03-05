# -*- coding: utf-8 -*-
"""
XGBoost + 双校准（Isotonic/Sigmoid） + “分层背景”（quantile 灰阶 + 可选正/负密度条）
只产出：
- Fig.7A  SHAP bar
- Fig.7B  SHAP summary
- Fig.8A  10特征 1D-PD + 阈值bootstrap（各自一张，带分层背景）
- Fig.8B  2D-PD（全部 45 组；如需只要 6 组，把 PAIRS_2D 改回固定列表即可）

不再绘制 ROC/PR/Top-k/校准等其它图。
"""

import os, json, warnings, argparse, time, joblib
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

plt.rcParams.update({
    "axes.grid": True, "grid.color": "#E5E7EB",
    "font.size": 11, "savefig.dpi": 180, "legend.frameon": False
})

# ------------------ 基础工具 ------------------
def load_meta(meta_path: str) -> Dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_pq(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pick_features(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    miss = [c for c in feats if c not in df.columns]
    if miss:
        raise ValueError(f"缺少特征列: {miss[:10]}{'...' if len(miss)>10 else ''}")
    X = df[feats].copy()
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X

def prob_of(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)

def make_calibrator(fitted_model):
    try:
        return CalibratedClassifierCV(estimator=fitted_model, method="isotonic", cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=fitted_model, method="isotonic", cv="prefit")

def make_calibrator_sigmoid(fitted_model):
    try:
        return CalibratedClassifierCV(estimator=fitted_model, method="sigmoid", cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=fitted_model, method="sigmoid", cv="prefit")

# ====== 内存安全版 1D / 2D PD ======
def batch_predict_mean(model, X_base: pd.DataFrame, feat: str, grid: np.ndarray,
                        batch_rows: int = 200_000) -> np.ndarray:
    n = len(X_base)
    out = np.empty(len(grid), dtype=np.float64)
    for gi, g in enumerate(grid):
        s = 0; acc = 0.0; cnt = 0
        while s < n:
            e = min(s + batch_rows, n)
            Xb = X_base.iloc[s:e].copy()
            Xb[feat] = g
            p = prob_of(model, Xb)
            acc += float(p.sum())
            cnt += (e - s)
            s = e
        out[gi] = acc / max(1, cnt)
    return out

def batch_predict_mean_2d(model, X_base: pd.DataFrame, f1: str, f2: str,
                          g1: np.ndarray, g2: np.ndarray, batch_rows: int = 200_000) -> np.ndarray:
    n1, n2 = len(g1), len(g2)
    Z = np.zeros((n2, n1), dtype=np.float64)
    for j, a in enumerate(g1):
        for i, b in enumerate(g2):
            s = 0; acc = 0.0; cnt = 0
            while s < len(X_base):
                e = min(s + batch_rows, len(X_base))
                Xb = X_base.iloc[s:e].copy()
                Xb[f1] = a; Xb[f2] = b
                p = prob_of(model, Xb)
                acc += float(p.sum()); cnt += (e - s); s = e
            Z[i, j] = acc / max(1, cnt)
    return Z

def _threshold_from_pd(xv: np.ndarray, yv: np.ndarray) -> Tuple[float, float]:
    if len(xv) < 3:
        return float(np.median(xv)), float(np.median(yv))
    dy = np.gradient(yv, xv)
    i = int(np.argmax(np.abs(dy)))
    return float(xv[i]), float(yv[i])

def bootstrap_pd_thresholds(model, X_base: pd.DataFrame, feat: str,
                            q_low=0.01, q_high=0.99, n_grid=40,
                            boot_n=200, boot_frac=0.8, seed=42) -> Dict:
    rng = np.random.RandomState(seed)
    base_vals = X_base[feat].values
    qs = np.linspace(q_low, q_high, n_grid)
    grid = np.quantile(base_vals, qs)
    y_mean = batch_predict_mean(model, X_base, feat, grid)
    x_thr, y_thr = _threshold_from_pd(grid, y_mean)

    boots = []
    idx_all = np.arange(len(X_base))
    m = max(10, int(len(idx_all) * boot_frac))
    for _ in range(boot_n):
        idx = rng.choice(idx_all, size=m, replace=True)
        Xb = X_base.iloc[idx].copy()
        yb = batch_predict_mean(model, Xb, feat, grid)
        xb, _ = _threshold_from_pd(grid, yb)
        boots.append(xb)
    boots = np.array(boots, dtype=float)

    q_pos = float((base_vals <= x_thr).mean())
    q_ci  = np.quantile([(base_vals <= v).mean() for v in boots], [0.025, 0.975])

    return {
        "grid": grid, "pd": y_mean,
        "thr_x": x_thr, "thr_y": y_thr,
        "thr_q": q_pos, "thr_q_lo": float(q_ci[0]), "thr_q_hi": float(q_ci[1]),
        "boot_x": boots
    }

# ====== 新增：1D PD “分层背景” ======
def _stratified_background(ax, x_all: np.ndarray, y_all: np.ndarray = None,
                           bins: int = 20, mode: str = "quantile+posneg",
                           base_alpha: float = 0.08):
    """
    在 ax 上绘制 1D 特征分布的“分层背景”：
    - mode = "quantile"            → 仅按分位分箱灰阶背景
    - mode = "quantile+posneg"     → 在 quantile 背景上叠加正/负样本的条形密度（半透明）
    """
    x = np.asarray(x_all, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    # 1) 分位分箱灰阶背景
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x)) + 1e-12
        edges = np.linspace(lo, hi, num=min(bins, 5))

    counts, _ = np.histogram(x, bins=edges)
    dens = counts / (counts.max() + 1e-12)
    for i in range(len(edges) - 1):
        depth = base_alpha + 0.25 * float(dens[i])
        ax.axvspan(edges[i], edges[i + 1], color="#9CA3AF", alpha=depth, lw=0, zorder=0)

    # 2) 正/负密度叠加（可选）
    if mode.endswith("+posneg") and y_all is not None:
        y = np.asarray(y_all).astype(int)
        pos = x_all[y_all == 1]
        neg = x_all[y_all == 0]
        c_pos, _ = np.histogram(pos, bins=edges)
        c_neg, _ = np.histogram(neg, bins=edges)
        dn = c_neg / (c_neg.max() + 1e-12)
        dp = c_pos / (c_pos.max() + 1e-12)
        width = np.diff(edges)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, dn, width=width*0.95, color="#1f77b4", alpha=0.20,
               align="center", edgecolor="none", zorder=1)
        ax.bar(centers, dp, width=width*0.95, color="#D1495B", alpha=0.20,
               align="center", edgecolor="none", zorder=2)

# ------------------ 主流程 ------------------
def main():
    t0 = time.time()
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="XGBoost + 校准 + Fig.9/10（定制输出）")

    # 数据与输出
    ap.add_argument("--meta", default="/home/dell/Daxinganling2025/output/Redata/step7_1112_1/step7_meta.json")
    ap.add_argument("--out",  default="/home/dell/Daxinganling2025/output/Figs9")

    # 特征（10 个）
    ap.add_argument("--features", default="avg_ie,t2m,ssr,sshf,SPEI_1,swvl1,K_index,VHI,lai_hv,si10")

    # xgb参数
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--min_child_weight", type=float, default=8.0)
    ap.add_argument("--reg_lambda", type=float, default=12.0)
    ap.add_argument("--learning_rate", type=float, default=0.04)
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--max_delta_step", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # PD/SHAP 控制
    ap.add_argument("--shap_max", type=int, default=8000)
    ap.add_argument("--pd_sample", type=int, default=120000)
    ap.add_argument("--pd_grid1d", type=int, default=40)
    ap.add_argument("--pd_grid2d", type=int, default=25)
    ap.add_argument("--thr_boot_n", type=int, default=200)
    ap.add_argument("--thr_boot_frac", type=float, default=0.8)

    # 背景样式（可选：quantile / quantile+posneg）
    ap.add_argument("--pd_bg_mode", default="quantile+posneg")
    ap.add_argument("--pd_bg_bins", type=int, default=20)

    args = ap.parse_args()
    ensure_dir(args.out)

    FEATURES = [c for c in args.features.split(",") if c.strip()]
    TARGET   = "fire_presence"

    # === 读数据 ===
    meta = load_meta(args.meta)
    arts = meta.get("artifacts", {})
    tr_p = arts.get("cls_train_balanced") or arts.get("cls_train") or arts.get("cls_train_std")
    vl_p  = arts.get("cls_val")   or arts.get("cls_val_std")
    if not (tr_p and vl_p):
        raise FileNotFoundError("step7_meta.json 缺少 cls_train/cls_val")

    df_tr = read_pq(tr_p); df_vl = read_pq(vl_p)

    for df in (df_tr, df_vl):
        if "valid_time" in df.columns and not np.issubdtype(df["valid_time"].dtype, np.datetime64):
            df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")

    X_tr = pick_features(df_tr, FEATURES); y_tr = df_tr[TARGET].astype("int8").to_numpy()
    X_vl = pick_features(df_vl, FEATURES); y_vl = df_vl[TARGET].astype("int8").to_numpy()

    pos = int(y_tr.sum()); neg = int(len(y_tr)-pos)
    spw = float(neg / max(1, pos))
    print(f"[WEIGHT] scale_pos_weight={spw:.1f} (neg={neg}, pos={pos})")

    # === 训练 ===
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        max_depth=args.max_depth, min_child_weight=args.min_child_weight, reg_lambda=args.reg_lambda,
        learning_rate=args.learning_rate, n_estimators=args.n_estimators,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        max_delta_step=args.max_delta_step, random_state=args.seed, n_jobs=-1,
        objective="binary:logistic", tree_method="hist", max_bin=256, verbosity=0,
        scale_pos_weight=spw
    )
    try:
        xgb.set_params(eval_metric="aucpr")
    except Exception:
        pass
    try:
        xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], early_stopping_rounds=100, verbose=False)
    except TypeError:
        xgb.fit(X_tr, y_tr)

    cal_iso = make_calibrator(xgb);  cal_iso.fit(X_vl, y_vl)
    cal_sig = make_calibrator_sigmoid(xgb); cal_sig.fit(X_vl, y_vl)

    # 保存基础指标（不画图）
    pv = prob_of(cal_iso, X_vl)
    metrics = {
        "AUC": float(roc_auc_score(y_vl, pv)),
        "PR_AUC": float(average_precision_score(y_vl, pv)),
        "Brier": float(brier_score_loss(y_vl, pv))
    }
    with open(os.path.join(args.out, "val_metrics_isotonic.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 保存模型
    joblib.dump(cal_iso, os.path.join(args.out, "model_xgb_isotonic.joblib"))
    joblib.dump(cal_sig, os.path.join(args.out, "model_xgb_sigmoid.joblib"))

    # ================= Fig.7A/7B：SHAP =================
    shap_order = FEATURES[:]
    try:
        import shap
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(X_tr), min(args.shap_max, len(X_tr)), replace=False) \
            if len(X_tr) > args.shap_max else np.arange(len(X_tr))
        shap_sample = X_tr.iloc[idx]

        expl = shap.TreeExplainer(xgb)
        shap_vals = expl.shap_values(shap_sample.values)

        abs_mean = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(abs_mean)[::-1]
        topk = min(10, len(FEATURES))
        top_feats = [FEATURES[i] for i in order[:topk]]
        top_vals = abs_mean[order[:topk]]
        shap_order = top_feats

        # Fig.7A bar
        plt.figure(figsize=(7, max(4, 0.45 * topk)))
        plt.barh(np.arange(topk)[::-1], top_vals[::-1])
        plt.yticks(np.arange(topk)[::-1], top_feats[::-1])
        plt.xlabel("mean(|SHAP|)")
        plt.title("Fig.7A — SHAP global importance")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig7A_shap_bar.png"), dpi=300)
        plt.close()

        # Fig.7B summary
        plt.figure(figsize=(8, max(4, 0.6 * len(FEATURES))))
        shap.summary_plot(shap_vals, shap_sample, feature_names=FEATURES, show=False)
        plt.title("Fig.7B — SHAP summary")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig7B_shap_summary.png"), dpi=300)
        plt.close()

        pd.DataFrame({"feature": FEATURES, "mean_abs_shap": abs_mean}) \
          .sort_values("mean_abs_shap", ascending=False) \
          .to_csv(os.path.join(args.out, "shap_importance.csv"), index=False)

    except Exception as e:
        print("[WARN] SHAP failed:", e)

    # ================= Fig.8A：1D PD + 阈值 bootstrap（全10特征，带分层背景） =================
    rng = np.random.RandomState(args.seed)
    base_idx = rng.choice(len(X_vl), min(args.pd_sample, len(X_vl)), replace=False) \
        if len(X_vl) > args.pd_sample else np.arange(len(X_vl))
    X_pd_base = X_vl.iloc[base_idx].copy()
    y_pd_base = y_vl[base_idx]  # 用于分层背景中的正/负密度

    thr_rows = []
    for f in FEATURES:
        try:
            out = bootstrap_pd_thresholds(
                model=cal_iso, X_base=X_pd_base, feat=f,
                q_low=0.01, q_high=0.99, n_grid=args.pd_grid1d,
                boot_n=args.thr_boot_n, boot_frac=args.thr_boot_frac, seed=args.seed
            )
            xv, yv = out["grid"], out["pd"]
            thr_x, thr_y = out["thr_x"], out["thr_y"]
            x_lo = float(np.quantile(out["boot_x"], 0.025))
            x_hi = float(np.quantile(out["boot_x"], 0.975))

            # —— 带“分层背景”的单图 —— #
            fig, ax = plt.subplots(figsize=(6,4))
            _stratified_background(
                ax,
                X_pd_base[f].values, y_pd_base,
                bins=args.pd_bg_bins,
                mode=args.pd_bg_mode,      # "quantile" 或 "quantile+posneg"
                base_alpha=0.08
            )
            # 叠加 PD 主曲线 + 阈值点与区间
            ax.plot(xv, yv, lw=2.0, color="black", label="PD")
            ax.scatter([thr_x], [thr_y], s=30, color="black", zorder=3)
            ax.axvspan(x_lo, x_hi, color="#D1495B", alpha=0.20, lw=0, zorder=1)

            ax.set_xlabel(f); ax.set_ylabel("Partial dependence (P)")
            ax.set_title(f"Fig.8A — PD (+stratified bg): {f}")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"fig8A_pd_{f}.png"), dpi=300)
            plt.close(fig)

            # 导出 CSV（曲线 + 阈值）
            pd.DataFrame({f: xv, "pd": yv}).to_csv(os.path.join(args.out, f"pd_{f}.csv"), index=False)
            pd.DataFrame({
                "feature":[f],
                "thr_unit":[thr_x], "thr_unit_lo":[x_lo], "thr_unit_hi":[x_hi],
                "thr_quantile":[out["thr_q"]], "thr_q_lo":[out["thr_q_lo"]], "thr_q_hi":[out["thr_q_hi"]],
                "pd_at_thr":[thr_y]
            }).to_csv(os.path.join(args.out, f"pd_{f}_thresholds.csv"), index=False)

            thr_rows.append([f, thr_x, x_lo, x_hi, out["thr_q"], out["thr_q_lo"], out["thr_q_hi"], thr_y])
        except Exception as e:
            print(f"[WARN][PD1D] {f} failed:", e)

    if thr_rows:
        pd.DataFrame(thr_rows, columns=[
            "feature","thr_unit","thr_unit_lo","thr_unit_hi",
            "thr_quantile","thr_q_lo","thr_q_hi","pd_at_thr"
        ]).to_csv(os.path.join(args.out, "fig8A_pd_threshold_candidates.csv"), index=False)

    # ================= Fig.8B：2D PD（全部 45 组；如需只要 6 组，改回固定列表） =================
    from itertools import combinations
    PAIRS_2D = [(a, b) for a, b in combinations(FEATURES, 2)]

    for (f1, f2) in PAIRS_2D:
        if f1 not in FEATURES or f2 not in FEATURES:
            print(f"[SKIP] pair {f1}×{f2} 不在 FEATURES 中")
            continue
        try:
            q = np.linspace(0.05, 0.95, args.pd_grid2d)
            g1 = np.quantile(X_pd_base[f1].values, q)
            g2 = np.quantile(X_pd_base[f2].values, q)
            Z = batch_predict_mean_2d(cal_iso, X_pd_base, f1, f2, g1, g2, batch_rows=200000)

            plt.figure(figsize=(6,5))
            cs = plt.contourf(g1, g2, Z, levels=12)
            plt.colorbar(cs)
            plt.xlabel(f1); plt.ylabel(f2)
            plt.title(f"Fig.8B — 2D PD: {f1} × {f2}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f"fig8B_pd2d_{f1}_x_{f2}.png"), dpi=300)
            plt.close()

            np.savez_compressed(os.path.join(args.out, f"pd2d_{f1}_x_{f2}.npz"), g1=g1, g2=g2, Z=Z)
        except Exception as e:
            print(f"[WARN][PD2D] {f1}×{f2} failed:", e)

    print("[OK] Outputs saved to", args.out, f"(elapsed {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
