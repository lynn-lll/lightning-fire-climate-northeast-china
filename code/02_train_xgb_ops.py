# -*- coding: utf-8 -*-
"""
Train XGBoost (10气象特征) + 双校准（Isotonic/Platt） + Top-k/分位运营阈值 + 分组Top-k + 可选小网格选型
- 读取 step7_meta.json -> cls_train/val/test
- 未下采样训练，设置 scale_pos_weight=neg/pos；验证集 aucpr 早停
- 两种校准：isotonic / sigmoid；输出两套结果与 top-k 表，最终选择isotonic结果更好
- 评估：ROC/PR/Calibration、Precision@k/Recall@k、概率直方图&CDF
- 输出：模型、metrics.json、pred_val/test.csv、topk_global_*.csv、topk_grouped_*.csv、ops_thresholds.json
"""

import os, json, warnings, argparse, joblib
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

plt.rcParams.update({
    "axes.grid": True, "grid.color": "#E5E7EB",
    "font.size": 11, "savefig.dpi": 180, "legend.frameon": False
})

# —— 运营阈值默认策略（可被运行时再次覆盖/重写）——
DEFAULT_POLICY = {
    "primary":  {"mode": "quantile", "q": 0.999,  "backup_threshold": 0.001613879},
    "fallback": {"mode": "topk",     "k": 5000,   "note": "如需固定条数"}
}

# ========= 基础工具 =========
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

def plot_curves(y_true, y_prob, tag: str, out_dir: str) -> Dict:
    ensure_dir(out_dir)
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    # ROC
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {tag}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{tag}.png")); plt.close()

    # PR
    plt.figure(); plt.plot(rec, prec, label=f"PR-AUC={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {tag}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{tag}.png")); plt.close()

    # Calibration（分位分箱）
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=20, strategy="quantile")
        plt.figure(); plt.plot(mean_pred, frac_pos, "o-")
        plt.plot([0,1],[0,1],"--"); plt.xlabel("Predicted"); plt.ylabel("Observed")
        plt.title(f"Calibration — {tag}"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"calib_{tag}.png")); plt.close()
    except Exception:
        pass

    # Prob hist
    plt.figure(); plt.hist(y_prob, bins=50)
    plt.title(f"{tag} prob histogram"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"prob_hist_{tag}.png")); plt.close()

    # Prob hist (log-x)
    xp = np.clip(y_prob, 1e-12, None)
    plt.figure(); plt.hist(xp, bins=50); plt.xscale("log")
    plt.title(f"{tag} prob histogram (log-x)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"prob_hist_log_{tag}.png")); plt.close()

    # CDF (log-x)
    xs = np.sort(xp); cdf = np.arange(1, len(xs)+1)/len(xs)
    plt.figure(); plt.plot(xs, cdf); plt.xscale("log")
    plt.xlabel("Probability (log)"); plt.ylabel("CDF"); plt.title(f"{tag} prob CDF")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"prob_cdf_{tag}.png")); plt.close()

    return {"AUC": float(auc), "PR_AUC": float(ap), "Brier": float(bs)}

def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, ks: List[int]) -> Tuple[List[float], List[float]]:
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    precs, recs = [], []
    cum_tp = np.cumsum(y_sorted)
    total_pos = max(1, int(y_true.sum()))
    n = len(y_true)
    for k in ks:
        kk = min(k, n)
        tp = int(cum_tp[kk-1]) if kk > 0 else 0
        precs.append(tp / max(kk, 1))
        recs.append(tp / total_pos)
    return precs, recs

def plot_pk_rk(y_true, y_prob, ks: List[int], tag: str, out_dir: str):
    precs, recs = precision_recall_at_k(y_true, y_prob, ks)
    xs = np.array(ks, dtype=float)

    plt.figure()
    plt.plot(xs, precs, marker="o")
    plt.xscale("log"); plt.xlabel("k (log)"); plt.ylabel("Precision@k"); plt.title(f"Precision@k — {tag}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"precision_at_k_{tag}.png")); plt.close()

    plt.figure()
    plt.plot(xs, recs, marker="o")
    plt.xscale("log"); plt.xlabel("k (log)"); plt.ylabel("Recall@k"); plt.title(f"Recall@k — {tag}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"recall_at_k_{tag}.png")); plt.close()

def topk_table_global(y_true: np.ndarray, y_prob: np.ndarray, ks: List[int], qs: List[float], split: str) -> pd.DataFrame:
    n = len(y_true); order = np.argsort(-y_prob); base_rows = []
    total_pos = max(1, int(y_true.sum()))

    # by k
    for k in ks:
        kk = min(int(k), n)
        idx = order[:kk]
        tp = int(y_true[idx].sum()); fp = int(kk - tp)
        thr = float(y_prob[idx[-1]]) if kk>0 else 1.0
        q   = float((y_prob <= thr).mean())
        base_rows.append(["k", split, k, tp/max(kk,1), tp/total_pos, tp, fp, int(total_pos-tp), thr, q])

    # by quantile
    s = np.sort(y_prob)
    for q in qs:
        thr = float(np.quantile(s, q))
        idx = np.where(y_prob >= thr)[0]
        tp = int(y_true[idx].sum()); fp = int(len(idx) - tp)
        base_rows.append(["quantile", split, q, tp/max(len(idx),1), tp/total_pos, tp, fp, int(total_pos-tp), thr, float(q)])

    return pd.DataFrame(base_rows, columns=["type","split","k_or_q","precision","recall","tp","fp","fn","threshold","quantile"])

def topk_table_grouped(df: pd.DataFrame,
                       prob_col: str, label_col: str,
                       groupby: List[str], k_per_group_list: List[int], split: str) -> pd.DataFrame:
    """对每个 group 取前 k_per_group，然后汇总评估；返回每个 k_per_group 的整体指标。"""
    rows = []
    if not groupby:
        return pd.DataFrame(columns=["split","groupby","k_per_group","alerts","tp","fp","fn","precision","recall"])
    gsize = df.groupby(groupby, dropna=False)[label_col].size().rename("n")

    for kpg in k_per_group_list:
        alerts_idx = []
        for g, sub in df.groupby(groupby, dropna=False):
            sub = sub.sort_values(prob_col, ascending=False)
            take = sub.head(max(0, int(kpg)))
            alerts_idx.append(take.index.values)
        if alerts_idx:
            sel = np.concatenate(alerts_idx)
        else:
            sel = np.array([], dtype=int)

        tp = int(df.loc[sel, label_col].sum())
        fp = int(len(sel) - tp)
        fn = int(df[label_col].sum()) - tp
        prec = tp / max(len(sel), 1)
        rec  = tp / max(int(df[label_col].sum()), 1)
        rows.append([split, "+".join(groupby), int(kpg), int(len(sel)), tp, fp, fn, float(prec), float(rec)])

    return pd.DataFrame(rows, columns=["split","groupby","k_per_group","alerts","tp","fp","fn","precision","recall"])

def make_calibrator(fitted_model):
    """兼容不同 sklearn 版本：CalibratedClassifierCV(estimator|base_estimator=, cv='prefit')"""
    try:
        return CalibratedClassifierCV(estimator=fitted_model, method="isotonic", cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=fitted_model, method="isotonic", cv="prefit")

def make_calibrator_sigmoid(fitted_model):
    try:
        return CalibratedClassifierCV(estimator=fitted_model, method="sigmoid", cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=fitted_model, method="sigmoid", cv="prefit")

# ========= 主流程 =========
def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="XGBoost + 双校准 + Top-k/分位阈值 + 分组Top-k")
    ap.add_argument("--meta", default="/home/dell/Daxinganling2025/output/Redata/step7_1112_1/step7_meta.json")
    ap.add_argument("--out",  default="/home/dell/Daxinganling2025/output/xgb_ops")
    # 只用 10 个气象特征
    ap.add_argument("--features", default="avg_ie,t2m,ssr,sshf,SPEI_1,swvl1,K_index,VHI,lai_hv,si10")
    # xgb 基础参数
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--min_child_weight", type=float, default=8.0)
    ap.add_argument("--reg_lambda", type=float, default=12.0)
    ap.add_argument("--learning_rate", type=float, default=0.04)
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--max_delta_step", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # 选择指标主要看 Precision@k（设一个锚点）
    ap.add_argument("--select_k", type=int, default=1000)
    ap.add_argument("--grid", action="store_true", help="开启小网格搜索（仅少量组合）")

    # Top-k & 分位
    ap.add_argument("--topk_list",  default="50,100,200,500,1000,2000,5000,10000,20000")
    ap.add_argument("--quantiles",  default="0.999,0.9995,0.9999,0.99995")
    # 分组 Top-k
    ap.add_argument("--groupby",    default="date,grid_id")  # 可选："date" 或 "date,grid_id" 或 ""
    ap.add_argument("--group_k",    default="5,10,20")

    # 目标列
    ap.add_argument("--target",     default="fire_presence")

    args = ap.parse_args()
    ensure_dir(args.out)

    FEATURES = [c for c in args.features.split(",") if c.strip()]
    TARGET   = args.target

    # ===== 读 meta & 数据 =====
    meta = load_meta(args.meta)
    arts = meta.get("artifacts", {})
    tr_p  = arts.get("cls_train") or arts.get("cls_train_std")
    vl_p  = arts.get("cls_val")   or arts.get("cls_val_std")
    te_p  = arts.get("cls_test")  or arts.get("cls_test_std")
    if not (tr_p and vl_p and te_p):
        raise FileNotFoundError("step7_meta.json 中未找到 cls_train/cls_val/cls_test 路径")

    df_tr = read_pq(tr_p)
    df_vl = read_pq(vl_p)
    df_te = read_pq(te_p)

    # 便于分组的时间列
    for df in (df_tr, df_vl, df_te):
        if "valid_time" in df.columns and not np.issubdtype(df["valid_time"].dtype, np.datetime64):
            df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
        if "valid_time" in df.columns:
            df["date"] = df["valid_time"].dt.date.astype("str")

    X_tr = pick_features(df_tr, FEATURES); y_tr = df_tr[TARGET].astype("int8").to_numpy()
    X_vl = pick_features(df_vl, FEATURES); y_vl = df_vl[TARGET].astype("int8").to_numpy()
    X_te = pick_features(df_te, FEATURES); y_te = df_te[TARGET].astype("int8").to_numpy()

    pos = int(y_tr.sum()); neg = int(len(y_tr)-pos)
    spw = float(neg / max(1, pos))
    print(f"[WEIGHT] scale_pos_weight={spw:.1f} (neg={neg}, pos={pos})")

    # ===== 训练（可选小网格，只看 Precision@k）=====
    from xgboost import XGBClassifier

    def make_xgb(md, mcw, rl, lr):
        model = XGBClassifier(
            max_depth=md,
            min_child_weight=mcw,
            reg_lambda=rl,
            learning_rate=lr,
            n_estimators=args.n_estimators,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            max_delta_step=args.max_delta_step,
            random_state=args.seed,
            n_jobs=-1,
            objective="binary:logistic",
            tree_method="hist",
            max_bin=256,
            verbosity=0,
            scale_pos_weight=spw
        )
        model.set_params(eval_metric="aucpr")
        return model

    grid_params = [(args.max_depth, args.min_child_weight, args.reg_lambda, args.learning_rate)]
    if args.grid:
        grid_params = []
        for md in [3, 4]:
            for mcw in [8.0, 12.0]:
                for rl in [10.0, 20.0]:
                    for lr in [0.03, 0.05]:
                        grid_params.append((md, mcw, rl, lr))

    best = None
    anchor_k = int(args.select_k)

    for (md, mcw, rl, lr) in grid_params:
        xgb = make_xgb(md, mcw, rl, lr)
        try:
            xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], early_stopping_rounds=100, verbose=False)
        except TypeError:
            xgb.fit(X_tr, y_tr)

        # 用两种校准都算一下在 val 的 Precision@k，取较优的那一个当“此参数的分数”
        iso = make_calibrator(xgb); iso.fit(X_vl, y_vl)
        sig = make_calibrator_sigmoid(xgb); sig.fit(X_vl, y_vl)
        pv_i = prob_of(iso, X_vl); pv_s = prob_of(sig, X_vl)

        prec_i, _ = precision_recall_at_k(y_vl, pv_i, [anchor_k])
        prec_s, _ = precision_recall_at_k(y_vl, pv_s, [anchor_k])
        score = max(float(prec_i[0]), float(prec_s[0]))
        if (best is None) or (score > best["score"]):
            best = dict(md=md, mcw=mcw, rl=rl, lr=lr, model=xgb, score=score,
                        iso=iso, sig=sig, pv_i=pv_i, pv_s=pv_s)

    xgb = best["model"]
    cal_iso = best["iso"]
    cal_sig = best["sig"]
    print(f"[SELECT] md={best['md']} mcw={best['mcw']} rl={best['rl']} lr={best['lr']}  Precision@{anchor_k}≈{best['score']:.4f} (val, best of iso/sig)")

    # ===== 评估两种校准：val/test =====
    for tag, cal, X, y in [
        ("val_iso",  cal_iso, X_vl, y_vl),
        ("test_iso", cal_iso, X_te, y_te),
        ("val_sig",  cal_sig, X_vl, y_vl),
        ("test_sig", cal_sig, X_te, y_te),
    ]:
        p = prob_of(cal, X)
        split = "val" if "val" in tag else "test"
        metrics = plot_curves(y, p, tag, args.out)
        plot_pk_rk(y, p, [int(k) for k in args.topk_list.split(",") if k], split, args.out)
        with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 保存预测
        out_pred = os.path.join(args.out, f"pred_{tag}.csv")
        pd.DataFrame({"prob": p, "label": y}).to_csv(out_pred, index=False)

    # ===== 生成 Top-k 表（全局）=====
    ks = [int(k) for k in args.topk_list.split(",") if k.strip()]
    qs = [float(q) for q in args.quantiles.split(",") if q.strip()]

    p_vl_iso = prob_of(cal_iso, X_vl); p_te_iso = prob_of(cal_iso, X_te)
    p_vl_sig = prob_of(cal_sig, X_vl); p_te_sig = prob_of(cal_sig, X_te)

    df_topk_val_iso = topk_table_global(y_vl, p_vl_iso, ks, qs, split="val_iso")
    df_topk_te_iso  = topk_table_global(y_te, p_te_iso, ks, qs, split="test_iso")
    df_topk_val_sig = topk_table_global(y_vl, p_vl_sig, ks, qs, split="val_sig")
    df_topk_te_sig  = topk_table_global(y_te, p_te_sig, ks, qs, split="test_sig")

    df_topk_global = pd.concat([df_topk_val_iso, df_topk_te_iso, df_topk_val_sig, df_topk_te_sig], ignore_index=True)
    df_topk_global.to_csv(os.path.join(args.out, "topk_global.csv"), index=False, encoding="utf-8-sig")

    # ===== 分组 Top-k（按天 / 按天×grid_id，可配置）=====
    groupby_cols = [c.strip() for c in args.groupby.split(",") if c.strip()]
    k_per_group_list = [int(x) for x in args.group_k.split(",") if x.strip()]

    def ensure_group_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
        return [c for c in cols if c in df.columns]

    gcols_vl = ensure_group_cols(df_vl, groupby_cols)
    gcols_te = ensure_group_cols(df_te, groupby_cols)

    rows_grouped = []
    if gcols_vl:
        dvl = df_vl[gcols_vl + [TARGET]].copy()
        dvl["prob_iso"] = p_vl_iso; dvl["prob_sig"] = p_vl_sig
        rows_grouped.append(topk_table_grouped(dvl, "prob_iso", TARGET, gcols_vl, k_per_group_list, "val_iso"))
        rows_grouped.append(topk_table_grouped(dvl, "prob_sig", TARGET, gcols_vl, k_per_group_list, "val_sig"))
    if gcols_te:
        dte = df_te[gcols_te + [TARGET]].copy()
        dte["prob_iso"] = p_te_iso; dte["prob_sig"] = p_te_sig
        rows_grouped.append(topk_table_grouped(dte, "prob_iso", TARGET, gcols_te, k_per_group_list, "test_iso"))
        rows_grouped.append(topk_table_grouped(dte, "prob_sig", TARGET, gcols_te, k_per_group_list, "test_sig"))

    if rows_grouped:
        df_grouped = pd.concat(rows_grouped, ignore_index=True)
        df_grouped.to_csv(os.path.join(args.out, "topk_grouped.csv"), index=False, encoding="utf-8-sig")

    # ===== 运营阈值（primary: 分位；fallback: 固定Top-k）=====
    # # 以 *验证集* 的 Isotonic 为主，选择你 anchor_k 对应的分位和阈值；同时记录一个常用分位（如 0.999）供手动选。
    s_vl_iso = np.sort(p_vl_iso)
    # # 若要固定“某个 k”的分位：
    k_anchor = min(anchor_k, len(p_vl_iso))
    thr_anchor = float(np.partition(p_vl_iso, -k_anchor)[-k_anchor]) if k_anchor>0 else 1.0
    q_anchor   = float((p_vl_iso <= thr_anchor).mean())

    # 也附上你配置里的一个高分位（比如第一个）
    q_primary = qs[0] if qs else 0.999
    thr_q_primary = float(np.quantile(s_vl_iso, q_primary))

    ops_cfg = {
        "primary":  {"mode": "quantile", "q": float(q_primary), "backup_threshold": thr_q_primary,
                     "note": "首选按验证集分位阈值以保持稳定告警量"},
        "fallback": {"mode": "topk", "k": int(anchor_k), "note": "运营如需固定条数时使用"},
        "anchor_on_val": {"k": int(anchor_k), "q_equiv": float(q_anchor), "threshold_equiv": float(thr_anchor)}
    }
    with open(os.path.join(args.out, "ops_thresholds.json"), "w") as f:
        json.dump(ops_cfg, f, indent=2)

    # ===== 保存模型（两套校准器） + 训练元信息 =====
    joblib.dump(cal_iso, os.path.join(args.out, "model_xgb_isotonic.joblib"))
    joblib.dump(cal_sig, os.path.join(args.out, "model_xgb_sigmoid.joblib"))

    meta_out = {
        "features_used": FEATURES,
        "train_pos": int(pos), "train_neg": int(neg), "scale_pos_weight": spw,
        "best_params": {"max_depth": best["md"], "min_child_weight": best["mcw"], "reg_lambda": best["rl"],
                        "learning_rate": best["lr"], "n_estimators": args.n_estimators,
                        "subsample": args.subsample, "colsample_bytree": args.colsample_bytree,
                        "max_delta_step": args.max_delta_step},
        "select_k": int(anchor_k),
        "topk_list": ks, "quantiles": qs,
        "groupby": groupby_cols, "group_k": k_per_group_list
    }
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print("[OK] Saved outputs to", args.out)

    # === 在脚本末尾追加：STEP3 — Climatic drivers & thresholds 可视化（Fig.6/7） ===
    # 放置位置：紧跟在保存 train_meta.json 之后，或在 main() 的最后 return 前

    # ------ 通用配置 ------
    FIG_OUT = args.out  # 直接复用你的 --out
    FEAT_NAMES = FEATURES
    SHAP_MAX = min(20000, len(X_tr))
    PD_GRID_1D = 50
    PD_GRID_2D = 40
    PAIRS_2D = [("t2m", "SPEI_1"), ("VHI", "SPEI_1"), ("K_index", "SPEI_1")]  # 可按需改

    os.makedirs(FIG_OUT, exist_ok=True)

    # ------ SHAP（Fig.6 左）------
    # why：在未校准 XGB 上做 SHAP；校准器仅重映射分数，对重要性与方向影响极小
    try:
        import shap
        shap_sample = X_tr.iloc[np.random.RandomState(args.seed).choice(len(X_tr), SHAP_MAX, replace=False)] if len(
            X_tr) > SHAP_MAX else X_tr
        expl = shap.TreeExplainer(best["model"])
        shap_vals = expl.shap_values(shap_sample.values)

        # Fig.6 — SHAP global bar
        abs_mean = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(abs_mean)[::-1]
        topk = min(10, len(FEAT_NAMES))
        top_feats = [FEAT_NAMES[i] for i in order[:topk]]
        top_vals = abs_mean[order[:topk]]

        plt.figure(figsize=(7, max(4, 0.45 * topk)))
        plt.barh(np.arange(topk)[::-1], top_vals[::-1])
        plt.yticks(np.arange(topk)[::-1], top_feats[::-1])
        plt.xlabel("mean(|SHAP|)")
        plt.title("Fig.6 — SHAP global importance")
        plt.tight_layout()
        fig6_shap_bar = os.path.join(FIG_OUT, "fig6_left_shap_bar.png")
        plt.savefig(fig6_shap_bar, dpi=300);
        plt.close()

        # Fig.6 — SHAP summary (beeswarm)
        plt.figure(figsize=(8, max(4, 0.5 * topk)))
        shap.summary_plot(shap_vals, shap_sample, feature_names=FEAT_NAMES, show=False)
        plt.title("Fig.6 — SHAP summary")
        fig6_shap_sum = os.path.join(FIG_OUT, "fig6_left_shap_summary.png")
        plt.tight_layout();
        plt.savefig(fig6_shap_sum, dpi=300);
        plt.close()
    except Exception as e:
        print("[WARN] SHAP failed:", e)

    # ------ 1D PD（Fig.6 右）------
    def pd_1d(cal_model, X_base_df: pd.DataFrame, feat: str, qgrid=PD_GRID_1D) -> pd.DataFrame:
        x = X_base_df.copy()
        qs = np.linspace(0.01, 0.99, qgrid)
        grid = np.quantile(x[feat].values, qs)
        out = []
        for g in grid:
            Xg = x.copy()
            Xg[feat] = g
            p = prob_of(cal_model, Xg).mean()
            out.append((g, p))
        return pd.DataFrame(out, columns=[feat, "pd"])

    def detect_thresholds(pd_df: pd.DataFrame, feat: str) -> pd.DataFrame:
        xv = pd_df[feat].values;
        yv = pd_df["pd"].values
        if len(xv) < 3:
            return pd.DataFrame(columns=["feature", "x", "pd", "note"])
        dy = np.gradient(yv, xv)
        i = int(np.argmax(np.abs(dy)))
        idxs = sorted({max(0, i - 1), i, min(len(xv) - 1, i + 1)})
        return pd.DataFrame({"feature": feat, "x": [xv[j] for j in idxs], "pd": [yv[j] for j in idxs],
                             "note": ["slope_peak/edge"] * len(idxs)})

    # 选 top 特征（沿用 SHAP 顺序，若 SHAP 失败则用原顺序）
    try:
        PD_FEATS = top_feats[:6]
    except Exception:
        PD_FEATS = FEAT_NAMES[:6]

    # PD 基样本：建议用验证集（更稳），若想更平滑可用 train+val
    X_pd_base = X_vl.copy()

    pd_records = []
    for f in PD_FEATS:
        df_pd = pd_1d(cal_iso, X_pd_base, f, qgrid=PD_GRID_1D)
        thr_df = detect_thresholds(df_pd, f)

        plt.figure(figsize=(6, 4))
        plt.plot(df_pd[f].values, df_pd["pd"].values, lw=1.5)
        if len(thr_df):
            plt.scatter(thr_df["x"].values, thr_df["pd"].values, marker="o")
        plt.xlabel(f);
        plt.ylabel("Partial dependence (P)")
        plt.title(f"Fig.6 — PD & thresholds: {f}")
        plt.tight_layout()
        outp = os.path.join(FIG_OUT, f"fig6_right_pd_{f}.png")
        plt.savefig(outp, dpi=300);
        plt.close()

        df_pd.to_csv(os.path.join(FIG_OUT, f"pd_{f}.csv"), index=False)
        thr_df.assign(figure=os.path.basename(outp)).to_csv(os.path.join(FIG_OUT, f"pd_{f}_thresholds.csv"),
                                                            index=False)
        pd_records.append(thr_df.assign(figure=os.path.basename(outp)))

    if pd_records:
        pd.concat(pd_records, ignore_index=True).to_csv(os.path.join(FIG_OUT, "fig6_pd_threshold_candidates.csv"),
                                                        index=False)

    # ------ 2D PD（Fig.7 左）------
    def pd_2d(cal_model, X_base_df: pd.DataFrame, f1: str, f2: str, qgrid=PD_GRID_2D):
        x = X_base_df.copy()
        q = np.linspace(0.05, 0.95, qgrid)
        g1 = np.quantile(x[f1].values, q);
        g2 = np.quantile(x[f2].values, q)
        Z = np.zeros((len(g2), len(g1)), dtype=float)
        for a_idx, a in enumerate(g1):
            x1 = x.copy();
            x1[f1] = a
            for b_idx, b in enumerate(g2):
                x2 = x1.copy();
                x2[f2] = b
                Z[b_idx, a_idx] = prob_of(cal_model, x2).mean()
        return g1, g2, Z

    delta_map = {}
    for (f1, f2) in PAIRS_2D:
        if f1 not in FEAT_NAMES or f2 not in FEAT_NAMES:
            continue
        g1, g2, Z = pd_2d(cal_iso, X_pd_base, f1, f2, qgrid=PD_GRID_2D)
        plt.figure(figsize=(6, 5))
        cs = plt.contourf(g1, g2, Z, levels=12)
        plt.colorbar(cs)
        plt.xlabel(f1);
        plt.ylabel(f2)
        plt.title(f"Fig.7 — 2D PD: {f1} × {f2}")
        plt.tight_layout()
        out2d = os.path.join(FIG_OUT, f"fig7_left_pd2d_{f1}_x_{f2}.png")
        plt.savefig(out2d, dpi=300);
        plt.close()
        np.savez_compressed(os.path.join(FIG_OUT, f"pd2d_{f1}_x_{f2}.npz"), g1=g1, g2=g2, Z=Z)
        delta_map[(f1, f2)] = float(Z.max() - Z.min())

    # ------ 路径草图（Fig.7 右；可选 networkx，回退文本）------
    try:
        import networkx as nx
        G = nx.Graph()
        for (a, b), d in delta_map.items():
            G.add_edge(a, b, weight=d)
        pos = nx.spring_layout(G, seed=0)
        plt.figure(figsize=(6, 5))
        widths = [1 + 6 * G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx(G, pos=pos, width=widths, with_labels=True, node_size=1200)
        plt.title("Fig.7 — pathway sketch (edge~Δ)")
        plt.tight_layout()
        fig7_pathway = os.path.join(FIG_OUT, "fig7_right_pathway.png")
        plt.savefig(fig7_pathway, dpi=300);
        plt.close()
    except Exception as e:
        print("[WARN] networkx not available, write text:", e)
        txt = os.path.join(FIG_OUT, "fig7_right_pathway.txt")
        with open(txt, "w", encoding="utf-8") as f:
            for (a, b), d in delta_map.items():
                f.write(f"{a} -- {b} : Δ={d:.4f}\n")

    print("[STEP3] Fig.6/7 generated into:", FIG_OUT)


if __name__ == "__main__":
    main()

