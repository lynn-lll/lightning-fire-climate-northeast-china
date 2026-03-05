# replot_iso_figs.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
AUX = "#9CA3AF"


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def read_pq(p: str) -> pd.DataFrame:
    return pd.read_parquet(p)


def pick_X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = df[feats].copy()
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def prob_of(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def new_fig_ax(fig_w: float, fig_h: float, margins: dict) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(**margins)
    return fig, ax


def save_close(fig: plt.Figure, path: str) -> None:
    fig.savefig(path)  # 不要 bbox_inches="tight"
    plt.close(fig)


def force_current_fig_layout(fig_w: float, fig_h: float, margins: dict) -> plt.Figure:
    fig = plt.gcf()
    fig.set_size_inches(fig_w, fig_h)
    fig.subplots_adjust(**margins)
    return fig


# ----------------- plots -----------------
def plot_roc(y, p, tag, out, *, fig_w: float, fig_h: float, margins: dict) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=AUX)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {tag}")
    ax.legend()

    save_close(fig, os.path.join(out, f"roc_{tag}.png"))
    return float(auc)


def plot_pr(y, p, tag, out, *, fig_w: float, fig_h: float, margins: dict) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.plot(rec, prec, label=f"PR-AUC={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — {tag}")
    ax.legend()

    save_close(fig, os.path.join(out, f"pr_{tag}.png"))
    return float(ap)


def plot_calibration_full(y, p, tag, out, *, fig_w: float, fig_h: float, margins: dict) -> float:
    frac, meanp = calibration_curve(y, p, n_bins=20, strategy="quantile")
    brier = brier_score_loss(y, p)

    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.plot(meanp, frac, "o-")
    ax.plot([0, 1], [0, 1], "--", color=AUX)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"Calibration — {tag}")

    save_close(fig, os.path.join(out, f"calibration_{tag}.png"))
    return float(brier)


def plot_prob_shapes(p, tag, out, *, fig_w: float, fig_h: float, margins: dict) -> None:
    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.hist(p, bins=50)
    ax.set_title(f"{tag} prob histogram")
    save_close(fig, os.path.join(out, f"prob_hist_{tag}.png"))

    ps = np.clip(p, 1e-15, 1.0)
    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.hist(ps, bins=50)
    ax.set_xscale("log")
    ax.set_title(f"{tag} prob histogram (log-x)")
    save_close(fig, os.path.join(out, f"prob_hist_log_{tag}.png"))

    s = np.sort(ps)
    cdf = np.arange(1, len(s) + 1) / len(s)
    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.plot(s, cdf)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probability (log)")
    ax.set_ylabel("CDF")
    ax.set_title(f"{tag} prob CDF")
    save_close(fig, os.path.join(out, f"prob_cdf_{tag}.png"))


def topk_table(y, p, ks, tag, out) -> pd.DataFrame:
    n = len(y)
    order = np.argsort(-p)
    y_sorted = y[order]
    base_rate = float(y.mean())
    rows = []
    for k in ks:
        kk = min(int(k), n)
        tp = int(y_sorted[:kk].sum())
        fp = int(kk - tp)
        prec = tp / max(1, kk)
        rec = tp / max(1, int(y.sum()))
        lift = prec / max(1e-12, base_rate)
        thr = float(p[order[kk - 1]]) if kk > 0 else 1.0
        rows.append([tag, kk, tp, fp, prec, rec, lift, thr, base_rate])
    df = pd.DataFrame(
        rows,
        columns=["split", "k", "tp", "fp", "precision", "recall", "lift", "threshold", "base_rate"],
    )
    df.to_csv(os.path.join(out, f"topk_{tag}.csv"), index=False, encoding="utf-8-sig")
    return df


def make_calibration_table(y_true, y_prob, *, n_bins=20, tail_q=None) -> pd.DataFrame:
    p = np.asarray(y_prob)
    y = np.asarray(y_true).astype(int)
    baseline = y.mean() + 1e-15

    if tail_q is not None:
        thr = np.quantile(p, tail_q)
        mask = p >= thr
        p, y = p[mask], y[mask]

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p, qs))
    if len(edges) <= 2:
        edges = np.linspace(p.min(), p.max() + 1e-15, num=min(n_bins, 5))

    bins = np.digitize(p, edges[1:-1], right=False)
    df = pd.DataFrame({"p": p, "y": y, "bin": bins})
    gb = df.groupby("bin")
    out = gb.agg(
        n=("y", "size"),
        pos=("y", "sum"),
        rate=("y", "mean"),
        mean_pred=("p", "mean"),
    ).reset_index(drop=True)
    out["bin_low"] = [edges[i] for i in range(len(out))]
    out["bin_high"] = [edges[i + 1] for i in range(len(out))]
    out["lift"] = out["rate"] / baseline
    return out[["bin_low", "bin_high", "n", "pos", "rate", "mean_pred", "lift"]].sort_values("bin_low")


def plot_calibration_tail(
    y_true,
    y_prob,
    tag,
    out_dir,
    *,
    tail_q=0.99,
    n_bins=12,
    log_x=True,
    fig_w: float,
    fig_h: float,
    margins: dict,
):
    ensure_dir(out_dir)
    tbl = make_calibration_table(y_true, y_prob, n_bins=n_bins, tail_q=tail_q)

    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.plot(tbl["mean_pred"], tbl["rate"], "o-")
    xmin, xmax = float(tbl["mean_pred"].min()), float(tbl["mean_pred"].max())
    xs = np.linspace(xmin, xmax, 100)
    ax.plot(xs, xs, "--", color=AUX)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Predicted (tail)")
    ax.set_ylabel("Observed")
    ax.set_title(f"Calibration (tail) — {tag}")
    save_close(fig, os.path.join(out_dir, f"calibration_tail_{tag}.png"))

    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    qpos = np.linspace(tail_q, 1, len(tbl))
    ax.plot(qpos, tbl["lift"], "o-")
    ax.axhline(1.0, ls="--", color=AUX)
    ax.set_xlabel("Quantile (tail)")
    ax.set_ylabel("Lift = rate / baseline")
    ax.set_title(f"Lift by tail quantile — {tag}")
    save_close(fig, os.path.join(out_dir, f"lift_by_quantile_{tag}.png"))

    tbl.to_csv(os.path.join(out_dir, f"calibration_bins_{tag}.csv"), index=False)
    return {
        "baseline_rate": float(np.mean(y_true)),
        "tail_q": float(tail_q),
        "n_bins_tail": int(len(tbl)),
        "max_lift_tail": float(tbl["lift"].max() if len(tbl) else 0.0),
    }


def plot_pk_rk(y, p, ks, tag, out, *, fig_w: float, fig_h: float, margins: dict) -> None:
    n = len(y)
    order = np.argsort(-p)
    y_sorted = y[order]
    precs, recs, ks_eff = [], [], []
    for k in ks:
        kk = min(int(k), n)
        tp = int(y_sorted[:kk].sum())
        precs.append(tp / max(1, kk))
        recs.append(tp / max(1, int(y.sum())))
        ks_eff.append(kk)

    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.plot(ks_eff, precs, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("k (log)")
    ax.set_ylabel("Precision@k")
    ax.set_title(f"Precision@k — {tag}")
    save_close(fig, os.path.join(out, f"precision_at_k_{tag}.png"))

    fig, ax = new_fig_ax(fig_w, fig_h, margins)
    ax.plot(ks_eff, recs, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("k (log)")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"Recall@k — {tag}")
    save_close(fig, os.path.join(out, f"recall_at_k_{tag}.png"))


def try_get_base_xgb(calibrated):
    for a in ("estimator", "base_estimator"):
        if hasattr(calibrated, a):
            return getattr(calibrated, a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/home/dell/Daxinganling2025/output/xgb_ops", help="包含 model_xgb_isotonic.joblib")
    ap.add_argument("--val_pq", default="/home/dell/Daxinganling2025/output/Redata/step7_1112_1/step5_cls_val.parquet")
    ap.add_argument("--test_pq", default="/home/dell/Daxinganling2025/output/Redata/step7_1112_1/step5_cls_test.parquet")
    ap.add_argument("--features", default="avg_ie,t2m,ssr,sshf,SPEI_1,swvl1,K_index,VHI,lai_hv,si10", help="逗号分隔")
    ap.add_argument("--target", default="fire_presence")

    ap.add_argument("--ks", default="100,200,500,1000,2000,5000,10000,20000")
    ap.add_argument("--tail_q", type=float, default=0.99)
    ap.add_argument("--tail_bins", type=int, default=12)

    ap.add_argument("--primary_q", type=float, default=0.9995)
    ap.add_argument("--fallback_k", type=int, default=5000)

    ap.add_argument("--out", default="/home/dell/Daxinganling2025/output/xgb_eval_explain_replot_0127")

    ap.add_argument("--fig_w", type=float, default=7.0)
    ap.add_argument("--fig_h", type=float, default=3.0)
    ap.add_argument("--font_size", type=float, default=16.0)

    # 固定边距：确保所有图片留白一致（拼大图不会漂移）
    ap.add_argument("--left", type=float, default=0.24)
    ap.add_argument("--right", type=float, default=0.96)
    ap.add_argument("--bottom", type=float, default=0.22)
    ap.add_argument("--top", type=float, default=0.88)

    args = ap.parse_args()

    ensure_dir(args.out)

    feats = [c.strip() for c in args.features.split(",") if c.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    df_val, df_test = read_pq(args.val_pq), read_pq(args.test_pq)
    Xv, yv = pick_X(df_val, feats), df_val[args.target].astype("int8").to_numpy()
    Xt, yt = pick_X(df_test, feats), df_test[args.target].astype("int8").to_numpy()

    model = joblib.load(os.path.join(args.model_dir, "model_xgb_isotonic.joblib"))

    ISO_RC = {
        "font.size": args.font_size,
        "axes.titlesize": args.font_size,
        "axes.labelsize": args.font_size,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size,
        "legend.frameon": False,
    }

    margins = dict(left=args.left, right=args.right, bottom=args.bottom, top=args.top)

    # --- iso folder ---
    iso_dir = os.path.join(args.out, "iso")
    ensure_dir(iso_dir)

    pv, pt = prob_of(model, Xv), prob_of(model, Xt)

    with plt.rc_context(ISO_RC):
        auc_v = plot_roc(yv, pv, "val_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)
        auc_t = plot_roc(yt, pt, "test_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)
        ap_v = plot_pr(yv, pv, "val_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)
        ap_t = plot_pr(yt, pt, "test_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)
        br_v = plot_calibration_full(yv, pv, "val_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)
        br_t = plot_calibration_full(yt, pt, "test_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)

        plot_prob_shapes(pv, "val_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)
        plot_prob_shapes(pt, "test_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)

        topk_table(yv, pv, ks, "val_iso", iso_dir)
        topk_table(yt, pt, ks, "test_iso", iso_dir)

        plot_pk_rk(yv, pv, ks, "val_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)
        plot_pk_rk(yt, pt, ks, "test_iso", iso_dir, fig_w=args.fig_w, fig_h=args.fig_h, margins=margins)

        tail_val = plot_calibration_tail(
            yv,
            pv,
            "val_iso",
            iso_dir,
            tail_q=args.tail_q,
            n_bins=args.tail_bins,
            fig_w=args.fig_w,
            fig_h=args.fig_h,
            margins=margins,
        )
        tail_test = plot_calibration_tail(
            yt,
            pt,
            "test_iso",
            iso_dir,
            tail_q=args.tail_q,
            n_bins=args.tail_bins,
            fig_w=args.fig_w,
            fig_h=args.fig_h,
            margins=margins,
        )

    # --- operating point (just computing threshold; no training) ---
    s_v = np.sort(np.clip(pv, 1e-15, 1.0))
    primary_thr = float(np.quantile(s_v, args.primary_q))
    operating_point = {
        "primary": {"mode": "quantile", "q": float(args.primary_q), "backup_threshold": primary_thr},
        "fallback": {"mode": "topk", "k": int(args.fallback_k)},
    }
    with open(os.path.join(args.out, "operating_point.json"), "w", encoding="utf-8") as f:
        json.dump(operating_point, f, indent=2, ensure_ascii=False)

    # --- explain folder (same style + fixed margins) ---
    explain_dir = os.path.join(args.out, "explain")
    ensure_dir(explain_dir)

    explain_meta = {}
    base_xgb = try_get_base_xgb(model)

    with plt.rc_context(ISO_RC):
        if base_xgb is not None and base_xgb.__class__.__name__.startswith("XGB"):
            try:
                import shap  # type: ignore

                explainer = shap.TreeExplainer(base_xgb, feature_perturbation="tree_path_dependent")
                ns = min(50000, len(Xv))
                Xv_sample = Xv.sample(ns, random_state=42) if len(Xv) > ns else Xv
                sv = explainer.shap_values(Xv_sample)

                vals = np.abs(sv).mean(axis=0)
                shap_imp = pd.DataFrame({"feature": Xv_sample.columns, "mean_abs_shap": vals}).sort_values(
                    "mean_abs_shap", ascending=False
                )
                shap_imp.to_csv(os.path.join(explain_dir, "shap_importance_val.csv"), index=False, encoding="utf-8-sig")

                # Summary
                plt.figure()
                try:
                    shap.summary_plot(
                        sv, Xv_sample, plot_type="dot", show=False, color_bar=True, plot_size=(args.fig_w, args.fig_h)
                    )
                except TypeError:
                    shap.summary_plot(sv, Xv_sample, plot_type="dot", show=False, color_bar=True)
                fig = force_current_fig_layout(args.fig_w, args.fig_h, margins)
                save_close(fig, os.path.join(explain_dir, "shap_summary_val.png"))

                # Dependence
                top10 = shap_imp["feature"].head(10).tolist()
                for f_name in top10:
                    try:
                        shap.dependence_plot(f_name, sv, Xv_sample, show=False)
                        fig = force_current_fig_layout(args.fig_w, args.fig_h, margins)
                        save_close(fig, os.path.join(explain_dir, f"shap_dependence_{f_name}.png"))
                    except Exception:
                        plt.close()

                explain_meta = {"method": "SHAP(TreeExplainer)", "top10": top10}
            except Exception as e:
                explain_meta = {"method": "SHAP_failed_fallback_perm_importance", "error": str(e)}
                base_xgb = None
        else:
            explain_meta = {"method": "permutation_importance_only"}

        if base_xgb is None:
            r = permutation_importance(model, Xv, yv, n_repeats=5, random_state=42, scoring="average_precision")
            perm_imp = pd.DataFrame(
                {"feature": Xv.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
            ).sort_values("importance_mean", ascending=False)
            perm_imp.to_csv(
                os.path.join(explain_dir, "permutation_importance_val.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            explain_meta = {"method": "permutation_importance", **explain_meta}

    report = {
        "packs": [
            {
                "name": "iso",
                "auc_val": auc_v,
                "auc_test": auc_t,
                "ap_val": ap_v,
                "ap_test": ap_t,
                "brier_val": br_v,
                "brier_test": br_t,
                "tail_val": tail_val,
                "tail_test": tail_test,
            }
        ],
        "explain": explain_meta,
        "operating_point": operating_point,
        "note": "Replot only (no training). Fixed margins for consistent alignment across PNGs.",
        "margins": margins,
        "figsize": [args.fig_w, args.fig_h],
        "font_size": args.font_size,
    }
    with open(os.path.join(args.out, "fit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[OK] Replotted figures in:", args.out)


if __name__ == "__main__":
    main()