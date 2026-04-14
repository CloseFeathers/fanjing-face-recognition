"""
说话检测模型训练 pipeline。

流程: 加载录制数据 → 标签对齐 → 特征工程 → 训练 LR/XGBoost → 评估报告

用法: python train_speaking_model.py
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path("data/recordings")
MODEL_OUT = Path("models/speaking")

# Blendshape columns relevant to mouth/jaw
MOUTH_BS = [
    "bs_jawOpen", "bs_jawForward", "bs_jawLeft", "bs_jawRight",
    "bs_mouthClose", "bs_mouthFunnel", "bs_mouthPucker",
    "bs_mouthLeft", "bs_mouthRight",
    "bs_mouthSmileLeft", "bs_mouthSmileRight",
    "bs_mouthFrownLeft", "bs_mouthFrownRight",
    "bs_mouthDimpleLeft", "bs_mouthDimpleRight",
    "bs_mouthStretchLeft", "bs_mouthStretchRight",
    "bs_mouthRollLower", "bs_mouthRollUpper",
    "bs_mouthShrugLower", "bs_mouthShrugUpper",
    "bs_mouthPressLeft", "bs_mouthPressRight",
    "bs_mouthUpperUpLeft", "bs_mouthUpperUpRight",
    "bs_mouthLowerDownLeft", "bs_mouthLowerDownRight",
]

RAW_FEATURES = MOUTH_BS + ["yaw", "pitch", "roll"]


# ======================================================================
# Step 1: Load all data
# ======================================================================

def load_all_data() -> pd.DataFrame:
    frames = []
    for sc_dir in sorted(DATA_ROOT.iterdir()):
        if not sc_dir.is_dir():
            continue
        csv_path = sc_dir / "features.csv"
        events_path = sc_dir / "events.jsonl"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["recording_id"] = sc_dir.name

        # Load key events for border ignore
        transitions = []
        if events_path.exists():
            with open(events_path, encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line)
                    if ev["type"] in ("key_down", "key_up"):
                        transitions.append(ev["timestamp_ms"])

        df["_transition_ms"] = [transitions] * len(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} frames from {len(frames)} recordings")
    return combined


# ======================================================================
# Step 2: Label alignment
# ======================================================================

def align_labels(df: pd.DataFrame, ignore_border_ms: float = 500.0) -> pd.DataFrame:
    labels = []
    for _, row in df.iterrows():
        ts = row["timestamp_ms"]
        transitions = row["_transition_ms"]
        cat = row["scenario_cat"]

        if cat == "SKIP":
            labels.append(-1)
            continue

        near_border = any(abs(ts - t) < ignore_border_ms for t in transitions)
        if near_border:
            labels.append(-1)
        elif row["key_pressed"] == 1:
            labels.append(1)
        else:
            labels.append(0)

    df["label"] = labels
    df.drop(columns=["_transition_ms"], inplace=True)

    n_spk = (df["label"] == 1).sum()
    n_nspk = (df["label"] == 0).sum()
    n_ign = (df["label"] == -1).sum()
    print(f"Labels: SPEAKING={n_spk}  NOT_SPEAKING={n_nspk}  IGNORE={n_ign}")
    return df


# ======================================================================
# Step 3: Feature engineering
# ======================================================================

def compute_window_features(df: pd.DataFrame, window: int = 16) -> pd.DataFrame:
    """For each recording, compute windowed stats over raw features."""
    all_feats = []
    for rec_id, group in df.groupby("recording_id", sort=False):
        group = group.sort_values("frame_id").reset_index(drop=True)
        raw = group[RAW_FEATURES].astype(float).values
        n_frames, n_raw = raw.shape

        feat_names = []
        feat_data = np.full((n_frames, n_raw * 6 + n_raw), np.nan)

        for i in range(n_frames):
            start = max(0, i - window + 1)
            win = raw[start:i + 1]

            current = raw[i]
            w_mean = np.mean(win, axis=0)
            w_std = np.std(win, axis=0)
            w_min = np.min(win, axis=0)
            w_max = np.max(win, axis=0)
            w_range = w_max - w_min

            if len(win) > 1:
                diffs = np.diff(win, axis=0)
                w_diff_mean = np.mean(np.abs(diffs), axis=0)
            else:
                w_diff_mean = np.zeros(n_raw)

            feat_data[i] = np.concatenate([
                current, w_mean, w_std, w_range, w_diff_mean,
                w_min, w_max,
            ])

        if not feat_names:
            for suffix in ["_cur", "_mean", "_std", "_range", "_dmean",
                           "_min", "_max"]:
                for fn in RAW_FEATURES:
                    feat_names.append(fn + suffix)

        feat_df = pd.DataFrame(feat_data, columns=feat_names)
        feat_df["label"] = group["label"].values
        feat_df["recording_id"] = rec_id
        feat_df["yaw_abs"] = np.abs(group["yaw"].astype(float).values)
        feat_df["scenario_id"] = group["scenario_id"].values
        all_feats.append(feat_df)

    result = pd.concat(all_feats, ignore_index=True)
    print(f"Features: {result.shape[1] - 3} dims, {len(result)} frames")
    return result


# ======================================================================
# Step 4: Train & evaluate
# ======================================================================

def train_and_evaluate(feat_df: pd.DataFrame, window: int):
    # Filter out IGNORE
    valid = feat_df[feat_df["label"] >= 0].copy()
    valid = valid.dropna()

    feature_cols = [c for c in valid.columns
                    if c not in ("label", "recording_id", "yaw_abs", "scenario_id")]
    X = valid[feature_cols].values
    y = valid["label"].values
    groups = valid["recording_id"].values

    print(f"\nTrainable data: {len(valid)} frames "
          f"(SPK={int(y.sum())}, NSPK={int((y == 0).sum())})")

    # GroupKFold by recording_id (3 folds)
    unique_groups = np.unique(groups)
    n_splits = min(3, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    results = {}

    for model_name, model_fn in [
        ("LogisticRegression", lambda: LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced")),
        ("XGBoost", lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))),
    ]:
        fold_f1s = []
        fold_precs = []
        fold_recs = []
        yaw_bucket_f1 = {b: [] for b in ["0-15", "15-30", "30-45", "45+"]}
        best_model = None
        best_f1 = -1
        best_scaler = None

        print(f"\n{'─' * 50}")
        print(f"  Model: {model_name}  |  Window: {window}")
        print(f"{'─' * 50}")

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if model_name == "LogisticRegression":
                model = model_fn()
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
            else:
                model = model_fn()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            fold_f1s.append(f1)
            fold_precs.append(prec)
            fold_recs.append(rec)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_scaler = scaler

            # Yaw-bucketed F1
            yaw_abs_test = valid.iloc[test_idx]["yaw_abs"].values
            for bname, lo, hi in [("0-15", 0, 15), ("15-30", 15, 30),
                                   ("30-45", 30, 45), ("45+", 45, 999)]:
                mask = (yaw_abs_test >= lo) & (yaw_abs_test < hi)
                if mask.sum() > 0 and len(np.unique(y_test[mask])) > 1:
                    yaw_bucket_f1[bname].append(
                        f1_score(y_test[mask], y_pred[mask], zero_division=0))

            print(f"  Fold {fold+1}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}"
                  f"  (test={len(y_test)}, train={len(y_train)})")

        avg_f1 = np.mean(fold_f1s)
        avg_p = np.mean(fold_precs)
        avg_r = np.mean(fold_recs)
        print(f"\n  Average: F1={avg_f1:.3f}  P={avg_p:.3f}  R={avg_r:.3f}")

        print(f"\n  Yaw-bucketed F1:")
        for bname in ["0-15", "15-30", "30-45", "45+"]:
            vals = yaw_bucket_f1[bname]
            if vals:
                print(f"    {bname:>5s}: {np.mean(vals):.3f}")
            else:
                print(f"    {bname:>5s}: (no data)")

        results[model_name] = {
            "avg_f1": avg_f1, "avg_p": avg_p, "avg_r": avg_r,
            "best_model": best_model, "best_scaler": best_scaler,
            "best_f1": best_f1, "feature_cols": feature_cols,
            "yaw_f1": {k: np.mean(v) if v else 0 for k, v in yaw_bucket_f1.items()},
        }

    return results


# ======================================================================
# Step 5: Feature importance (XGBoost)
# ======================================================================

def show_feature_importance(results: dict, top_n: int = 20):
    xgb_res = results.get("XGBoost")
    if not xgb_res:
        return
    model = xgb_res["best_model"]
    cols = xgb_res["feature_cols"]
    imp = model.feature_importances_
    indices = np.argsort(imp)[::-1][:top_n]

    print(f"\n{'═' * 50}")
    print(f"  Top {top_n} Feature Importance (XGBoost)")
    print(f"{'═' * 50}")
    for rank, i in enumerate(indices):
        print(f"  {rank+1:2d}. {cols[i]:40s}  {imp[i]:.4f}")


# ======================================================================
# Step 6: Save best model
# ======================================================================

def save_best_model(results: dict, window: int):
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    best_name = max(results, key=lambda k: results[k]["avg_f1"])
    best = results[best_name]

    model_path = MODEL_OUT / "speaking_model.json"
    meta_path = MODEL_OUT / "speaking_meta.json"

    best["best_model"].get_booster().save_model(str(model_path))

    import json as _json
    meta = {
        "model_type": best_name,
        "feature_cols": best["feature_cols"],
        "raw_features": RAW_FEATURES,
        "window_size": window,
        "metrics": {
            "f1": best["avg_f1"],
            "precision": best["avg_p"],
            "recall": best["avg_r"],
            "yaw_f1": best["yaw_f1"],
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        _json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'═' * 50}")
    print(f"  Best model: {best_name}")
    print(f"  F1={best['avg_f1']:.3f}  P={best['avg_p']:.3f}  R={best['avg_r']:.3f}")
    print(f"  Saved to: {model_path} + {meta_path}")
    print(f"{'═' * 50}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  Speaking Detection Model Training")
    print("=" * 60)

    # Load
    df = load_all_data()

    # Label alignment with border sweep
    best_window = None
    best_border = None
    best_overall_f1 = -1
    all_results = {}

    for border_ms in [300, 500, 750]:
        print(f"\n{'━' * 60}")
        print(f"  Border ignore: {border_ms}ms")
        print(f"{'━' * 60}")
        labeled = align_labels(df.copy(), ignore_border_ms=border_ms)

        for window in [10, 12, 16, 20]:
            feat_df = compute_window_features(labeled, window=window)
            results = train_and_evaluate(feat_df, window)

            key = f"border{border_ms}_win{window}"
            all_results[key] = results

            xgb_f1 = results.get("XGBoost", {}).get("avg_f1", 0)
            if xgb_f1 > best_overall_f1:
                best_overall_f1 = xgb_f1
                best_window = window
                best_border = border_ms

    # Re-run best config for final model
    print(f"\n{'━' * 60}")
    print(f"  BEST CONFIG: border={best_border}ms, window={best_window}")
    print(f"  XGBoost F1 = {best_overall_f1:.3f}")
    print(f"{'━' * 60}")

    labeled = align_labels(df.copy(), ignore_border_ms=best_border)
    feat_df = compute_window_features(labeled, window=best_window)
    final_results = train_and_evaluate(feat_df, best_window)

    show_feature_importance(final_results)
    save_best_model(final_results, best_window)


if __name__ == "__main__":
    main()
