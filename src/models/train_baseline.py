"""Fixed five-fold baseline experiment. Run from any directory with -m.

No splitting, parameter search, holdout evaluation, or external data fetching.
The saved cv621 checkpoints are development models, not final all-row models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data.loader import (CSV_NAME, EXPECTED_FEATURES, ID_COL, TARGET,
                             cv_pos, iter_cv, load_folds, load_raw)

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"dummy_mean": "Mean baseline", "linear_regression": "Linear regression",
          "random_forest": "Random forest", "ridge_reference": "Ridge reference"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def build_models(config):
    return {
        "dummy_mean": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "linear_regression": Pipeline([("scaler", StandardScaler()),
                                        ("model", LinearRegression())]),
        "random_forest": Pipeline([("model", RandomForestRegressor(
            **config["random_forest"], random_state=config["seed"]))]),
        "ridge_reference": Pipeline([("scaler", StandardScaler()),
                                      ("model", Ridge(alpha=config["ridge_alpha"]))]),
    }


def metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)),
            "r2": float(r2_score(y, pred))}


def load_inputs(root, config):
    data_path = root / "data/raw" / CSV_NAME
    folds_path = root / "data/processed/folds.csv"
    if digest(folds_path) != config["folds_sha256"]:
        raise ValueError("Frozen folds.csv fingerprint mismatch; do not regenerate folds.")
    version = json.loads((root / "reports/data_version.json").read_text(encoding="utf-8"))
    if version["data_version"] != config["data_version"]:
        raise ValueError("data_version mismatch")
    meta = json.loads((root / "data/processed/folds.meta.json").read_text(encoding="utf-8"))
    if meta["version"] != config["folds_version"]:
        raise ValueError("folds_version mismatch")
    df = load_raw(str(data_path.parent))
    folds = load_folds(df, str(folds_path))
    if len(df) != 777 or not folds["id"].is_unique:
        raise ValueError("Invalid row count or duplicate fold IDs")
    if set(folds["fold_id"]) != {-1, 0, 1, 2, 3, 4}:
        raise ValueError("Invalid fold assignments")
    if not np.array_equal(folds["split"].eq("holdout"), folds["fold_id"].eq(-1)):
        raise ValueError("split and fold_id disagree")
    cv = cv_pos(folds)
    if len(cv) != 621 or int(folds["fold_id"].eq(-1).sum()) != 156:
        raise ValueError("CV/holdout sizes changed")
    if not np.isfinite(df[EXPECTED_FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite input values")
    return df, folds, cv


def source_info(root):
    tracked = [*root.glob("src/**/*.py"), root / "configs/baseline.json",
               root / "requirements.txt", root / "requirements-baseline.txt",
               root / "tests/test_baseline.py", root / "README_BASELINE.md"]
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(tracked) if p.is_file()}
    source_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    # Only this extracted project may supply the Git identity, never a parent repository.
    git_commit = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    return {"git_commit": git_commit, "source_sha256": source_sha, "files_sha256": files}


def plot_results(out, predictions, scores):
    colors = ["#355C7D", "#C67C45", "#687D6D"]
    keys = ["linear_regression", "random_forest", "ridge_reference"]
    fig, ax = plt.subplots(figsize=(8.1, 3.3), layout="constrained")
    vals = [next(r["oof_mae"] for r in scores if r["model"] == k) for k in keys]
    bars = ax.barh([LABELS[k] for k in keys], vals, color=colors, height=0.57)
    ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=11)
    ax.set_xlim(0, max(vals)*1.20)
    ax.set_xlabel("OOF MAE (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    for edge in ["top", "right", "left"]:
        ax.spines[edge].set_visible(False)
    fig.savefig(out / "figures/oof_mae.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), layout="constrained")
    y = predictions["yield"].to_numpy()
    # Include every predicted value, including negative OLS extrapolations.
    all_values = np.concatenate([y]+[predictions[k].to_numpy() for k in keys[:2]])
    lim = [float(all_values.min())-500, float(all_values.max())+500]
    for col, key in enumerate(keys[:2]):
        p = predictions[key].to_numpy()
        axes[0,col].scatter(y, p, s=12, alpha=.55, color=colors[col], edgecolors="none")
        axes[0,col].plot(lim, lim, "--", color="#777777", linewidth=1)
        axes[0,col].set(xlabel="Actual yield", ylabel="OOF predicted yield",
                        title=LABELS[key], xlim=lim, ylim=lim)
        axes[1,col].scatter(p, p-y, s=12, alpha=.55, color=colors[col], edgecolors="none")
        axes[1,col].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[1,col].set(xlabel="OOF predicted yield", ylabel="Residual (predicted - actual)")
        worst = int(np.argmax(np.abs(p-y)))
        axes[0,col].annotate(f"Row {int(predictions.iloc[worst]['Row#'])}",
                            (y[worst],p[worst]),xytext=(8,12),textcoords="offset points",fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    residual_limit = max(abs(predictions[k]-predictions["yield"]).max() for k in keys[:2])*1.08
    for ax in axes[1]: ax.set_ylim(-residual_limit, residual_limit)
    fig.savefig(out / "figures/oof_diagnostics.png", dpi=200)
    plt.close(fig)


def run(root=ROOT, output=None):
    config_path = root / "configs/baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(output) if output else root / "reports/baseline_models"
    model_dir = out / "models"
    for p in [out, model_dir, out / "figures"]: p.mkdir(parents=True, exist_ok=True)
    df, folds, cv = load_inputs(root, config)
    # Drop holdout rows before building any feature matrices or target arrays.
    X = df.iloc[cv][EXPECTED_FEATURES].copy().reset_index(drop=True)
    y = df.iloc[cv][TARGET].to_numpy(dtype=float)
    ids = df.iloc[cv][ID_COL].to_numpy()
    fold_ids = folds.iloc[cv]["fold_id"].to_numpy()
    full_to_cv = {int(p): i for i, p in enumerate(cv)}
    splits = [(np.array([full_to_cv[int(i)] for i in tr]),
               np.array([full_to_cv[int(i)] for i in va])) for tr, va in iter_cv(folds)]
    predictions = pd.DataFrame({"Row#": ids, "fold_id": fold_ids, "yield": y})
    all_fold_rows, scores, checkpoints = [], [], {}
    with threadpool_limits(limits=1):
        for key, template in build_models(config).items():
            oof = np.full(len(cv), np.nan)
            coverage = np.zeros(len(cv), dtype=int)
            rows = []
            for fold, (tr, va) in enumerate(splits):
                model = clone(template)
                start = time.perf_counter(); model.fit(X.iloc[tr], y[tr])
                fit_s = time.perf_counter()-start
                start = time.perf_counter(); pred = model.predict(X.iloc[va])
                predict_s = time.perf_counter()-start
                oof[va] = pred; coverage[va] += 1
                val = metrics(y[va], pred)
                train = metrics(y[tr], model.predict(X.iloc[tr]))
                row = {"model": key, "fold": fold, "train_n": len(tr), "valid_n": len(va),
                       **{"valid_"+k:v for k,v in val.items()},
                       **{"train_"+k:v for k,v in train.items()},
                       "fit_seconds": fit_s, "valid_predict_seconds": predict_s,
                       "scaler_fit_n": int(model.named_steps["scaler"].n_samples_seen_)
                           if "scaler" in model.named_steps else None}
                if key == "linear_regression":
                    row["ols_design_rank"] = int(model.named_steps["model"].rank_)
                rows.append(row)
                joblib.dump(model, model_dir / f"{key}_fold{fold}.joblib", compress=3)
            if not np.isfinite(oof).all() or not (coverage == 1).all():
                raise ValueError("OOF must cover each CV row exactly once")
            predictions[key] = oof
            summary = {"model": key, **{"oof_"+k:v for k,v in metrics(y,oof).items()},
                "fold_mae_mean": float(np.mean([r["valid_mae"] for r in rows])),
                "fold_mae_std": float(np.std([r["valid_mae"] for r in rows],ddof=1)),
                "train_mae_mean": float(np.mean([r["train_mae"] for r in rows])),
                "cv_fit_seconds": sum(r["fit_seconds"] for r in rows),
                "oof_predict_seconds": sum(r["valid_predict_seconds"] for r in rows)}
            # Refit solely on CV rows for a reusable development checkpoint.
            model = clone(template)
            start = time.perf_counter(); model.fit(X,y)
            summary["refit_cv621_seconds"] = time.perf_counter()-start
            path = model_dir / f"{key}_cv621.joblib"
            joblib.dump(model, path, compress=3)
            reloaded = joblib.load(path)
            np.testing.assert_allclose(reloaded.predict(X.iloc[:8]), model.predict(X.iloc[:8]), atol=1e-10)
            checkpoints[key] = {"path": path.relative_to(out).as_posix(),
                "sha256": digest(path), "trained_rows": 621, "includes_holdout": False,
                "parameters": model.get_params(deep=True)}
            # Model objects are represented separately; keep complete estimator parameters.
            checkpoints[key]["parameters"] = {k:v for k,v in model.get_params(deep=True).items()
                if k.startswith("model__") or k.startswith("scaler__")}
            if key == "linear_regression":
                estimator = model.named_steps["model"]
                checkpoints[key]["rank"] = int(estimator.rank_)
                checkpoints[key]["singular_values"] = estimator.singular_.tolist()
            if key in {"linear_regression", "ridge_reference"}:
                pd.DataFrame({"feature": EXPECTED_FEATURES,
                              "coefficient_on_standardized_feature": model.named_steps["model"].coef_}
                    ).to_csv(out / f"{key}_coefficients.csv", index=False)
            scores.append(summary); all_fold_rows.extend(rows)
            print(f"{key:20s} OOF MAE={summary['oof_mae']:.6f} RMSE={summary['oof_rmse']:.6f}",flush=True)
    predictions.to_csv(out / "oof_predictions.csv", index=False, float_format="%.12g")
    pd.DataFrame(all_fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    # A tiny label-free input for verifying the inference interface; CV rows only.
    df.iloc[cv[:5]][[ID_COL]+EXPECTED_FEATURES].to_csv(out / "inference_example.csv", index=False)
    plot_results(out,predictions,scores)
    reference = json.loads((root / "reports/baselines.json").read_text(encoding="utf-8"))["results"]
    agreement = {}
    for key,index in [("dummy_mean",0),("ridge_reference",1),("random_forest",2)]:
        actual = next(r["oof_mae"] for r in scores if r["model"]==key)
        agreement[key] = {"supplied_oof_mae": reference[index]["oof_mae"],
                          "actual_oof_mae": actual,
                          "absolute_delta": abs(actual-reference[index]["oof_mae"])}
    cv_analysis = {"fruitset_yield_pearson_cv_only": float(X["fruitset"].corr(pd.Series(y))),
        "temperature_max_abs_pearson_cv_only": float(
            X[EXPECTED_FEATURES[5:11]].corr().where(~np.eye(6,dtype=bool)).abs().max().max())}
    errors = predictions.copy()
    errors["linear_absolute_error"] = np.abs(errors["linear_regression"]-errors["yield"])
    errors.nlargest(10,"linear_absolute_error").to_csv(out/"largest_linear_errors.csv",index=False)
    worst_pos = int(np.argmax(np.abs(predictions["linear_regression"]-y)))
    worst_fold = int(fold_ids[worst_pos])
    train_X = X.loc[fold_ids != worst_fold]
    out_of_range = {feature:{"value":float(X.iloc[worst_pos][feature]),
                            "train_min":float(train_X[feature].min()),
                            "train_max":float(train_X[feature].max())}
        for feature in EXPECTED_FEATURES
        if X.iloc[worst_pos][feature] < train_X[feature].min()
           or X.iloc[worst_pos][feature] > train_X[feature].max()}
    cv_analysis["largest_ols_error"] = {
        "row_id":int(ids[worst_pos]), "fold":worst_fold,
        "actual":float(y[worst_pos]),
        "predictions":{key:float(predictions.iloc[worst_pos][key]) for key in LABELS},
        "absolute_error":float(abs(predictions.iloc[worst_pos]["linear_regression"]-y[worst_pos])),
        "out_of_training_range_features":out_of_range,
        "note":"Retained without deletion or clipping; diagnosis uses CV training fold only"}
    manifest = {
        "experiment": config["experiment"],
        "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "timezone": "Asia/Tokyo", "data_version": config["data_version"],
        "folds_version": config["folds_version"], "seed": config["seed"],
        "data_md5": hashlib.md5((root/"data/raw"/CSV_NAME).read_bytes()).hexdigest(),
        "folds_sha256": digest(root/"data/processed/folds.csv"),
        "config_sha256": digest(config_path), "source": source_info(root),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p:importlib.metadata.version(p) for p in
                ["numpy","pandas","scipy","scikit-learn","joblib","threadpoolctl","matplotlib"]}},
        "feature_columns": EXPECTED_FEATURES, "target": TARGET,
        "n_cv":621, "n_holdout":156, "holdout_used":False,
        "holdout_mae":None, "final_all777_model":False,
        "metric_protocol":"MAE/RMSE/R2 on 621 concatenated OOF predictions; fold SD uses ddof=1",
        "timing_protocol":"fit and validation predict separately; train scoring, serialization, plotting excluded",
        "results":scores, "reference_reproduction":agreement,
        "cv_analysis":cv_analysis, "checkpoints":checkpoints,
    }
    write_json(out/"metrics.json",manifest)
    print(f"Saved {out}; holdout remains frozen",flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Results directory (default reports/baseline_models)")
    args = parser.parse_args()
    run(output=args.output)


if __name__ == "__main__": main()
