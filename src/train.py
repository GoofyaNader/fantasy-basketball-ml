"""Train the per-game fantasy points model and compare it to simple baselines.

Usage:
    python -m src.train
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, REPORTS_DIR, TEST_SEASON, TRAIN_SEASONS
from src.data import load_all
from src.features import FEATURES, TARGET, build_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    df = build_features(load_all())
    # Skip each player's first game of a season-run: no history to predict from
    df = df[df["fp_last10"].notna()]

    train = df[df["SEASON"].isin(TRAIN_SEASONS)]
    test = df[df["SEASON"] == TEST_SEASON]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Train: {len(train):,} player-games | Test ({TEST_SEASON}): {len(test):,}")

    results = {}

    # Baselines: what a casual fantasy player might do by eye
    results["Baseline: last 10 games avg"] = mean_absolute_error(y_test, test["fp_last10"])
    season_avg = test["fp_season"].fillna(test["fp_prev_season"]).fillna(test["fp_last10"])
    results["Baseline: season avg"] = mean_absolute_error(y_test, season_avg)

    ridge = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
    ridge.fit(X_train, y_train)
    results["Ridge regression"] = mean_absolute_error(y_test, ridge.predict(X_test))

    gbm = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        loss="absolute_error", random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    results["Gradient boosting"] = mean_absolute_error(y_test, gbm_pred)

    print("\nMean absolute error per game (lower is better):")
    for name, mae in results.items():
        print(f"  {name:<30} {mae:6.2f} FP")

    # Retrain the best model on everything (incl. test season) for live use
    final = HistGradientBoostingRegressor(**gbm.get_params())
    full = df[df["SEASON"].isin(TRAIN_SEASONS + [TEST_SEASON])]
    final.fit(full[FEATURES], full[TARGET])

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODELS_DIR / "fp_model.joblib")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    _plot(test.assign(pred=gbm_pred))
    print(f"\nSaved model to {MODELS_DIR / 'fp_model.joblib'} and charts to {REPORTS_DIR}/")


def _plot(test: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hexbin(test["pred"], test[TARGET], gridsize=50, cmap="Blues", mincnt=1)
    lim = [test[TARGET].min(), test[TARGET].quantile(0.999)]
    ax.plot(lim, lim, color="#d9480f", lw=1)
    ax.set(xlabel="Predicted FP", ylabel="Actual FP", title=f"Per-game predictions, {TEST_SEASON}")

    # Weekly view: does the model rank players correctly within each week?
    test = test.assign(week=test["GAME_DATE"].dt.to_period("W-SUN"))
    weekly = test.groupby(["week", "PLAYER_ID"]).agg(pred=("pred", "sum"), actual=(TARGET, "sum"))
    corr = weekly.groupby("week").apply(lambda w: w["pred"].corr(w["actual"], method="spearman"))
    ax = axes[1]
    ax.plot(corr.index.to_timestamp(), corr.values, marker="o", color="#1c7ed6")
    ax.axhline(np.nanmean(corr.values), ls="--", color="gray")
    ax.set(ylim=(0, 1), ylabel="Rank correlation", title="Weekly ranking accuracy (1.0 = perfect)")
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "evaluation.png", dpi=120)
    print(f"Weekly ranking correlation (avg): {np.nanmean(corr.values):.3f}")


if __name__ == "__main__":
    main()
