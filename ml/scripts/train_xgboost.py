"""XGBoost regressor over all 3 buildings, one-step-ahead.

building_id is one-hot encoded so a single model handles all three scales.
Early-stopping on the val split picks the best iteration; final predictions
are on the held-out test split.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from eval_utils import (
    evaluate_predictions,
    feature_columns,
    get_train_targets_by_building,
    load_features_and_splits,
    save_metrics,
    save_predictions,
    split_mask,
)

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "models" / "results"


def build_design_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = features.reset_index()
    dummies = pd.get_dummies(df["building_id"], prefix="bld", dtype="int8")
    df = pd.concat([df, dummies], axis=1)
    cols = feature_columns(features) + list(dummies.columns)
    return df, cols


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    features, split = load_features_and_splits()
    df, cols = build_design_matrix(features)

    train_idx = split_mask(features, split, "train").to_numpy()
    val_idx = split_mask(features, split, "val").to_numpy()
    test_idx = split_mask(features, split, "test").to_numpy()

    X_train, y_train = df.loc[train_idx, cols], df.loc[train_idx, "consumption"]
    X_val, y_val = df.loc[val_idx, cols], df.loc[val_idx, "consumption"]
    X_test, y_test = df.loc[test_idx, cols], df.loc[test_idx, "consumption"]

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="mae",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"best iteration: {model.best_iteration}, best val MAE: {model.best_score:.3f}")

    test_meta = df.loc[test_idx, ["building_id", "timestamp"]]
    preds_df = pd.DataFrame({
        "building_id": test_meta["building_id"].values,
        "timestamp": test_meta["timestamp"].values,
        "y_true": y_test.values,
        "y_pred": model.predict(X_test),
    })

    y_train_by_building = get_train_targets_by_building(features, split)
    metrics = evaluate_predictions(preds_df, y_train_by_building)

    save_predictions("xgboost", preds_df)
    save_metrics("xgboost", metrics)

    with (CHECKPOINT_DIR / "xgboost.pkl").open("wb") as f:
        pickle.dump({"model": model, "feature_cols": cols}, f)

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    importances.plot(kind="barh", ax=ax, color="#3a6ea5")
    ax.set_title("XGBoost — top 20 feature importances")
    ax.set_xlabel("gain-weighted importance")
    fig.tight_layout()
    (RESULTS_DIR / "feature_importance").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "feature_importance" / "xgboost.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== xgboost ===")
    print(f"macro MAE:  {metrics['macro']['mae']:.2f}")
    print(f"macro RMSE: {metrics['macro']['rmse']:.2f}")
    print(f"macro MAPE: {metrics['macro']['mape']:.2f}%")
    print(f"macro R2:   {metrics['macro']['r2']:.3f}")
    print(f"peak F1:    {metrics['macro']['peak_f1']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
