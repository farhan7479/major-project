"""ARIMA(2,1,2) per building, 1-step-ahead rolling forecast on the test set.

Each building gets its own model since consumption scales differ by ~3x.
Parameters are estimated on the train split only; predictions on the test
split use `dynamic=False` so each forecast at time t conditions on the
actual y[1..t-1] (not on previous forecasts) — true 1-step-ahead.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from eval_utils import (
    evaluate_predictions,
    get_train_targets_by_building,
    load_features_and_splits,
    save_metrics,
    save_predictions,
)

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
ORDER = (2, 1, 2)
SEASONAL_ORDER = (0, 0, 0, 0)


def fit_and_predict(train_y: pd.Series, full_y: pd.Series, test_start: pd.Timestamp) -> pd.Series:
    model = SARIMAX(
        train_y,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    extended = res.apply(full_y, refit=False)
    preds = extended.predict(start=test_start, end=full_y.index[-1], dynamic=False)
    return preds, res


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    features, split = load_features_and_splits()
    test_ts = split[split == "test"].index

    all_preds = []
    for bid, g in features.groupby(level="building_id"):
        series = g["consumption"].droplevel("building_id").sort_index()
        train_y = series.loc[split[split == "train"].index]
        test_y = series.loc[test_ts]
        full_y = series.loc[split.index]
        test_start = test_y.index[0]

        print(f"fitting SARIMAX{ORDER}x{SEASONAL_ORDER} on {bid} (n_train={len(train_y)})...")
        preds, fitted = fit_and_predict(train_y, full_y, test_start)

        with (CHECKPOINT_DIR / f"arima_{bid}.pkl").open("wb") as f:
            pickle.dump(fitted, f)

        df = pd.DataFrame({
            "building_id": bid,
            "timestamp": preds.index,
            "y_true": test_y.values,
            "y_pred": preds.values,
        })
        all_preds.append(df)

    preds_df = pd.concat(all_preds, ignore_index=True)
    y_train_by_building = get_train_targets_by_building(features, split)
    metrics = evaluate_predictions(preds_df, y_train_by_building)

    save_predictions("arima", preds_df)
    save_metrics("arima", metrics)

    print(f"\n=== arima ===")
    print(f"macro MAE:  {metrics['macro']['mae']:.2f}")
    print(f"macro RMSE: {metrics['macro']['rmse']:.2f}")
    print(f"macro MAPE: {metrics['macro']['mape']:.2f}%")
    print(f"macro R2:   {metrics['macro']['r2']:.3f}")
    print(f"peak F1:    {metrics['macro']['peak_f1']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
