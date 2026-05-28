"""Naive baselines: last-value (h-1) and seasonal-naive (h-24).

These set the floor every smarter model has to beat. If a fancy neural net
can't outperform "today's hour = yesterday's hour at the same time" it's
not doing useful work.
"""

from __future__ import annotations

import sys

from eval_utils import (
    evaluate_predictions,
    get_train_targets_by_building,
    load_features_and_splits,
    save_metrics,
    save_predictions,
    split_mask,
)


def run(model_name: str, lag_column: str) -> None:
    features, split = load_features_and_splits()
    test_mask = split_mask(features, split, "test")
    test = features[test_mask]

    preds = (
        test.reset_index()[["building_id", "timestamp", "consumption", lag_column]]
        .rename(columns={"consumption": "y_true", lag_column: "y_pred"})
    )

    y_train_by_building = get_train_targets_by_building(features, split)
    metrics = evaluate_predictions(preds, y_train_by_building)

    save_predictions(model_name, preds)
    save_metrics(model_name, metrics)

    print(f"\n=== {model_name} ===")
    print(f"macro MAE:  {metrics['macro']['mae']:.2f}")
    print(f"macro RMSE: {metrics['macro']['rmse']:.2f}")
    print(f"macro MAPE: {metrics['macro']['mape']:.2f}%")
    print(f"macro R2:   {metrics['macro']['r2']:.3f}")
    print(f"peak F1:    {metrics['macro']['peak_f1']:.3f}")


def main() -> int:
    run("naive_last", "lag_1")
    run("naive_seasonal_24h", "lag_24")
    return 0


if __name__ == "__main__":
    sys.exit(main())
