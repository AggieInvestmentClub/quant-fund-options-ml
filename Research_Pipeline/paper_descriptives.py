"""Raw class and option-characteristic summaries for paper Table 2/4 checks."""

from __future__ import annotations

import pandas as pd

from .data import PAPER_OPTIONS


def describe_panel(panel: pd.DataFrame) -> dict:
    labeled = panel[panel.label.notna()].copy()
    if labeled.empty:
        return {"status": "unavailable: no matured 12-month labels", "labeled_rows": 0}
    counts = labeled.label.value_counts().reindex(["jump", "crash", "normal"], fill_value=0)
    option_means = {}
    for label, group in labeled.groupby("label"):
        option_means[label] = {
            name: float(group[name].mean()) if group[name].notna().any() else None
            for name in PAPER_OPTIONS
        }
    return {
        "labeled_rows": len(labeled),
        "first_observation_month": str(labeled.period.min()),
        "last_observation_month": str(labeled.period.max()),
        "class_counts": {label: int(count) for label, count in counts.items()},
        "class_proportions": {label: float(count / len(labeled)) for label, count in counts.items()},
        "raw_option_feature_means_by_future_class": option_means,
        "method": "raw stock-month mean by subsequent 12-month log-return class; no model predictions",
    }
