"""Paper specification: 19 inputs, monthly normalization, and expanding refits."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .data import PAPER_CONTROLS, PAPER_OPTIONS
from .models import CLASS_IDS, _training_rows


def paper_feature_sets() -> dict[str, list[str]]:
    """Table 3: B plus six option groups for both, call, and put sides."""
    groups = {"B": list(PAPER_CONTROLS)}
    for side_name, sides in (("cp", ("call", "put")), ("c", ("call",)), ("p", ("put",))):
        for group_name, metrics in (
            ("SDGTV", ("iv", "delta", "gamma", "theta", "vega")),
            ("S", ("iv",)), ("D", ("delta",)), ("G", ("gamma",)),
            ("T", ("theta",)), ("V", ("vega",)),
        ):
            groups[f"{group_name}_{side_name}"] = PAPER_CONTROLS + [
                f"avg_{side}_{metric}" for metric in metrics for side in sides
            ]
    return groups


def normalize_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Use only same-month cross sections; fill missing values with that month's median."""
    result = panel.copy()
    for column in PAPER_CONTROLS + PAPER_OPTIONS:
        if column not in result:
            raise ValueError(f"missing paper feature {column}")
        values = pd.to_numeric(result[column], errors="coerce")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"infinite paper feature {column}")
        groups = values.groupby(result.period)
        filled = values.fillna(groups.transform("median"))
        if filled.isna().any():
            raise ValueError(f"{column}: no values in at least one month")
        mean = filled.groupby(result.period).transform("mean")
        std = filled.groupby(result.period).transform(lambda part: part.std(ddof=0))
        result[column] = ((filled - mean) / std.replace(0, 1)).astype(float)
    return result


def _fit_paper(kind: str, train: pd.DataFrame, cols: list[str], seed: int):
    y = train.label.map(CLASS_IDS).to_numpy(dtype=int)
    if len(set(y)) != 3:
        raise ValueError("paper training window must contain all three classes")
    if kind == "logistic":
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(train[cols], y)
        return model, None
    if kind != "lightgbm":
        raise ValueError(kind)
    periods = sorted(train.period.unique())
    split = max(1, min(len(periods) - 1, int(np.floor(0.8 * len(periods)))))
    if split >= len(periods):
        raise ValueError("LightGBM needs at least two training months")
    fit = train[train.period.isin(periods[:split])]
    validation = train[train.period.isin(periods[split:])]
    if len(set(fit.label)) != 3:
        raise ValueError("LightGBM fitting window must contain all three classes")
    model = lgb.LGBMClassifier(
        objective="multiclass", class_weight="balanced", random_state=seed,
        n_jobs=1, verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")
        model.fit(
            fit[cols], fit.label.map(CLASS_IDS).to_numpy(dtype=int),
            eval_set=[(validation[cols], validation.label.map(CLASS_IDS).to_numpy(dtype=int))],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(5, verbose=False)],
        )
    return model, {"fit_months": split, "validation_months": len(periods) - split,
                   "best_iteration": int(model.best_iteration_)}


def train_paper(
    panel: pd.DataFrame, *, first_test: str = "2006-01", last_test: str = "2022-12",
    label_policy: str = "point_in_time", min_train_months: int = 120, seed: int = 17,
    specifications: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit each month. The paper-stated schedule is explicitly hindsight-only."""
    if label_policy not in {"point_in_time", "paper_stated"}:
        raise ValueError("label_policy must be point_in_time or paper_stated")
    if min_train_months < 1:
        raise ValueError("min_train_months must be positive")
    prepared = normalize_by_month(panel)
    all_specs = paper_feature_sets()
    specs = specifications or list(all_specs)
    unknown = set(specs) - set(all_specs)
    if unknown:
        raise ValueError(f"unknown paper specifications: {sorted(unknown)}")
    start, end = pd.Period(first_test), pd.Period(last_test)
    if end < start:
        raise ValueError("last_test precedes first_test")
    test = prepared[prepared.period.between(start, end) & prepared.label.notna()]
    months = sorted(test.period.unique())
    if not months:
        raise ValueError("no labeled test months in requested paper window")
    predictions, windows = [], []
    for period in months:
        score = test[test.period.eq(period)]
        cutoff = score.rebalance_close_at.max()
        if label_policy == "point_in_time":
            train = _training_rows(prepared, cutoff)
        else:
            train = prepared[prepared.period.lt(period) & prepared.label.notna()].copy()
        count = train.period.nunique()
        if count < min_train_months:
            raise ValueError(f"{period}: only {count} training months; need {min_train_months}")
        windows.append({"period": str(period), "training_months": int(count),
                        "latest_training_period": str(train.period.max())})
        for specification in specs:
            cols = all_specs[specification]
            for kind in ("logistic", "lightgbm"):
                model, validation = _fit_paper(kind, train, cols, seed)
                probabilities = model.predict_proba(score[cols])
                scored = score[["period", "trade_date", "rebalance_date", "security_id",
                                "market_cap", "sp500_member", "label", "forward_log_return"]].copy()
                scored["feature_set"] = specification
                scored["model"] = kind
                for name, index in CLASS_IDS.items():
                    scored[f"p_{name}"] = probabilities[:, list(model.classes_).index(index)]
                predictions.append(scored)
                if validation is not None:
                    windows[-1].setdefault("lightgbm", {})[specification] = validation
    report = {
        "experiment": "paper_table_3", "label_policy": label_policy,
        "hindsight_training_labels": label_policy == "paper_stated",
        "first_test_month": str(months[0]), "last_test_month": str(months[-1]),
        "test_months": len(months), "specifications": {name: all_specs[name] for name in specs},
        "candidate_models": {f"{name}/{kind}": {"features": all_specs[name]}
                             for name in specs for kind in ("logistic", "lightgbm")},
        "training_windows": windows,
    }
    return pd.concat(predictions, ignore_index=True), report
