"""Chronological model selection and monthly refitting."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .data import PAPER_CONTROLS, PAPER_OPTIONS, SKEW_FEATURES


CLASS_NAMES = ["crash", "normal", "jump"]
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def feature_sets(panel: pd.DataFrame) -> dict[str, list[str]]:
    groups = {"options_only": PAPER_OPTIONS}
    has_controls = panel[PAPER_CONTROLS].notna().any().all()
    if has_controls:
        groups["stock_only"] = PAPER_CONTROLS
        groups["combined"] = PAPER_CONTROLS + PAPER_OPTIONS
    extra = [col for col in SKEW_FEATURES if col in panel and panel[col].notna().any()]
    if extra:
        name = "combined_plus_skew" if has_controls else "options_plus_skew"
        groups[name] = (PAPER_CONTROLS if has_controls else []) + PAPER_OPTIONS + extra
    return groups


def candidates(kind: str) -> list[dict]:
    if kind == "logistic":
        return [{"C": c} for c in (0.1, 1.0, 10.0)]
    if kind == "lightgbm":
        return [{"num_leaves": leaves, "learning_rate": rate}
                for leaves in (15, 31) for rate in (0.03, 0.1)]
    if kind == "mlp":
        return [{"hidden_layer_sizes": (32, 16), "alpha": 0.001}]
    raise ValueError(kind)


def make_model(kind: str, params: dict, seed: int) -> Pipeline:
    if kind == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=params["C"], class_weight="balanced", max_iter=1000, random_state=seed)),
        ])
    if kind == "lightgbm":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", lgb.LGBMClassifier(
                objective="multiclass", class_weight="balanced", n_estimators=200,
                num_leaves=params["num_leaves"], learning_rate=params["learning_rate"],
                n_jobs=1, random_state=seed, verbosity=-1,
            )),
        ])
    if kind == "mlp":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"], alpha=params["alpha"],
                learning_rate_init=0.001, max_iter=150, random_state=seed,
                early_stopping=False,
            )),
        ])
    raise ValueError(kind)


def _fit(model: Pipeline, kind: str, x: pd.DataFrame, y: np.ndarray) -> Pipeline:
    if len(set(y)) != 3:
        raise ValueError("training sample must contain crash, normal and jump labels")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if kind == "mlp":
            model.fit(x, y, model__sample_weight=compute_sample_weight("balanced", y))
        else:
            model.fit(x, y)
    return model


def _probs(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    result = np.zeros((len(x), 3), dtype=float)
    for position, class_id in enumerate(model.named_steps["model"].classes_):
        result[:, int(class_id)] = probabilities[:, position]
    return result


def _training_rows(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    eligible = panel.label.notna() & panel.label_available_at.le(cutoff) & panel.rebalance_close_at.lt(cutoff)
    return panel.loc[eligible].copy()


def train_and_score(
    panel: pd.DataFrame, *, test_months: int = 24, min_train_months: int = 120,
    validation_months: int = 24, seed: int = 17,
) -> tuple[pd.DataFrame, dict]:
    labeled_months = sorted(panel.loc[panel.label.notna(), "period"].unique())
    if len(labeled_months) < test_months:
        raise ValueError(f"only {len(labeled_months)} labeled months; need {test_months} test months")
    test_periods = labeled_months[-test_months:]
    first_test = panel.loc[panel.period.eq(test_periods[0])]
    first_cutoff = first_test.rebalance_close_at.max()
    initial_train = _training_rows(panel, first_cutoff)
    mature_months = sorted(initial_train.period.unique())
    if len(mature_months) < min_train_months:
        raise ValueError(
            f"only {len(mature_months)} matured training months before {test_periods[0]}; "
            f"need {min_train_months}"
        )
    groups = feature_sets(panel)
    variants = [(name, kind) for name in groups for kind in ("logistic", "lightgbm")]
    extended = "combined_plus_skew" if "combined_plus_skew" in groups else "options_plus_skew"
    if extended in groups:
        variants.append((extended, "mlp"))
    selections = {}
    for feature_name, kind in variants:
        columns = groups[feature_name]
        if not initial_train[columns].notna().any().all():
            raise ValueError(f"{feature_name}: at least one feature has no training observations")
        val_periods = set(mature_months[-min(validation_months, max(1, len(mature_months) // 5)):])
        fit_rows = initial_train[~initial_train.period.isin(val_periods)]
        val_rows = initial_train[initial_train.period.isin(val_periods)]
        if len(set(fit_rows.label)) != 3:
            raise ValueError(f"{feature_name}/{kind}: fitting window must contain all three labels")
        fit_y = fit_rows.label.map(CLASS_IDS).to_numpy(dtype=int)
        val_y = val_rows.label.map(CLASS_IDS).to_numpy(dtype=int)
        scores = []
        for params in candidates(kind):
            model = _fit(make_model(kind, params, seed), kind, fit_rows[columns], fit_y)
            score = log_loss(val_y, _probs(model, val_rows[columns]), labels=[0, 1, 2])
            scores.append((float(score), params))
        best_score, best_params = min(scores, key=lambda item: item[0])
        selections[f"{feature_name}/{kind}"] = {
            "features": columns, "params": best_params,
            "validation_log_loss": best_score,
            "initial_mature_training_months": len(mature_months),
        }
    prediction_frames = []
    for period in test_periods:
        score_rows = panel[panel.period.eq(period)].copy()
        cutoff = score_rows.rebalance_close_at.max()
        train_rows = _training_rows(panel, cutoff)
        if len(train_rows.period.unique()) < min_train_months:
            raise ValueError(f"{period}: too few matured training months")
        y = train_rows.label.map(CLASS_IDS).to_numpy(dtype=int)
        for name, kind in variants:
            specification = selections[f"{name}/{kind}"]
            columns = specification["features"]
            model = _fit(make_model(kind, specification["params"], seed), kind, train_rows[columns], y)
            probs = _probs(model, score_rows[columns])
            scored = score_rows[[
                "period", "trade_date", "rebalance_date", "security_id", "market_cap",
                "sp500_member", "label", "forward_log_return",
            ]].copy()
            scored["feature_set"] = name
            scored["model"] = kind
            scored["p_crash"] = probs[:, 0]
            scored["p_normal"] = probs[:, 1]
            scored["p_jump"] = probs[:, 2]
            prediction_frames.append(scored)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selection_report = {
        "first_test_month": str(test_periods[0]),
        "last_test_month": str(test_periods[-1]),
        "test_months": len(test_periods),
        "candidate_models": selections,
    }
    return predictions, selection_report


def prediction_metrics(predictions: pd.DataFrame) -> dict:
    result = {}
    for (group, kind, universe), frame in _metric_groups(predictions):
        labeled = frame[frame.label.notna()]
        row = {"rows": len(labeled)}
        for label_name, probability in (("jump", "p_jump"), ("crash", "p_crash")):
            truth = labeled.label.eq(label_name).astype(int)
            row[f"{label_name}_accuracy_ratio"] = (
                float(2 * roc_auc_score(truth, labeled[probability]) - 1)
                if truth.nunique() == 2 else None
            )
            top = pd.concat([
                month.sort_values([probability, "security_id"], ascending=[False, True]).head(
                    max(1, int(np.ceil(len(month) * 0.05)))
                )
                for _, month in labeled.groupby("period")
            ], ignore_index=True) if len(labeled) else labeled
            ppv = float(top.label.eq(label_name).mean()) if len(top) else None
            row[f"{label_name}_ppv_top5"] = ppv
            row[f"{label_name}_fdr_top5"] = 1 - ppv if ppv is not None else None
        result[f"{group}/{kind}/{universe}"] = row
    return result


def _metric_groups(predictions: pd.DataFrame):
    for (group, kind), frame in predictions.groupby(["feature_set", "model"]):
        yield (group, kind, "full"), frame
        sp500 = frame[frame.sp500_member.fillna(False)]
        if frame.sp500_member.notna().all() and sp500.period.nunique() == frame.period.nunique():
            yield (group, kind, "sp500"), sp500
