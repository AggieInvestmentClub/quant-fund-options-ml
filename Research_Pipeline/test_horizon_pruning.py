import json
import numpy as np
import pandas as pd
import pytest

from .horizon_pruning import fit_pruned, verify_control, require_same_candidates
from .horizon_data import horizon_panel
from .test_horizon_strategy import fixture


def test_pruned_horizon_fit_ignores_future_values_and_preserves_schema(tmp_path, monkeypatch):
    from . import horizon_pruning as hp
    p, prices = fixture()
    h, schedule = horizon_panel(p, prices, 4, min_known=2)
    h['ret_20'] = 2*h.ret_5
    monkeypatch.setitem(hp.FEATURES, 'full', ['ret_5', 'ret_20'])
    schedule = schedule[schedule.period.isin(h.period)].tail(1)
    a = fit_pruned(h, schedule, tmp_path/'first', {'source':'test'}, min_weeks=1, diagnostics=False)
    altered = h.copy()
    future = altered.label_available_at.ge(schedule.decision_at.iloc[0])
    altered.loc[future, 'target_relative'] = 99999.
    # Scoring rows stay unchanged; only unavailable later feature observations change.
    future_features = altered.decision_at.gt(schedule.decision_at.iloc[0])
    altered.loc[future_features, ['ret_5','ret_20']] = 99999.
    b = fit_pruned(altered, schedule, tmp_path/'second', {'source':'test'}, min_weeks=1, diagnostics=False)
    np.testing.assert_array_equal(a.score, b.score)
    window = json.loads((tmp_path/'first/training_windows.json').read_text())[0]
    assert len(window['columns']) == 1
    assert len(window['removed']) == 1
    assert pd.Timestamp(window['latest_outcome']) < schedule.decision_at.iloc[0]
    repeat = fit_pruned(h, schedule, tmp_path/'first', {'source':'test'}, min_weeks=1, diagnostics=False)
    pd.testing.assert_frame_equal(a, repeat)
    with pytest.raises(ValueError, match='cache mismatch'):
        fit_pruned(h, schedule, tmp_path/'first', {'source':'changed'}, min_weeks=1, diagnostics=False)


def test_control_gate_rejects_changed_scores_targets_and_keys():
    f = pd.DataFrame(dict(period=['a','a'],security_id=['A','B'],score=[.1,.2],target=[.01,np.nan]))
    assert verify_control(f,f)['max_prediction_error'] == 0
    for column in ('score','target'):
        changed=f.copy();changed.loc[0,column] += .01
        with pytest.raises(AssertionError):verify_control(f,changed)
    with pytest.raises(AssertionError):verify_control(f,f.iloc[:1])


def test_candidates_and_unknown_outcomes_cannot_change_after_pruning():
    f = pd.DataFrame(dict(period=['a','a'],security_id=['A','B'],market_cap=[10.,20.],
                          target=[np.nan,.1],entry_date=pd.to_datetime(['2026-05-11']*2),
                          exit_date=pd.to_datetime(['2026-06-08']*2),score=[1.,2.]))
    changed=f.iloc[::-1].copy();changed['score']=0.
    require_same_candidates(f,changed)
    with pytest.raises(AssertionError):require_same_candidates(f,changed.iloc[:1])
    changed.loc[0,'target']=0.
    with pytest.raises(AssertionError):require_same_candidates(f,changed)
