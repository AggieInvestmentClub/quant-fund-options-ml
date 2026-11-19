import json
from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from .daily_ledger import select_weights, ledger, InvalidPortfolio
from .portfolio_ledger import PortfolioSpec, prepare_replay
from .horizon_strategy import weekly_boundaries, GROSS, NET
from .horizon_statistics import bootstrap_indices
from .passive_benchmarks import (case_spec, monthly_intervals, latest_snapshot, fixed_model_rows,
    passive_rows, evaluate, model_inference, validate_spy, sessions, decision_at, fetch_spy)


def fixture():
    from .strategy_data import schedule
    days=sessions('2026-04-01','2026-08-10');ids=[f'S{i:02}' for i in range(25)]
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,close=50*(1+.0005*i+.005*np.sin(i+j)),
        adjusted_close=100*(1+.0005*i+.005*np.sin(i+j))) for i,d in enumerate(days) for j,s in enumerate(ids)])
    sch=schedule('2026-04-03','2026-08-07',True)
    panel=pd.DataFrame([dict(period=r.period,trade_date=r.signal_date,decision_at=r.decision_at,
        available_at=r.decision_at-pd.Timedelta(minutes=30),security_id=s,market_cap=1e9+j*1e8)
        for r in sch.itertuples() for j,s in enumerate(ids)])
    entries=pd.to_datetime(['2026-05-11','2026-06-08','2026-07-06','2026-08-03'])
    lookup=prices.set_index(['trade_date','security_id']).adjusted_close
    pred=pd.DataFrame([dict(period=f'p{i}',security_id=s,market_cap=1e9+j*1e8,
        entry_date=a,exit_date=b,score=float(np.sin(i+j)),target=float(lookup[b,s]/lookup[a,s]-1))
        for i,(a,b) in enumerate(zip(entries,entries[1:])) for j,s in enumerate(ids)])
    return panel,pred,prices,ids


def test_explicit_weights_ignore_scores_or_select_top20_with_deterministic_ties():
    ids=np.array([f'S{i:02}' for i in range(25)]);score=np.arange(25.);caps=np.ones(25)
    a=select_weights(score,caps,ids,portfolio=case_spec('fixed25_monthly'))
    b=select_weights(-score,caps,ids,portfolio=case_spec('fixed25_monthly'))
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a,.04)
    w=select_weights(score,caps,ids,portfolio=case_spec('ml25'))
    assert (w>0).sum()==20 and w.sum()==pytest.approx(1.)
    assert np.all(w[-20:]==.05) and not w[:5].any()
    tied=select_weights(np.zeros(25),caps,ids,portfolio=case_spec('ml25'))
    assert np.all(tied[:20]==.05)
    with pytest.raises(InvalidPortfolio):select_weights(score[:19],caps[:19],ids[:19],portfolio=case_spec('ml25'))
    small=select_weights(score[:10],caps[:10],ids[:10],portfolio=case_spec('fixed25_monthly'))
    assert small.sum()==pytest.approx(.5) # cap retained for generic small all-stock portfolios


@pytest.mark.parametrize('kwargs',[
    dict(selection_mode='all',mode='long_short'),dict(selection_mode='all',mode='long_only'),
    dict(selection_mode='fixed_count',mode='long_only',weighting='equal'),
    dict(selection_mode='fixed_count',mode='long_only',weighting='equal',selection_count=True),
    dict(selection_mode='ranked',selection_count=20)])
def test_invalid_explicit_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):PortfolioSpec(**kwargs)


def test_calendar_months_holidays_and_available_snapshot():
    intervals=monthly_intervals('2026-05-11','2026-10-26')
    assert [x[1] for x in intervals]==list(pd.to_datetime(['2026-05-11','2026-06-01','2026-07-01','2026-08-03','2026-09-01','2026-10-01']))
    assert intervals[-1][2]==pd.Timestamp('2026-10-26')
    panel,_,_,_=fixture();f=latest_snapshot(panel,pd.Timestamp('2026-06-01'))
    assert f.available_at.max()<decision_at('2026-06-01')
    changed=panel.copy();changed.loc[changed.decision_at.eq(f.decision_at.iloc[0]),'available_at']=decision_at('2026-06-01')+pd.Timedelta(minutes=1)
    older=latest_snapshot(changed,pd.Timestamp('2026-06-01'))
    assert older.decision_at.max()<f.decision_at.min() and len(older)==25


def test_fixed_candidates_require_all_25_and_passive_never_uses_outcomes():
    panel,pred,_,ids=fixture()
    pd.testing.assert_frame_equal(fixed_model_rows(pred,ids),pred)
    with pytest.raises(ValueError):fixed_model_rows(pred.iloc[1:],ids)
    pred.loc[0,'score']=np.nan
    with pytest.raises(ValueError):fixed_model_rows(pred,ids)
    a=passive_rows(panel,monthly_intervals('2026-05-11','2026-08-03'),ids)
    panel['target']=99999.;b=passive_rows(panel,monthly_intervals('2026-05-11','2026-08-03'),ids)
    pd.testing.assert_frame_equal(a,b);assert a.groupby('period').size().eq(25).all()
    # Missing fresh option observations must not remove a fixed passive holding.
    panel=panel[~(panel.security_id.eq(ids[0])&panel.trade_date.gt('2026-04-20'))]
    kept=passive_rows(panel,monthly_intervals('2026-05-11','2026-08-03'),ids)
    assert kept.groupby('period').size().eq(25).all()


def test_synthetic_ledger_and_simulations_reconcile_and_reproduce(tmp_path):
    panel,pred,prices,ids=fixture();bounds=weekly_boundaries('2026-05-11','2026-08-03')
    for name,p in [('ml25',pred),('fixed25_monthly',passive_rows(panel,monthly_intervals('2026-05-11','2026-08-03'),ids))]:
        result=evaluate(p,prices,name,tmp_path/name,bounds)
        assert result['net']['status']=='complete'
        trades=pd.read_parquet(tmp_path/name/'net_trades.parquet')
        assert 'liquidation' in set(trades.event)
        assert trades.raw_close.lt(60).all() # commission raw prices, not the adjusted units near 100
        holding=pd.read_parquet(tmp_path/name/'net_holdings.parquet')
        assert holding.groupby('period').size().eq(20 if name=='ml25' else 25).all()
    observed=json.loads((tmp_path/'ml25/metrics.json').read_text());idx=bootstrap_indices(12,4)
    a=model_inference('ml25',pred,prices,tmp_path/'stats1',bounds,observed,idx,4)
    b=model_inference('ml25',pred,prices,tmp_path/'stats2',bounds,observed,idx,4)
    assert a==b
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'stats1/random_draws.parquet'),pd.read_parquet(tmp_path/'stats2/random_draws.parquet'))
    selected=pd.read_parquet(tmp_path/'ml25/net_holdings.parquet').iloc[0]
    bad=prices.copy();bad.loc[bad.security_id.eq(selected.security_id)&bad.trade_date.eq(selected.entry_date+pd.Timedelta(days=1)),'adjusted_close']=np.nan
    assert evaluate(pred,bad,'ml25',tmp_path/'bad',bounds)['net']['status']=='invalid'


def test_spy_gross_is_adjusted_total_return_without_double_counting_dividends():
    _,_,prices,_=fixture();p=prices[prices.security_id.eq('S00')].copy();p['security_id']='benchmark:SPY';p['ticker']='SPY';p['dividend']=100.
    p=validate_spy(p,'2026-05-11','2026-08-03')
    candidates=pd.DataFrame([dict(period='hold',entry_date=p.trade_date.min(),exit_date=p.trade_date.max(),
                                 security_id='benchmark:SPY',market_cap=1.,score=0.)])
    r=ledger(prepare_replay(candidates,p),portfolio=case_spec('spy'),costs=GROSS,initial_capital=100000.)
    assert r['final_equity']/100000-1==pytest.approx(p.adjusted_close.iloc[-1]/p.adjusted_close.iloc[0]-1)
    assert r['holdings'].weight.eq(1).all()
    with pytest.raises(ValueError):validate_spy(pd.concat([p,p.iloc[:1]]),'2026-05-11','2026-08-03')
    p.loc[1,'close']=np.nan
    with pytest.raises(ValueError):validate_spy(p,'2026-05-11','2026-08-03')


def test_failed_spy_fetch_is_recorded_without_other_downloads(tmp_path,monkeypatch):
    calls=[]
    def ticker(name):
        calls.append(name)
        def fail(**kwargs):raise TimeoutError('provider unavailable')
        return SimpleNamespace(history=fail)
    monkeypatch.setitem(sys.modules,'yfinance',SimpleNamespace(Ticker=ticker))
    fetch_spy(tmp_path/'fetch')
    assert calls==['SPY']
    assert json.loads((tmp_path/'fetch/metadata.json').read_text())['status']=='unavailable'
    assert not (tmp_path/'fetch/spy_prices.parquet').exists()


def test_complete_synthetic_command_keeps_source_hashes_and_missing_spy(tmp_path,monkeypatch):
    from . import passive_benchmarks as module
    from .reevaluate import sha, dump
    panel,pred,prices,ids=fixture()
    monkeypatch.setattr(module,'WINDOWS',{'development':('2026-05-11','2026-08-03'),
                                        'historical':('2026-05-11','2026-08-03')})
    monkeypatch.setattr(module,'configured_universe',lambda p:ids)
    panel.to_parquet(tmp_path/'panel.parquet',index=False);prices.to_parquet(tmp_path/'prices.parquet',index=False)
    reference=tmp_path/'reference';reference.mkdir()
    for window in module.WINDOWS:
        f=reference/window/'fits/h4_relative_full_ridge';f.mkdir(parents=True)
        pred.to_parquet(f/'predictions.parquet',index=False)
    control=evaluate(pred,prices,'ml_broad',tmp_path/'control',weekly_boundaries('2026-05-11','2026-08-03'))
    (reference/'historical/rules').mkdir()
    dump(reference/'historical/rules/metrics.json',{'ml_buffer/long_only':control})
    dump(reference/'manifest.json',dict(sources=dict(panel=sha(tmp_path/'panel.parquet'),prices=sha(tmp_path/'prices.parquet'))))
    module.run(tmp_path/'panel.parquet',tmp_path/'prices.parquet',reference,tmp_path/'results',cores=1,iterations=2)
    out=tmp_path/'results';assert json.loads((out/'complete.json').read_text())['status']=='complete'
    assert all(json.loads((out/'source_verification.json').read_text()).values())
    assert json.loads((out/'manifest.json').read_text())['engineering_pilot']
    comparison=json.loads((out/'historical/paired_comparisons.json').read_text())
    assert comparison['primary_ml25']['status']=='complete'
    assert comparison['ml25_vs_spy']['status'].startswith('unavailable')
