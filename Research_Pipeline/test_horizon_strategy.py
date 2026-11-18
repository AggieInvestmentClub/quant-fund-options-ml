from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from .portfolio_ledger import PortfolioSpec,SelectionPolicy,prepare_replay
from .daily_ledger import select_weights,ledger,InvalidPortfolio
from .horizon_data import horizon_panel,window_schedule,matured_training,momentum_predictions
from .horizon_strategy import (POLICIES,NET,GROSS,weekly_boundaries,weekly_equity,
    fit_configuration,choose_configuration,performance,portfolio_case)
from .horizon_statistics import shuffled_periods,bootstrap_indices,bounded_p,simulate_case
from .reevaluate import audit


def fixture(n=400):
    from .test_feature_experiments import panel_fixture
    from .feature_experiments import features
    p=features(panel_fixture());p['ret_63']=p.ret_20
    # Use the full exchange calendar, not business-day approximations, for valuation.
    import exchange_calendars as xc
    dates=xc.get_calendar('XNYS',start=p.trade_date.min()-pd.Timedelta(days=5),end=p.exit_date.max()+pd.Timedelta(days=50)).sessions.tz_localize(None)
    ids=p.security_id.unique()
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100*(1+.001*np.sin(i+j)),close=100*(1+.001*np.sin(i+j))) for i,d in enumerate(dates) for j,s in enumerate(ids)])
    return p,prices


def test_horizon_calendar_labels_relative_centering_and_maturity():
    p,prices=fixture();h,master=horizon_panel(p,prices,4,min_known=2)
    first=h.iloc[0];week=master[master.period.eq(first.period)].iloc[0]
    assert first.exit_date==week.exit_date
    assert h.groupby('period').target_relative.mean().abs().max()<1e-12
    tr=matured_training(h,h.decision_at.max(),'relative',min_weeks=1)
    assert tr.label_available_at.max()<h.decision_at.max()
    missing=prices[~(prices.security_id.eq(first.security_id)&prices.trade_date.eq(first.exit_date))]
    other,_=horizon_panel(p,missing,4,min_known=2)
    a=other[(other.period==first.period)&(other.security_id==first.security_id)]
    assert len(a)==1 and a.target_absolute.isna().all() # never changes candidate keys
    assert h.exit_date.max()>p.exit_date.max()


def test_holiday_entry_schedule_uses_same_week_not_next_observation():
    from .strategy_data import schedule
    s=schedule('2025-01-01','2025-04-01',True)
    s['exit_date']=s.entry_date.shift(-2)
    result=window_schedule(s,2,'2025-01-20','2025-02-18')
    assert result.entry_date.iloc[0]==pd.Timestamp('2025-01-21')
    assert result.exit_date.iloc[-1]==pd.Timestamp('2025-02-18')
    # MLK day Monday correctly maps to Tuesday; the next two-week anchor stays calendar based.
    first=s[s.signal_date.eq(pd.Timestamp('2025-01-17'))].iloc[0]
    assert first.entry_date==pd.Timestamp('2025-01-21')
    dates=weekly_boundaries('2026-05-11','2026-10-26')
    assert len(dates)==25 and pd.Timestamp('2026-05-26') in dates


def test_retention_strict_entries_filters_and_static_caps():
    ids=np.array([f'S{i:03}' for i in range(400)]);scores=np.arange(400.);caps=np.ones(400)
    baseline=PortfolioSpec('equal',selection=POLICIES['baseline'])
    a=select_weights(scores,caps,ids,portfolio=baseline)
    assert np.sum(a>0)==20 and a[a>0].sum()==pytest.approx(1)
    strict=select_weights(scores,caps,ids,portfolio=replace(baseline,selection=POLICIES['entry']))
    assert np.sum(strict>0)==10 and strict[strict>0].sum()==pytest.approx(.5)
    previous=np.zeros(400);previous[365:385]=1 # ranks 16..35, some beyond entry range
    b=select_weights(scores,caps,ids,portfolio=replace(baseline,selection=POLICIES['buffer']),previous=previous)
    assert np.all(b[365:385]>0) and np.all(b[385:]==0)
    allowed=np.ones(400,bool);allowed[365]=False
    c=select_weights(scores,caps,ids,portfolio=replace(baseline,selection=POLICIES['buffer']),previous=previous,allow_long=allowed)
    assert c[365]==0 and c[399]>0
    d=select_weights(scores,caps,ids,portfolio=baseline,allow_long=np.zeros(400,bool),allow_short=np.zeros(400,bool))
    assert not d.any()
    assert np.array_equal(a,select_weights(scores,caps,ids,portfolio=baseline))
    assert not (set(np.flatnonzero(b>0))&set(np.flatnonzero(b<0)))


def test_momentum_confirmations_keep_rows_and_missing_does_not_impute():
    p,prices=fixture();h,s=horizon_panel(p,prices,1,min_known=2);s=s[s.period.isin(h.period)].tail(2)
    p=h.copy();p['ret_63']=.1;p['iv_gap']=np.linspace(-1,1,len(p))
    p.loc[p.security_id.eq(p.security_id.iloc[0]),'put_skew_10_25_30d']=np.nan
    plain=momentum_predictions(p,s,'plain',1);filtered=momentum_predictions(p,s,'gap_skew',1)
    assert len(plain)==len(filtered)
    assert plain.allow_long.all() and not plain.allow_short.any()
    assert not filtered.loc[filtered.security_id.eq(p.security_id.iloc[0]),'allow_long'].any()


def test_empty_portfolio_and_capture_parity_and_all_costs(tmp_path):
    from .test_strategy import fixture as tiny
    pred,prices=tiny();prices['close']=prices.adjusted_close
    pred['allow_long']=False;pred['allow_short']=False
    ps=prepare_replay(pred,prices);spec=PortfolioSpec('equal',selection=POLICIES['both'])
    r=ledger(ps,portfolio=spec,costs=NET,initial_capital=100000.)
    audit(r);assert r['final_equity']==100000 and r['total_transaction_cost']==0
    fast=ledger(ps,portfolio=spec,costs=NET,initial_capital=100000.,capture=False)
    assert r['equity_path']==fast['equity_path']


def test_shared_permutations_and_conservative_missing_draws():
    from .test_strategy import fixture as tiny
    pred,prices=tiny();prices['close']=prices.adjusted_close
    pred['allow_long']=pred.score>0;pred['allow_short']=pred.score<0
    ps=prepare_replay(pred,prices)
    a=shuffled_periods(ps,7);b=shuffled_periods(ps,7)
    for x,y in zip(a,b):
        assert np.array_equal(x['score'],y['score'])
        assert np.array_equal(x['allow_long'],x['score']>0)
    indices=bootstrap_indices(24,20)
    assert np.array_equal(indices,bootstrap_indices(24,20))
    assert bounded_p([1,np.nan,3],2,3,24)['p_bounds']==[.5,.75]
    assert bounded_p([1,np.nan,3],2,3,24)['p_raw'] is None


def test_changed_future_labels_do_not_change_fit(tmp_path):
    p,prices=fixture();h,s=horizon_panel(p,prices,2,min_known=2);s=s[s.period.isin(h.period)].tail(1)
    a=fit_configuration(h,s,'h2/relative/full/ridge',tmp_path/'a',{'fixture':'1'},min_weeks=1,diagnostics=False)
    other=h.copy();other.loc[other.label_available_at.ge(s.decision_at.iloc[0]),'target_relative']=999
    b=fit_configuration(other,s,'h2/relative/full/ridge',tmp_path/'b',{'fixture':'1'},min_weeks=1,diagnostics=False)
    assert a.score.to_numpy()==pytest.approx(b.score.to_numpy(),abs=1e-12)
    with pytest.raises(ValueError):fit_configuration(h,s,'h2/relative/full/ridge',tmp_path/'a',{'fixture':'changed'},min_weeks=1,diagnostics=False)


def test_selection_never_uses_historical_results():
    reports={'a':{'net':{'status':'complete','cumulative_return':.1,'sharpe':1.,'mean_weekly_turnover':1}},
             'b':{'net':{'status':'invalid'}}}
    rank={'a':pd.DataFrame({'rank_ic':[.01,.02]}),'b':pd.DataFrame({'rank_ic':[.5]})}
    assert choose_configuration(reports,rank)['variant']=='a'
    reports['a']['net']['cumulative_return']=-.1
    assert 'no qualifying' in choose_configuration(reports,rank)['status']


def test_synthetic_end_to_end_portfolio_and_statistics(tmp_path):
    p,prices=fixture();h,s=horizon_panel(p,prices,2,min_known=2)
    # Eight weekly marks, four two-week holding intervals for a small engineering pilot.
    start=s[s.period.isin(h.period)].entry_date.iloc[-12];end=start+pd.Timedelta(weeks=8)
    sched=window_schedule(s,2,start,end);bounds=weekly_boundaries(start,end)
    pred=fit_configuration(h,sched,'h2/absolute/stock_only/ridge',tmp_path/'fit',{'fixture':'1'},min_weeks=1,diagnostics=False)
    reports,runs,ps=portfolio_case(pred,prices,tmp_path/'ledger',boundaries=bounds)
    assert reports['net']['status']=='complete'
    assert len(weekly_equity(runs['net'],bounds))==8
    job=('stage1/h2/absolute/stock_only/ridge',dict(policy='both',mode='long_short'),pred,prices,
         str(tmp_path/'stats'),2,24,bounds,bootstrap_indices(8,2))
    simulate_case(job)
    import json
    stats=json.loads((tmp_path/'stats/statistics.json').read_text())
    assert stats['iterations']==2 and (tmp_path/'stats/random_draws.parquet').exists()
    # Missing selected prices still fail; the filter never sees future-price availability.
    held=runs['net']['holdings'].iloc[0]
    broken=prices.copy();broken.loc[broken.security_id.eq(held.security_id)&broken.trade_date.eq(bounds[1]),'adjusted_close']=np.nan
    bad,_,_=portfolio_case(pred,broken,tmp_path/'bad',boundaries=bounds)
    assert bad['net']['status']=='invalid'
