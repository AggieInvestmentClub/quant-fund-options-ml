"""Chronology, fixed allocation, paired inference and accounting regressions."""
import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from .daily_ledger import select_weights,ledger,InvalidPortfolio
from .portfolio_ledger import prepare_replay
from .fixed_universe_forecast import (portfolio_spec,score_metrics,assert_common,fit_variant,
                                      fixed_rows,stock_baselines,evaluate,model_pipeline)
from .fixed_forecast_statistics import hold_blocks,forecast_values,infer
from .test_passive_benchmarks import fixture
from .horizon_strategy import NET,weekly_boundaries
from .reevaluate import audit

def test_slots_use_each_forecast_without_ranking_or_redistribution():
    ids=np.array([f'S{i}' for i in range(25)]);caps=np.ones(25);scores=np.r_[np.ones(10),np.zeros(5),-np.ones(10)]
    w=select_weights(scores,caps,ids,portfolio=portfolio_spec())
    np.testing.assert_array_equal(w[:10],.04);assert w.sum()==pytest.approx(.4);assert not w[10:].any()
    altered=scores.copy();altered[:10]*=100
    np.testing.assert_array_equal(w,select_weights(altered,caps,ids,portfolio=portfolio_spec()))
    passive=select_weights(np.ones(25),caps,ids,portfolio=portfolio_spec())
    assert passive.sum()==pytest.approx(1.)
    matched=select_weights(scores,caps,ids,portfolio=portfolio_spec(comparison=True))
    np.testing.assert_allclose(matched,.4/25)
    assert not select_weights(scores,caps,ids,portfolio=portfolio_spec(1.)).any()
    with pytest.raises(ValueError):select_weights(np.ones(26),np.ones(26),np.arange(26),portfolio=portfolio_spec())
    with pytest.raises(ValueError):portfolio_spec(np.nan)

def test_cash_and_positive_controls_drift_costs_and_capture_parity(tmp_path):
    _,pred,prices,_=fixture();spec=portfolio_spec();bounds=weekly_boundaries('2026-05-11','2026-08-03')
    for score in (0.,-1.,1.):
        f=pred.copy();f['score']=score;ps=prepare_replay(f,prices)
        a=ledger(ps,portfolio=spec,costs=NET,initial_capital=100000.)
        b=ledger(ps,portfolio=spec,costs=NET,initial_capital=100000.,capture=False)
        audit(a);assert a['equity_path']==b['equity_path']
        if score<=0:assert a['final_equity']==100000 and not a['total_transaction_cost'] and a['holdings'].empty
        else:
            assert a['holdings'].weight.eq(.04).all()
            assert 'liquidation'in set(a['trades'].event)
            assert not a['daily'].max_abs_weight.eq(.04).all() # daily price drift is retained
            from .passive_benchmarks import case_spec
            control=ledger(ps,portfolio=replace(case_spec('fixed25_four_week'),max_position=.04),costs=NET,initial_capital=100000.)
            assert a['equity_path']==control['equity_path']
    f=pred.copy();f['score']=1.
    bad=prices.copy();bad.loc[bad.security_id.eq('S00')&bad.trade_date.eq(pd.Timestamp('2026-05-12')),'adjusted_close']=np.nan
    result,_=evaluate(f,bad,tmp_path/'bad',bounds)
    assert result['net']['status']=='invalid'
    # Bad marks for a never-held stock cannot invalidate an all-cash portfolio.
    f['score']=0.;assert evaluate(f,bad,tmp_path/'cash',bounds)[0]['net']['status']=='complete'

def forecast_fixture():
    _,pred,prices,ids=fixture()
    pred['baseline_mean']=.001;pred['baseline_up_probability']=.6
    pred['trade_date']=pred.entry_date-pd.Timedelta(days=3);pred['missing_input_count']=0
    pred['variant']='full/ridge';pred['score']=pred.target*.5
    other=pred.copy();other['variant']='pruned/ridge';other['score']=-pred.target
    return pd.concat([pred,other],ignore_index=True),prices

def test_forecast_masks_metrics_and_common_keys_do_not_depend_on_model():
    f,_=forecast_fixture();assert_common(f)
    pm=score_metrics(f)
    assert pm[pm.variant.eq('full/ridge')].direction_accuracy.eq(1).all()
    assert pm[pm.variant.eq('pruned/ridge')].direction_accuracy.eq(0).all()
    lost=f.copy();lost.loc[lost.security_id.isin(['S00','S01','S02','S03','S04','S05']),'target']=np.nan
    assert score_metrics(lost).mse.isna().all()
    assert len(lost)==len(f) # future missingness never removes candidates
    changed=f.copy();changed.loc[0,'target']+=.1
    with pytest.raises(AssertionError):assert_common(changed)
    with pytest.raises(ValueError):fixed_rows(f,'p0',[f'S{i:02}' for i in range(25)])

def test_training_baseline_fallback_and_maturity_pruning(tmp_path,monkeypatch):
    from .test_horizon_strategy import fixture as history
    from .horizon_data import horizon_panel
    from . import fixed_universe_forecast as module
    p,prices=history();panel,sched=horizon_panel(p,prices,4,min_known=2)
    panel['ret_20']=-panel.ret_5
    monkeypatch.setitem(module.FEATURES,'full',['ret_5','ret_20'])
    ids=panel.security_id.unique().tolist();sched=sched[sched.period.isin(panel.period)].tail(1)
    a=fit_variant(panel,sched,ids,'pruned/ridge',{},tmp_path/'a',{'source':'a'},min_weeks=1)
    changed=panel.copy();changed.loc[changed.label_available_at.ge(sched.decision_at.iloc[0]),'target_absolute']=999.
    changed.loc[changed.decision_at.gt(sched.decision_at.iloc[0]),'ret_5']=999.
    b=fit_variant(changed,sched,ids,'pruned/ridge',{},tmp_path/'b',{'source':'a'},min_weeks=1)
    np.testing.assert_array_equal(a.score,b.score)
    saved=json.loads((tmp_path/'a/training_windows.json').read_text())[0]
    assert pd.Timestamp(saved['latest_outcome'])<sched.decision_at.iloc[0]
    assert len(saved['columns'])==1
    with pytest.raises(ValueError,match='incompatible'):
        fit_variant(panel,sched,ids,'pruned/ridge',{},tmp_path/'a',{'source':'b'},min_weeks=1)
    baseline=stock_baselines(pd.DataFrame(dict(security_id=['A','A'],target=[.1,-.1])),pd.DataFrame(dict(security_id=['A','B'])))
    assert baseline.baseline_fallback.tolist()==[False,True]
    assert baseline.baseline_mean.eq(0).all()

def test_shared_monte_carlo_and_block_bootstrap_and_costed_synthetic_end_to_end(tmp_path):
    pred,prices=forecast_fixture();bounds=weekly_boundaries('2026-05-11','2026-08-03')
    from .fixed_universe_forecast import window_backtests
    window_backtests(pred,prices,tmp_path/'portfolios',bounds)
    a=infer(pred,prices,tmp_path,bounds,iterations=4,cores=1)
    b=infer(pred,prices,tmp_path,bounds,iterations=4,cores=1)
    assert a==b
    blocks=hold_blocks(6,20)
    np.testing.assert_array_equal(blocks,hold_blocks(6,20))
    assert (blocks[:,1::2]-blocks[:,::2]==1).all()
    frame=pd.read_parquet(tmp_path/'inference/full_ridge/random_draws.parquet')
    assert frame.net_failure.isna().all() and frame.net_turnover.ge(0).all()
    assert a['full/ridge']['mse_improvement']['p_raw']>=1/5
    prepared=prepare_replay(pred[pred.variant.eq('full/ridge')],prices)
    lookup={p:(g.sort_values('security_id').target.to_numpy(),g.sort_values('security_id').baseline_mean.to_numpy())
            for p,g in pred[pred.variant.eq('full/ridge')].groupby('period')}
    observed=forecast_values(prepared,lookup)
    assert observed['direction_accuracy']==1.

def test_all_missing_features_keep_schema_and_deterministic_fit():
    x=pd.DataFrame({'x':[1.,2.,3.,4.],'missing':[np.nan]*4});y=np.array([.1,.2,-.1,-.2])
    a=model_pipeline('ridge').fit(x,y);b=model_pipeline('ridge').fit(x,y)
    np.testing.assert_array_equal(a.predict(x),b.predict(x))
    assert a.feature_names_in_.tolist()==['x','missing']


def test_report_rejects_changed_frozen_source_before_replaying_benchmark(tmp_path):
    from .fixed_forecast_report import build_report
    from .reevaluate import dump,sha
    source=tmp_path/'source';source.write_text('frozen')
    dump(tmp_path/'manifest.json',dict(paths={'spy':str(source)},sources={'spy':sha(source)}))
    folder=tmp_path/'historical/inference';folder.mkdir(parents=True)
    dump(folder/'statistics.json',{})
    source.write_text('changed')
    with pytest.raises(ValueError,match='report source hash changed'):
        build_report(tmp_path)


def test_small_synthetic_complete_command_and_latest_scoring(tmp_path,monkeypatch):
    from . import fixed_universe_forecast as module
    from .test_feature_experiments import panel_fixture
    from .horizon_data import horizon_panel,window_schedule
    from .passive_benchmarks import sessions
    from .reevaluate import sha,dump
    p=panel_fixture(80,25);p['security_id']='ticker_proxy:'+p.security_id
    ids=sorted(p.security_id.unique())
    days=sessions('2024-01-02','2025-08-15')
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,close=50*np.exp(.0001*i+.01*np.sin(i+j)),
                             adjusted_close=100*np.exp(.0001*i+.01*np.sin(i+j)))
                          for i,d in enumerate(days) for j,s in enumerate(ids)])
    windows={'development':('2025-02-03','2025-03-31'),'historical':('2025-03-31','2025-05-27')}
    monkeypatch.setattr(module,'WINDOWS',windows)
    p.to_parquet(tmp_path/'panel.parquet',index=False);prices.to_parquet(tmp_path/'prices.parquet',index=False)
    spy=prices[prices.security_id.eq(ids[0])].copy();spy['security_id']='benchmark:SPY';spy['ticker']='SPY'
    spy.to_parquet(tmp_path/'spy.parquet',index=False)
    config=tmp_path/'config.py';config.write_text('STOCK_UNIVERSE = '+repr([s.split(':')[1] for s in ids]))
    h,master=horizon_panel(p,prices,4);reference=tmp_path/'reference';reference.mkdir()
    for w,dates in windows.items():
        sched=window_schedule(master,4,*dates)
        for g in ('stock_only','full'):
            for m in ('ridge','lightgbm'):
                module.fit_variant(h,sched,ids,g+'/'+m,{},reference/w/'fits'/f'h4_absolute_{g}_{m}',{'synthetic':'reference'})
    dump(reference/'manifest.json',dict(sources=dict(panel=sha(tmp_path/'panel.parquet'),prices=sha(tmp_path/'prices.parquet'))))
    out=tmp_path/'results'
    module.run(tmp_path/'panel.parquet',tmp_path/'prices.parquet',reference,config,out,
               spy=tmp_path/'spy.parquet',cores=1,iterations=2)
    assert json.loads((out/'complete.json').read_text())['status']=='complete'
    assert all(json.loads((out/'source_verification.json').read_text()).values())
    assert len(pd.read_parquet(out/'historical/predictions.parquet'))==6*25*2
    assert len(pd.read_csv(out/'latest/forecasts.csv'))==25
    spy_metrics=json.loads((out/'historical/portfolios/spy/metrics.json').read_text())
    assert spy_metrics['net']['status']=='complete'
    assert spy_metrics['net']['transaction_cost']>0
    assert spy_metrics['gross']['transaction_cost']==0
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['engineering_pilot']
    # Resume guards cannot silently run different capital/data configurations.
    with pytest.raises(ValueError,match='incompatible experiment resume'):
        module.run(tmp_path/'panel.parquet',tmp_path/'prices.parquet',reference,config,out,
                   spy=tmp_path/'spy.parquet',cores=1,iterations=3,stage='audit')
