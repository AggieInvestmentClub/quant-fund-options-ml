from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from .feature_diagnostics import prune,diagnostic_validation,accumulated_effects,permutation_importance
from .sensitivity_data import reject_future_columns,strict_stock_features,legacy_slopes,GROUPS
from .data_feature_sensitivity import fit_variant,portfolio_sensitivities,model_sensitivities
from .portfolio_ledger import PortfolioSpec,CostSpec,prepare_replay
from .daily_ledger import select_weights,ledger,InvalidPortfolio
from .reevaluate import audit
from .test_feature_experiments import panel_fixture
from .feature_experiments import features


def test_pruning_coverage_constant_and_stable_correlation():
    rng=np.random.default_rng(3);n=300
    f=pd.DataFrame({'a':rng.normal(size=n),'b':rng.normal(size=n),'c':np.nan,'constant':1.})
    f['duplicate']=f.a*2;f.loc[:160,'b']=np.nan
    kept,removed=prune(f,list(f))
    assert kept==['a'] and 'duplicate' in removed and 'constant' in removed and 'b' in removed
    small=f.iloc[:100].drop(columns=['b','constant','c'])
    assert len(prune(small,list(small))[0])==2 # insufficient paired support cannot prune
    with pytest.raises(ValueError):prune(f,['c'])


def test_future_target_and_selection_flags_cannot_enter_features():
    for name in ['forward_5d_return','target_tradable_5d_return','jump_label_5d','is_target_valid_5d','is_options_investable']:
        with pytest.raises(ValueError):reject_future_columns(['avg_call_iv',name])
    reject_future_columns([c for cs in GROUPS.values() for c in cs])


def test_session_lookbacks_do_not_bridge_missing_prices_or_stocks():
    import exchange_calendars as xc
    dates=xc.get_calendar('XNYS',start='2023-01-01',end='2025-01-01').sessions.tz_localize(None)[:280]
    p=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100+i,volume=1000.) for s in ['A','B'] for i,d in enumerate(dates)])
    p=p[~(p.security_id.eq('A')&p.trade_date.eq(dates[-20]))]
    f=strict_stock_features(p);a=f[f.security_id.eq('A')].iloc[-1];b=f[f.security_id.eq('B')].iloc[-1]
    assert pd.isna(a.ret_63) and pd.isna(a.rv_90) and pd.notna(b.ret_252)
    assert b.volume_ratio_20==1


def test_sparse_filled_slopes_are_not_treated_as_original_endpoints():
    raw=pd.DataFrame({'atm_iv_90d':[np.nan,.4],'atm_iv_30d':[.3,.3],
        'risk_reversal_25_delta_90d':[.1,.2],'risk_reversal_25_delta_30d':[np.nan,.1],
        'atm_term_slope':[.2,.7],'rr_term_slope':[.4,.6]})
    f=legacy_slopes(raw)
    assert pd.isna(f.atm_term_slope.iloc[0]) and f.atm_term_slope.iloc[1]==pytest.approx(.1)
    assert pd.isna(f.rr_term_slope.iloc[0])


def test_exposure_selection_standalone_legs_and_cost_split():
    ids=np.array([f'S{i:03}' for i in range(500)]);s=np.arange(500.);caps=np.ones(500)
    p=PortfolioSpec('equal',selection_fraction=.1,side_exposure=.5)
    w=select_weights(s,caps,ids,portfolio=p)
    assert (w>0).sum()==50 and w[w>0].sum()==pytest.approx(.5)
    for mode in ['long_only','short_only']:
        x=select_weights(s,caps,ids,portfolio=replace(p,mode=mode))
        assert not ((x<0).any() and (x>0).any())
    from .test_strategy import fixture
    pred,prices=fixture();prices['close']=prices.adjusted_close
    pred.loc[pred.period.eq('1'),'score']*=-1
    for mode in ['long_short','long_only','short_only']:
        r=ledger(prepare_replay(pred,prices),portfolio=PortfolioSpec('equal',mode=mode),costs=CostSpec(mode='ibkr_fixed'),initial_capital=100000)
        audit(r);t=r['trades']
        assert np.allclose(t.long_transaction_cost+t.short_transaction_cost,t.transaction_cost)
        if mode=='long_only':assert r['total_borrow']==0 and t.short_traded_notional.sum()==0
        if mode=='short_only':assert t.long_traded_notional.sum()==0


def test_new_fits_future_invariance_cache_and_fixed_sensitivity_sets(tmp_path):
    p=features(panel_fixture());sched=p[['period','trade_date','decision_at','entry_date','exit_date','label_available_at']].drop_duplicates().iloc[-1:]
    a,windows=fit_variant(p,sched,'pruned_full/ridge',tmp_path,{'source':'test'})
    altered=p.copy();altered.loc[altered.label_available_at.ge(sched.decision_at.iloc[0]),'target']=9999
    b,_=fit_variant(altered,sched,'pruned_full/ridge',tmp_path/'other',{'source':'test'})
    assert np.allclose(a.score,b.score)
    reused,_=fit_variant(p,sched,'pruned_full/ridge',tmp_path,{'source':'test'})
    pd.testing.assert_frame_equal(a,reused)
    assert len(model_sensitivities())==12 and len(portfolio_sensitivities())==13
    fit,val,status=diagnostic_validation(p)
    assert fit.label_available_at.max()<val.decision_at.min()


def test_synthetic_pruned_fit_local_effects_and_importance(tmp_path):
    p=features(panel_fixture());tr=p[p.period.isin(sorted(p.period.unique())[:52])];val=p[p.period.isin(sorted(p.period.unique())[54:])]
    from .strategy_experiments import pipeline
    cols=['ret_5','ret_20'];m=pipeline('ridge').fit(tr[cols],tr.target)
    a=permutation_importance(m,tr,val,cols,repetitions=2);b=permutation_importance(m,tr,val,cols,repetitions=2)
    pd.testing.assert_frame_equal(a,b)
    effects=accumulated_effects(m,tr,cols)
    assert effects.support.sum()==len(tr)*len(cols) and np.isfinite(effects.centered_effect).all()


def test_twelve_variant_synthetic_end_to_end_and_paired_simulations(tmp_path):
    from .data_feature_sensitivity import fit_variants,MAIN,PRIMARY,REFERENCE,replay_cases
    from .feature_experiments import predict_and_evaluate,simulations
    p=features(panel_fixture());rng=np.random.default_rng(9)
    for c in GROUPS['pruned_extended']:
        if c not in p:p[c]=rng.normal(size=len(p))
    sched=p[['period','trade_date','decision_at','entry_date','exit_date','label_available_at']].drop_duplicates().iloc[-4:]
    pred=fit_variants(p,sched,MAIN,tmp_path/'fits',{'fixture':'frozen'},1)
    dates=sorted(set(sched.entry_date)|set(sched.exit_date));ids=sorted(p.security_id.unique())
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100*(1+.002*np.sin(i+j)),close=100*(1+.002*np.sin(i+j))) for i,d in enumerate(dates) for j,s in enumerate(ids)])
    reports,runs,prepared,pm=predict_and_evaluate(pred,prices,tmp_path/'eval')
    assert len(reports)==12 and all(m['portfolio']['net']['status']=='complete' for m in reports.values())
    sim=tmp_path/'sim';sim.mkdir()
    stats,ci=simulations(pred,prepared,runs,pm,sim,iterations=2,primary=PRIMARY,reference=REFERENCE)
    assert all(x['family_size']==12 for x in stats.values())
    replay=replay_cases(pred[pred.variant.isin([PRIMARY,REFERENCE])],prices,tmp_path/'replay')
    assert len(replay)==26 and all(x['status']=='complete' for x in replay.values())
    prices.loc[prices.security_id.eq(runs[PRIMARY,'net']['holdings'].security_id.iloc[0]),'close']=np.nan
    bad=replay_cases(pred[pred.variant.eq(PRIMARY)],prices,tmp_path/'bad',{'main':portfolio_sensitivities()['main']})
    assert bad[PRIMARY+'/main']['status']=='invalid'
