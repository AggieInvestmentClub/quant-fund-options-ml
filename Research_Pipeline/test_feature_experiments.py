import json
import numpy as np
import pandas as pd
import pytest
from .feature_experiments import (features,training,schedules,fit_scores,reproduce,prediction_metrics,
    assert_common_keys,predict_and_evaluate,simulations,ranked_correlation,GROUPS,VARIANTS)


def panel_fixture(weeks=64,n=220):
    rng=np.random.default_rng(4);dates=pd.date_range('2024-01-05',periods=weeks,freq='W-FRI');rows=[]
    for i,d in enumerate(dates):
        entry=d+pd.Timedelta(days=3);exit=entry+pd.Timedelta(days=7);decision=entry.tz_localize('UTC')+pd.Timedelta(hours=16)
        for j in range(n):
            values={c:float(rng.normal(.2,.03)) for c in GROUPS['full']}
            values['iv_gap']=values['avg_call_iv']-values['avg_put_iv'];values['vol_20']=.02
            rows.append(dict(**values,security_id=f'S{j:03d}',period=str(d.to_period('W-FRI')),trade_date=d,
                entry_date=entry,exit_date=exit,decision_at=decision,available_at=decision-pd.Timedelta(hours=1),
                label_available_at=exit.tz_localize('UTC')+pd.Timedelta(hours=20),market_cap=float(1e8+j*1e6),
                target=float(.01*np.sin(i+j))))
    return pd.DataFrame(rows)


def test_exact_week_lags_holidays_missing_weeks_and_security_isolation():
    p=panel_fixture(6,2)
    # July 3 is the last exchange session of the July 4 holiday week.
    dates=pd.date_range('2025-06-06',periods=6,freq='W-FRI')
    p['trade_date']=[dates[i//2]-pd.Timedelta(days=1) if dates[i//2]==pd.Timestamp('2025-07-04') else dates[i//2] for i in range(len(p))]
    p['period']=p.trade_date.dt.to_period('W-FRI').astype(str)
    p['entry_date']=p.trade_date.dt.to_period('W-FRI').dt.end_time.dt.normalize()+pd.Timedelta(days=3)
    p['exit_date']=p.entry_date+pd.Timedelta(days=7)
    p['decision_at']=p.entry_date.dt.tz_localize('UTC')+pd.Timedelta(hours=16)
    p['available_at']=p.decision_at-pd.Timedelta(hours=1)
    p['label_available_at']=p.exit_date.dt.tz_localize('UTC')+pd.Timedelta(hours=20)
    p['avg_call_iv']=np.arange(len(p))+.5;p['avg_put_iv']=.5;p['iv_gap']=p.avg_call_iv-p.avg_put_iv
    p=p.drop(index=4)  # Missing scheduled week for S000, not for S001.
    f=features(p);a=f[f.security_id.eq('S000')].set_index('period');b=f[f.security_id.eq('S001')].set_index('period')
    week3=str(dates[3].to_period('W-FRI'));week4=str(dates[4].to_period('W-FRI'))
    assert pd.isna(a.loc[week3,'iv_gap_change_1w'])
    assert b.loc[week3,'iv_gap_change_1w']==pytest.approx(2.)
    assert a.loc[week4,'iv_gap_change_4w']==pytest.approx(8.)
    assert len(f)==len(p) and f.index.equals(p.index)
    assert f.iv_realized_variance_proxy.iloc[0]==pytest.approx(f.iv_level.iloc[0]**2-252*.02**2)


def test_features_ignore_labels_and_future_changes():
    p=panel_fixture(6,2);before=features(p);alter=p.copy();alter.target=999
    alter.loc[alter.trade_date.eq(alter.trade_date.max()),'avg_call_iv']=100
    alter['iv_gap']=alter.avg_call_iv-alter.avg_put_iv;after=features(alter)
    cols=['iv_level','iv_gap_change_1w','iv_gap_change_4w','rr60_change_1w','iv_realized_variance_proxy']
    mask=p.trade_date.lt(p.trade_date.max());pd.testing.assert_frame_equal(before.loc[mask,cols],after.loc[mask,cols])
    late=p.copy();late.loc[0,'available_at']=late.loc[0,'decision_at']+pd.Timedelta(seconds=1)
    with pytest.raises(ValueError,match='available after'):features(late)
    with pytest.raises(ValueError,match='duplicate'):features(pd.concat([p,p.iloc[:1]]))


def test_label_maturity_strict_before_decision():
    p=panel_fixture();decision=p.decision_at.max();tr=training(p,decision)
    assert (tr.label_available_at<decision).all()
    edge=p.copy();edge.loc[0,'label_available_at']=decision
    assert 0 not in training(edge,decision).index


def test_predefined_schedules_and_sizes():
    from .strategy_data import schedule
    s=schedule('2024-04-05','2026-11-06',True).rename(columns={'signal_date':'trade_date'})
    chosen=schedules(s)
    assert len(chosen['assessment'])==52 and len(chosen['historical'])==26
    assert chosen['historical'].entry_date.min()==pd.Timestamp('2026-05-11')
    assert chosen['historical'].exit_date.max()==pd.Timestamp('2026-11-09')
    assert [len(v) for v in GROUPS.values()]==[5,25,13,17]


def test_known_outcome_masks_and_quintile_ties():
    p=panel_fixture(1,220);p['variant']='full/ridge';p['score']=np.arange(len(p),dtype=float)
    p.loc[0,'target']=np.nan;m=prediction_metrics(p)
    assert m.known_outcomes.iloc[0]==219 and m.missing_outcomes.iloc[0]==1
    f=p.copy();f['variant']='compact/ridge';combined=pd.concat([p,f],ignore_index=True);assert_common_keys(combined,p)
    x=prediction_metrics(combined);assert x.rank_ic.iloc[0]==x.rank_ic.iloc[1]
    assert np.isfinite(x.quintile_spread).all()
    low=p.iloc[:199];assert np.isnan(prediction_metrics(low).rank_ic.iloc[0])
    assert np.isnan(ranked_correlation(np.ones((220,1)),p.target.to_numpy())[0])
    bad=f.copy();bad.loc[1,'target']=np.nan
    with pytest.raises(AssertionError):assert_common_keys(pd.concat([p,bad],ignore_index=True),p)


def test_fit_preprocessing_schema_and_control_roundtrip(tmp_path):
    p=features(panel_fixture());p['atm_term_slope']=np.nan
    s=p[['period','trade_date','decision_at','entry_date','exit_date','label_available_at']].drop_duplicates().iloc[-1:]
    pred,windows=fit_scores(p,s,['stock_only/ridge','full/ridge'],tmp_path,{'fixture':'frozen'})
    reference=pred.copy();reference['variant']=reference.variant.str.replace('full/','combined/',regex=False)
    checks=reproduce(pred,reference);assert max(v['max_absolute_prediction_error'] for v in checks.values())==0
    corrupted=reference.copy();corrupted.loc[0,'score']+=1e-4
    with pytest.raises(AssertionError,match='control predictions differ'):reproduce(pred,corrupted)
    import joblib
    artifact=joblib.load(next((tmp_path/'models').glob('*full_ridge.joblib')))
    assert artifact['features']==GROUPS['full']
    tr=training(p,s.decision_at.iloc[0]);imputer=artifact['model'].steps[0][1]
    assert imputer.statistics_[0]==pytest.approx(tr[GROUPS['full'][0]].median())
    assert len(pred)==440 and len(windows)==1
    assert windows[0]['latest_outcome']<windows[0]['decision_at']


def test_synthetic_feature_pipeline_backtest_and_paired_inference(tmp_path):
    p=features(panel_fixture());sched=p[['period','trade_date','decision_at','entry_date','exit_date','label_available_at']].drop_duplicates().iloc[-6:]
    out=tmp_path/'fits';pred,windows=fit_scores(p,sched,VARIANTS,out,{'fixture':'frozen'})
    assert_common_keys(pred,p[p.period.isin(sched.period)])
    dates=sorted(set(sched.entry_date)|set(sched.exit_date));ids=sorted(p.security_id.unique())
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100*(1+.002*np.sin(i+j)),close=100*(1+.002*np.sin(i+j))) for i,d in enumerate(dates) for j,s in enumerate(ids)])
    reports,observed,prepared,pm=predict_and_evaluate(pred,prices,tmp_path/'eval')
    assert len(reports)==8 and len(pm)==48
    assert all(x['portfolio']['net']['status']=='complete' for x in reports.values())
    one=tmp_path/'sims1';one.mkdir();two=tmp_path/'sims2';two.mkdir()
    a,ci=simulations(pred,prepared,observed,pm,one,iterations=4,seed=17)
    b,ci2=simulations(pred,prepared,observed,pm,two,iterations=4,seed=17)
    assert a==b and ci==ci2
    pd.testing.assert_frame_equal(pd.read_parquet(one/'random_draws.parquet'),pd.read_parquet(two/'random_draws.parquet'))
    assert all(v['family_size']==8 for v in a.values())
    assert np.array_equal(np.load(one/'bootstrap_week_indices.npy'),np.load(two/'bootstrap_week_indices.npy'))
    # Identical score rankings must receive identical donors and trade selections.
    shared=pred.copy();shared['score']=shared.security_id.str[1:].astype(float)
    same_reports,same_runs,same_prepared,same_pm=predict_and_evaluate(shared,prices,tmp_path/'same_scores')
    third=tmp_path/'same_sims';third.mkdir()
    simulations(shared,same_prepared,same_runs,same_pm,third,iterations=2,seed=17)
    draws=pd.read_parquet(third/'random_draws.parquet')
    assert draws.groupby('draw').mean_rank_ic.nunique().eq(1).all()
    assert draws.groupby('draw').net_mean_return.nunique().eq(1).all()
    # A missing selected daily price invalidates portfolios, not predictive scoring.
    sid=observed['full/ridge','net']['holdings'].security_id.iloc[0]
    prices.loc[prices.security_id.eq(sid) & prices.trade_date.eq(dates[1]),'adjusted_close']=np.nan
    altered,_,_,pm2=predict_and_evaluate(pred,prices,tmp_path/'missing')
    pd.testing.assert_frame_equal(pm,pm2)
    assert altered['full/ridge']['portfolio']['net']['status']=='invalid'
