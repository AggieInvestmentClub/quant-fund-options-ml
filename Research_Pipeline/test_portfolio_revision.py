from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from .daily_ledger import ledger,metrics,select_weights,InvalidPortfolio
from .portfolio_ledger import PortfolioSpec,CostSpec,order_costs,prepare_replay
from .reevaluate import audit,simulation
from .test_strategy import fixture


def test_equal_weights_caps_ties_and_reduced_exposure():
    ids=np.array([f'S{i:03d}' for i in range(200)])
    s=np.ones(200);cap=np.arange(1,201,dtype=float)
    w=select_weights(s,cap,ids,portfolio=PortfolioSpec('equal'))
    assert len(np.flatnonzero(w))==20
    assert w.max()==pytest.approx(.05);assert w.min()==pytest.approx(-.05)
    assert w[w>0].sum()==pytest.approx(.5);assert w[w<0].sum()==pytest.approx(-.5)
    assert np.array_equal(w,select_weights(s,cap,ids,portfolio=PortfolioSpec('equal')))
    w=select_weights(s,cap,ids,crash=s,portfolio=PortfolioSpec('equal'))
    assert np.sum(w>0)==10 and np.sum(w<0)==10
    ids=np.array([f'S{i:03d}' for i in range(500)]);s=np.arange(500.)
    w=select_weights(s,np.ones(500),ids,portfolio=PortfolioSpec('equal'))
    assert w[w>0].sum()==pytest.approx(1);assert max(abs(w))==pytest.approx(.04)


def test_commission_minimum_cap_and_no_trade():
    c=CostSpec(mode='ibkr_fixed',execution_bps=1)
    fees=order_costs(np.array([100000.,100.,2.,0.,-100000.]),np.array([100.,100.,.2,np.nan,100.]),c)
    assert fees['commission']==pytest.approx([5,1,.02,0,5])
    assert fees['execution'][0]==pytest.approx(10)
    assert fees['regulatory'][-1]==pytest.approx(2.06+.003)
    assert fees['regulatory'][0]==pytest.approx(.003)
    with pytest.raises(InvalidPortfolio,match='raw execution'):
        order_costs(np.array([1.]),np.array([np.nan]),c)


def test_dollar_ledger_reconciliation_and_capture_parity():
    pred,prices=fixture();prices['close']=prices.adjusted_close*2
    ps=prepare_replay(pred,prices);p=PortfolioSpec('equal');c=CostSpec(mode='ibkr_fixed',execution_bps=1)
    r=ledger(ps,portfolio=p,costs=c,initial_capital=100000.)
    audit(r);fast=ledger(ps,portfolio=p,costs=c,initial_capital=100000.,capture=False)
    assert r['final_equity']==pytest.approx(fast['final_equity'],abs=1e-8)
    assert r['period_returns']==pytest.approx(fast['period_returns'])
    assert all(r['holdings'].weight.abs()<=.05)
    t=r['trades'];assert np.allclose(t.share_equivalent,t.notional/t.raw_close)
    assert t.event.eq('liquidation').any()
    assert metrics(r,52)['cumulative_return']==pytest.approx(r['final_equity']/100000-1)
    # Changed adjusted-unit scale must not change share-cost estimates or returns.
    altered=prices.copy();altered.adjusted_close*=10
    other=ledger(prepare_replay(pred,altered),portfolio=p,costs=c,initial_capital=100000.)
    assert other['final_equity']==pytest.approx(r['final_equity'],abs=1e-7)
    larger=ledger(ps,portfolio=p,costs=c,initial_capital=1000000.)
    assert larger['total_transaction_cost']/1000000<r['total_transaction_cost']/100000


def test_raw_prices_required_only_for_actual_broker_orders():
    pred,prices=fixture();prices['close']=prices.adjusted_close
    prices.loc[prices.security_id.eq('B'),'close']=np.nan
    p=PortfolioSpec('equal');c=CostSpec(mode='ibkr_fixed')
    ledger(prepare_replay(pred,prices),portfolio=p,costs=c,initial_capital=100000.)
    prices.loc[prices.security_id.eq('A') & prices.trade_date.eq(prices.trade_date.min()),'close']=np.nan
    with pytest.raises(InvalidPortfolio,match='raw execution'):
        ledger(prepare_replay(pred,prices),portfolio=p,costs=c,initial_capital=100000.)
    ledger(prepare_replay(pred,prices),portfolio=p,costs=CostSpec(),initial_capital=100000.)


def test_reversal_is_one_net_order_and_borrow_reconciles():
    pred,prices=fixture();prices['close']=prices.adjusted_close
    second=pred.period.eq('1');pred.loc[second,'score']=-pred.loc[second,'score']
    r=ledger(prepare_replay(pred,prices),portfolio=PortfolioSpec('equal'),costs=CostSpec(mode='ibkr_fixed'),initial_capital=100000.)
    audit(r);tr=r['trades'];reversal=tr[tr.period.eq('1') & tr.event.eq('rebalance') & tr.security_id.eq('A')]
    assert len(reversal)==1 and reversal.signed_notional.iloc[0]<-9000
    assert r['total_borrow']>0


def test_missing_selected_daily_prices_remain_invalid():
    pred,prices=fixture();prices['close']=prices.adjusted_close
    prices.loc[prices.security_id.eq('A') & prices.trade_date.eq(prices.trade_date.unique()[1]),'adjusted_close']=np.nan
    with pytest.raises(InvalidPortfolio,match='selected daily'):
        ledger(prepare_replay(pred,prices),portfolio=PortfolioSpec('equal'),costs=CostSpec(),initial_capital=100000.)


def test_simulation_reproducibility_and_identical_gross_net_permutations():
    pred,prices=fixture();prices['close']=prices.adjusted_close
    # Six monthly intervals provide a valid three-period bootstrap fixture.
    days=pd.bdate_range('2024-01-02',periods=13);ids=['A','B','C','D']
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100+i,close=100+i) for i,d in enumerate(days) for s in ids])
    pred=pd.DataFrame([dict(period=str(k),security_id=s,market_cap=100.,entry_date=days[2*k],exit_date=days[2*k+2],score=j) for k in range(6) for j,s in enumerate(ids)])
    ps=prepare_replay(pred,prices);p=PortfolioSpec('equal');c=CostSpec(flat_bps=0,borrow_rate=0)
    observed=ledger(ps,portfolio=p,costs=c,initial_capital=100000.)
    a,draws,boot=simulation(ps,observed,p,c,100000.,12,10,17,4)
    b,other,boot2=simulation(ps,observed,p,c,100000.,12,10,17,4)
    assert a==b;pd.testing.assert_frame_equal(draws,other);pd.testing.assert_frame_equal(boot,boot2)
    assert np.allclose(draws.net_mean_return,draws.gross_mean_return)


def test_legacy_proportional_path_scales_and_daily_drift_is_reported():
    from .daily_ledger import prepare
    pred,prices=fixture();prices['close']=prices.adjusted_close
    old=ledger(prepare(pred,prices))
    new=ledger(prepare_replay(pred,prices),portfolio=PortfolioSpec(),costs=CostSpec(),initial_capital=100000.)
    assert new['final_equity']/100000==pytest.approx(old['final_equity'],abs=1e-10)
    assert new['daily'].equity.to_numpy()/100000==pytest.approx(old['daily'].equity.to_numpy(),abs=1e-10)
    audit(new)
    revised=ledger(prepare_replay(pred,prices),portfolio=PortfolioSpec('equal'),costs=CostSpec(flat_bps=0,borrow_rate=0),initial_capital=100000.)
    assert revised['daily'].cap_breach.any()
    assert len(revised['holdings'])==4  # no extra daily cap trades


def test_short_loss_insolvency_fails_instead_of_reporting_performance():
    pred,prices=fixture();prices['close']=prices.adjusted_close
    prices.loc[prices.security_id.eq('D') & prices.trade_date.gt(prices.trade_date.min()),'adjusted_close']=10000
    with pytest.raises(InvalidPortfolio,match='insolvent'):
        ledger(prepare_replay(pred,prices),portfolio=PortfolioSpec('equal'),costs=CostSpec(flat_bps=0),initial_capital=100000.)


def test_synthetic_reevaluation_end_to_end(tmp_path):
    import json
    from .daily_ledger import prepare
    from .reevaluate import run,sha
    root=tmp_path/'source';root.mkdir();days=pd.bdate_range('2024-01-02',periods=13);ids=['A','B','C','D']
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100+i*(j+1)*.1,close=100+i*(j+1)*.1) for i,d in enumerate(days) for j,s in enumerate(ids)])
    pred=pd.DataFrame([dict(period=str(k),security_id=s,market_cap=100.,entry_date=days[2*k],exit_date=days[2*k+2],score=float(j),target=float(j)*.01,variant='stock_only/ridge') for k in range(6) for j,s in enumerate(ids)])
    price_path=root/'stock_prices.parquet';prices.to_parquet(price_path,index=False)
    hashes={'handoff/stock_prices.parquet':sha(price_path)}
    for track,freq in [('weekly',52),('paper_diagnostic',12)]:
        d=root/track;d.mkdir();pred.to_parquet(d/'predictions.parquet',index=False);hashes[track+'/predictions.parquet']=sha(d/'predictions.parquet')
        original=ledger(prepare(pred,prices));(d/'metrics.json').write_text(json.dumps({'stock_only/ridge':{'primary':metrics(original,freq)}}))
        for name in ['holdings','daily']:
            frame=original[name].copy();frame['variant']='stock_only/ridge';frame.to_parquet(d/('daily_equity.parquet' if name=='daily' else name+'.parquet'),index=False)
    audit_dir=root/'final_audit';audit_dir.mkdir();(audit_dir/'artifact_hashes.json').write_text(json.dumps(hashes))
    out=tmp_path/'replay';run(root,price_path,out,{'iterations':4},max_cores=1)
    assert json.loads((out/'source_verification.json').read_text())['hashes_unchanged']
    summary=pd.read_csv(out/'comparison.csv');assert len(summary)==22 and summary.status.eq('complete').all()
    assert (out/'weekly/stock_only_ridge/random_draws.parquet').exists()
