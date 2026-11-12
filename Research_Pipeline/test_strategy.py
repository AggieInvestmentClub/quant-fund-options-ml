import numpy as np
import pandas as pd
import pytest
from Research_Pipeline.daily_ledger import prepare,ledger,InvalidPortfolio
from Research_Pipeline.strategy_data import schedule

def fixture():
    dates=pd.bdate_range('2024-01-02',periods=5);ids=['A','B','C','D']
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100*(1+.01*i) if s=='A' else 100) for i,d in enumerate(dates) for s in ids])
    pred=pd.DataFrame([dict(period=str(k),security_id=s,market_cap=100.,entry_date=dates[k*2],exit_date=dates[k*2+2],score=4-j) for k in range(2) for j,s in enumerate(ids)])
    return pred,prices

def test_daily_ledger_self_financing_costs_and_drift():
    pred,prices=fixture();p=prepare(pred,prices);r=ledger(p,bps=0,borrow=0)
    assert r['final_equity']==pytest.approx(1.02*(1+.02/1.02))
    assert np.allclose(r['daily'].equity,r['daily'].cash+r['daily'].position_value)
    assert np.prod(1+r['periodic'].net_return)==pytest.approx(r['final_equity'])
    net=ledger(p);assert net['final_equity']<r['final_equity']
    assert net['total_transaction_cost']>0
    assert net['daily'].transaction_cost.sum()==pytest.approx(net['total_transaction_cost'])
    assert net['daily'].borrow_cost.sum()==pytest.approx(net['total_borrow'])
    assert np.allclose(net['daily'].equity,net['daily'].cash+net['daily'].position_value)
    assert ledger(p,capture=False)['final_equity']==pytest.approx(net['final_equity'])

def test_missing_selected_price_invalidates_but_unused_missing_does_not():
    pred,prices=fixture();bad=prices.copy();bad.loc[bad.security_id.eq('A')&bad.trade_date.eq(bad.trade_date.unique()[1]),'adjusted_close']=np.nan
    with pytest.raises(InvalidPortfolio,match='selected'):ledger(prepare(pred,bad))
    unused=prices.copy();unused.loc[unused.security_id.eq('B'),'adjusted_close']=np.nan
    ledger(prepare(pred,unused))

def test_calendar_holiday_early_close_and_decision():
    s=schedule('2024-11-01','2024-12-06',True)
    row=s[s.signal_date.eq(pd.Timestamp('2024-11-29'))].iloc[0]
    assert row.entry_date==pd.Timestamp('2024-12-02')
    assert row.decision_at==pd.Timestamp('2024-12-02T17:00:00Z')
    assert row.entry_close_at==pd.Timestamp('2024-12-02T21:00:00Z')

def test_actual_calendar_month_schedule():
    s=schedule('2024-03-01','2024-05-31',False)
    row=s[s.signal_date.eq(pd.Timestamp('2024-03-28'))].iloc[0]
    assert row.entry_date==pd.Timestamp('2024-04-01')

def test_weekly_maturity_and_common_universe(tmp_path):
    from Research_Pipeline.strategy_data import weekly_panel
    from Research_Pipeline.strategy_experiments import weekly_models
    from Research_Pipeline.data import PAPER_OPTIONS
    import exchange_calendars as xc
    cal=xc.get_calendar('XNYS',start='2023-01-01',end='2026-01-01')
    days=cal.sessions[(cal.sessions>=pd.Timestamp('2023-01-03'))&(cal.sessions<=pd.Timestamp('2025-04-01'))]
    rng=np.random.default_rng(4);ids=[f'S{i:03d}' for i in range(200)]
    records=[];options=[]
    for i,day in enumerate(days):
        close=100*np.exp(.0001*i+rng.normal(0,.001,len(ids)))
        nxt=cal.next_session(day);avail=nxt.tz_localize('America/New_York')+pd.Timedelta(hours=11)
        for j,sid in enumerate(ids):
            records.append(dict(trade_date=day,security_id=sid,close=close[j],adjusted_close=close[j],volume=10000,market_cap=1e8))
            options.append(dict(trade_date=day,security_id=sid,available_at=avail.tz_convert('UTC'),**{c:.2+rng.normal(0,.001) for c in PAPER_OPTIONS}))
    prices=pd.DataFrame(records);o=pd.DataFrame(options)
    panel,sch=weekly_panel(dict(stock_prices=prices,options_features=o));panel.attrs['price_last']=days.max()
    (tmp_path/'models').mkdir()
    # Only run Ridge here; the end-to-end command exercises the challengers.
    complete=sch[sch.exit_date.le(days.max())].tail(26)
    decision=panel[panel.period.eq(complete.period.iloc[0])].decision_at.iloc[0]
    train=panel[panel.label_available_at.lt(decision)&panel.target.notna()]
    assert train.period.nunique()>=52
    assert (train.label_available_at<decision).all()
    assert panel.groupby('period').size().min()==200
