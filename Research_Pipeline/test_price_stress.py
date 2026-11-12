import numpy as np
import pandas as pd
import pytest
from Research_Pipeline.price_stress import apply_shock,check_entries,missing_cells
from Research_Pipeline.daily_ledger import prepare,ledger,InvalidPortfolio

def data():
    dates=pd.bdate_range('2024-01-02',periods=3)
    prices=pd.DataFrame([dict(trade_date=d,security_id=s,adjusted_close=100.) for d in dates for s in 'ABCD'])
    prices=prices[~(prices.trade_date.eq(dates[-1])&prices.security_id.eq('D'))]
    pred=pd.DataFrame([dict(period='week',variant='model',security_id=s,score=4-i,market_cap=100.,entry_date=dates[0],exit_date=dates[-1]) for i,s in enumerate('ABCD')])
    return pred,prices

def test_stress_changes_only_missing_prices_without_mutating_source():
    pred,prices=data();old=prices.copy(deep=True);cells=missing_cells(pred,prices)
    altered,patch=apply_shock(prices,cells,.25)
    pd.testing.assert_frame_equal(prices,old)
    joined=prices.merge(altered,on=['trade_date','security_id'],suffixes=('_original','_scenario'),validate='one_to_one')
    assert np.array_equal(joined.adjusted_close_original,joined.adjusted_close_scenario)
    assert len(patch)==1 and patch.adjusted_close.iloc[0]==125
    assert patch.anchor_date.iloc[0]<patch.trade_date.iloc[0]

def test_upward_missing_price_shock_hurts_short_position():
    pred,prices=data();cells=missing_cells(pred,prices)
    values=[]
    for shock in (-.25,0,.25):
        altered,_=apply_shock(prices,cells,shock);run=ledger(prepare(pred,altered),bps=0,borrow=0)
        values.append(run['final_equity']-1)
    assert values==pytest.approx([.25,0,-.25])

def test_missing_entry_cannot_become_synthetic_trade():
    pred,prices=data();entry=pred.entry_date.iloc[0]
    prices=prices[~(prices.trade_date.eq(entry)&prices.security_id.eq('A'))]
    with pytest.raises(InvalidPortfolio,match='entry'):
        check_entries(prepare(pred,prices))
