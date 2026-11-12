"""Calendar-aware schedules and weekly features from the existing handoff."""
import numpy as np
import pandas as pd
import exchange_calendars as xc
from .data import PAPER_OPTIONS,SKEW_FEATURES
STOCK=['ret_5','ret_20','vol_20','dollar_volume_20','log_cap']
OPTION=PAPER_OPTIONS+['iv_gap']+SKEW_FEATURES

def schedule(start,end,weekly):
    cal=xc.get_calendar('XNYS',start=pd.Timestamp(start)-pd.Timedelta(days=60),end=pd.Timestamp(end)+pd.Timedelta(days=60))
    sessions=cal.sessions.tz_localize(None) if cal.sessions.tz is not None else cal.sessions
    groups=pd.Series(sessions,index=sessions).groupby(sessions.to_period('W-FRI' if weekly else 'M')).max()
    rows=[]
    for period,signal in groups.items():
        pos=sessions.get_loc(signal)+1
        if pos>=len(sessions): continue
        entry=sessions[pos]
        decision=(entry.tz_localize('America/New_York')+pd.Timedelta(hours=12)).tz_convert('UTC')
        rows.append(dict(period=str(period),signal_date=signal,entry_date=entry,decision_at=decision,
                         entry_close_at=cal.session_close(entry)))
    f=pd.DataFrame(rows); f['exit_date']=f.entry_date.shift(-1)
    f['label_available_at']=f.exit_date.map(lambda d:cal.session_close(d) if pd.notna(d) else pd.NaT)
    return f[f.signal_date.between(pd.Timestamp(start),pd.Timestamp(end))].copy()

def weekly_panel(tables):
    prices=tables['stock_prices'].sort_values(['security_id','trade_date']).copy()
    chunks=[]
    for _,f in prices.groupby('security_id',sort=False):
        f=f.copy(); r=f.adjusted_close.pct_change(fill_method=None)
        f['ret_5']=f.adjusted_close.pct_change(5,fill_method=None);f['ret_20']=f.adjusted_close.pct_change(20,fill_method=None)
        f['vol_20']=r.rolling(20,min_periods=20).std()
        f['dollar_volume_20']=(f.close*f.volume).rolling(20,min_periods=20).mean()
        chunks.append(f)
    stock=pd.concat(chunks);o=tables['options_features'].copy()
    sched=schedule(o.trade_date.min(),o.trade_date.max(),True)
    f=o.merge(sched,left_on='trade_date',right_on='signal_date',validate='many_to_one')
    f=f[f.available_at.le(f.decision_at)]
    f=f.merge(stock[['trade_date','security_id','close','market_cap',*STOCK[:-1]]],on=['trade_date','security_id'],validate='one_to_one')
    f['log_cap']=np.log(f.market_cap.where(f.market_cap>0)); f['iv_gap']=f.avg_call_iv-f.avg_put_iv
    f=f[f.avg_call_iv.notna()&f.avg_put_iv.notna()&f.close.ge(5)&f.market_cap.gt(0)&f[STOCK].notna().all(axis=1)].copy()
    lookup=prices.set_index(['trade_date','security_id']).adjusted_close
    for name,date in [('entry_price','entry_date'),('exit_price','exit_date')]:
        keys=pd.MultiIndex.from_arrays([f[date],f.security_id]); f[name]=lookup.reindex(keys).to_numpy()
    # Missing future prices remain in the prediction universe; only training labels are missing.
    f['target']=f.exit_price/f.entry_price-1
    f.loc[f.entry_price.le(0),'target']=np.nan
    for c in OPTION:
        if c not in f:f[c]=np.nan
    return f,sched
