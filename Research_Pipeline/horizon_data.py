"""Exchange-calendar horizons and fixed momentum confirmations; no downloads."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import exchange_calendars as xc
from .strategy_data import schedule, STOCK, OPTION

FEATURES={'stock_only':STOCK,'full':STOCK+OPTION}
WINDOWS={'development':('2025-06-02','2026-03-09'),
         'historical':('2026-05-11','2026-10-26')}
HORIZONS=(1,2,4)
TARGETS=('absolute','relative')
MODELS=('ridge','lightgbm')
VARIANTS=[f'h{h}/{t}/{g}/{m}' for h in HORIZONS for t in TARGETS
          for g in FEATURES for m in MODELS]


def horizon_panel(panel,prices,horizon,min_known=200):
    if horizon not in HORIZONS:raise ValueError('horizon must be 1, 2 or 4 weeks')
    if panel.duplicated(['period','security_id']).any() or prices.duplicated(['trade_date','security_id']).any():
        raise ValueError('ambiguous input keys')
    if not panel.available_at.le(panel.decision_at).all():raise ValueError('future feature availability')
    cal=xc.get_calendar('XNYS',start=panel.trade_date.min()-pd.Timedelta(days=60),
                        end=panel.trade_date.max()+pd.Timedelta(days=120))
    master=schedule(panel.trade_date.min(),panel.trade_date.max()+pd.Timedelta(weeks=horizon+2),True)
    master['exit_date']=master.entry_date.shift(-horizon)
    master['label_available_at']=master.exit_date.map(lambda x:cal.session_close(x) if pd.notna(x) else pd.NaT)
    columns=['period','entry_date','exit_date','decision_at','label_available_at']
    f=panel.drop(columns=[c for c in columns[1:] if c in panel]).merge(master[columns],on='period',how='left',validate='many_to_one')
    if f.decision_at.isna().any():raise ValueError('signal week absent from exchange calendar')
    if not f.available_at.le(f.decision_at).all():raise ValueError('feature not ready for horizon decision')
    lookup=prices.set_index(['trade_date','security_id']).adjusted_close
    entry=lookup.reindex(pd.MultiIndex.from_arrays([f.entry_date,f.security_id])).to_numpy(float)
    exit=lookup.reindex(pd.MultiIndex.from_arrays([f.exit_date,f.security_id])).to_numpy(float)
    finite=np.isfinite(entry)&np.isfinite(exit)&(entry>0)&(exit>=0)
    absolute=np.full(len(f),np.nan);absolute[finite]=exit[finite]/entry[finite]-1
    f['target_absolute']=absolute
    counts=f.groupby('period').target_absolute.transform('count')
    means=f.groupby('period').target_absolute.transform('mean').where(counts>=min_known)
    f['benchmark_return']=means
    f['benchmark_known_outcomes']=counts
    f['target_relative']=f.target_absolute-means
    # Save the canonical execution target separately from the fitted objective.
    f['target']=f.target_absolute
    f['horizon_weeks']=horizon
    return f,master


def window_schedule(master,horizon,start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    start_anchor=start.to_period('W-SUN').start_time;end_anchor=end.to_period('W-SUN').start_time
    if (end_anchor-start_anchor).days%(7*horizon):raise ValueError('window must contain complete holding periods')
    entries=pd.date_range(start_anchor,end_anchor-pd.Timedelta(days=1),freq=f'{horizon}W-MON')
    # Calendar anchor is a week, so holiday Mondays use that week's actual entry.
    entry_weeks=master.entry_date.dt.to_period('W-SUN')
    wanted=entries.to_period('W-SUN')
    f=master[entry_weeks.isin(wanted)].copy()
    final=master.loc[entry_weeks.eq(end.to_period('W-SUN')),'entry_date']
    if len(f)!=len(entries) or len(final)!=1 or f.exit_date.iloc[-1]!=final.iloc[0]:
        raise ValueError('incomplete or changed horizon window')
    return f.sort_values('entry_date')


def matured_training(panel,decision,target,min_weeks=52):
    f=panel[panel.label_available_at.lt(decision)&np.isfinite(panel['target_'+target])].copy()
    if f.period.nunique()<min_weeks:raise ValueError('fewer than 52 matured training signal weeks')
    f['target']=f['target_'+target]
    return f


def momentum_predictions(panel,schedule,kind,horizon):
    f=panel[panel.period.isin(schedule.period)].copy()
    if 'ret_63' not in f:raise ValueError('frozen panel requires verified ret_63')
    finite=f.ret_63.notna()&np.isfinite(f.ret_63)
    f['score']=f.ret_63.where(finite,0.)
    f['allow_long']=finite&f.ret_63.gt(0)
    f['allow_short']=finite&f.ret_63.lt(0)
    if kind in ('gap','gap_skew'):
        med=f.groupby('period').iv_gap.transform('median')
        f['allow_long'] &= f.iv_gap.notna()&f.iv_gap.gt(med)
        f['allow_short'] &= f.iv_gap.notna()&f.iv_gap.lt(med)
    if kind=='gap_skew':
        c='put_skew_10_25_30d';med=f.groupby('period')[c].transform('median')
        f['allow_long'] &= f[c].notna()&f[c].lt(med)
        f['allow_short'] &= f[c].notna()&f[c].gt(med)
    if kind not in ('plain','gap','gap_skew'):raise ValueError('unknown momentum rule')
    f['variant']=f'h{horizon}/momentum/{kind}'
    return prediction_columns(f)


def prediction_columns(f):
    cols=['period','security_id','trade_date','decision_at','entry_date','exit_date',
          'market_cap','target','target_absolute','target_relative','benchmark_return',
          'benchmark_known_outcomes','score','variant','horizon_weeks']
    cols += [c for c in ('allow_long','allow_short') if c in f]
    return f[cols].copy()
