"""Self-financing daily adjusted-price ledger shared by both research tracks."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

class InvalidPortfolio(ValueError):
    pass

def select_weights(scores, caps, ids, crash=None, portfolio=None):
    n=max(1,math.ceil(len(ids)*.05))
    if len(ids)<2*n:
        raise InvalidPortfolio('universe too small for two disjoint sides')
    lo=np.lexsort((ids,-scores)); so=np.lexsort((ids, scores if crash is None else -crash))
    longs=set(lo[:n]); shorts=set(so[:n])
    for i in longs & shorts:
        if crash is None or scores[i]>=crash[i]: shorts.remove(i)
        else: longs.remove(i)
    for order,side,other in ((lo,longs,shorts),(so,shorts,longs)):
        for i in order:
            if len(side)>=n: break
            if i not in side|other: side.add(i)
    w=np.zeros(len(ids))
    for side,sign in ((longs,1),(shorts,-1)):
        ix=np.array(sorted(side))
        if portfolio is not None and portfolio.weighting=='equal':
            w[ix]=sign*min(1/len(ix),portfolio.max_position)
        else:
            w[ix]=sign*caps[ix]/caps[ix].sum()
    return w

def prepare(predictions, prices):
    """Keep every candidate; missing future prices never change selection."""
    pivot=prices.pivot(index='trade_date',columns='security_id',values='adjusted_close').sort_index()
    result=[]
    for key,f in predictions.groupby('period',sort=True):
        f=f.sort_values('security_id'); entry=pd.Timestamp(f.entry_date.iloc[0]); exit=pd.Timestamp(f.exit_date.iloc[0])
        days=pivot.index[(pivot.index>=entry)&(pivot.index<=exit)]
        if len(days)<2 or days[0]!=entry or days[-1]!=exit: raise InvalidPortfolio('missing interval boundary')
        result.append(dict(period=str(key), ids=f.security_id.to_numpy(str), caps=f.market_cap.to_numpy(float),
            score=f.score.to_numpy(float), crash=f.p_crash.to_numpy(float) if 'p_crash' in f else None,
            prices=pivot.reindex(index=days,columns=f.security_id).to_numpy(float), days=days, marks=pivot.loc[entry].to_dict(),
            bins=np.array_split(np.argsort(f.market_cap.to_numpy(),kind='stable'),min(10,len(f)))))
    return result

def ledger(periods, *, bps=30, borrow=.03, rng=None, capture=True,
           portfolio=None, costs=None, initial_capital=None):
    # The no-configuration path remains the historical implementation.
    if portfolio is not None or costs is not None or initial_capital is not None:
        from .portfolio_ledger import replay_ledger, PortfolioSpec, CostSpec
        return replay_ledger(periods, portfolio=portfolio or PortfolioSpec(),
            costs=costs or CostSpec(flat_bps=bps,borrow_rate=borrow),
            initial_capital=1. if initial_capital is None else initial_capital,
            rng=rng,capture=capture)
    return _legacy_ledger(periods,bps=bps,borrow=borrow,rng=rng,capture=capture)

def _legacy_ledger(periods, *, bps=30, borrow=.03, rng=None, capture=True):
    nav=1.; cash=1.; held={}; records=[]; returns=[]; holdings=[]; total_cost=total_borrow=turnover=0.
    for pi,p in enumerate(periods):
        ids,caps=p['ids'],p['caps']; scores=p['score'].copy(); crash=None if p['crash'] is None else p['crash'].copy()
        if rng is not None:
            for bucket in p['bins']:
                donor=rng.permutation(bucket); scores[bucket]=p['score'][donor]
                if crash is not None: crash[bucket]=p['crash'][donor]
        w=select_weights(scores,caps,ids,crash); ix=np.flatnonzero(w); selected=set(ids[ix])
        path=p['prices'][:,ix]; dates=p['days']
        if not np.isfinite(path).all() or (path[0]<=0).any() or (path<0).any():
            bad=(~np.isfinite(path))|(path<0); bad[0]|=path[0]<=0
            examples=[dict(security_id=str(ids[ix[j]]),date=str(dates[i].date())) for i,j in zip(*np.where(bad))][:8]
            raise InvalidPortfolio(f"missing/invalid selected daily price in {p['period']}: {examples}")
        marks=p['marks']; old={}
        for sid,units in held.items():
            if sid not in marks or not np.isfinite(marks[sid]):
                # Prior holdings must appear in the next candidate matrix for their exit mark.
                raise InvalidPortfolio(f'missing carried holding at rebalance: {sid}')
            old[sid]=units*marks[sid]
        pre=cash+sum(old.values())
        if pre<=0: raise InvalidPortfolio('insolvent before rebalance')
        # Solve post-cost equity since target exposures are defined on that equity.
        post=pre
        for _ in range(30):
            target=dict(zip(ids[ix],w[ix]*post))
            traded=sum(abs(target.get(k,0)-old.get(k,0)) for k in target.keys()|old.keys())
            updated=pre-bps/10000*traded
            if abs(updated-post)<1e-13: break
            post=updated
        post=updated
        if post<=0: raise InvalidPortfolio('insolvent after costs')
        target=dict(zip(ids[ix],w[ix]*post)); traded=sum(abs(target.get(k,0)-old.get(k,0)) for k in target.keys()|old.keys())
        cost=bps/10000*traded; total_cost+=cost; turnover+=traded/pre
        units=w[ix]*post/path[0]; cash=pre-cost-float(np.sum(units*path[0])); held=dict(zip(ids[ix],units))
        if capture:
            holdings.extend(dict(period=p['period'],security_id=sid,weight=float(ww),units=float(u),entry_date=dates[0],exit_date=dates[-1]) for sid,ww,u in zip(ids[ix],w[ix],units))
            records.append(dict(date=dates[0],equity=post,cash=cash,position_value=float(np.sum(units*path[0])),transaction_cost=cost,borrow_cost=0.))
        interval_borrow=0.
        for di in range(1,len(dates)):
            days=(dates[di]-dates[di-1]).days
            fee=borrow*days/365*float(np.abs(units[units<0]*path[di-1,units<0]).sum())
            cash-=fee; total_borrow+=fee; interval_borrow+=fee
            nav=cash+float(units@path[di])
            exit_cost=0.
            if pi==len(periods)-1 and di==len(dates)-1:
                exit_cost=bps/10000*float(np.abs(units*path[di]).sum()); cash+=float(units@path[di])-exit_cost
                total_cost+=exit_cost; turnover+=float(np.abs(units*path[di]).sum())/nav; nav=cash; held={}
            if nav<=0: raise InvalidPortfolio(f"insolvent in {p['period']}")
            if capture: records.append(dict(date=dates[di],equity=nav,cash=cash,position_value=0. if not held else float(units@path[di]),transaction_cost=exit_cost,borrow_cost=fee))
        returns.append(dict(period=p['period'],entry_date=dates[0],exit_date=dates[-1],net_return=nav/pre-1,
            transaction_cost=cost+exit_cost,borrow_cost=interval_borrow,turnover=traded/pre,
            long_return=float(np.sum(w[ix][w[ix]>0]*(path[-1,w[ix]>0]/path[0,w[ix]>0]-1))),
            short_return=float(np.sum(w[ix][w[ix]<0]*(path[-1,w[ix]<0]/path[0,w[ix]<0]-1)))))
    daily=pd.DataFrame(records)
    if capture: daily=daily.groupby('date',as_index=False).agg(equity=('equity','last'),cash=('cash','last'),position_value=('position_value','last'),transaction_cost=('transaction_cost','sum'),borrow_cost=('borrow_cost','sum')).sort_values('date')
    return dict(final_equity=nav,total_transaction_cost=total_cost,total_borrow=total_borrow,total_turnover=turnover,
                periodic=pd.DataFrame(returns),daily=daily,holdings=pd.DataFrame(holdings))

def metrics(run, frequency):
    capital=run.get('initial_capital',1.)
    r=run['periodic'].net_return.to_numpy(); daily=run['daily']; wealth=np.r_[capital,daily.equity.to_numpy()]
    elapsed=(daily.date.iloc[-1]-daily.date.iloc[0]).days
    sd=np.std(r,ddof=1)
    return dict(periods=len(r),cumulative_return=run['final_equity']/capital-1,
        annualized_return=(run['final_equity']/capital)**(365/max(1,elapsed))-1,
        sharpe=float(np.mean(r)/sd*np.sqrt(frequency)) if sd else None,
        annualized_volatility=float(sd*np.sqrt(frequency)),mean_periodic_return=float(r.mean()),
        max_drawdown=float(np.min(wealth/np.maximum.accumulate(wealth)-1)),hit_rate=float(np.mean(r>0)),
        mean_turnover=run['total_turnover']/len(r),transaction_cost=run['total_transaction_cost'],borrow_cost=run['total_borrow'])
