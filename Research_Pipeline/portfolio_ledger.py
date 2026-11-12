"""Dollar ledger for frozen-score experiments; never submits broker orders."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .daily_ledger import InvalidPortfolio, select_weights


@dataclass(frozen=True)
class PortfolioSpec:
    weighting: str = 'market_cap'
    max_position: float = .05

    def __post_init__(self):
        if self.weighting not in ('market_cap','equal') or not 0 < self.max_position <= 1:
            raise ValueError('invalid portfolio specification')


@dataclass(frozen=True)
class CostSpec:
    mode: str = 'flat'
    flat_bps: float = 30.
    execution_bps: float = 0.
    borrow_rate: float = .03
    commission_per_share: float = .005
    minimum_order: float = 1.
    maximum_order_fraction: float = .01
    sec_sell_fraction: float = .0000206
    finra_sell_per_share: float = 0.
    cat_per_share: float = .000003
    schedule_date: str = '2026-11-12'
    schedule_url: str = 'https://www.interactivebrokers.com/en/pricing/commissions-stocks.php'

    def __post_init__(self):
        if self.mode not in ('flat','ibkr_fixed'):
            raise ValueError('invalid cost mode')
        for name in ('flat_bps','execution_bps','borrow_rate','commission_per_share',
                     'minimum_order','maximum_order_fraction','sec_sell_fraction',
                     'finra_sell_per_share','cat_per_share'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError('invalid cost parameter: '+name)


def order_costs(delta, raw, spec):
    """One net order per security; whole-share schedule on ideal share equivalents.

    This deliberately approximates fractional execution. Current rates, including
    the current FINRA TAF holiday, are held constant across historical dates.
    """
    delta=np.asarray(delta,float); value=np.abs(delta); active=value>0
    commission=np.zeros_like(value); regulatory=np.zeros_like(value)
    execution=value*spec.execution_bps/10000
    flat=value*spec.flat_bps/10000 if spec.mode=='flat' else np.zeros_like(value)
    shares=np.zeros_like(value)
    if spec.mode=='ibkr_fixed':
        raw=np.asarray(raw,float)
        if np.any(active & (~np.isfinite(raw)|(raw<=0))):
            raise InvalidPortfolio('missing/invalid raw execution price for traded security')
        shares[active]=value[active]/raw[active]
        commission[active]=np.minimum(np.maximum(shares[active]*spec.commission_per_share,
            spec.minimum_order),value[active]*spec.maximum_order_fraction)
        sell=delta<0
        regulatory=sell*(value*spec.sec_sell_fraction+shares*spec.finra_sell_per_share)+shares*spec.cat_per_share
    return dict(commission=commission,regulatory=regulatory,execution=execution,
                flat=flat,shares=shares,total=commission+regulatory+execution+flat)


def prepare_replay(predictions,prices):
    """Precompute arrays once, retaining every candidate and missing future price."""
    if predictions.duplicated(['period','security_id']).any():
        raise ValueError('duplicate prediction key')
    if prices.duplicated(['trade_date','security_id']).any():
        raise ValueError('duplicate price key')
    ids=np.array(sorted(set(prices.security_id)|set(predictions.security_id)),dtype=str)
    adj=prices.pivot(index='trade_date',columns='security_id',values='adjusted_close').sort_index().reindex(columns=ids)
    raw=prices.pivot(index='trade_date',columns='security_id',values='close').reindex(index=adj.index,columns=ids) if 'close'in prices else adj*np.nan
    result=[]
    for key,f in predictions.groupby('period',sort=True):
        f=f.sort_values('security_id');entry=pd.Timestamp(f.entry_date.iloc[0]);exit=pd.Timestamp(f.exit_date.iloc[0])
        if f.entry_date.nunique()!=1 or f.exit_date.nunique()!=1:
            raise ValueError('ambiguous period boundaries')
        days=adj.index[(adj.index>=entry)&(adj.index<=exit)]
        if len(days)<2 or days[0]!=entry or days[-1]!=exit:
            raise InvalidPortfolio('missing interval boundary')
        scores=f.score.to_numpy(float);caps=f.market_cap.to_numpy(float)
        if not np.isfinite(scores).all() or not np.isfinite(caps).all() or (caps<=0).any():
            raise ValueError('invalid scores or market caps')
        crash=f.p_crash.to_numpy(float) if 'p_crash'in f else None
        if crash is not None and not np.isfinite(crash).all():raise ValueError('invalid crash probabilities')
        result.append(dict(period=str(key),ids=f.security_id.to_numpy(str),caps=caps,
            score=scores,crash=crash,global_ids=ids,index=np.searchsorted(ids,f.security_id),
            all_prices=adj.loc[days].to_numpy(float),raw_entry=raw.loc[entry].to_numpy(float),
            raw_exit=raw.loc[exit].to_numpy(float),days=days,
            calendar_days=np.diff(days.to_numpy()).astype('timedelta64[D]').astype(float),
            bins=np.array_split(np.argsort(caps,kind='stable'),min(10,len(f)))))
    if not result:raise ValueError('no prediction periods')
    for a,b in zip(result,result[1:]):
        if a['days'][-1]!=b['days'][0]:raise ValueError('noncontiguous holding periods')
    return result


def replay_ledger(periods, *, portfolio, costs, initial_capital=100000., rng=None, capture=True):
    if not np.isfinite(initial_capital) or initial_capital<=0:raise ValueError('invalid initial capital')
    if not periods or 'global_ids' not in periods[0]:
        raise ValueError('configured ledger requires prepare_replay arrays')
    ids=periods[0]['global_ids'];size=len(ids);held=np.zeros(size);cash=float(initial_capital)
    daily=[];returns=[];holdings=[];trades=[];exposures=[]
    totals=dict(commission=0.,regulatory=0.,execution=0.,flat=0.,borrow=0.,traded=0.,turnover=0.)
    pnl=np.zeros(size);leg_long=leg_short=0.;period_returns=[]
    tol=max(1e-10,initial_capital*1e-12)

    def transact(target,old,raw,day,period,event):
        delta=target-old
        delta[np.abs(delta)<tol*.01]=0.
        fees=order_costs(delta,raw,costs)
        if capture:
            for j in np.flatnonzero(delta):
                trades.append(dict(date=day,period=period,event=event,security_id=ids[j],
                    signed_notional=delta[j],notional=abs(delta[j]),raw_close=raw[j],
                    share_equivalent=fees['shares'][j],commission=fees['commission'][j],
                    regulatory=fees['regulatory'][j],execution=fees['execution'][j],
                    flat_cost=fees['flat'][j],transaction_cost=fees['total'][j]))
        return delta,fees

    def record(day,values,nav,fees,borrow_fee):
        if not capture:return
        pos=np.flatnonzero(values);weights=values[pos]/nav
        lng=weights[weights>0];sht=-weights[weights<0]
        eff=lambda w:float(w.sum()**2/(w@w)) if len(w) else 0.
        nv=np.flatnonzero(np.char.endswith(ids,':NVDA')|(ids=='NVDA'))
        max_w=float(np.max(np.abs(weights))) if len(weights) else 0.
        daily.append(dict(date=day,equity=nav,cash=cash,position_value=float(values.sum()),
            transaction_cost=float(fees['total'].sum()),borrow_cost=borrow_fee,
            commission=float(fees['commission'].sum()),regulatory=float(fees['regulatory'].sum()),
            execution=float(fees['execution'].sum()),flat_cost=float(fees['flat'].sum()),
            gross_exposure=float(np.abs(weights).sum()),net_exposure=float(weights.sum()),
            max_abs_weight=max_w,cap_breach=max_w>portfolio.max_position+1e-10,
            effective_long=eff(lng),effective_short=eff(sht),
            nvda_weight=float(values[nv].sum()/nav)))

    for pi,p in enumerate(periods):
        scores=p['score'].copy();crash=None if p['crash'] is None else p['crash'].copy()
        if rng is not None:
            for bucket in p['bins']:
                donor=rng.permutation(bucket);scores[bucket]=p['score'][donor]
                if crash is not None:crash[bucket]=p['crash'][donor]
        local_w=select_weights(scores,p['caps'],p['ids'],crash,portfolio)
        ix=p['index'][local_w!=0];w=local_w[local_w!=0];path=p['all_prices'][:,ix];dates=p['days']
        if not np.isfinite(path).all() or (path<0).any() or (path[0]<=0).any():
            raise InvalidPortfolio('missing/invalid selected daily price in '+p['period'])
        old_ix=np.flatnonzero(held);marks=p['all_prices'][0,old_ix]
        if not np.isfinite(marks).all() or (marks<0).any():raise InvalidPortfolio('missing carried holding mark')
        old=np.zeros(size);old[old_ix]=held[old_ix]*marks;pre=cash+old.sum()
        if pre<=0:raise InvalidPortfolio('insolvent before rebalance')
        global_w=np.zeros(size);global_w[ix]=w
        post=pre
        for _ in range(100):
            delta=global_w*post-old;delta[np.abs(delta)<tol*.01]=0.
            fee=order_costs(delta,p['raw_entry'],costs)['total'].sum()
            updated=pre-fee
            if abs(updated-post)<=tol:break
            post=updated
        else:raise InvalidPortfolio('post-cost sizing did not converge')
        post=updated
        if post<=0:raise InvalidPortfolio('insolvent after costs')
        target=global_w*post;delta,fees=transact(target,old,p['raw_entry'],dates[0],p['period'],'rebalance')
        fee=float(fees['total'].sum())
        if abs(pre-fee-post)>tol*2:raise InvalidPortfolio('post-cost sizing residual')
        traded=float(np.abs(delta).sum());totals['turnover']+=traded/pre
        for k in ('commission','regulatory','execution','flat'):totals[k]+=fees[k].sum()
        totals['traded']+=traded;cash=pre-fee-target.sum();held=np.zeros(size);held[ix]=target[ix]/path[0]
        if capture:
            holdings.extend(dict(period=p['period'],security_id=ids[j],weight=float(ww),
                units=float(held[j]),entry_date=dates[0],exit_date=dates[-1],entry_equity=post,
                pnl=float(held[j]*(path[-1,z]-path[0,z]))) for z,(j,ww) in enumerate(zip(ix,w)))
        record(dates[0],target,post,fees,0.)
        units=held[ix];part=units*(path[-1]-path[0]);pnl[ix]+=part
        long_pnl=float(part[w>0].sum());short_pnl=float(part[w<0].sum());leg_long+=long_pnl;leg_short+=short_pnl
        interval_borrow=0.;exit_fee=0.;exit_turnover=0.
        for di in range(1,len(dates)):
            borrow_fee=costs.borrow_rate*p['calendar_days'][di-1]/365*float(np.abs(units[units<0]*path[di-1,units<0]).sum())
            cash-=borrow_fee;interval_borrow+=borrow_fee;totals['borrow']+=borrow_fee
            values=np.zeros(size);values[ix]=units*path[di];nav=cash+values.sum()
            if nav<=0:raise InvalidPortfolio('insolvent in '+p['period'])
            nofees={k:np.zeros(0) for k in ('total','commission','regulatory','execution','flat')}
            if pi==len(periods)-1 and di==len(dates)-1:
                delta,fees=transact(np.zeros(size),values,p['raw_exit'],dates[di],p['period'],'liquidation')
                exit_fee=float(fees['total'].sum());exit_traded=float(np.abs(delta).sum());exit_turnover=exit_traded/nav
                totals['turnover']+=exit_turnover;totals['traded']+=exit_traded
                for k in ('commission','regulatory','execution','flat'):totals[k]+=fees[k].sum()
                cash+=values.sum()-exit_fee;nav=cash;values[:]=0;held[:]=0
                if nav<=0:raise InvalidPortfolio('insolvent at liquidation')
            else:fees=nofees
            record(dates[di],values,nav,fees,borrow_fee)
        r=nav/pre-1;period_returns.append(r)
        if capture:returns.append(dict(period=p['period'],entry_date=dates[0],exit_date=dates[-1],net_return=r,
            transaction_cost=fee+exit_fee,borrow_cost=interval_borrow,turnover=traded/pre+exit_turnover,
            long_pnl=long_pnl,short_pnl=short_pnl))
    result=dict(initial_capital=initial_capital,final_equity=float(nav),total_transaction_cost=float(sum(totals[k] for k in ('commission','regulatory','execution','flat'))),
        total_borrow=float(totals['borrow']),total_turnover=float(totals['turnover']),period_returns=np.array(period_returns),
        cost_totals={k:float(v) for k,v in totals.items()},long_pnl=leg_long,short_pnl=leg_short,
        stock_pnl=dict(zip(ids[pnl!=0],pnl[pnl!=0])),portfolio=asdict(portfolio),costs=asdict(costs))
    if capture:
        df=pd.DataFrame(daily);sum_cols=['transaction_cost','borrow_cost','commission','regulatory','execution','flat_cost']
        df=df.groupby('date',as_index=False).agg({c:'sum' if c in sum_cols else 'last' for c in df.columns if c!='date'})
        result.update(daily=df,holdings=pd.DataFrame(holdings),trades=pd.DataFrame(trades),periodic=pd.DataFrame(returns))
    return result
