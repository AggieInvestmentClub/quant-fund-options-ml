"""Paired size-conditioned nulls and calendar-week block bootstrap."""
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from .horizon_strategy import (SEED,POLICIES,NET,GROSS,WINDOWS,weekly_boundaries,
    weekly_equity,portfolio_case)
from .portfolio_ledger import PortfolioSpec,prepare_replay
from .daily_ledger import ledger,InvalidPortfolio
from .feature_experiments import ranked_correlation
from .reevaluate import dump


def shuffled_periods(periods,draw,seed=SEED):
    result=[]
    for p in periods:
        value=int.from_bytes(hashlib.sha256(f'{seed}/{draw}/{p["period"]}'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(value);donor=np.arange(len(p['ids']))
        for bucket in p['bins']:donor[bucket]=rng.permutation(bucket)
        q=dict(p,score=p['score'][donor])
        for k in ('allow_long','allow_short'):
            if k in p:q[k]=p[k][donor]
        result.append(q)
    return result


def bootstrap_indices(n,iterations,block=8,seed=SEED+1):
    if n<block:raise ValueError('fewer than eight weekly observations')
    rng=np.random.default_rng(seed);starts=rng.integers(0,n-block+1,size=(iterations,int(np.ceil(n/block))))
    return (starts[:,:,None]+np.arange(block)).reshape(iterations,-1)[:,:n]


def bounded_p(values,observed,iterations,family):
    values=np.asarray(values,float);valid=np.isfinite(values);invalid=int((~valid).sum())
    if observed is None or not np.isfinite(observed):return dict(p_raw=None,p_adjusted=None,p_bounds=None,invalid_draws=invalid)
    hits=int(np.sum(values[valid]>=observed));bounds=[(1+hits)/(iterations+1),(1+hits+invalid)/(iterations+1)]
    return dict(p_raw=bounds[0] if not invalid else None,p_adjusted=min(1,bounds[0]*family) if not invalid else None,
       p_bounds=bounds,adjusted_p_bounds=[min(1,p*family) for p in bounds],invalid_draws=invalid)


def ic_series(pred,ps,min_known=200):
    lookup={p:f.sort_values('security_id').target.to_numpy(float) for p,f in pred.groupby('period')}
    return np.array([ranked_correlation(p['score'][:,None],lookup[p['period']],min_known)[0] for p in ps])


def simulate_case(job):
    key,c,pred,prices,dest,iterations,family,boundaries,indices=job
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'statistics.json').exists():return key
    prepared=prepare_replay(pred,prices)
    spec=PortfolioSpec('equal',mode=c['mode'],selection=POLICIES[c['policy']])
    observed={};weekly={};records=[]
    for label,cost in [('net',NET),('gross',GROSS)]:
        try:
            r=ledger(prepared,portfolio=spec,costs=cost,initial_capital=100000.,capture=False)
            weekly[label]=weekly_equity(r,boundaries).to_numpy();observed[label]=float(weekly[label].mean())
        except InvalidPortfolio:observed[label]=None
    do_rank=key.startswith('stage1/')
    ic=ic_series(pred,prepared) if do_rank else None
    rank_observed=float(ic.mean()) if do_rank and np.isfinite(ic).all() else None
    for draw in range(iterations):
        ps=shuffled_periods(prepared,draw);row=dict(draw=draw)
        if do_rank:row['mean_rank_ic']=float(ic_series(pred,ps).mean())
        for label,cost in [('net',NET),('gross',GROSS)]:
            try:
                r=ledger(ps,portfolio=spec,costs=cost,initial_capital=100000.,capture=False)
                row[label+'_mean_weekly_return']=float(weekly_equity(r,boundaries).mean())
                row[label+'_status']='complete';row[label+'_turnover']=r['total_turnover']/len(boundaries[1:])
                for component in ['commission','regulatory','execution','flat','borrow']:row[label+'_'+component]=r['cost_totals'][component]
            except InvalidPortfolio as exc:
                row[label+'_status']=str(exc);row[label+'_mean_weekly_return']=np.nan
        records.append(row)
        if (draw+1)%500==0:print('null',key,draw+1,'/',iterations,flush=True)
    draws=pd.DataFrame(records);draws.to_parquet(dest/'random_draws.parquet',index=False)
    stats=dict(iterations=iterations,portfolio_family_size=family,seed=SEED,
       interpretation='exploratory retrospective; independent period shuffles are not turnover matched',
       net=bounded_p(draws.net_mean_weekly_return,observed['net'],iterations,family),
       gross=bounded_p(draws.gross_mean_weekly_return,observed['gross'],iterations,family))
    boot={}
    for label,r in weekly.items():
        samples=r[indices];sd=samples.std(axis=1,ddof=1)
        means=samples.mean(axis=1);annual=np.prod(1+samples,axis=1)**(365/(7*len(r)))-1
        sharpe=np.divide(means*np.sqrt(52),sd,out=np.full(iterations,np.nan),where=sd>0)
        for name,x in [('mean_weekly_return',means),('annualized_return',annual),('sharpe',sharpe)]:
            boot[label+'_'+name]=x;stats[label+'_'+name+'_95pct']=np.nanquantile(x,[.025,.975]).tolist() if np.isfinite(x).any() else None
        pd.DataFrame({'date':boundaries[1:],'return':r}).to_parquet(dest/(label+'_weekly.parquet'),index=False)
    if do_rank:
        stats['ranking']=bounded_p(draws.mean_rank_ic,rank_observed,iterations,24)
        positions=pd.DatetimeIndex([p['days'][0] for p in prepared]);weekly_ic=np.full(len(boundaries)-1,np.nan)
        weekly_ic[boundaries[:-1].get_indexer(positions)]=ic
        sampled=weekly_ic[indices];counts=np.isfinite(sampled).sum(axis=1)
        means=np.divide(np.nansum(sampled,axis=1),counts,out=np.full(iterations,np.nan),where=counts>0)
        boot['mean_ic']=means
        stats['mean_ic_95pct']=np.nanquantile(means,[.025,.975]).tolist() if np.isfinite(means).any() else None
    pd.DataFrame(boot,index=np.arange(iterations)).to_parquet(dest/'bootstrap.parquet')
    stats['bootstrap']=dict(block_calendar_weeks=8,resamples=iterations,
          limitation='only 24 calendar weeks and six four-week holding periods; intervals have limited precision')
    dump(dest/'statistics.json',stats)
    return key


def run_inference(out,predictions,prices,cases,cores,iterations):
    out=Path(out);folder=out/'inference';folder.mkdir(exist_ok=True)
    boundaries=weekly_boundaries(*WINDOWS['historical']);indices=bootstrap_indices(len(boundaries)-1,iterations)
    np.save(folder/'paired_block_indices.npy',indices)
    jobs=[];aliases={};seen={}
    for key,c in cases.items():
        signature=(c['variant'],c['policy'],c['mode'])
        if signature in seen:aliases[key]=seen[signature];continue
        seen[signature]=key
        jobs.append((key,c,predictions[c['variant']],prices,str(folder/key.replace('/','_')),
                     iterations,len(cases),boundaries,indices))
    if cores==1:list(map(simulate_case,jobs))
    else:
        with ProcessPoolExecutor(max_workers=cores) as pool:list(pool.map(simulate_case,jobs))
    dump(folder/'aliases.json',aliases)
    reports={}
    for key in cases:
        canonical=aliases.get(key,key)
        reports[key]=__import__('json').loads((folder/canonical.replace('/','_')/'statistics.json').read_text())
    dump(folder/'statistics.json',reports)
    # Predeclared policy improvement and incremental-options comparisons share calendar blocks.
    selection=__import__('json').loads((out/'selection.json').read_text())['variant']
    h,t,_,model=selection.split('/');stock=f'{h}/{t}/stock_only/{model}'
    contrasts={}
    for name,a,b in [('entry_and_buffer','stage2/ml_both/long_short','stage2/ml_baseline/long_short'),
                      ('incremental_options','paired/full','paired/stock_only')]:
        def boot(key):
            key=aliases.get(key,key);return pd.read_parquet(folder/key.replace('/','_')/'bootstrap.parquet')
        x,y=boot(a),boot(b)
        if 'net_mean_weekly_return' in x and 'net_mean_weekly_return' in y:
            d=x.net_mean_weekly_return-y.net_mean_weekly_return
            contrasts[name]=dict(mean_return_difference_95pct=np.quantile(d,[.025,.975]).tolist())
        else:contrasts[name]=dict(status='unavailable: incomplete portfolio')
    dump(folder/'paired_contrasts.json',contrasts)
