"""Paired fixed-universe forecast randomizations and two-hold block resamples."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .daily_ledger import ledger,InvalidPortfolio
from .portfolio_ledger import prepare_replay
from .horizon_statistics import shuffled_periods,bounded_p
from .horizon_strategy import NET,GROSS,weekly_equity
from .reevaluate import dump

def hold_blocks(n,iterations,seed=18):
    if n<2:raise ValueError('need at least two holding periods for block bootstrap')
    starts=np.random.default_rng(seed).integers(0,n-1,size=(iterations,int(np.ceil(n/2))))
    return (starts[:,:,None]+np.arange(2)).reshape(iterations,-1)[:,:n]

def forecast_values(periods,lookup,min_known=20):
    records=[]
    for p in periods:
        y,baseline=lookup[p['period']];mask=np.isfinite(y)
        if mask.sum()<min_known:continue
        y=y[mask];score=p['score'][mask];base=baseline[mask]
        mse=np.mean((score-y)**2);bmse=np.mean((base-y)**2)
        a=rankdata(score);a-=a.mean();b=rankdata(y);b-=b.mean();den=np.sqrt((a@a)*(b@b))
        records.append([bmse-mse,np.mean((score>0)==(y>0)),(a@b)/den if den>0 else np.nan])
    if not records:return dict(mse_improvement=np.nan,direction_accuracy=np.nan,rank_ic=np.nan)
    a=np.asarray(records);return dict(mse_improvement=float(np.mean(a[:,0])),
             direction_accuracy=float(np.mean(a[:,1])),rank_ic=float(np.nanmean(a[:,2])) if np.isfinite(a[:,2]).any() else np.nan)

def simulate_variant(job):
    from .fixed_universe_forecast import portfolio_spec
    variant,pred,prices,dest,bounds,iterations,family=job
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    periods=prepare_replay(pred,prices);lookup={}
    for p,f in pred.groupby('period',sort=True):
        f=f.sort_values('security_id');lookup[p]=(f.target.to_numpy(float),f.baseline_mean.to_numpy(float))
    observed=forecast_values(periods,lookup);observed_returns={}
    for name,cost in [('net',NET),('gross',GROSS)]:
        try:
            r=ledger(periods,portfolio=portfolio_spec(),costs=cost,initial_capital=100000.,capture=False)
            observed_returns[name]=float(weekly_equity(r,bounds).mean())
        except InvalidPortfolio:observed_returns[name]=None
    records=[]
    for draw in range(iterations):
        ps=shuffled_periods(periods,draw);record=dict(draw=draw,**forecast_values(ps,lookup))
        for name,cost in [('net',NET),('gross',GROSS)]:
            try:
                r=ledger(ps,portfolio=portfolio_spec(),costs=cost,initial_capital=100000.,capture=False)
                record[name+'_mean_weekly_return']=float(weekly_equity(r,bounds).mean())
                record[name+'_return']=r['final_equity']/100000-1
                record[name+'_turnover']=r['total_turnover']
                for k,value in r['cost_totals'].items():record[name+'_'+k]=value
                record[name+'_failure']=None
            except InvalidPortfolio as exc:
                record[name+'_mean_weekly_return']=np.nan;record[name+'_failure']=str(exc)
        records.append(record)
        if (draw+1)%500==0:print('Monte Carlo',variant,draw+1,'/',iterations,flush=True)
    frame=pd.DataFrame(records);frame.to_parquet(dest/'random_draws.parquet',index=False)
    result=dict(observed=observed,observed_mean_weekly_return=observed_returns,iterations=iterations,family=family)
    for key in ('mse_improvement','direction_accuracy','rank_ic'):
        result[key]=bounded_p(frame[key],observed[key],iterations,family)
    for key in ('net','gross'):
        result[key]=bounded_p(frame[key+'_mean_weekly_return'],observed_returns[key],iterations,family)
    dump(dest/'statistics.json',result)
    return variant,result

def confidence(values):
    a=np.asarray(values,float)
    return np.quantile(a,[.025,.975]).tolist() if np.isfinite(a).all() else None

def infer(pred,prices,root,bounds,iterations=2000,cores=2):
    root=Path(root);out=root/'inference';out.mkdir(exist_ok=True)
    variants=sorted(pred.variant.unique());family=len(variants)
    # validate equal candidates, outcomes and base forecasts before paired inference
    from .fixed_universe_forecast import assert_common,score_metrics
    assert_common(pred)
    jobs=[(v,f,prices,str(out/v.replace('/','_')),bounds,iterations,family) for v,f in pred.groupby('variant',sort=True)]
    if cores==1:results=dict(map(simulate_variant,jobs))
    else:
        with ProcessPoolExecutor(max_workers=cores) as pool:results=dict(pool.map(simulate_variant,jobs))
    pm=score_metrics(pred);periods=sorted(pred.period.unique());indices=hold_blocks(len(periods),iterations)
    np.save(out/'paired_hold_indices.npy',indices);boot={}
    for v in variants+['passive_25','historical_mean','zero_forecast']:
        records={}
        if v in variants:
            f=pm[pm.variant.eq(v)].set_index('period').reindex(periods)
            for key in ['mse','mae','direction_accuracy','balanced_direction_accuracy','rank_ic','mse_improvement']:
                values=f[key].to_numpy(float)
                # Preserve missing decision statistics, rather than resampling a smaller period universe.
                records[key]=values[indices].mean(axis=1)
        for label in ('net','gross'):
            path=root/'portfolios'/v.replace('/','_')/f'{label}_weekly.parquet'
            if not path.exists():continue
            weekly=pd.read_parquet(path).iloc[:,0].to_numpy(float)
            if len(weekly)!=4*len(periods):raise ValueError('bootstrap needs four weekly returns per hold')
            sampled=weekly.reshape(len(periods),4)[indices].reshape(iterations,-1)
            records[label+'_mean_weekly_return']=sampled.mean(axis=1)
            records[label+'_return']=np.prod(1+sampled,axis=1)-1
            years=(bounds[-1]-bounds[0]).days/365.
            records[label+'_annualized_return']=np.power(1+records[label+'_return'],1/years)-1
            sd=sampled.std(axis=1,ddof=1)
            records[label+'_sharpe']=np.divide(sampled.mean(axis=1)*np.sqrt(52),sd,
                  out=np.full(iterations,np.nan),where=sd>0)
        frame=pd.DataFrame(records);folder=out/v.replace('/','_');folder.mkdir(exist_ok=True)
        frame.to_parquet(folder/'bootstrap.parquet',index=False)
        intervals={c:confidence(frame[c]) for c in frame}
        dump(folder/'bootstrap_intervals.json',intervals);boot[v]=frame
        if v in results:results[v]['bootstrap_95pct']=intervals
    contrasts={}
    for kind in ('ridge','lightgbm'):
        for a,b in [('full','stock_only'),('pruned','full')]:
            va,vb=a+'/'+kind,b+'/'+kind
            if va not in boot or vb not in boot:continue
            diff=boot[vb].mse-boot[va].mse
            name=va+' minus '+vb
            contrasts[name]=dict(mse_reduction_95pct=confidence(diff),
                 observed_mse_reduction=float(pm[pm.variant.eq(vb)].mse.mean()-pm[pm.variant.eq(va)].mse.mean()))
    for v in variants:
        if 'net_return'in boot[v] and 'net_return'in boot['passive_25']:
            contrasts[v+' minus passive_25']=dict(net_cumulative_difference_95pct=confidence(boot[v].net_return-boot['passive_25'].net_return),
                 net_mean_weekly_difference_95pct=confidence(boot[v].net_mean_weekly_return-boot['passive_25'].net_mean_weekly_return))
    dump(out/'statistics.json',results);dump(out/'paired_contrasts.json',contrasts)
    dump(out/'method.json',dict(iterations=iterations,seed=17,bootstrap_seed=18,
         forecast_family=family,portfolio_family=family,bootstrap_block_holds=2,
         permutation='within each date and market-cap decile; shared donor permutations across variants',
         limitation='small size groups; conditions on active slot count, does not independently test aggregate market timing; all dates previously inspected',
         bootstrap='resample observed forecast statistics and four-week bundles of weekly returns; no new execution simulation',
         multiplicity='six main variants only; sensitivities descriptive; not a correction for all prior research trials'))
    return results
