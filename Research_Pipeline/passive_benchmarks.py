"""Frozen-score passive/ML comparisons. Only fetch-spy performs a download."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'

import argparse
import ast
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import exchange_calendars as xc

from .daily_ledger import ledger, InvalidPortfolio
from .portfolio_ledger import PortfolioSpec, prepare_replay
from .horizon_data import WINDOWS
from .horizon_strategy import NET, GROSS, POLICIES, weekly_boundaries, weekly_equity, performance
from .horizon_statistics import shuffled_periods, bootstrap_indices, bounded_p, ic_series
from .reevaluate import sha, dump, audit, alpha_proxy
from .run import _limit_cores

PORTFOLIO_FAMILY=53
RANKING_FAMILY=27


def configured_universe(path):
    """Read just the literal list; importing production config is unnecessary."""
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STOCK_UNIVERSE' for t in node.targets):
            names=ast.literal_eval(node.value)
            if len(names)!=25 or len(set(names))!=25:raise ValueError('expected 25 unique configured stocks')
            return ['ticker_proxy:'+x for x in names]
    raise ValueError('configured stock universe absent')


def sessions(start,end):
    return xc.get_calendar('XNYS',start=start,end=end).sessions.tz_localize(None)


def decision_at(day):
    return pd.Timestamp(day).tz_localize('America/New_York').replace(hour=12).tz_convert('UTC')


def monthly_intervals(start,end):
    days=sessions(start,end)
    if days[0]!=pd.Timestamp(start) or days[-1]!=pd.Timestamp(end):raise ValueError('non-session window boundary')
    first=pd.Series(days,index=days.to_period('M')).groupby(level=0).first()
    entries=[pd.Timestamp(start)]+[d for d in first if pd.Timestamp(start)<d<pd.Timestamp(end)]
    edges=entries+[pd.Timestamp(end)]
    return [(f'monthly_{i:02}',a,b) for i,(a,b) in enumerate(zip(edges,edges[1:]))]


def latest_snapshot(panel,day):
    cutoff=decision_at(day)
    eligible=panel[panel.decision_at.le(cutoff)]
    if eligible.empty:raise ValueError('no decision-time weekly eligibility snapshot')
    ready=eligible.groupby('decision_at').available_at.max()
    ready=ready[ready.le(cutoff)]
    if ready.empty:raise ValueError('no complete snapshot available before decision')
    latest=ready.index.max()
    f=eligible[eligible.decision_at.eq(latest)].copy()
    if f.security_id.duplicated().any():raise ValueError('ambiguous eligibility snapshot')
    return f


def passive_rows(panel,intervals,ids=None):
    rows=[]
    for period,entry,exit in intervals:
        if ids is not None:
            cutoff=decision_at(entry)
            # Fixed passive membership never depends on current option coverage.
            # Lagged observed caps are metadata only: they do not affect allocation.
            f=panel[panel.security_id.isin(ids)&panel.decision_at.le(cutoff)&panel.available_at.le(cutoff)]
            f=f.sort_values('decision_at').groupby('security_id',sort=False).tail(1).copy()
            if set(f.security_id)!=set(ids):raise ValueError('fixed 25 historical metadata incomplete; do not shrink universe')
        else:f=latest_snapshot(panel,entry)
        f['snapshot_decision_at']=f.decision_at
        f=f[['security_id','market_cap','available_at','trade_date','snapshot_decision_at']].copy()
        f['decision_at']=decision_at(entry);f['period']=period;f['entry_date']=entry;f['exit_date']=exit
        f['score']=0.;f['target']=np.nan;rows.append(f)
    return pd.concat(rows,ignore_index=True)


def fixed_model_rows(pred,ids):
    f=pred[pred.security_id.isin(ids)].copy()
    for _,g in f.groupby('period'):
        if set(g.security_id)!=set(ids) or not np.isfinite(g.score).all():
            raise ValueError('all 25 configured stocks require finite decision-time model scores')
    if f.period.nunique()!=pred.period.nunique():raise ValueError('missing fixed-universe decision')
    return f


def validate_spy(frame,start,end):
    needed={'trade_date','ticker','security_id','close','adjusted_close'}
    if not needed.issubset(frame):raise ValueError('SPY close and adjusted_close required')
    if set(frame.ticker)!= {'SPY'} or set(frame.security_id)!= {'benchmark:SPY'}:
        raise ValueError('SPY adapter accepts no other instrument')
    if frame.trade_date.duplicated().any():raise ValueError('duplicate SPY date')
    expect=sessions(start,end)
    q=frame.set_index('trade_date').reindex(expect)
    if not np.isfinite(q[['close','adjusted_close']]).all().all() or (q[['close','adjusted_close']]<=0).any().any():
        raise ValueError('incomplete/nonpositive SPY raw or adjusted prices')
    return q.reset_index(names='trade_date')


def fetch_spy(out):
    """Explicit narrow network operation; never repairs stock or option data."""
    out=Path(out)
    if out.exists():raise FileExistsError('SPY freeze requires a fresh directory')
    out.mkdir(parents=True,mode=0o700)
    start=min(x[0] for x in WINDOWS.values());end=max(x[1] for x in WINDOWS.values())
    metadata=dict(ticker='SPY',purpose='passive S&P 500 ETF total-return proxy only',
                  start=start,end=end,retrieved_at=datetime.now(timezone.utc).isoformat(),
                  provider='Yahoo Finance through yfinance',auto_adjust=False,repair=False)
    try:
        import yfinance as yf
        metadata['provider_version']=version('yfinance')
        q=yf.Ticker('SPY').history(start=start,end=str((pd.Timestamp(end)+pd.Timedelta(days=1)).date()),
                                 auto_adjust=False,actions=True,repair=False,raise_errors=True)
        if q.empty or 'Adj Close' not in q:raise ValueError('SPY adjusted history unavailable')
        q=q.reset_index().rename(columns={'Date':'trade_date','Close':'close','Adj Close':'adjusted_close',
                                         'Dividends':'dividend','Stock Splits':'split'})
        q['trade_date']=pd.to_datetime(q.trade_date).dt.tz_localize(None).dt.normalize()
        q['ticker']='SPY';q['security_id']='benchmark:SPY'
        q=validate_spy(q,start,end)
        q.to_parquet(out/'spy_prices.parquet',index=False)
        metadata.update(status='complete',rows=len(q),sha256=sha(out/'spy_prices.parquet'))
    except Exception as exc:
        metadata.update(status='unavailable',reason=str(exc))
    dump(out/'metadata.json',metadata)
    print(json.dumps(metadata,indent=2),flush=True)


def case_spec(name):
    if name=='ml_broad':return PortfolioSpec('equal',mode='long_only',selection=POLICIES['buffer'])
    if name=='ml25':return PortfolioSpec('equal',mode='long_only',selection_mode='fixed_count',selection_count=20)
    return PortfolioSpec('equal',mode='long_only',selection_mode='all',max_position=1. if name=='spy' else .05)


def case_inputs(panel,pred,ids,start,end,spy):
    intervals=[(g.period.iloc[0],g.entry_date.iloc[0],g.exit_date.iloc[0]) for _,g in pred.groupby('period',sort=True)]
    months=monthly_intervals(start,end)
    builders=dict(fixed25_monthly=lambda:passive_rows(panel,months,ids),fixed25_four_week=lambda:passive_rows(panel,intervals,ids),
                  broad_monthly=lambda:passive_rows(panel,months),broad_four_week=lambda:pred.assign(score=0.),
                  ml25=lambda:fixed_model_rows(pred,ids),ml_broad=lambda:pred)
    result={};failures={}
    for name,builder in builders.items():
        try:result[name]=builder()
        except ValueError as exc:failures[name]={k:dict(status='invalid',reason=str(exc)) for k in ('net','gross')}
    if spy is not None:
        result['spy']=pd.DataFrame([dict(period='buy_hold',security_id='benchmark:SPY',market_cap=1.,score=0.,
                                        entry_date=pd.Timestamp(start),exit_date=pd.Timestamp(end))])
    return result,failures


def price_gaps(pred,prices):
    """Audit candidate coverage without using it to change selection."""
    pivot=prices.pivot(index='trade_date',columns='security_id',values='adjusted_close')
    rows=[]
    for period,f in pred.groupby('period',sort=True):
        expected=sessions(f.entry_date.iloc[0],f.exit_date.iloc[0]);a=pivot.reindex(index=expected,columns=f.security_id)
        for i,j in zip(*np.where(~np.isfinite(a.to_numpy())|(a.to_numpy()<0))):
            rows.append(dict(period=period,date=expected[i],security_id=str(a.columns[j]),reason='missing/invalid candidate adjusted mark'))
    return rows


def evaluate(pred,prices,name,out,boundaries):
    out.mkdir(parents=True,exist_ok=True);pred.to_parquet(out/'candidates.parquet',index=False)
    dump(out/'candidate_price_gaps.json',price_gaps(pred,prices))
    expected=sessions(pred.entry_date.min(),pred.exit_date.max())
    if not expected.isin(pd.DatetimeIndex(prices.trade_date.unique())).all():
        result={k:dict(status='invalid',reason='entire required exchange session absent') for k in ('gross','net')}
        dump(out/'metrics.json',result);return result
    prepared=prepare_replay(pred,prices);result={}
    for label,cost in [('gross',GROSS),('net',NET)]:
        try:
            run=ledger(prepared,portfolio=case_spec(name),costs=cost,initial_capital=100000.)
            checks=audit(run);result[label]=performance(run,boundaries)
            result[label]['average_invested_exposure']=float(run['daily'].gross_exposure.mean())
            result[label]['net_profit']=run['final_equity']-100000.
            for field in ('daily','periodic','holdings','trades'):run[field].to_parquet(out/(label+'_'+field+'.parquet'),index=False)
            weekly_equity(run,boundaries).to_frame('return').to_parquet(out/(label+'_weekly.parquet'))
            dump(out/(label+'_verification.json'),checks)
            fast=ledger(prepared,portfolio=case_spec(name),costs=cost,initial_capital=100000.,capture=False)
            if fast['equity_path']!=run['equity_path']:
                raise AssertionError('capture-free ledger differs')
        except InvalidPortfolio as exc:result[label]=dict(status='invalid',reason=str(exc))
    dump(out/'metrics.json',result);return result


def model_inference(name,pred,prices,out,bounds,observed,indices,iterations):
    out.mkdir(parents=True,exist_ok=True);ps=prepare_replay(pred,prices);minimum=20 if name=='ml25' else 200
    observed_ic=ic_series(pred,ps,minimum);records=[]
    for draw in range(iterations):
        randomized=shuffled_periods(ps,draw);record=dict(draw=draw,mean_rank_ic=float(ic_series(pred,randomized,minimum).mean()))
        for label,cost in [('net',NET),('gross',GROSS)]:
            try:
                r=ledger(randomized,portfolio=case_spec(name),costs=cost,initial_capital=100000.,capture=False)
                record.update({label+'_mean_weekly_return':float(weekly_equity(r,bounds).mean()),label+'_status':'complete',
                               label+'_turnover':r['total_turnover']/len(bounds[1:])})
                for component,value in r['cost_totals'].items():record[label+'_'+component]=value
            except InvalidPortfolio as exc:record.update({label+'_mean_weekly_return':np.nan,label+'_status':str(exc)})
        records.append(record)
        if (draw+1)%500==0:print('null',name,draw+1,'/',iterations,flush=True)
    draws=pd.DataFrame(records);draws.to_parquet(out/'random_draws.parquet',index=False)
    stats=dict(iterations=iterations,portfolio_family=PORTFOLIO_FAMILY,ranking_family=RANKING_FAMILY,
               mean_rank_ic=float(observed_ic.mean()),ranking=bounded_p(draws.mean_rank_ic,float(observed_ic.mean()),iterations,RANKING_FAMILY),
               limitation='exploratory; period shuffling changes persistence and turnover; 25-stock size deciles contain only 2-3 names')
    for label in ('net','gross'):
        mean=observed[label].get('mean_weekly_return')
        stats[label]=bounded_p(draws[label+'_mean_weekly_return'],mean,iterations,PORTFOLIO_FAMILY)
    # Place signal-date IC on shared weekly blocks, without pretending there are 24 decisions.
    weeks=np.full(len(bounds)-1,np.nan);pos=bounds[:-1].get_indexer(pd.DatetimeIndex([p['days'][0] for p in ps]))
    weeks[pos]=observed_ic;sample=weeks[indices];counts=np.isfinite(sample).sum(axis=1)
    means=np.divide(np.nansum(sample,axis=1),counts,out=np.full(iterations,np.nan),where=counts>0)
    pd.DataFrame(dict(mean_rank_ic=means)).to_parquet(out/'ranking_bootstrap.parquet',index=False)
    stats['mean_ic_95pct']=np.nanquantile(means,[.025,.975]).tolist() if np.isfinite(means).any() else None
    dump(out/'statistics.json',stats);return stats


def paired_comparisons(root,reports,indices,iterations):
    results={}
    pairs=[('primary_ml25','ml25','fixed25_four_week'),('broader_ml','ml_broad','broad_four_week'),
           ('monthly_vs_four_week25','fixed25_monthly','fixed25_four_week')]
    for name in reports:
        if name!='spy':pairs.append((name+'_vs_spy',name,'spy'))
    for label,a,b in pairs:
        if a not in reports or b not in reports or any(reports[x]['net']['status']!='complete' for x in (a,b)):
            results[label]=dict(status='unavailable: incomplete portfolio or benchmark');continue
        x=pd.read_parquet(root/a/'net_weekly.parquet')['return'].to_numpy()
        y=pd.read_parquet(root/b/'net_weekly.parquet')['return'].to_numpy()
        delta=x-y;boot=delta[indices].mean(axis=1)
        result=dict(status='complete',mean_weekly_excess=float(delta.mean()),
                    cumulative_return_difference=reports[a]['net']['cumulative_return']-reports[b]['net']['cumulative_return'],
                    mean_excess_95pct=np.quantile(boot,[.025,.975]).tolist())
        compound=np.prod(1+x[indices],axis=1)-np.prod(1+y[indices],axis=1)
        result['cumulative_difference_95pct']=np.quantile(compound,[.025,.975]).tolist()
        pd.DataFrame(dict(mean_weekly_excess=boot,cumulative_difference=compound)).to_parquet(root/'inference'/(label+'_paired_bootstrap.parquet'),index=False)
        if a in ('ml25','ml_broad'):
            d=pd.read_parquet(root/'inference'/a/'random_draws.parquet')
            result['selection_randomization']=bounded_p(d.net_mean_weekly_return-y.mean(),float(delta.mean()),iterations,PORTFOLIO_FAMILY)
        results[label]=result
    dump(root/'paired_comparisons.json',results);return results


def run(panel_path,price_path,reference,out,spy_path=None,spy_metadata=None,cores=2,iterations=2000):
    _limit_cores(cores)
    if iterations<2:raise ValueError('need at least two simulations')
    out=Path(out).resolve();reference=Path(reference).resolve()
    paths=dict(panel=Path(panel_path).resolve(),prices=Path(price_path).resolve(),reference_manifest=reference/'manifest.json',
               reference_metrics=reference/'historical/rules/metrics.json',config=Path(__file__).parents[1]/'Data_Pipeline/config.py')
    for window in WINDOWS:paths[window+'_predictions']=reference/window/'fits/h4_relative_full_ridge/predictions.parquet'
    if spy_path:paths['spy']=Path(spy_path).resolve()
    if spy_metadata:paths['spy_metadata']=Path(spy_metadata).resolve()
    if out.exists():raise FileExistsError('fresh result directory required')
    if any(out==p or out in p.parents for p in paths.values()) or out==reference or reference in out.parents:
        raise ValueError('outputs must remain separate from inputs')
    hashes={k:sha(p) for k,p in paths.items()};previous=json.loads(paths['reference_manifest'].read_text())
    if any(hashes[k]!=previous['sources'][k] for k in ('panel','prices')):raise ValueError('frozen source hash differs')
    ids=configured_universe(paths['config']);out.mkdir(parents=True,mode=0o700)
    dump(out/'manifest.json',dict(sources=hashes,paths={k:str(p) for k,p in paths.items()},configured_ids=ids,windows=WINDOWS,
         capital=100000,costs=asdict(NET),iterations=iterations,engineering_pilot=iterations<2000,max_cores=cores,seed=17,
         portfolio_family=PORTFOLIO_FAMILY,ranking_family=RANKING_FAMILY,
         code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         software={k:version(k) for k in ('numpy','pandas','scikit-learn','exchange-calendars')},
         interpretation='exploratory frozen-score comparison; no training; fixed configured names are not a historical index universe'))
    panel=pd.read_parquet(paths['panel']);prices=pd.read_parquet(paths['prices'])
    spy=None;spy_status=dict(status='unavailable',reason='no validated frozen SPY prices')
    if spy_path:
        try:
            if not spy_metadata:raise ValueError('SPY provenance metadata required')
            meta=json.loads(paths['spy_metadata'].read_text())
            if meta.get('status')!='complete' or meta.get('sha256')!=hashes['spy']:raise ValueError('SPY hash/provenance mismatch')
            spy=validate_spy(pd.read_parquet(paths['spy']),min(x[0] for x in WINDOWS.values()),max(x[1] for x in WINDOWS.values()))
            spy_status=dict(status='complete',definition='SPY adjusted-price total-return proxy; not the index itself')
        except ValueError as exc:spy_status=dict(status='unavailable',reason=str(exc))
    elif spy_metadata:spy_status=json.loads(paths['spy_metadata'].read_text())
    dump(out/'spy_status.json',spy_status)
    all_reports={};all_inputs={};summaries=[]
    for window,(start,end) in WINDOWS.items():
        folder=out/window;folder.mkdir();pred=pd.read_parquet(paths[window+'_predictions'])
        candidates,failures=case_inputs(panel,pred,ids,start,end,spy);bounds=weekly_boundaries(start,end);reports=dict(failures)
        for name,f in candidates.items():
            reports[name]=evaluate(f,spy if name=='spy' else prices,name,folder/name,bounds)
            print(window,name,reports[name]['net']['status'],reports[name]['net'].get('cumulative_return'),flush=True)
            if reports[name]['net']['status']=='complete' and spy is not None:
                q=spy.set_index('trade_date').adjusted_close.reindex(bounds).to_numpy()
                benchmark=q[1:]/q[:-1]-1
                for cost in ('net','gross'):
                    values=pd.read_parquet(folder/name/(cost+'_weekly.parquet'))['return'].to_numpy()
                    a=alpha_proxy(values,benchmark);a['definition']='SPY total-return benchmark alpha, zero cash rate; not FF5 plus momentum alpha'
                    reports[name][cost]['spy_alpha']=a
                    dump(folder/name/'metrics.json',reports[name])
            for cost in ('net','gross'):
                v=reports[name][cost]
                summaries.append(dict(window=window,portfolio=name,cost=cost,status=v['status'],return_=v.get('cumulative_return'),
                                      profit=v.get('net_profit'),sharpe=v.get('sharpe'),drawdown=v.get('max_drawdown'),
                                      volatility=v.get('annualized_volatility'),reason=v.get('reason')))
        if 'spy' not in reports:reports['spy']={k:dict(status='unavailable',reason=spy_status.get('reason')) for k in ('net','gross')}
        for name in set(reports)-set(candidates):
            for cost,v in reports[name].items():
                summaries.append(dict(window=window,portfolio=name,cost=cost,status=v['status'],reason=v.get('reason')))
        dump(folder/'metrics.json',reports);all_reports[window]=reports;all_inputs[window]=candidates
    # The original valid portfolio is the numerical control for ledger changes.
    control=json.loads(paths['reference_metrics'].read_text())['ml_buffer/long_only'];checks={}
    for cost in ('net','gross'):
        for key in ('cumulative_return','sharpe','max_drawdown','transaction_cost','borrow_cost'):
            error=abs(all_reports['historical']['ml_broad'][cost][key]-control[cost][key]);checks[cost+'/'+key]=error
            if error>1e-8:raise AssertionError('original portfolio control differs: '+cost+'/'+key)
    dump(out/'control_verification.json',checks)
    folder=out/'historical';(folder/'inference').mkdir();bounds=weekly_boundaries(*WINDOWS['historical'])
    indices=bootstrap_indices(len(bounds)-1,iterations);np.save(folder/'paired_block_indices.npy',indices)
    inference={}
    for name in ('ml25','ml_broad'):
        if name not in all_inputs['historical']:
            inference[name]=dict(status='unavailable',reason=all_reports['historical'][name]['net'].get('reason'))
            continue
        inference[name]=model_inference(name,all_inputs['historical'][name],prices,folder/'inference'/name,bounds,
                                         all_reports['historical'][name],indices,iterations)
    dump(folder/'inference/statistics.json',inference)
    for name,m in all_reports['historical'].items():
        if m['net']['status']!='complete':continue
        boot={};intervals={}
        for cost in ('net','gross'):
            r=pd.read_parquet(folder/name/(cost+'_weekly.parquet'))['return'].to_numpy();sample=r[indices]
            means=sample.mean(axis=1);sd=sample.std(axis=1,ddof=1)
            values=dict(mean_weekly_return=means,annualized_return=np.prod(1+sample,axis=1)**(52/len(r))-1,
                        sharpe=np.divide(means*np.sqrt(52),sd,out=np.full(iterations,np.nan),where=sd>0))
            for metric,a in values.items():
                boot[cost+'_'+metric]=a;intervals[cost+'_'+metric+'_95pct']=np.nanquantile(a,[.025,.975]).tolist()
        pd.DataFrame(boot).to_parquet(folder/name/'bootstrap.parquet',index=False);dump(folder/name/'bootstrap_intervals.json',intervals)
    paired_comparisons(folder,all_reports['historical'],indices,iterations)
    pd.DataFrame(summaries).to_csv(out/'comparison.csv',index=False)
    verified={k:sha(p)==hashes[k] for k,p in paths.items()};dump(out/'source_verification.json',verified)
    if not all(verified.values()):raise ValueError('input changed during replay')
    dump(out/'complete.json',dict(status='complete',invalid_results_retained=True,spy=spy_status))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    fetch=sub.add_parser('fetch-spy');fetch.add_argument('--output-dir',required=True)
    runp=sub.add_parser('run')
    for key in ('panel','prices','reference-results','output-dir'):runp.add_argument('--'+key,required=True)
    runp.add_argument('--spy-prices');runp.add_argument('--spy-metadata')
    runp.add_argument('--max-cores',type=int,default=2);runp.add_argument('--iterations',type=int,default=2000)
    a=p.parse_args()
    if a.command=='fetch-spy':fetch_spy(a.output_dir)
    else:run(a.panel,a.prices,a.reference_results,a.output_dir,a.spy_prices,a.spy_metadata,a.max_cores,a.iterations)


if __name__=='__main__':main()
