"""Reevaluate immutable saved predictions. No fitting or market-data acquisition.

python -m Research_Pipeline.reevaluate --help
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict,replace
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from .daily_ledger import prepare,ledger,metrics,InvalidPortfolio
from .portfolio_ledger import PortfolioSpec,CostSpec,prepare_replay
from .factors import newey_west_paired_mean
from .run import _limit_cores

SEED=17
DEFAULT=dict(initial_capital=100000.,max_position=.05,execution_bps=1.,borrow_rate=.03,
             iterations=2000,seed=SEED)


def dump(path,obj):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,(np.floating,float)) and not np.isfinite(x):return None
        if isinstance(x,np.generic):return x.item()
        return x
    Path(path).write_text(json.dumps(clean(obj),indent=2,sort_keys=True,default=str,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions(cfg):
    cap=cfg['initial_capital'];market=PortfolioSpec();equal=PortfolioSpec('equal',cfg['max_position'])
    flat=CostSpec(borrow_rate=cfg['borrow_rate'])
    broker=CostSpec(mode='ibkr_fixed',execution_bps=cfg['execution_bps'],borrow_rate=cfg['borrow_rate'])
    return {
        'original_control':(market,flat,cap),
        'cost_only':(market,broker,cap),
        'weight_only':(equal,flat,cap),
        'both_changes':(equal,broker,cap),
        'gross_market_cap':(market,CostSpec(flat_bps=0,borrow_rate=0),cap),
        'gross_equal':(equal,CostSpec(flat_bps=0,borrow_rate=0),cap),
        'capital_10000':(equal,broker,10000.),
        'capital_1000000':(equal,broker,1000000.),
        'execution_0bps':(equal,replace(broker,execution_bps=0),cap),
        'execution_5bps':(equal,replace(broker,execution_bps=5),cap),
        'zero_borrow':(equal,replace(broker,borrow_rate=0),cap),
    }


def verify_legacy(root,prices):
    checks={}
    for track,freq in [('weekly',52),('paper_diagnostic',12)]:
        pred=pd.read_parquet(root/track/'predictions.parquet')
        old=json.loads((root/track/'metrics.json').read_text())
        saved_h=pd.read_parquet(root/track/'holdings.parquet')
        daily_path=root/'final_audit'/f'{track}_daily_equity.parquet'
        saved_d=pd.read_parquet(daily_path if daily_path.exists() else root/track/'daily_equity.parquet')
        for v,f in pred.groupby('variant'):
            key=track+'/'+v
            try:
                run=ledger(prepare(f,prices));new=metrics(run,freq)
            except InvalidPortfolio as exc:
                if 'primary'in old[v]:raise AssertionError('legacy result unexpectedly invalid: '+key) from exc
                checks[key]={'status':'same invalid strategy','reason':str(exc)};continue
            if 'primary'not in old[v]:raise AssertionError('legacy invalid result became valid: '+key)
            for name,val in old[v]['primary'].items():
                if isinstance(val,(int,float)):
                    assert np.isclose(new[name],val,rtol=1e-9,atol=1e-11),(key,name,new[name],val)
            h=saved_h[saved_h.variant.eq(v)].drop(columns='variant').sort_values(['period','security_id']).reset_index(drop=True)
            n=run['holdings'].sort_values(['period','security_id']).reset_index(drop=True)
            for col in ['entry_date','exit_date']:
                h[col]=h[col].astype('datetime64[ns]');n[col]=n[col].astype('datetime64[ns]')
            pd.testing.assert_frame_equal(h[n.columns],n,check_exact=False,rtol=1e-9,atol=1e-11)
            d=saved_d[saved_d.variant.eq(v)].sort_values('date').reset_index(drop=True)
            for col in ['equity','cash','position_value','transaction_cost','borrow_cost']:
                assert np.allclose(d[col],run['daily'][col],rtol=1e-9,atol=1e-11),(key,col)
            checks[key]={'status':'reproduced','metrics_holdings_daily_equity_and_fees':True}
    return checks


def audit(run):
    capital=run['initial_capital'];tol=max(1e-8,capital*1e-9)
    d=run['daily'];t=run['trades'];cost=run['cost_totals']
    assert np.allclose(d.equity,d.cash+d.position_value,rtol=1e-10,atol=tol)
    for k in ['commission','regulatory','execution']:
        assert np.isclose(t[k].sum(),cost[k],rtol=1e-10,atol=tol)
        assert np.isclose(d[k].sum(),cost[k],rtol=1e-10,atol=tol)
    assert np.isclose(t.flat_cost.sum(),cost['flat'],atol=tol)
    assert np.isclose(d.transaction_cost.sum(),run['total_transaction_cost'],atol=tol)
    assert np.isclose(d.borrow_cost.sum(),run['total_borrow'],atol=tol)
    assert np.isclose(run['periodic'].transaction_cost.sum(),run['total_transaction_cost'],atol=tol)
    assert np.isclose(run['periodic'].borrow_cost.sum(),run['total_borrow'],atol=tol)
    assert np.isclose(np.prod(1+run['period_returns']),run['final_equity']/capital,atol=1e-10)
    assert np.isclose(run['long_pnl']+run['short_pnl']-run['total_transaction_cost']-run['total_borrow'],run['final_equity']-capital,atol=tol)
    assert np.isclose(sum(run['stock_pnl'].values()),run['long_pnl']+run['short_pnl'],atol=tol)
    if run['portfolio']['weighting']=='equal':
        assert run['holdings'].weight.abs().max()<=run['portfolio']['max_position']+1e-10
    return dict(cash_reconciles=True,trades_and_costs_reconcile=True,pnl_reconciles=True,
                returns_compound=True,target_caps_pass=True)


def extra_metrics(run):
    c=run['cost_totals'];h=run['holdings'];d=run['daily'];periods=run['periodic'].period
    nv=h[h.security_id.str.endswith(':NVDA')|h.security_id.eq('NVDA')]
    nw=nv.set_index('period').weight.reindex(periods,fill_value=0).abs()
    out=dict(costs_dollars=c,costs_bps_per_traded_dollar={k:10000*c[k]/c['traded'] if c['traded'] else 0 for k in ['commission','regulatory','execution','flat','borrow']},
        costs_pct_starting_capital={k:c[k]/run['initial_capital'] for k in ['commission','regulatory','execution','flat','borrow']},
        largest_target_position=float(h.weight.abs().max()),largest_daily_position=float(d.max_abs_weight.max()),
        daily_cap_breach_days=int(d.cap_breach.sum()),minimum_effective_long=float(d.loc[d.gross_exposure.gt(0),'effective_long'].min()),
        minimum_effective_short=float(d.loc[d.gross_exposure.gt(0),'effective_short'].min()),
        nvda=dict(periods_held=len(nv),total_periods=len(periods),average_abs_target_weight=float(nw.mean()),
                  maximum_abs_target_weight=float(nw.max()),maximum_abs_daily_weight=float(d.nvda_weight.abs().max()),
                  pnl_dollars=float(nv.pnl.sum()),pnl_pct_starting_capital=float(nv.pnl.sum()/run['initial_capital'])),
        long_pnl_dollars=run['long_pnl'],short_pnl_dollars=run['short_pnl'],stock_pnl_dollars=run['stock_pnl'])
    return out


def alpha_proxy(r,b):
    y=np.asarray(r);x=np.column_stack([np.ones(len(y)),b]);beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta;z=x*res[:,None];meat=z.T@z
    for lag in range(1,min(4,len(y)-1)+1):
        cross=z[lag:].T@z[:-lag];meat+=(1-lag/5)*(cross+cross.T)
    inv=np.linalg.pinv(x.T@x);cov=inv@meat@inv*len(y)/(len(y)-2);se=np.sqrt(max(0,cov[0,0]))
    return dict(alpha_weekly=float(beta[0]),alpha_annualized_linear=float(52*beta[0]),beta=float(beta[1]),newey_west_lags=4,t=float(beta[0]/se) if se else None,definition='internal-universe alpha proxy; not factor alpha')


def simulation(prepared,observed,portfolio,cost,capital,freq,iterations,seed,family):
    rng=np.random.default_rng(seed);draws=[];net_values=[];gross_values=[]
    gross_cost=CostSpec(flat_bps=0,borrow_rate=0)
    # The same per-draw seed gives net and gross identical score permutations.
    for i in range(iterations):
        draw_seed=int(rng.integers(0,2**63-1));row={'draw':i,'seed':draw_seed}
        for label,spec,values in [('net',cost,net_values),('gross',gross_cost,gross_values)]:
            try:
                r=ledger(prepared,portfolio=portfolio,costs=spec,initial_capital=capital,
                         rng=np.random.default_rng(draw_seed),capture=False)
                row[label+'_mean_return']=float(r['period_returns'].mean());values.append(row[label+'_mean_return'])
                row[label+'_turnover']=r['total_turnover']/len(prepared)
                for k in ['commission','regulatory','execution','flat','borrow']:
                    row[label+'_'+k]=r['cost_totals'][k]
                row[label+'_status']='complete'
            except InvalidPortfolio as exc:
                row[label+'_status']=str(exc)
        draws.append(row)
        if (i+1)%250==0:print(f'  simulations {i+1}/{iterations}',flush=True)
    invalid=iterations-len(net_values);out=dict(iterations=iterations,seed=seed,invalid_null_draws=invalid,
        invalid_gross_draws=iterations-len(gross_values),status='complete' if invalid==0 and observed is not None else 'inconclusive',
        p_raw=None,p_adjusted=None,p_bounds=None,adjusted_p_bounds=None,
        limitation='period-wise shuffling changes persistence and turnover; size-conditioned null only; inspected historical holdout')
    if observed is not None:
        r=observed['period_returns'];hits=int(np.sum(np.array(net_values)>=r.mean()))
        bounds=[(1+hits)/(iterations+1),(1+hits+invalid)/(iterations+1)]
        out.update(p_bounds=bounds,adjusted_p_bounds=[min(1,x*family) for x in bounds],
                   p_raw=bounds[0] if invalid==0 else None,p_adjusted=min(1,bounds[0]*family) if invalid==0 else None)
        block=4 if freq==52 else 3;boot=[]
        for _ in range(iterations):
            starts=rng.integers(0,len(r)-block+1,size=int(np.ceil(len(r)/block)))
            sample=np.concatenate([r[j:j+block] for j in starts])[:len(r)];sd=sample.std(ddof=1)
            boot.append([sample.mean(),np.prod(1+sample)**(freq/len(sample))-1,sample.mean()/sd*np.sqrt(freq) if sd else np.nan])
        a=np.array(boot);out.update(block_periods=block,mean_return_95pct=np.quantile(a[:,0],[.025,.975]).tolist(),
            annualized_return_95pct=np.quantile(a[:,1],[.025,.975]).tolist(),
            sharpe_95pct=np.nanquantile(a[:,2],[.025,.975]).tolist() if np.isfinite(a[:,2]).any() else [None,None])
    else:a=np.empty((0,3))
    return out,pd.DataFrame(draws),pd.DataFrame(a,columns=['mean_return','annualized_return','sharpe'])


def run(root,prices_path,out,config=None,max_cores=4):
    _limit_cores(max_cores);root=Path(root);prices_path=Path(prices_path);out=Path(out)
    cfg={**DEFAULT,**(config or {})}
    if set(cfg)!=set(DEFAULT):raise ValueError('unrecognized configuration key')
    if cfg['iterations']<1:raise ValueError('iterations must be positive')
    for _,(p,c,capital) in conditions(cfg).items():
        if not np.isfinite(capital) or capital<=0:raise ValueError('invalid capital')
    if out.exists():raise FileExistsError('fresh output directory required')
    sources=[prices_path,*[root/t/'predictions.parquet' for t in ['weekly','paper_diagnostic']]]
    source_hashes={str(p):sha(p) for p in sources}
    expected=json.loads((root/'final_audit/artifact_hashes.json').read_text())
    for p in sources:
        key='handoff/stock_prices.parquet' if p==prices_path else str(p.relative_to(root))
        if source_hashes[str(p)]!=expected[key]:raise ValueError('source differs from frozen audit: '+key)
    prices=pd.read_parquet(prices_path);checks=verify_legacy(root,prices)
    print('Legacy predictions, holdings, equity and fees reproduced',flush=True)
    out.mkdir(parents=True);dump(out/'legacy_verification.json',checks)
    manifest=dict(configuration=cfg,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes=source_hashes,source_results=str(root),max_cores=max_cores,
        fee_schedule=asdict(CostSpec(mode='ibkr_fixed')),assumptions=[
        'exploratory revision on inspected test period; no refitting or new market data',
        'current commission/regulatory schedule applied as constant scenario, not historical charges',
        'whole-share commission schedule on fractional share equivalents; not exact broker execution',
        '1 bp joint spread/slippage allowance is assumed, not measured',
        '200% target gross exposure; zero cash interest; financing and borrow availability unverified',
        'target caps at rebalance only; daily drift is measured, not traded',
        'missing prices and corporate actions unchanged; no historical repairs'])
    dump(out/'manifest.json',manifest);summary=[];all_metrics={}
    for track,freq in [('paper_diagnostic',12),('weekly',52)]:
        dest=out/track;dest.mkdir();pred=pd.read_parquet(root/track/'predictions.parquet');reports={};sims={};periodic={}
        old_metrics=json.loads((root/track/'metrics.json').read_text())
        for variant,f in pred.groupby('variant',sort=True):
            print(track,variant,flush=True);prepared=prepare_replay(f,prices);reports[variant]={};main=None
            for condition,(portfolio,cost,capital) in conditions(cfg).items():
                location=dest/variant.replace('/','_')/condition;location.mkdir(parents=True)
                try:
                    result=ledger(prepared,portfolio=portfolio,costs=cost,initial_capital=capital)
                    verification=audit(result);m=metrics(result,freq);m.update(extra_metrics(result));m['status']='complete'
                    m['alpha_proxy']={'status':'unavailable: factor data not supplied'}
                    if freq==52:
                        benchmark=[float(np.mean(p['all_prices'][-1,p['index']]/p['all_prices'][0,p['index']]-1)) for p in prepared]
                        m['alpha_proxy']=alpha_proxy(result['period_returns'],benchmark) if np.isfinite(benchmark).all() else {'status':'unavailable: missing benchmark prices'}
                    reports[variant][condition]=m
                    for name in ['daily','periodic','holdings','trades']:result[name].to_parquet(location/f'{name}.parquet',index=False)
                    dump(location/'verification.json',verification)
                    periodic[(variant,condition)]=result['periodic']
                    if condition=='both_changes':main=result
                    if condition=='original_control' and 'primary'in old_metrics[variant]:
                        for field in ['cumulative_return','sharpe','max_drawdown','mean_periodic_return','mean_turnover']:
                            assert np.isclose(m[field],old_metrics[variant]['primary'][field],rtol=1e-8,atol=1e-10),(variant,field)
                    summary.append(dict(track=track,variant=variant,condition=condition,net_return=m['cumulative_return'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],status='complete'))
                except InvalidPortfolio as exc:
                    reports[variant][condition]={'status':'invalid','reason':str(exc)}
                    summary.append(dict(track=track,variant=variant,condition=condition,status='invalid',reason=str(exc)))
            if freq==52:
                ic={str(p):float(g[['score','target']].corr(method='spearman').iloc[0,1]) for p,g in f.groupby('period')}
                if 'rank_ic_by_week'in old_metrics[variant]:
                    assert ic==old_metrics[variant]['rank_ic_by_week'],'rank IC changed'
                reports[variant]['rank_ic_by_week']=ic
            portfolio,cost,capital=conditions(cfg)['both_changes']
            s,draws,bootstrap=simulation(prepared,main,portfolio,cost,capital,freq,cfg['iterations'],cfg['seed'],7 if freq==52 else 4)
            sims[variant]=s;loc=dest/variant.replace('/','_');draws.to_parquet(loc/'random_draws.parquet',index=False);bootstrap.to_parquet(loc/'bootstrap.parquet',index=False)
            dump(dest/'metrics.json',reports);dump(dest/'monte_carlo.json',sims)
        paired={}
        if freq==52:
            for (v,condition),ret in periodic.items():
                baseline=('stock_only/'+v.split('/')[1],condition)
                if v.startswith(('options_only/','combined/')) and baseline in periodic:
                    a=ret.rename(columns={'net_return':'gross_return'});b=periodic[baseline].rename(columns={'net_return':'gross_return'})
                    paired[v+'/'+condition]=newey_west_paired_mean(a,b,lags=4)
        dump(dest/'incremental_options_value.json',paired);all_metrics[track]=reports
    for path,value in source_hashes.items():assert sha(path)==value,'frozen input changed'
    dump(out/'source_verification.json',dict(hashes_unchanged=True,predictions_unchanged=True,no_model_fitting=True))
    pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
    lines=['# Portfolio and cost revision','', '**Exploratory: the historical holdout has already been inspected.**',
           '', 'Saved predictions and input prices are unchanged. See manifest.json for fee assumptions.',
           '', '| Track | Model | Condition | Net return | Sharpe | Daily drawdown |', '|---|---|---|---:|---:|---:|']
    for row in summary:
        if row['status']=='complete':lines.append(f"| {row['track']} | {row['variant']} | {row['condition']} | {row['net_return']:.2%} | {row['sharpe']:.2f} | {row['max_drawdown']:.2%} |")
        else:lines.append(f"| {row['track']} | {row['variant']} | {row['condition']} | Unavailable | | |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Finished:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--prices',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--config');p.add_argument('--max-cores',type=int,default=4)
    a=p.parse_args();config=json.loads(Path(a.config).read_text()) if a.config else None
    run(a.results,a.prices,a.output_dir,config,a.max_cores)

if __name__=='__main__':main()
