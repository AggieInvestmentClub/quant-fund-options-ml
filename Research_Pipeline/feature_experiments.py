"""Fixed weekly feature ablation on frozen data; all inference is exploratory."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse,json,hashlib,subprocess,time
from pathlib import Path
from dataclasses import asdict
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .strategy_data import STOCK,OPTION
from .strategy_experiments import pipeline
from .daily_ledger import ledger,metrics,InvalidPortfolio
from .portfolio_ledger import prepare_replay,PortfolioSpec,CostSpec
from .reevaluate import dump,sha,audit,extra_metrics,alpha_proxy
from .run import _limit_cores

SEED=17
COMPACT=['iv_level','iv_gap','avg_call_delta','avg_call_gamma','avg_call_vega',
         'risk_reversal_25_delta_60d','put_skew_10_25_30d','atm_term_slope']
ALTERNATIVES=['iv_gap_change_1w','iv_gap_change_4w','rr60_change_1w','iv_realized_variance_proxy']
GROUPS={'stock_only':STOCK,'full':STOCK+OPTION,'compact':STOCK+COMPACT,
        'compact_plus':STOCK+COMPACT+ALTERNATIVES}
VARIANTS=[g+'/'+m for m in ('ridge','lightgbm') for g in GROUPS]
DEFINITIONS={'iv_level':'(avg_call_iv + avg_put_iv) / 2',
 'iv_gap':'avg_call_iv - avg_put_iv',
 'iv_gap_change_1w':'iv_gap[t] - iv_gap[t-1 scheduled week], same security',
 'iv_gap_change_4w':'iv_gap[t] - iv_gap[t-4 scheduled weeks], same security',
 'rr60_change_1w':'risk_reversal_25_delta_60d[t] - value[t-1 scheduled week], same security',
 'iv_realized_variance_proxy':'iv_level**2 - 252 * vol_20**2; unmatched horizons, not measured variance risk premium'}
PRIMARY='compact_plus/ridge';REFERENCE='full/ridge'


def features(panel):
    """Preserve original row order and eligibility; no outcome enters a feature."""
    f=panel.copy()
    if f.duplicated(['period','security_id']).any():raise ValueError('duplicate stock-week')
    if not (f.available_at<=f.decision_at).all():raise ValueError('feature available after decision')
    if not (f.trade_date.dt.tz_localize('UTC')<=f.decision_at).all():raise ValueError('observation after decision')
    periods=pd.PeriodIndex(f.period,freq='W-FRI')
    if not np.array_equal(periods.asi8,pd.PeriodIndex(f.trade_date.dt.to_period('W-FRI')).asi8):
        raise ValueError('signal date inconsistent with scheduled week')
    source=f.set_index(['security_id','period'])
    f['iv_level']=(f.avg_call_iv+f.avg_put_iv)/2
    if not np.allclose(f.iv_gap,f.avg_call_iv-f.avg_put_iv,equal_nan=True):raise ValueError('saved IV gap differs')
    for name,column,lag in [('iv_gap_change_1w','iv_gap',1),('iv_gap_change_4w','iv_gap',4),
                            ('rr60_change_1w','risk_reversal_25_delta_60d',1)]:
        keys=pd.MultiIndex.from_arrays([f.security_id,(periods-lag).astype(str)])
        old=source.reindex(keys)
        values=old[column].to_numpy(float,copy=True)
        ready=old.available_at.to_numpy()<=f.decision_at.to_numpy()
        values[~ready]=np.nan
        f[name]=f[column].to_numpy(float)-values
    f['iv_realized_variance_proxy']=f.iv_level**2-252*f.vol_20**2
    assert len(f)==len(panel)
    for group,cols in GROUPS.items():
        if len(cols)!=len(set(cols)):raise ValueError('duplicate feature in '+group)
        if np.isinf(f[cols].to_numpy(float)).any():raise ValueError('infinite feature in '+group)
    return f


def schedules(panel):
    s=panel[['period','trade_date','decision_at','entry_date','exit_date','label_available_at']].drop_duplicates().sort_values('trade_date')
    if s.period.duplicated().any():raise ValueError('ambiguous weekly schedule')
    earlier=s[s.trade_date.between('2025-05-02','2026-04-24')].copy()
    historical=s[s.entry_date.ge('2026-05-11') & s.exit_date.le('2026-11-09')].copy()
    if len(earlier)!=52 or len(historical)!=26:raise ValueError('expected 52 assessment and 26 historical weeks')
    if earlier.trade_date.iloc[0]!=pd.Timestamp('2025-05-02') or earlier.trade_date.iloc[-1]!=pd.Timestamp('2026-04-24'):
        raise ValueError('assessment boundaries changed')
    if not earlier.label_available_at.lt(historical.decision_at.min()).all():raise ValueError('assessment outcomes not mature at historical start')
    return {'assessment':earlier,'historical':historical}


def training(panel,decision):
    f=panel[panel.label_available_at.lt(decision)&panel.target.notna()]
    if not np.isfinite(f.target).all():raise ValueError('nonfinite training label')
    if f.period.nunique()<52:raise ValueError('fewer than 52 matured training weeks')
    assert f.label_available_at.max()<decision
    return f


def fit_scores(panel,schedule,variants,out,source_hashes):
    frames=[];windows=[];out=Path(out);(out/'models').mkdir(parents=True,exist_ok=True)
    diagnostics=out/'training_diagnostics';diagnostics.mkdir(exist_ok=True)
    union=list(dict.fromkeys(GROUPS['full']+GROUPS['compact_plus']))
    for n,row in enumerate(schedule.itertuples(index=False)):
        tr=training(panel,row.decision_at);f=panel[panel.period.eq(row.period)]
        if len(f)<200:raise ValueError('fewer than 200 eligible stocks')
        window=dict(period=row.period,decision_at=row.decision_at,training_weeks=tr.period.nunique(),training_rows=len(tr),
                    first_signal=tr.trade_date.min(),last_signal=tr.trade_date.max(),latest_outcome=tr.label_available_at.max())
        windows.append(window);key=row.period.replace('/','_')
        if not (diagnostics/(key+'.json')).exists():
            finite=tr[union].notna().astype(np.int64)
            dump(diagnostics/(key+'.json'),dict(window=window,feature_order=union,
                 nonmissing=tr[union].notna().sum().to_dict(),missing_fraction=tr[union].isna().mean().to_dict(),
                 pearson=tr[union].corr(min_periods=50).to_dict(),pair_counts=(finite.T@finite).to_numpy(),
                 scoring_missing_fraction=f[union].isna().mean().to_dict(),
                 interpretation='training window only, before imputation; pairwise counts differ; no automatic selection'))
        for variant in variants:
            group,kind=variant.split('/');cols=GROUPS[group]
            model=pipeline(kind).fit(tr[cols],tr.target)
            pred=f[['period','security_id','market_cap','trade_date','entry_date','exit_date','target']].copy()
            pred['score']=model.predict(f[cols]);pred['variant']=variant
            if not np.isfinite(pred.score).all():raise ValueError('nonfinite prediction')
            frames.append(pred)
            joblib.dump(dict(model=model,features=cols,definitions=DEFINITIONS,training=window,
                source_hashes=source_hashes,target='next scheduled entry-to-exit adjusted stock return',
                schema_version=1,feature_group=group),out/'models'/f'{key}_{group}_{kind}.joblib')
        print(f'fit {out.name} week {n+1}/{len(schedule)} ({len(variants)} variants)',flush=True)
    return pd.concat(frames,ignore_index=True),windows


def reproduce(pred,reference):
    checks={}
    for v,f in pred.groupby('variant'):
        group,kind=v.split('/');old=('combined' if group=='full' else group)+'/'+kind
        saved=reference[reference.variant.eq(old)]
        joined=f.merge(saved[['period','security_id','score','target']],on=['period','security_id'],how='outer',
                       validate='one_to_one',suffixes=('_new','_saved'),indicator=True)
        if not joined._merge.eq('both').all():raise AssertionError('control stock-week keys changed: '+v)
        if not np.allclose(joined.target_new,joined.target_saved,rtol=0,atol=0,equal_nan=True):raise AssertionError('targets changed')
        error=float(np.max(np.abs(joined.score_new-joined.score_saved)))
        if error>1e-8:raise AssertionError(f'control predictions differ: {v}/{error}')
        checks[v]=dict(rows=len(joined),max_absolute_prediction_error=error,tolerance=1e-8)
    return checks


def ranked_correlation(scores,targets,min_known=200):
    mask=np.isfinite(targets)
    if mask.sum()<min_known:return np.full(scores.shape[1],np.nan)
    a=rankdata(scores[mask],axis=0);a-=a.mean(axis=0)
    b=rankdata(targets[mask]);b-=b.mean();den=np.sqrt(np.sum(a*a,axis=0)*(b@b))
    return np.divide(b@a,den,out=np.full(scores.shape[1],np.nan),where=den>0)


def prediction_metrics(pred,min_known=200):
    rows=[]
    for (v,period),f in pred.groupby(['variant','period'],sort=True):
        f=f.sort_values('security_id');mask=np.isfinite(f.target);known=f[mask]
        valid=len(known)>=min_known
        ic=ranked_correlation(f.score.to_numpy()[:,None],f.target.to_numpy(),min_known)[0]
        row=dict(variant=v,period=period,signal_date=f.trade_date.iloc[0],eligible=len(f),known_outcomes=len(known),
                 missing_outcomes=int((~mask).sum()),rank_ic=ic,
                 mse=float(np.mean((known.score-known.target)**2)) if valid else np.nan)
        row.update({f'quintile_{j}':np.nan for j in range(1,6)});row['quintile_spread']=np.nan
        if valid:
            # Stable id tie handling; all variants share the same known-outcome keys.
            ordered=known.sort_values(['score','security_id']);buckets=np.array_split(np.arange(len(ordered)),5)
            for j,ix in enumerate(buckets):row[f'quintile_{j+1}']=float(ordered.iloc[ix].target.mean())
            row['quintile_spread']=row['quintile_5']-row['quintile_1']
        rows.append(row)
    return pd.DataFrame(rows)


def assert_common_keys(pred,expected):
    keys=['period','security_id'];base=expected[keys+['target']].sort_values(keys).reset_index(drop=True)
    for _,f in pred.groupby('variant'):
        a=f[keys+['target']].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,base,check_dtype=False,check_exact=True)


def predict_and_evaluate(pred,prices,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);pm=prediction_metrics(pred);pm.to_parquet(out/'ranking_by_week.parquet',index=False)
    reports={};runs={};prepared={}
    portfolio=PortfolioSpec('equal',.05);cost=CostSpec(mode='ibkr_fixed',execution_bps=1,borrow_rate=.03)
    for variant,f in pred.groupby('variant',sort=True):
        rows=pm[pm.variant.eq(variant)];reports[variant]=dict(mean_ic=float(rows.rank_ic.mean()),
            mean_mse=float(rows.mse.mean()),mean_quintile_spread=float(rows.quintile_spread.mean()),
            scoring_weeks=int(rows.rank_ic.notna().sum()),periods=len(rows),portfolio={})
        prepared[variant]=prepare_replay(f,prices);d=out/variant.replace('/','_');d.mkdir(exist_ok=True)
        for label,spec in [('net',cost),('gross',CostSpec(flat_bps=0,borrow_rate=0))]:
            try:
                r=ledger(prepared[variant],portfolio=portfolio,costs=spec,initial_capital=100000.)
                audit(r);m=metrics(r,52);m.update(extra_metrics(r));m['status']='complete'
                benchmark=[np.mean(p['all_prices'][-1,p['index']]/p['all_prices'][0,p['index']]-1) for p in prepared[variant]]
                m['alpha_proxy']=alpha_proxy(r['period_returns'],benchmark) if np.isfinite(benchmark).all() else {'status':'unavailable: missing benchmark prices'}
                reports[variant]['portfolio'][label]=m;runs[(variant,label)]=r
                for name in ['daily','periodic','holdings','trades']:r[name].to_parquet(d/f'{label}_{name}.parquet',index=False)
                dump(d/f'{label}_verification.json',audit(r))
            except InvalidPortfolio as exc:reports[variant]['portfolio'][label]={'status':'invalid','reason':str(exc)}
    dump(out/'metrics.json',reports)
    # Reuse shared immutable price arrays after validating identical candidate keys.
    first=sorted(prepared)[0];base=prepared[first]
    for v,ps in prepared.items():
        for a,b in zip(base,ps):
            assert a['period']==b['period'] and np.array_equal(a['ids'],b['ids'])
            assert np.array_equal(a['caps'],b['caps'])
    return reports,runs,prepared,pm


def paired_statistics(pm):
    pivot=pm.pivot(index='period',columns='variant',values='rank_ic').sort_index();out={}
    for kind in ['ridge','lightgbm']:
        pairs=[('compact','full'),('compact_plus','compact'),('compact_plus','full')]+[(g,'stock_only') for g in ['full','compact','compact_plus']]
        for a,b in pairs:
            diff=pivot[a+'/'+kind]-pivot[b+'/'+kind];out[a+'/'+kind+' minus '+b+'/'+kind]=dict(mean_ic_difference=float(diff.mean()),weekly=diff.to_dict())
    return out


def simulations(pred,prepared,observed,pm,out,iterations=2000,seed=SEED,min_known=200):
    """Common permutations and block indices across all eight fitted variants."""
    out=Path(out);variants=sorted(prepared);family=len(variants);base=prepared[variants[0]]
    matrices=[];targets=[]
    for pi,p in enumerate(base):
        matrices.append(np.column_stack([prepared[v][pi]['score'] for v in variants]))
        f=pred[(pred.variant.eq(variants[0])) & pred.period.eq(p['period'])].sort_values('security_id')
        targets.append(f.target.to_numpy(float))
    observed_ic=pm.pivot(index='period',columns='variant',values='rank_ic').reindex(index=[p['period'] for p in base],columns=variants).to_numpy(float)
    rank_complete=np.isfinite(observed_ic).all(axis=0);rank_obs=np.mean(observed_ic,axis=0)
    null_ic=np.full((iterations,family),np.nan);records=[];rng=np.random.default_rng(seed)
    portfolio=PortfolioSpec('equal',.05);net=CostSpec(mode='ibkr_fixed',execution_bps=1,borrow_rate=.03)
    gross=CostSpec(flat_bps=0,borrow_rate=0)
    for draw in range(iterations):
        shuffled=[];ics=[]
        for pi,p in enumerate(base):
            donor=np.arange(len(p['ids']))
            for bucket in p['bins']:donor[bucket]=rng.permutation(bucket)
            s=matrices[pi][donor];shuffled.append(s);ics.append(ranked_correlation(s,targets[pi],min_known))
        null_ic[draw]=np.mean(ics,axis=0)
        for j,v in enumerate(variants):
            ps=[dict(p,score=shuffled[i][:,j]) for i,p in enumerate(base)]
            row=dict(draw=draw,variant=v,mean_rank_ic=null_ic[draw,j])
            for label,cost in [('net',net),('gross',gross)]:
                try:
                    r=ledger(ps,portfolio=portfolio,costs=cost,initial_capital=100000.,capture=False)
                    row[label+'_mean_return']=float(r['period_returns'].mean());row[label+'_status']='complete'
                    row[label+'_turnover']=r['total_turnover']/len(base)
                    for k in ['commission','regulatory','execution','flat','borrow']:row[label+'_'+k]=r['cost_totals'][k]
                except InvalidPortfolio as exc:row[label+'_status']=str(exc)
            records.append(row)
        if (draw+1)%250==0:print(f'randomization {draw+1}/{iterations}',flush=True)
    draws=pd.DataFrame(records);draws.to_parquet(out/'random_draws.parquet',index=False)
    reports={}
    for j,v in enumerate(variants):
        f=draws[draws.variant.eq(v)];valid=f.net_status.eq('complete');invalid=int((~valid).sum())
        rank_valid=np.isfinite(null_ic[:,j]);rank_invalid=int((~rank_valid).sum());rank_p=None
        if rank_complete[j] and not rank_invalid:rank_p=(1+int(np.sum(null_ic[:,j]>=rank_obs[j])))/(iterations+1)
        p=None;bounds=None
        if (v,'net')in observed:
            obs=observed[v,'net']['period_returns'].mean();hits=int((f.loc[valid,'net_mean_return']>=obs).sum())
            bounds=[(1+hits)/(iterations+1),(1+hits+invalid)/(iterations+1)]
            if invalid==0:p=bounds[0]
        reports[v]=dict(iterations=iterations,family_size=family,invalid_rank_draws=rank_invalid,
            ranking_p_raw=rank_p,ranking_p_adjusted=min(1,rank_p*family) if rank_p is not None else None,
            invalid_net_draws=invalid,invalid_gross_draws=int(f.gross_status.ne('complete').sum()),
            portfolio_p_raw=p,portfolio_p_adjusted=min(1,p*family) if p is not None else None,
            portfolio_p_bounds=bounds,portfolio_adjusted_p_bounds=[min(1,x*family) for x in bounds] if bounds else None,
            limitation='size-conditioned period-wise shuffles alter persistence and turnover; not a turnover-matched null; inspected historical sample')
    # The same block indices resample every model and every paired contrast.
    n=len(base);blocks=4;idx=[];boot_rows=[];delta=[]
    for draw in range(iterations):
        starts=rng.integers(0,n-blocks+1,size=int(np.ceil(n/blocks)))
        ix=np.concatenate([np.arange(s,s+blocks) for s in starts])[:n];idx.append(ix)
        vals=np.mean(observed_ic[ix],axis=0)
        delta.append(vals[variants.index(PRIMARY)]-vals[variants.index(REFERENCE)])
        for j,v in enumerate(variants):
            row=dict(draw=draw,variant=v,mean_ic=vals[j])
            for label in ['net','gross']:
                if (v,label)in observed:
                    r=observed[v,label]['period_returns'][ix];sd=r.std(ddof=1)
                    row[label+'_mean_return']=r.mean();row[label+'_annualized_return']=np.prod(1+r)**(52/n)-1
                    row[label+'_sharpe']=r.mean()/sd*np.sqrt(52) if sd else np.nan
            boot_rows.append(row)
    boot=pd.DataFrame(boot_rows);boot.to_parquet(out/'bootstrap.parquet',index=False)
    np.save(out/'bootstrap_week_indices.npy',np.asarray(idx))
    pd.DataFrame({'draw':np.arange(iterations),'mean_ic_difference':delta}).to_parquet(out/'primary_ic_difference_bootstrap.parquet',index=False)
    for v,f in boot.groupby('variant'):
        reports[v]['bootstrap_95pct']={c:np.quantile(f[c].dropna(),[.025,.975]).tolist() if f[c].notna().any() else None
            for c in f.columns if c not in ['draw','variant']}
    primary_ci=np.quantile(delta,[.025,.975]).tolist() if np.isfinite(delta).all() else None
    dump(out/'monte_carlo.json',reports);dump(out/'primary_ic_difference.json',dict(primary=PRIMARY,reference=REFERENCE,
        observed=float(rank_obs[variants.index(PRIMARY)]-rank_obs[variants.index(REFERENCE)]),bootstrap_95pct=primary_ci,
        iterations=iterations,block_weeks=4,seed=seed))
    return reports,primary_ci


def run(panel_path,prices_path,reference_root,out,max_cores=4,iterations=2000):
    _limit_cores(max_cores);out=Path(out);reference_root=Path(reference_root)
    if out.exists():raise FileExistsError('fresh output directory required')
    if iterations<1:raise ValueError('iterations must be positive')
    sources=[Path(panel_path),Path(prices_path),reference_root/'weekly/predictions.parquet']
    hashes={str(p):sha(p) for p in sources};expected=json.loads((reference_root/'final_audit/artifact_hashes.json').read_text())
    for p,k in zip(sources,['weekly/weekly_panel.parquet','handoff/stock_prices.parquet','weekly/predictions.parquet']):
        if hashes[str(p)]!=expected[k]:raise ValueError('frozen source mismatch: '+k)
    panel=features(pd.read_parquet(panel_path));prices=pd.read_parquet(prices_path);reference=pd.read_parquet(sources[-1]);schedule=schedules(panel)
    out.mkdir(parents=True);manifest=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes=hashes,groups=GROUPS,definitions=DEFINITIONS,seed=SEED,iterations=iterations,max_cores=max_cores,
        software={p:version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','exchange-calendars']},
        primary_challenger=PRIMARY,reference=REFERENCE,original_primary='combined/ridge',
        portfolio=asdict(PortfolioSpec('equal')),costs=asdict(CostSpec(mode='ibkr_fixed',execution_bps=1)),initial_capital=100000.,
        status='exploratory retrospective feature experiment; monthly paper diagnostic unchanged; no new data or tuning')
    dump(out/'manifest.json',manifest);dump(out/'split_schedule.json',{k:v.to_dict('records') for k,v in schedule.items()})
    controls=[v for v in VARIANTS if v.split('/')[0]in ['stock_only','full']]
    control,windows=fit_scores(panel,schedule['historical'],controls,out/'historical',hashes)
    checks=reproduce(control,reference);dump(out/'control_reproduction.json',checks)
    print('All four controls reproduced within 1e-8',flush=True)
    other=[v for v in VARIANTS if v not in controls]
    extra,_=fit_scores(panel,schedule['historical'],other,out/'historical',hashes)
    historical=pd.concat([control,extra],ignore_index=True)
    earlier,earlier_windows=fit_scores(panel,schedule['assessment'],VARIANTS,out/'assessment',hashes)
    outputs={};runs={};rankings={};prepareds={}
    for name,pred in [('assessment',earlier),('historical',historical)]:
        dest=out/name;wanted=panel[panel.period.isin(schedule[name].period)]
        assert_common_keys(pred,wanted)
        pred.to_parquet(dest/'predictions.parquet',index=False)
        dump(dest/'training_windows.json',earlier_windows if name=='assessment' else windows)
        reports,observed,prepared,pm=predict_and_evaluate(pred,prices,dest)
        if name=='assessment':
            pm=pm.merge(schedule[name][['period']].assign(block=np.arange(52)//13+1),on='period',validate='many_to_one')
            pm.groupby(['variant','block']).agg(mean_ic=('rank_ic','mean'),mean_mse=('mse','mean'),mean_quintile_spread=('quintile_spread','mean')).reset_index().to_csv(dest/'block_metrics.csv',index=False)
        dump(dest/'paired_ic_differences.json',paired_statistics(pm))
        outputs[name]=reports;runs[name]=observed;rankings[name]=pm;prepareds[name]=prepared
    sims,ci=simulations(historical,prepareds['historical'],runs['historical'],rankings['historical'],out/'historical',iterations,SEED)
    earlier_delta=outputs['assessment'][PRIMARY]['mean_ic']-outputs['assessment'][REFERENCE]['mean_ic']
    block=rankings['assessment'].pivot(index=['block','period'],columns='variant',values='rank_ic')
    diffs=(block[PRIMARY]-block[REFERENCE]).groupby('block').mean()
    gates=dict(assessment_ic_improves=earlier_delta>0,at_least_three_blocks_improve=int(diffs.gt(0).sum())>=3,
        historical_positive_ic=outputs['historical'][PRIMARY]['mean_ic']>0,
        paired_bootstrap_above_zero=ci is not None and ci[0]>0,
        adjusted_ranking_p_below_005=sims[PRIMARY]['ranking_p_adjusted'] is not None and sims[PRIMARY]['ranking_p_adjusted']<.05)
    dump(out/'success_gates.json',dict(gates=gates,all_pass=all(gates.values()),assessment_ic_difference=earlier_delta,
        block_differences=diffs.to_dict(),conclusion='promising for further prediction research' if all(gates.values()) else 'primary challenger not supported by the declared research screens',
        limitation='exploratory; not proof of profitability, fresh independent evidence, or paper replication'))
    for p,digest in hashes.items():assert sha(p)==digest,'input mutated'
    dump(out/'source_verification.json',dict(hashes_unchanged=True,monthly_reference_untouched=True))
    lines=['# Weekly feature experiment','', '**Exploratory retrospective analysis; no independent holdout claim.**','',
           '| Variant | Assessment mean IC | Historical mean IC | Net return | Adjusted ranking p |', '|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        m=outputs['historical'][v];p=m['portfolio']['net'];ret=f"{p['cumulative_return']:.2%}" if p['status']=='complete' else 'Unavailable'
        lines.append(f"| {v} | {outputs['assessment'][v]['mean_ic']:.4f} | {m['mean_ic']:.4f} | {ret} | {sims[v]['ranking_p_adjusted']} |")
    lines+=['',json.loads((out/'success_gates.json').read_text())['conclusion'], '', 'Missing selected prices still invalidate portfolios. Monthly reference and all source files are unchanged.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Finished:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',required=True);p.add_argument('--prices',required=True)
    p.add_argument('--reference-results',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--max-cores',type=int,default=4)
    p.add_argument('--iterations',type=int,default=2000,help='2000 for declared research; smaller counts are engineering pilots only')
    a=p.parse_args();run(a.panel,a.prices,a.reference_results,a.output_dir,a.max_cores,a.iterations)

if __name__=='__main__':main()
