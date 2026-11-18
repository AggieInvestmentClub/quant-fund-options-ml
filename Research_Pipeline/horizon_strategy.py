"""Staged horizon/relative-return research on frozen inputs. Never trades live."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,json,subprocess
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from .horizon_data import (FEATURES,WINDOWS,HORIZONS,VARIANTS,horizon_panel,
    window_schedule,matured_training,momentum_predictions,prediction_columns)
from .strategy_experiments import pipeline
from .portfolio_ledger import PortfolioSpec,SelectionPolicy,CostSpec,prepare_replay
from .daily_ledger import ledger,InvalidPortfolio
from .reevaluate import dump,sha,audit,extra_metrics,alpha_proxy
from .feature_experiments import prediction_metrics
from .feature_diagnostics import correlations,diagnostic_validation,permutation_importance,accumulated_effects
from .run import _limit_cores

SEED=17
POLICIES={'baseline':SelectionPolicy(), 'entry':SelectionPolicy(.025,.05),
          'buffer':SelectionPolicy(.05,.10),'both':SelectionPolicy(.025,.10)}
NET=CostSpec(mode='ibkr_fixed',execution_bps=1,borrow_rate=.03)
GROSS=CostSpec(flat_bps=0,borrow_rate=0)
DEFAULT_FALLBACK='h4/relative/full/ridge'


def fit_configuration(panel,sched,variant,out,hashes,min_weeks=52,diagnostics=True):
    _,target,group,kind=variant.split('/');cols=FEATURES[group]
    out=Path(out);out.mkdir(parents=True,exist_ok=True);frames=[];windows=[]
    for row in sched.itertuples(index=False):
        tr=matured_training(panel,row.decision_at,target,min_weeks);sc=panel[panel.period.eq(row.period)]
        if sc.empty:raise ValueError('empty scheduled signal universe: '+row.period)
        if len(sc)<200 and min_weeks==52:raise ValueError('fewer than 200 decision-time eligible stocks')
        window=dict(period=row.period,decision_at=str(row.decision_at),training_rows=len(tr),
          training_weeks=tr.period.nunique(),latest_outcome=str(tr.label_available_at.max()),
          first_signal=str(tr.trade_date.min()),last_signal=str(tr.trade_date.max()))
        path=out/(row.period.replace('/','_')+'.joblib')
        metadata=dict(variant=variant,columns=cols,window=window,sources=hashes,
          parameters=pipeline(kind).get_params(),schema=1)
        # A stable JSON representation avoids estimator object equality in cache checks.
        signature=json.dumps(metadata,sort_keys=True,default=str)
        if path.exists():
            saved=joblib.load(path)
            if saved['signature']!=signature:raise ValueError('model cache mismatch: '+str(path))
            model=saved['model']
        else:
            model=pipeline(kind).fit(tr[cols],tr.target)
            joblib.dump(dict(model=model,signature=signature,metadata=metadata),path)
        sc=sc.copy();sc['score']=model.predict(sc[cols]);sc['variant']=variant
        sc['target']=sc['target_'+target]
        if not np.isfinite(sc.score).all():raise ValueError('nonfinite scores')
        frames.append(prediction_columns(sc));windows.append(window)
        if diagnostics and group=='full' and kind=='ridge':
            d=out/'diagnostics'/row.period.replace('/','_')
            if not (d/'coverage.json').exists():correlations(tr,sc,cols,d,window)
        if diagnostics and row.period==sched.period.iloc[-1]:
            d=out/'effects';d.mkdir(exist_ok=True)
            if not (d/'importance.csv').exists():
                fitting,validation,status=diagnostic_validation(tr)
                dump(d/'validation.json',status)
                if status['eligible']:
                    independent=pipeline(kind).fit(fitting[cols],fitting.target)
                    permutation_importance(independent,fitting,validation,cols).to_csv(d/'importance.csv',index=False)
                    accumulated_effects(independent,fitting,cols).to_csv(d/'local_effects.csv',index=False)
        print('fit',variant,row.period,flush=True)
    pred=pd.concat(frames,ignore_index=True);pred.to_parquet(out/'predictions.parquet',index=False)
    dump(out/'training_windows.json',windows)
    return pred


def weekly_equity(run,entries):
    points=pd.DataFrame(run['equity_path'],columns=['date','equity']).groupby('date').equity.last()
    dates=pd.DatetimeIndex(entries)
    nav=points.reindex(dates).to_numpy(float,copy=True)
    if not np.isfinite(nav).all():raise InvalidPortfolio('missing weekly equity boundary')
    # Opening trades are included against initial capital, not the post-cost first mark.
    nav[0]=run['initial_capital']
    return pd.Series(nav[1:]/nav[:-1]-1,index=dates[1:],name='net_return')


def weekly_boundaries(start,end):
    from .strategy_data import schedule
    s=schedule(pd.Timestamp(start)-pd.Timedelta(days=10),pd.Timestamp(end),True)
    start_week=pd.Timestamp(start).to_period('W-SUN');end_week=pd.Timestamp(end).to_period('W-SUN')
    weeks=s.entry_date.dt.to_period('W-SUN')
    dates=s.entry_date[weeks.between(start_week,end_week)].drop_duplicates().sort_values()
    if dates.iloc[0].to_period('W-SUN')!=start_week or dates.iloc[-1].to_period('W-SUN')!=end_week:raise ValueError('weekly boundary changed')
    return pd.DatetimeIndex(dates)


def performance(run,boundaries):
    r=weekly_equity(run,boundaries);sd=r.std(ddof=1);equity=np.r_[run['initial_capital'],[v for _,v in run['equity_path']]]
    elapsed=(boundaries[-1]-boundaries[0]).days
    result=dict(status='complete',cumulative_return=run['final_equity']/run['initial_capital']-1,
       annualized_return=(run['final_equity']/run['initial_capital'])**(365/elapsed)-1,
       sharpe=float(r.mean()/sd*np.sqrt(52)) if sd>0 else None,
       annualized_volatility=float(sd*np.sqrt(52)),mean_weekly_return=float(r.mean()),
       max_drawdown=float(np.min(equity/np.maximum.accumulate(equity)-1)),
       hit_rate=float(r.gt(0).mean()),weeks=len(r),holding_periods=len(run['period_returns']),
       mean_weekly_turnover=run['total_turnover']/len(r),transaction_cost=run['total_transaction_cost'],
       borrow_cost=run['total_borrow'],cost_components=run['cost_totals'])
    if 'daily'in run:result.update(extra_metrics(run))
    return result


def portfolio_case(pred,prices,out,policy='baseline',mode='long_short',boundaries=None):
    ps=prepare_replay(pred,prices);spec=PortfolioSpec('equal',selection=POLICIES[policy],mode=mode)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);reports={};runs={}
    for label,cost in [('net',NET),('gross',GROSS)]:
        try:
            r=ledger(ps,portfolio=spec,costs=cost,initial_capital=100000.)
            audit(r);reports[label]=performance(r,boundaries);runs[label]=r
            bench=benchmark_weekly(ps,boundaries)
            reports[label]['alpha_proxy']=alpha_proxy(weekly_equity(r,boundaries),bench) if bench is not None else dict(status='unavailable: incomplete benchmark prices')
            for name in ['daily','periodic','holdings','trades']:r[name].to_parquet(out/(label+'_'+name+'.parquet'),index=False)
            weekly_equity(r,boundaries).to_frame().to_parquet(out/(label+'_weekly.parquet'))
            dump(out/(label+'_verification.json'),audit(r))
        except InvalidPortfolio as exc:reports[label]=dict(status='invalid',reason=str(exc))
    dump(out/'metrics.json',reports)
    return reports,runs,ps


def benchmark_weekly(periods,boundaries):
    points={};wealth=1.
    for p in periods:
        path=p['all_prices'][:,p['index']]
        if not np.isfinite(path).all() or (path[0]<=0).any():return None
        values=wealth*np.mean(path/path[0],axis=1)
        points.update(zip(p['days'],values));wealth=values[-1]
    values=np.array([points.get(x,np.nan) for x in boundaries])
    return values[1:]/values[:-1]-1 if np.isfinite(values).all() else None


def fit_job(args):
    panel,sched,variant,out,hashes=args
    path=Path(out)/'predictions.parquet'
    # Model metadata is verified by fit_configuration even on resume.
    return variant,fit_configuration(panel,sched,variant,out,hashes)


def fit_stage(panels,masters,window,out,hashes,cores,variants=VARIANTS):
    jobs=[]
    for v in variants:
        h=int(v.split('/')[0][1:]);s=window_schedule(masters[h],h,*WINDOWS[window])
        jobs.append((panels[h],s,v,str(Path(out)/v.replace('/','_')),hashes))
    if cores==1:fits=map(fit_job,jobs)
    else:
        # Numerical libraries stay single-threaded; only independent configurations run concurrently.
        pool=ProcessPoolExecutor(max_workers=cores);fits=pool.map(fit_job,jobs)
    try:return dict(fits)
    finally:
        if cores!=1:pool.shutdown()


def choose_configuration(reports,ranking):
    candidates=[]
    for v,m in reports.items():
        n=m['net'];ic=ranking[v].rank_ic.mean()
        if n['status']=='complete' and n['cumulative_return']>0 and pd.notna(ic) and ic>0 and n['sharpe'] is not None:
            candidates.append((-n['sharpe'],n['mean_weekly_turnover'],v))
    if candidates:
        return dict(status='qualified on development only',variant=sorted(candidates)[0][2],
                    criterion='positive net return and mean IC; highest net weekly Sharpe, then turnover/name')
    return dict(status='no qualifying ML strategy; diagnostic fallback only',variant=DEFAULT_FALLBACK)


def rule_cases(selected):
    h=selected.split('/')[0]
    cases={f'ml_{p}':dict(variant=selected,policy=p) for p in POLICIES}
    for k in ('plain','gap','gap_skew'):cases['momentum_'+k]=dict(variant=f'{h}/momentum/{k}',policy='baseline')
    cases['momentum_gap_skew_both']=dict(variant=f'{h}/momentum/gap_skew',policy='both')
    return cases


def control_reproduction(panel,prices,reference,out,hashes):
    from .feature_experiments import features,schedules,fit_scores,reproduce
    f=features(panel);s=schedules(f)['historical']
    p,_=fit_scores(f,s,['stock_only/ridge','full/ridge','stock_only/lightgbm','full/lightgbm'],out,hashes)
    saved=pd.read_parquet(Path(reference)/'historical/predictions.parquet')
    # Current updated-data results already name this group full rather than combined.
    saved=saved.copy();saved['variant']=saved.variant.str.replace('full/','combined/',regex=False)
    checks=reproduce(p,saved)
    dump(Path(out)/'verification.json',checks)
    return checks


def run(panel_path,price_path,reference,out,cores=4,iterations=2000,stage='all'):
    _limit_cores(cores);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths={'panel':Path(panel_path).resolve(),'prices':Path(price_path).resolve(),
           'reference_predictions':Path(reference).resolve()/'historical/predictions.parquet'}
    if any(out==p.parent or out in p.parents for p in paths.values()):raise ValueError('outputs must be separate from immutable inputs')
    hashes={k:sha(p) for k,p in paths.items()}
    spec=dict(schema=1,sources=hashes,paths={k:str(v) for k,v in paths.items()},windows=WINDOWS,
       variants=VARIANTS,policies={k:asdict(v) for k,v in POLICIES.items()},costs=asdict(NET),
       seed=SEED,iterations=iterations,cores=cores,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       software={k:version(k) for k in ['numpy','pandas','scikit-learn','lightgbm','exchange-calendars']},
       interpretation='exploratory retrospective; inspected historical period; not paper replication')
    spec=json.loads(json.dumps(spec))
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=spec:raise ValueError('run manifest differs; use a fresh output directory')
    dump(manifest,spec)
    panel=pd.read_parquet(paths['panel']);prices=pd.read_parquet(paths['prices'])
    if not (out/'controls/verification.json').exists():control_reproduction(panel,prices,reference,out/'controls',hashes)
    if stage=='controls':return
    panels={};masters={}
    for h in HORIZONS:
        panels[h],masters[h]=horizon_panel(panel,prices,h)
        panels[h].to_parquet(out/f'labeled_h{h}.parquet',index=False)
    coverage=panel.groupby('period').agg(eligible=('security_id','size'),signal_date=('trade_date','first'))
    coverage.to_csv(out/'universe_coverage.csv')
    summaries={};allpred={};rankings={};case_catalog={};historical={}
    for window in WINDOWS:
        folder=out/window;folder.mkdir(exist_ok=True)
        fits=fit_stage(panels,masters,window,folder/'fits',hashes,cores)
        allpred[window]=fits;reports={};ranking={};boundaries=weekly_boundaries(*WINDOWS[window])
        for v,p in fits.items():
            ranking[v]=prediction_metrics(p)
            ranking[v].to_parquet(folder/(v.replace('/','_')+'_ranking.parquet'),index=False)
            reports[v],_,_=portfolio_case(p,prices,folder/'portfolios'/v.replace('/','_'),boundaries=boundaries)
        summaries[window]=reports;rankings[window]=ranking;dump(folder/'metrics.json',reports)
        if window=='development':
            selected=choose_configuration(reports,ranking);dump(out/'selection.json',selected)
            if stage=='horizons':
                # Stage name stops after BOTH horizon windows below; it never freezes a new selection on historical outcomes.
                pass
    if stage=='horizons':return
    cases=rule_cases(selected['variant']);h=int(selected['variant'].split('/')[0][1:])
    # Freeze the complete inference family before any Stage 2 historical replay.
    for v in VARIANTS:case_catalog['stage1/'+v]=dict(variant=v,policy='baseline',mode='long_short')
    for key,c in cases.items():
        for mode in ('long_short','long_only','short_only'):
            case_catalog['stage2/'+key+'/'+mode]={**c,'mode':mode}
    ht,tt,_,mm=selected['variant'].split('/')
    controls={g:dict(variant=f'{ht}/{tt}/{g}/{mm}',policy='both',mode='long_short') for g in FEATURES}
    for g,c in controls.items():case_catalog['paired/'+g]=c
    dump(out/'inference_family.json',dict(portfolio_family_size=len(case_catalog),
         ranking_family_size=len(VARIANTS),cases=case_catalog))
    for window in WINDOWS:
        sched=window_schedule(masters[h],h,*WINDOWS[window])
        for k in ('plain','gap','gap_skew'):
            p=momentum_predictions(panels[h],sched,k,h);allpred[window][p.variant.iloc[0]]=p
        dest=out/window/'rules';dest.mkdir(exist_ok=True);reports={};boundaries=weekly_boundaries(*WINDOWS[window])
        for key,c in cases.items():
            p=allpred[window][c['variant']]
            for mode in ('long_short','long_only','short_only'):
                name=key+'/'+mode
                reports[name],_,_=portfolio_case(p,prices,dest/name.replace('/','_'),c['policy'],mode,boundaries)
        for group,c in controls.items():
            reports['paired/'+group],_,_=portfolio_case(allpred[window][c['variant']],prices,
                dest/('paired_'+group),'both','long_short',boundaries)
        dump(dest/'metrics.json',reports)
        pd.concat(allpred[window].values(),ignore_index=True).to_parquet(out/window/'predictions.parquet',index=False)
    if stage=='rules':return
    from .horizon_statistics import run_inference
    run_inference(out,allpred['historical'],prices,case_catalog,cores,iterations)
    if hashes!={k:sha(p) for k,p in paths.items()}:raise ValueError('immutable sources changed during run')
    dump(out/'source_verification.json',dict(hashes_unchanged=True,sha256=hashes))
    from .horizon_report import write_report
    write_report(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','prices','reference-results','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-cores',type=int,choices=range(1,5),default=4)
    p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--stage',choices=['controls','horizons','rules','report','all'],default='all')
    a=p.parse_args()
    if a.iterations<1:raise ValueError('iterations must be positive')
    if a.stage=='report':
        from .horizon_report import write_report
        write_report(Path(a.output_dir));return
    run(a.panel,a.prices,a.reference_results,a.output_dir,a.max_cores,a.iterations,a.stage)

if __name__=='__main__':main()
