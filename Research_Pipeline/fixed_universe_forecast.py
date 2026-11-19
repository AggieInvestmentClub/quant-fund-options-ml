"""Four-week absolute-return forecasts and independent fixed-capital slots.

Offline only. Research output, fitted models and input data must remain private.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .daily_ledger import ledger, InvalidPortfolio
from .feature_diagnostics import prune, correlations, accumulated_effects, correlation_groups
from .horizon_data import FEATURES, WINDOWS, horizon_panel, window_schedule, matured_training
from .horizon_pruning import verify_control
from .horizon_strategy import weekly_boundaries, weekly_equity, performance, GROSS, NET
from .passive_benchmarks import configured_universe, validate_spy
from .portfolio_ledger import PortfolioSpec, prepare_replay
from .reevaluate import dump, sha, audit, alpha_proxy
from .run import _limit_cores

SEED=17
MAIN=[f'{g}/{m}' for g in ('stock_only','full','pruned') for m in ('ridge','lightgbm')]
PRIMARY='full/ridge'

def model_pipeline(kind, overrides=None):
    params=dict(overrides or {})
    if kind=='ridge': model=Ridge(alpha=params.get('alpha',10))
    elif kind=='lightgbm':
        defaults=dict(n_estimators=100,num_leaves=15,learning_rate=.05,min_child_samples=50,
                      n_jobs=1,random_state=SEED,verbosity=-1)
        defaults.update(params);model=LGBMRegressor(**defaults)
    else:raise ValueError('unknown model family')
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         StandardScaler(),model)

def columns_for(train, group, threshold=.90):
    if group=='pruned':return prune(train,FEATURES['full'],correlation_threshold=threshold)
    if group not in FEATURES:raise ValueError('unknown feature group')
    return list(FEATURES[group]),{}

def stock_baselines(train, score):
    grouped=train.groupby('security_id').target
    means=grouped.mean();positive=grouped.apply(lambda x:float((x>0).mean()))
    result=score.copy();result['baseline_fallback']=~result.security_id.isin(means.index)
    result['baseline_mean']=result.security_id.map(means).fillna(train.target.mean())
    result['baseline_up_probability']=result.security_id.map(positive).fillna((train.target>0).mean())
    return result

def fixed_rows(panel, period, ids):
    score=panel[panel.period.eq(period)&panel.security_id.isin(ids)].copy()
    if score.security_id.duplicated().any() or set(score.security_id)!=set(ids):
        raise ValueError(f'{period}: all fixed stocks require unambiguous decision-time rows')
    return score.sort_values('security_id').reset_index(drop=True)

def fit_variant(panel, schedule, ids, variant, overrides, destination, sources, min_weeks=52,
                correlation_threshold=.90):
    group,kind=variant.split('/');dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    records=[];windows=[]
    for row in schedule.itertuples(index=False):
        train=matured_training(panel,row.decision_at,'absolute',min_weeks)
        score=stock_baselines(train,fixed_rows(panel,row.period,ids))
        cols,removed=columns_for(train,group,correlation_threshold)
        metadata=dict(variant=variant,overrides=overrides,columns=cols,removed=removed,
                      decision_at=str(row.decision_at),training_rows=len(train),
                      training_weeks=int(train.period.nunique()),latest_outcome=str(train.label_available_at.max()),
                      first_signal=str(train.trade_date.min()),last_signal=str(train.trade_date.max()),
                      sources=sources,correlation_threshold=correlation_threshold,target='four_week_absolute_return',
                      schema=1)
        signature=json.dumps(metadata,sort_keys=True,default=str)
        path=dest/(row.period.replace('/','_')+'.joblib')
        if path.exists():
            saved=joblib.load(path)
            if saved['signature']!=signature:raise ValueError('incompatible model cache: '+str(path))
            model=saved['model']
        else:
            model=model_pipeline(kind,overrides).fit(train[cols],train.target)
            joblib.dump(dict(model=model,signature=signature,metadata=metadata),path)
        keep=['period','security_id','trade_date','decision_at','available_at','entry_date','exit_date',
              'market_cap','target_absolute','baseline_mean','baseline_up_probability','baseline_fallback']
        frame=score[keep].rename(columns={'target_absolute':'target'}).copy()
        frame['score']=model.predict(score[cols]);frame['variant']=variant
        frame['missing_input_count']=score[cols].isna().sum(axis=1).to_numpy()
        if not np.isfinite(frame.score).all():raise ValueError('nonfinite forecast')
        records.append(frame);windows.append(metadata)
    result=pd.concat(records,ignore_index=True)
    result.to_parquet(dest/'predictions.parquet',index=False);dump(dest/'training_windows.json',windows)
    print('fitted',variant,len(schedule),'decisions',flush=True)
    return result

def _fit_job(job):
    return fit_variant(*job)

def fit_main(panel,schedule,ids,dest,sources,cores):
    jobs=[(panel,schedule,ids,v,{},str(Path(dest)/v.replace('/','_')),sources) for v in MAIN]
    if cores==1:return pd.concat([_fit_job(j) for j in jobs],ignore_index=True)
    with ProcessPoolExecutor(max_workers=cores) as pool:
        return pd.concat(list(pool.map(_fit_job,jobs)),ignore_index=True)

def assert_common(pred):
    base=None
    for _,f in pred.groupby('variant',sort=True):
        current=f[['period','security_id','target','baseline_mean']].sort_values(['period','security_id']).reset_index(drop=True)
        if base is None:base=current
        else:pd.testing.assert_frame_equal(current,base,check_exact=True)

def score_metrics(pred,min_known=20):
    from scipy.stats import spearmanr
    rows=[]
    for (variant,period),f in pred.groupby(['variant','period'],sort=True):
        known=f[np.isfinite(f.target)];valid=len(known)>=min_known
        row=dict(variant=variant,period=period,eligible=len(f),known=len(known),
                 missing_outcomes=len(f)-len(known),missing_inputs=int(f.missing_input_count.sum()) if 'missing_input_count'in f else 0)
        for k in ('mse','mae','rmse','direction_accuracy','balanced_direction_accuracy','rank_ic',
                  'baseline_mse','zero_mse','baseline_direction_accuracy','mse_improvement','mse_skill'):row[k]=np.nan
        if valid:
            y=known.target.to_numpy(float);p=known.score.to_numpy(float);base=known.baseline_mean.to_numpy(float)
            up=y>0;guess=p>0;both=np.unique(up).size==2
            mse=float(np.mean((p-y)**2));bmse=float(np.mean((base-y)**2))
            row.update(mse=mse,mae=float(np.mean(np.abs(p-y))),rmse=float(np.sqrt(mse)),
                       direction_accuracy=float(np.mean(guess==up)),
                       balanced_direction_accuracy=float((np.mean(guess[up])+np.mean(~guess[~up]))/2) if both else np.nan,
                       rank_ic=float(spearmanr(p,y).statistic) if np.unique(p).size>1 and np.unique(y).size>1 else np.nan,
                       baseline_mse=bmse,zero_mse=float(np.mean(y*y)),mse_improvement=bmse-mse,
                       baseline_direction_accuracy=float(np.mean((known.baseline_up_probability.to_numpy()>.5)==up)),
                       mse_skill=1-mse/bmse if bmse>0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_accuracy(metrics):
    return {v:dict(scoring_decisions=int(f.mse.notna().sum()),decisions=len(f),
                   **{c:float(f[c].mean()) for c in ['mse','mae','direction_accuracy',
                     'balanced_direction_accuracy','rank_ic','baseline_mse','zero_mse','mse_improvement','baseline_direction_accuracy']},
                   rmse=float(np.sqrt(f.mse.mean())),
                   mse_skill=float(1-f.mse.mean()/f.baseline_mse.mean()) if f.baseline_mse.mean()>0 else None)
            for v,f in metrics.groupby('variant')}

def portfolio_spec(threshold=0.,comparison=False):
    return PortfolioSpec(weighting='equal',max_position=.04,mode='long_only',
                         selection_mode='exposure_matched' if comparison else 'fixed_slots',entry_threshold=threshold)

def evaluate(pred,prices,out,boundaries,spec=None,cost=NET,capital=100000.,spy_returns=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);ps=prepare_replay(pred,prices)
    results={};spec=spec or portfolio_spec()
    for name,fees in [('net',cost),('gross',GROSS)]:
        try:
            run=ledger(ps,portfolio=spec,costs=fees,initial_capital=capital)
            checks=audit(run);m=performance(run,boundaries)
            m.update(net_profit=run['final_equity']-capital,
                     mean_invested_exposure=float(run['daily'].gross_exposure.mean()),
                     mean_cash_fraction=float((run['daily'].cash/run['daily'].equity).mean()))
            m['spy_alpha']=dict(status='unavailable: no frozen benchmark')
            if spy_returns is not None:
                m['spy_alpha']=alpha_proxy(weekly_equity(run,boundaries),spy_returns)
                m['spy_alpha']['definition']='SPY benchmark alpha; not paper factor alpha'
            results[name]=m
            for field in ('daily','periodic','holdings','trades'):run[field].to_parquet(out/f'{name}_{field}.parquet',index=False)
            weekly_equity(run,boundaries).to_frame().to_parquet(out/f'{name}_weekly.parquet')
            dump(out/f'{name}_verification.json',checks)
        except InvalidPortfolio as exc:results[name]=dict(status='invalid',reason=str(exc))
    dump(out/'metrics.json',results)
    return results,ps

def window_backtests(pred,prices,dest,bounds,spy_returns=None):
    reports={};prepared={}
    for v,f in pred.groupby('variant',sort=True):
        folder=Path(dest)/v.replace('/','_')
        reports[v],prepared[v]=evaluate(f,prices,folder,bounds,spy_returns=spy_returns)
        reports[v]['exposure_matched'],_=evaluate(f,prices,folder/'exposure_matched',bounds,
                                                 spec=portfolio_spec(comparison=True),spy_returns=spy_returns)
    reference=pred[pred.variant.eq(PRIMARY)].copy();reference['score']=1.
    reports['passive_25'],_=evaluate(reference,prices,Path(dest)/'passive_25',bounds,spy_returns=spy_returns)
    # Baseline forecasts get the same fixed-slot execution; they are not fitted ML variants.
    for name,col in [('historical_mean','baseline_mean'),('zero_forecast',None)]:
        f=reference.copy();f['score']=f[col] if col else 0.
        reports[name],_=evaluate(f,prices,Path(dest)/name,bounds,spy_returns=spy_returns)
    dump(Path(dest)/'metrics.json',reports)
    return reports,prepared

def diagnostics(panel,sched,ids,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for row in sched.itertuples(index=False):
        tr=matured_training(panel,row.decision_at,'absolute');sc=fixed_rows(panel,row.period,ids)
        folder=out/row.period.replace('/','_');folder.mkdir(exist_ok=True)
        cols,removed=columns_for(tr,'pruned');meta=dict(decision_at=str(row.decision_at),removed=removed,retained=cols)
        for name,train,columns in [('broad_before',tr,FEATURES['full']),('broad_after',tr,cols),
                                   ('fixed25_training',tr[tr.security_id.isin(ids)],FEATURES['full'])]:
            correlations(train,sc,columns,folder/name,meta)
    dump(out/'complete.json',dict(decisions=len(sched),status='complete'))

def importance_effects(panel,sched,ids,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for row in sched.iloc[[0,-1]].drop_duplicates('period').itertuples(index=False):
        tr=matured_training(panel,row.decision_at,'absolute');weeks=sorted(tr.period.unique())[-13:]
        validation=tr[tr.period.isin(weeks)&tr.security_id.isin(ids)].copy()
        cutoff=tr[tr.period.isin(weeks)].decision_at.min()
        fitting=tr[tr.label_available_at.lt(cutoff)]
        scorable=sum(np.isfinite(f.target).sum()>=20 for _,f in validation.groupby('period'))
        eligible=fitting.period.nunique()>=39 and scorable>=8
        for group in ('full','pruned'):
            for kind in ('ridge','lightgbm'):
                dest=out/row.period.replace('/','_')/(group+'_'+kind);dest.mkdir(parents=True,exist_ok=True)
                dump(dest/'validation.json',dict(cutoff=str(cutoff),fitting_weeks=fitting.period.nunique(),
                     validation_weeks=len(weeks),scorable=scorable,eligible=eligible))
                if not eligible:continue
                cols,removed=columns_for(fitting,group);model=model_pipeline(kind).fit(fitting[cols],fitting.target)
                known=validation[np.isfinite(validation.target)].copy()
                # Equal date weighting, with coverage fixed before any feature permutations.
                known=known[known.period.isin([p for p,f in known.groupby('period') if len(f)>=20])]
                def loss(x):
                    f=known[['period','target']].copy();f['error']=(model.predict(x)-f.target.to_numpy())**2
                    return float(f.groupby('period').error.mean().mean())
                baseline=loss(known[cols]);rows=[];rng=np.random.default_rng(SEED)
                groups=[([c],'individual') for c in cols]+[(g,'correlated_group') for g in correlation_groups(fitting,cols)]
                for group_cols,mode in groups:
                    values=[]
                    for _ in range(20):
                        changed=known[cols].copy()
                        for ix in known.groupby('period').indices.values():
                            changed.iloc[ix,changed.columns.get_indexer(group_cols)]=known.iloc[rng.permutation(ix)][group_cols].to_numpy()
                        values.append(loss(changed)-baseline)
                    rows.append(dict(feature='|'.join(group_cols),kind=mode,mse_increase=float(np.mean(values)),
                                     standard_deviation=float(np.std(values)),repetitions=20,baseline_mse=baseline))
                pd.DataFrame(rows).to_csv(dest/'importance.csv',index=False)
                accumulated_effects(model,fitting,cols).to_csv(dest/'local_effects.csv',index=False)
                dump(dest/'schema.json',dict(columns=cols,removed=removed,fit_cutoff=str(cutoff)))

def sensitivity_variants():
    result={}
    for group in ('full','pruned'):
        for alpha in (1,100):result[f'{group}_ridge_alpha_{alpha}']=(f'{group}/ridge',{'alpha':alpha},.90)
        for key,values in [('num_leaves',(7,31)),('n_estimators',(50,200))]:
            for value in values:result[f'{group}_lightgbm_{key}_{value}']=(f'{group}/lightgbm',{key:value},.90)
    for threshold in (.80,.95):
        for kind in ('ridge','lightgbm'):result[f'pruned_{kind}_pearson_{threshold}']=(f'pruned/{kind}',{},threshold)
    return result

def sensitivity(panel,sched,ids,prices,main_pred,dest,sources,bounds,spy_returns=None):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);records=[]
    for label,(variant,overrides,threshold) in sensitivity_variants().items():
        pred=fit_variant(panel,sched,ids,variant,overrides,dest/'fits'/label,sources,
                         correlation_threshold=threshold)
        pred['variant']=label;pm=score_metrics(pred);pm.to_csv(dest/f'{label}_accuracy.csv',index=False)
        report,_=evaluate(pred,prices,dest/'portfolios'/label,bounds,spy_returns=spy_returns)
        records.append(dict(case=label,kind='model_or_pruning',forecast=summarize_accuracy(pm)[label],portfolio=report))
    for variant in ('full/ridge','pruned/ridge'):
        pred=main_pred[main_pred.variant.eq(variant)]
        settings=[('entry_0.005',.005,NET,100000.),('entry_0.01',.01,NET,100000.),
                  ('execution_0',0.,replace(NET,execution_bps=0),100000.),
                  ('execution_5',0.,replace(NET,execution_bps=5),100000.),
                  ('capital_10000',0.,NET,10000.),('capital_1000000',0.,NET,1000000.)]
        for label,threshold,cost,capital in settings:
            report,_=evaluate(pred,prices,dest/'portfolios'/(variant.replace('/','_')+'_'+label),
                              bounds,spec=portfolio_spec(threshold),cost=cost,capital=capital,spy_returns=spy_returns)
            records.append(dict(case=variant+'/'+label,kind='trading',portfolio=report))
    dump(dest/'comparison.json',records)

def _spy_returns(path,bounds):
    if path is None:return None
    frame=validate_spy(pd.read_parquet(path),bounds[0],bounds[-1])
    x=frame.set_index('trade_date').adjusted_close.reindex(bounds).to_numpy(float)
    return x[1:]/x[:-1]-1

def prepare_inputs(panel_path,prices_path,reference,config,spy,out,cores,iterations):
    _limit_cores(cores);out=Path(out);out.mkdir(parents=True,exist_ok=True,mode=0o700)
    paths=dict(panel=Path(panel_path).resolve(),prices=Path(prices_path).resolve(),config=Path(config).resolve())
    if spy is not None:paths['spy']=Path(spy).resolve()
    reference=Path(reference).resolve()
    for w in WINDOWS:
        for g in ('stock_only','full'):
            for m in ('ridge','lightgbm'):
                paths[f'{w}_{g}_{m}']=reference/w/'fits'/f'h4_absolute_{g}_{m}'/'predictions.parquet'
    sources={k:sha(p) for k,p in paths.items()}
    ref=json.loads((reference/'manifest.json').read_text())
    for k in ('panel','prices'):
        if ref['sources'][k]!=sources[k]:raise ValueError('reference input hash mismatch: '+k)
    code_paths=[Path(__file__),*[Path(__file__).with_name(n+'.py') for n in
                ('fixed_forecast_statistics','portfolio_ledger','daily_ledger','feature_diagnostics','horizon_data')]]
    code={p.name:sha(p) for p in code_paths}
    manifest=dict(schema=1,sources=sources,code_sha256=code,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  configured_ids=configured_universe(config),target='four_week_absolute_return',training_universe='broader covered universe',
                  windows=WINDOWS,variants=MAIN,primary=PRIMARY,capital=100000,slot_weight=.04,entry_threshold=0.,
                  pruning=dict(missing_threshold=.5,correlation_threshold=.90,min_pairs=200),
                  costs=asdict(NET),seed=SEED,iterations=iterations,max_cores=cores,engineering_pilot=iterations<2000,
                  packages={k:version(k) for k in ('numpy','pandas','scikit-learn','lightgbm','exchange-calendars')},
                  interpretation='exploratory retrospective; no top-k selection; no live orders',paths={k:str(v) for k,v in paths.items()})
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=json.loads(json.dumps(manifest,default=str)):
            raise ValueError('incompatible experiment resume')
    else:dump(manifest_path,manifest)
    panel,master=horizon_panel(pd.read_parquet(panel_path),pd.read_parquet(prices_path),4)
    prices=pd.read_parquet(prices_path);ids=manifest['configured_ids'];schedules={}
    for w,dates in WINDOWS.items():
        schedules[w]=window_schedule(master,4,*dates)
        for p in schedules[w].period:fixed_rows(panel,p,ids)
    panel[['period','security_id','target_absolute','entry_date','exit_date']].to_parquet(out/'label_audit.parquet',index=False)
    dump(out/'coverage.json',dict(panel_rows=len(panel),training_identifiers=panel.security_id.nunique(),
          universe=ids,decisions={w:len(s) for w,s in schedules.items()},
          fixed25=[dict(window=w,period=p,known_outcomes=int(np.isfinite(fixed_rows(panel,p,ids).target_absolute).sum()))
                   for w,s in schedules.items() for p in s.period]))
    return panel,prices,ids,schedules,sources,paths

def run(panel_path,prices_path,reference,config,out,spy=None,cores=2,iterations=2000,stage='all'):
    out=Path(out)
    panel,prices,ids,schedules,sources,paths=prepare_inputs(panel_path,prices_path,reference,config,spy,out,cores,iterations)
    if stage=='audit':return
    predictions={};controls={}
    for w,sched in schedules.items():
        dest=out/w;dest.mkdir(exist_ok=True)
        # Re-fit four controls before testing new feature or portfolio policies.
        control_frames=[]
        for g in ('stock_only','full'):
            for m in ('ridge','lightgbm'):
                v=g+'/'+m
                actual=fit_variant(panel,sched,ids,v,{},dest/'fits'/v.replace('/','_'),sources)
                saved=pd.read_parquet(paths[f'{w}_{g}_{m}']);saved=saved[saved.security_id.isin(ids)]
                controls[w+'/'+v]=verify_control(actual,saved)
                control_frames.append(actual)
        predictions[w]=pd.concat(control_frames,ignore_index=True)
    dump(out/'control_verification.json',controls)
    if stage=='controls':return
    for w,sched in schedules.items():
        dest=out/w
        pruned=[]
        jobs=[(panel,sched,ids,'pruned/'+m,{},str(dest/'fits'/('pruned_'+m)),sources) for m in ('ridge','lightgbm')]
        if cores==1:pruned=[_fit_job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=cores) as pool:pruned=list(pool.map(_fit_job,jobs))
        predictions[w]=pd.concat([predictions[w],*pruned],ignore_index=True)
        assert_common(predictions[w]);predictions[w].to_parquet(dest/'predictions.parquet',index=False)
        pm=score_metrics(predictions[w]);pm.to_csv(dest/'accuracy_by_decision.csv',index=False)
        dump(dest/'accuracy.json',summarize_accuracy(pm))
        per_stock=predictions[w][np.isfinite(predictions[w].target)].copy()
        per_stock['squared_error']=(per_stock.score-per_stock.target)**2
        per_stock['absolute_error']=(per_stock.score-per_stock.target).abs()
        per_stock.groupby(['variant','security_id']).agg(mse=('squared_error','mean'),mae=('absolute_error','mean'),known=('target','size')).to_csv(dest/'accuracy_by_stock.csv')
    if stage=='fit':return
    for w,sched in schedules.items():diagnostics(panel,sched,ids,out/w/'diagnostics')
    importance_effects(panel,schedules['historical'],ids,out/'historical/feature_effects')
    if stage=='diagnostics':return
    for w,pred in predictions.items():
        bounds=weekly_boundaries(*WINDOWS[w]);spyret=_spy_returns(spy,bounds)
        window_backtests(pred,prices,out/w/'portfolios',bounds,spyret)
    if stage=='backtest':return
    from .fixed_forecast_statistics import infer
    infer(predictions['historical'],prices,out/'historical',weekly_boundaries(*WINDOWS['historical']),iterations,cores)
    if stage=='inference':return
    for w,sched in schedules.items():
        bounds=weekly_boundaries(*WINDOWS[w])
        sensitivity(panel,sched,ids,prices,predictions[w],out/w/'sensitivity',sources,bounds,_spy_returns(spy,bounds))
    if stage=='sensitivity':return
    from .fixed_forecast_report import build_report
    build_report(out)
    latest_score(panel,ids,out/'latest',sources)
    checks={k:sha(p)==sources[k] for k,p in paths.items()};dump(out/'source_verification.json',checks)
    if not all(checks.values()):raise ValueError('frozen source changed during run')
    dump(out/'complete.json',dict(status='complete',exploratory=True,iterations=iterations,main_variants=len(MAIN)))

def latest_score(panel,ids,out,sources):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    latest=panel.sort_values('decision_at').iloc[-1]
    sched=panel[panel.period.eq(latest.period)][['period','decision_at']].drop_duplicates()
    if len(sched)!=1:raise ValueError('ambiguous latest decision')
    pred=fit_variant(panel,sched,ids,PRIMARY,{},out/'model',sources)
    pred['target_weight']=np.where(pred.score>0,.04,0.)
    pred['action']=np.where(pred.score>0,'long','cash')
    pred.to_parquet(out/'forecasts.parquet',index=False)
    pred[['security_id','decision_at','entry_date','exit_date','score','action','target_weight']].to_csv(out/'forecasts.csv',index=False)
    dump(out/'status.json',dict(as_of=str(latest.decision_at),source='latest frozen snapshot; not live data',
                               outcome_known=int(np.isfinite(pred.target).sum()),training_target='four_week_absolute'))
    return pred

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',required=True);p.add_argument('--prices',required=True)
    p.add_argument('--reference-results',required=True);p.add_argument('--config',default='Data_Pipeline/config.py')
    p.add_argument('--spy');p.add_argument('--output-dir',required=True);p.add_argument('--max-cores',type=int,default=2)
    p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--stage',choices=['audit','controls','fit','diagnostics','backtest','inference','sensitivity','report','latest','all'],default='all')
    args=p.parse_args()
    if args.iterations<1:p.error('iterations must be positive')
    if args.stage=='report':
        from .fixed_forecast_report import build_report
        build_report(Path(args.output_dir));return
    if args.stage=='latest':
        _limit_cores(args.max_cores)
        prices=pd.read_parquet(args.prices);panel,_=horizon_panel(pd.read_parquet(args.panel),prices,4)
        latest_score(panel,configured_universe(args.config),Path(args.output_dir)/'latest',
                     dict(panel=sha(args.panel),prices=sha(args.prices)));return
    run(args.panel,args.prices,args.reference_results,args.config,args.output_dir,args.spy,
        args.max_cores,args.iterations,args.stage)

if __name__=='__main__':main()
