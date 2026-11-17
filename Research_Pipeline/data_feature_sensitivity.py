"""Frozen-input data, feature and portfolio sensitivity research.

Stages are restartable only with identical input hashes and configuration.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,json,subprocess,time
from pathlib import Path
from dataclasses import asdict,replace
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor
import joblib
import numpy as np
import pandas as pd
from .sensitivity_data import GROUPS,DEFINITIONS,EXTRA,build_panel,data_changes,reject_future_columns
from .feature_diagnostics import (prune,correlations,diagnostic_validation,permutation_importance,
                                  accumulated_effects)
from .feature_experiments import (features,schedules,training,fit_scores,reproduce,assert_common_keys,
    prediction_metrics,predict_and_evaluate,simulations)
from .strategy_experiments import pipeline,verify_snapshot
from .portfolio_ledger import PortfolioSpec,CostSpec,prepare_replay
from .daily_ledger import ledger,metrics,InvalidPortfolio
from .reevaluate import dump,sha,audit,extra_metrics,verify_legacy
from .run import _limit_cores

PRIMARY='pruned_extended/ridge';REFERENCE='full/ridge'
MAIN=[g+'/'+m for m in ('ridge','lightgbm') for g in GROUPS]
SEED=17


def model_spec(variant):
    group,kind=variant.split('/');base=kind.split('_')[0];params={}
    if kind.startswith('ridge_alpha_'):params['alpha']=float(kind.rsplit('_',1)[1])
    elif '_leaves_' in kind:params['num_leaves']=int(kind.rsplit('_',1)[1])
    elif '_trees_' in kind:params['n_estimators']=int(kind.rsplit('_',1)[1])
    return group,base,params


def model_sensitivities():
    return [g+'/'+kind for g in ('full','pruned_extended') for kind in
            ('ridge_alpha_1','ridge_alpha_100','lightgbm_leaves_7','lightgbm_leaves_31',
             'lightgbm_trees_50','lightgbm_trees_200')]


def fit_variant(panel,schedule,variant,out,hashes):
    out=Path(out);group,kind,params=model_spec(variant);cols=GROUPS[group];frames=[];windows=[]
    reject_future_columns(cols)
    for row in schedule.itertuples(index=False):
        tr=training(panel,row.decision_at);f=panel[panel.period.eq(row.period)]
        chosen,removed=prune(tr,cols) if group.startswith('pruned_') else (list(cols),{})
        window=dict(period=row.period,decision_at=str(row.decision_at),training_weeks=int(tr.period.nunique()),
                    training_rows=len(tr),first_signal=str(tr.trade_date.min()),last_signal=str(tr.trade_date.max()),
                    latest_outcome=str(tr.label_available_at.max()))
        name=row.period.replace('/','_')+'_'+variant.replace('/','_')
        model_path=out/'models'/(name+'.joblib');model_path.parent.mkdir(parents=True,exist_ok=True)
        metadata=dict(variant=variant,features=chosen,definitions=DEFINITIONS,training=window,
                      source_hashes=hashes,params=params,removed=removed,schema_version=2)
        if model_path.exists():
            artifact=joblib.load(model_path)
            if artifact['metadata']!=metadata:raise ValueError('cached model metadata differs: '+name)
            model=artifact['model']
        else:
            model=pipeline(kind);model.steps[-1][1].set_params(**params);model.fit(tr[chosen],tr.target)
            joblib.dump(dict(model=model,metadata=metadata),model_path)
        pred=f[['period','security_id','market_cap','trade_date','entry_date','exit_date','target']].copy()
        pred['score']=model.predict(f[chosen]);pred['variant']=variant
        if not np.isfinite(pred.score).all():raise ValueError('nonfinite predictions')
        frames.append(pred);windows.append(dict(**window,variant=variant,features=chosen,removed=removed))
    print('Fitted',out.name,variant,len(schedule),flush=True)
    return pd.concat(frames,ignore_index=True),windows


def fit_variants(panel,sched,variants,out,hashes,cores):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    jobs=[(panel,sched,v,out,hashes) for v in variants]
    if cores==1:result=[fit_variant(*a) for a in jobs]
    else:
        with ProcessPoolExecutor(max_workers=cores) as pool:
            futures=[pool.submit(fit_variant,*a) for a in jobs]
            result=[f.result() for f in futures]
    pred=pd.concat([r[0] for r in result],ignore_index=True)
    assert_common_keys(pred,panel[panel.period.isin(sched.period)])
    dump(out/'training_windows.json',[w for _,ws in result for w in ws])
    return pred


def run_diagnostics(panel,schedule,out,hashes,cores):
    out=Path(out);columns=GROUPS['pruned_extended']
    periods=schedule.period.tolist()
    # Six predeclared reporting endpoints: four assessment blocks, first/last historical weeks.
    diagnostic_periods={periods[i] for i in (12,25,38,51,52,len(periods)-1)}
    tasks=[]
    for row in schedule.itertuples(index=False):
        tr=training(panel,row.decision_at);f=panel[panel.period.eq(row.period)]
        dest=out/row.period.replace('/','_');dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'coverage.json').exists():
            correlations(tr,f,columns,dest,dict(period=row.period,decision_at=row.decision_at))
        fit,val,status=diagnostic_validation(tr);dump(dest/'validation.json',status)
        if row.period in diagnostic_periods and status['eligible']:
            for variant in MAIN:tasks.append((fit,val,variant,dest,hashes))
    if cores==1:
        for args in tasks:importance_job(*args)
    else:
        with ProcessPoolExecutor(max_workers=cores) as pool:
            futures=[pool.submit(importance_job,*a) for a in tasks]
            for f in futures:f.result()


def importance_job(fit,val,variant,out,hashes):
    dest=Path(out)/variant.replace('/','_');dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    group,kind,_=model_spec(variant);cols=GROUPS[group]
    chosen,removed=prune(fit,cols) if group.startswith('pruned_') else (cols,{})
    model=pipeline(kind).fit(fit[chosen],fit.target)
    importance=permutation_importance(model,fit,val,chosen)
    importance.to_csv(dest/'permutation_importance.csv',index=False)
    effects=accumulated_effects(model,fit,chosen)
    effects.to_csv(dest/'local_effects.csv',index=False)
    estimator=model.steps[-1][1];prep=model[:-1];names=prep.get_feature_names_out(chosen)
    reliance=estimator.coef_ if kind=='ridge' else estimator.feature_importances_
    pd.DataFrame({'feature':names,'fitted_reliance':reliance}).to_csv(dest/'fitted_reliance.csv',index=False)
    joblib.dump(dict(model=model,features=chosen,removed=removed,source_hashes=hashes,
                    latest_outcome=str(fit.label_available_at.max()),validation_first_decision=str(val.decision_at.min())),
                dest/'diagnostic_model.joblib')
    dump(dest/'complete.json',dict(repetitions=20,validation_weeks=val.period.nunique(),
         interpretation='chronological validation model reliance; local effects are descriptive, not causal'))
    print('Diagnostics',Path(out).name,variant,flush=True)


def portfolio_sensitivities():
    p=PortfolioSpec('equal');c=CostSpec(mode='ibkr_fixed',execution_bps=1,borrow_rate=.03)
    cases={'main':(p,c,100000.)}
    for f in (.025,.10):cases['selection_'+str(f)]=(replace(p,selection_fraction=f),c,100000.)
    cases['cap_0025']=(replace(p,max_position=.025),c,100000.)
    cases['half_exposure']=(replace(p,side_exposure=.5),c,100000.)
    for b in (0,5):cases['execution_'+str(b)]=(p,replace(c,execution_bps=b),100000.)
    for b in (0,.06):cases['borrow_'+str(b)]=(p,replace(c,borrow_rate=b),100000.)
    for capital in (10000.,1000000.):cases['capital_'+str(int(capital))]=(p,c,capital)
    for mode in ('long_only','short_only'):cases[mode]=(replace(p,mode=mode),c,100000.)
    return cases


def replay_cases(pred,prices,out,cases=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];reports={}
    cases=cases or portfolio_sensitivities()
    for v,f in pred.groupby('variant',sort=True):
        try:prepared=prepare_replay(f,prices)
        except InvalidPortfolio as exc:
            for name in cases:rows.append(dict(variant=v,condition=name,status='invalid',reason=str(exc)))
            continue
        for name,(p,c,capital) in cases.items():
            dest=out/v.replace('/','_')/name;dest.mkdir(parents=True,exist_ok=True)
            try:
                r=ledger(prepared,portfolio=p,costs=c,initial_capital=capital);audit(r)
                m=dict(**metrics(r,52),**extra_metrics(r),status='complete',portfolio=asdict(p),cost=asdict(c))
                for key in ('daily','holdings','trades','periodic'):r[key].to_parquet(dest/(key+'.parquet'),index=False)
                dump(dest/'verification.json',audit(r));reports[v+'/'+name]=m
                rows.append(dict(variant=v,condition=name,status='complete',**metrics(r,52)))
            except InvalidPortfolio as exc:
                reports[v+'/'+name]=dict(status='invalid',reason=str(exc));rows.append(dict(variant=v,condition=name,status='invalid',reason=str(exc)))
    dump(out/'metrics.json',reports);pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    return reports


def compare_panels(old,new,out):
    keys=['period','security_id'];common=old[keys].merge(new[keys],on=keys,validate='one_to_one')
    dump(Path(out)/'panel_comparison.json',dict(old_rows=len(old),new_rows=len(new),common_keys=len(common),
         added_keys=len(new)-len(common),removed_keys=len(old)-len(common)))
    return common


def source_files(snapshot):
    snapshot=Path(snapshot)
    return [snapshot/'wide.parquet',snapshot/'snapshot.json',*sorted((snapshot/'stocks').glob('*.csv'))]


def run(old_snapshot,new_snapshot,reference_results,out,max_cores=4,iterations=2000,stage='all'):
    _limit_cores(max_cores)
    if iterations<1:raise ValueError('iterations must be positive')
    old_snapshot,new_snapshot,root,out=map(Path,(old_snapshot,new_snapshot,reference_results,out))
    sources=source_files(old_snapshot)+source_files(new_snapshot)
    sources += [root/'weekly/weekly_panel.parquet',root/'private_inputs/stock_prices.parquet',root/'weekly/predictions.parquet']
    sources += sorted(p for p in (root/'paper_diagnostic').glob('*') if p.is_file())
    hashes={str(p):sha(p) for p in sources}
    config=dict(old_snapshot=str(old_snapshot),new_snapshot=str(new_snapshot),reference_results=str(root),
                max_cores=max_cores,iterations=iterations,seed=17,groups=GROUPS,definitions=DEFINITIONS,
                software={p:version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','exchange-calendars']})
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=dict(config=config,source_hashes=hashes,code_commit=commit,primary=PRIMARY,reference=REFERENCE,
                  status='exploratory retrospective; inspected periods; monthly paper diagnostic unchanged')
    if out.exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('resume manifest differs')
    else:out.mkdir(parents=True);dump(out/'manifest.json',manifest)
    verify_snapshot(old_snapshot);verify_snapshot(new_snapshot)
    if stage=='report':
        from .sensitivity_report import build_charts,write_markdown
        build_charts(out);write_markdown(out)
        return
    if not (out/'audit/weekly_panel.parquet').exists():
        (out/'audit').mkdir(exist_ok=True);data_changes(old_snapshot,new_snapshot,out/'audit')
        # build_panel requires a new destination and all handoff writes stay private.
        panel,prices=build_panel(new_snapshot,out/'prepared')
        for p in ('weekly_panel.parquet','stock_prices.parquet'):
            import shutil;shutil.copy2(out/'prepared'/p,out/'audit'/p)
    panel=pd.read_parquet(out/'audit/weekly_panel.parquet');prices=pd.read_parquet(out/'audit/stock_prices.parquet')
    old=features(pd.read_parquet(root/'weekly/weekly_panel.parquet'));old_prices=pd.read_parquet(root/'private_inputs/stock_prices.parquet')
    schedule=schedules(panel);dump(out/'split_schedule.json',{k:v.to_dict('records') for k,v in schedule.items()})
    if stage=='audit':return
    if not (out/'baseline/complete.json').exists():
        dest=out/'baseline';dest.mkdir(exist_ok=True)
        dump(dest/'legacy_ledger_verification.json',verify_legacy(root,old_prices))
        control=[v for v in MAIN if v.split('/')[0] in ('stock_only','full')]
        old_pred,_=fit_scores(old,schedule['historical'],control,dest/'original_fits',hashes)
        saved=pd.read_parquet(root/'weekly/predictions.parquet')
        dump(dest/'control_reproduction.json',reproduce(old_pred,saved))
        replay_cases(saved,prices,dest/'price_change',{'main':portfolio_sensitivities()['main']})
        common=compare_panels(old,panel,dest)
        current=fit_variants(panel,schedule['historical'],control,dest/'updated_fits',hashes,max_cores)
        old_pred.to_parquet(dest/'old_predictions.parquet',index=False);current.to_parquet(dest/'updated_predictions.parquet',index=False)
        for label,pred in [('old',old_pred),('updated',current)]:
            pred=pred.merge(common,on=['period','security_id'],validate='many_to_one')
            prediction_metrics(pred).to_csv(dest/(label+'_common_key_metrics.csv'),index=False)
        dump(dest/'complete.json',dict(original_predictions_reproduced=True,price_change_replay=True))
    if stage=='baseline':return
    for name,sched in schedule.items():
        dest=out/name;dest.mkdir(exist_ok=True)
        if not (dest/'predictions.parquet').exists():
            pred=fit_variants(panel,sched,MAIN,dest,hashes,max_cores);pred.to_parquet(dest/'predictions.parquet',index=False)
        pred=pd.read_parquet(dest/'predictions.parquet')
        reports,runs,prepared,pm=predict_and_evaluate(pred,prices,dest)
        if name=='assessment':
            pm=pm.merge(sched[['period']].assign(block=np.arange(len(sched))//13+1),on='period',validate='many_to_one')
            pm.groupby(['variant','block']).agg(mean_ic=('rank_ic','mean'),scoring_weeks=('rank_ic','count'),
                scheduled_weeks=('period','size')).reset_index().to_csv(dest/'block_metrics.csv',index=False)
        if name=='historical' and not (dest/'monte_carlo.json').exists():
            simulations(pred,prepared,runs,pm,dest,iterations,17,primary=PRIMARY,reference=REFERENCE)
        save_contrasts(pm,dest)
    if stage=='features':return
    if not (out/'sensitivity/predictions.parquet').exists():
        dest=out/'sensitivity';dest.mkdir(exist_ok=True)
        combined=pd.concat(schedule.values(),ignore_index=True)
        sensitivity=fit_variants(panel,combined,model_sensitivities(),dest,hashes,max_cores)
        sensitivity.to_parquet(dest/'predictions.parquet',index=False)
        m=prediction_metrics(sensitivity);m['window']=np.where(m.period.isin(schedule['historical'].period),'historical','assessment')
        m.to_csv(dest/'ranking_by_week.csv',index=False)
        for name,sched in schedule.items():
            subset=sensitivity[sensitivity.period.isin(sched.period)]
            predict_and_evaluate(subset,prices,dest/name)
        pred=pd.read_parquet(out/'historical/predictions.parquet')
        replay_cases(pred[pred.variant.isin([REFERENCE,PRIMARY])],prices,dest/'portfolio')
    if stage=='sensitivity':return
    run_diagnostics(panel,pd.concat(schedule.values(),ignore_index=True),out/'diagnostics',hashes,max_cores)
    from .sensitivity_report import build_charts,write_markdown
    build_charts(out);write_markdown(out)
    for p,digest in hashes.items():
        if sha(p)!=digest:raise ValueError('source changed during experiment: '+p)
    dump(out/'source_verification.json',dict(hashes_unchanged=True,monthly_reference_unchanged=True))
    dump(out/'complete.json',dict(status='complete',iterations=iterations,main_variants=len(MAIN),
         model_sensitivity_variants=len(model_sensitivities()),code_commit=commit,exploratory=True))
    print('Completed data/feature sensitivity experiment',out,flush=True)


def save_contrasts(pm,out):
    p=pm.pivot(index='period',columns='variant',values='rank_ic');rows=[]
    for kind in ('ridge','lightgbm'):
        pairs=[('compact','full'),('compact_plus','compact'),('pruned_full','full'),('pruned_extended','pruned_full')]
        pairs += [(g,'stock_only') for g in GROUPS if g!='stock_only']
        for a,b in pairs:
            d=p[a+'/'+kind]-p[b+'/'+kind]
            rows.append(dict(comparison=a+' minus '+b,model=kind,mean_ic_difference=d.mean(),
                             scorable_weeks=d.notna().sum(),scheduled_weeks=len(d)))
    pd.DataFrame(rows).to_csv(Path(out)/'paired_ic_differences.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('old-snapshot','new-snapshot','reference-results','output-dir'):p.add_argument('--'+arg,required=True)
    p.add_argument('--max-cores',type=int,default=4);p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--stage',choices=['audit','baseline','features','sensitivity','report','all'],default='all')
    a=p.parse_args();run(a.old_snapshot,a.new_snapshot,a.reference_results,a.output_dir,a.max_cores,a.iterations,a.stage)

if __name__=='__main__':main()
