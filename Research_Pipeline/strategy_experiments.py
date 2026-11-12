"""Frozen-data paper diagnostic, weekly strategy, and reusable latest scoring.

python -m Research_Pipeline.strategy_experiments --help
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMRegressor,LGBMClassifier,early_stopping
from .ben_handoff import convert_ben_file
from .data import load_handoff,build_monthly_panel,PAPER_OPTIONS
from .strategy_data import weekly_panel,schedule,STOCK,OPTION
from .daily_ledger import prepare,ledger,metrics,InvalidPortfolio
from .run import _limit_cores
from .factors import newey_west_paired_mean

SEED=17

def dump(p,obj):
    
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
        if isinstance(x,np.generic):return x.item()
        return x
    Path(p).write_text(json.dumps(clean(obj),indent=2,sort_keys=True,default=str,allow_nan=False)+'\n')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def freeze(wide,stocks,out):
    out=Path(out)
    if out.exists():raise FileExistsError('snapshot must be a fresh directory')
    (out/'stocks').mkdir(parents=True); records=[]
    for source,target in [(Path(wide),out/'wide.parquet')]+[(p,out/'stocks'/p.name) for p in sorted(Path(stocks).glob('*.csv'))]:
        before=sha(source);shutil.copy2(source,target)
        if before!=sha(source) or before!=sha(target):raise ValueError(f'input changed while freezing: {source}')
        records.append(dict(source=str(source),copy=str(target.relative_to(out)),sha256=before,bytes=target.stat().st_size))
    dump(out/'snapshot.json',dict(files=records))

def verify_snapshot(root):
    spec=json.loads((root/'snapshot.json').read_text())
    for f in spec['files']:
        if sha(root/f['copy'])!=f['sha256']:raise ValueError('frozen input hash changed')
    return spec

def pipeline(kind):
    model=Ridge(alpha=10) if kind=='ridge' else LGBMRegressor(n_estimators=100,num_leaves=15,learning_rate=.05,min_child_samples=50,n_jobs=1,random_state=SEED,verbosity=-1)
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),model)

def weekly_models(panel,sched,out):
    complete=sched[sched.exit_date.notna() & sched.exit_date.le(panel.attrs['price_last'])]
    periods=complete.period.tolist()[-26:]
    if len(periods)!=26:raise ValueError('need 26 completed weekly periods')
    groups={'stock_only':STOCK,'options_only':OPTION,'combined':STOCK+OPTION};frames=[];windows=[]
    for period in periods:
        f=panel[panel.period.eq(period)].copy()
        if len(f)<200:raise ValueError(f'{period}: {len(f)} eligible stocks; need 200')
        decision=f.decision_at.iloc[0];train=panel[panel.label_available_at.lt(decision)&panel.target.notna()]
        if train.period.nunique()<52:raise ValueError('need 52 matured training weeks')
        windows.append(dict(period=period,training_weeks=int(train.period.nunique()),training_rows=len(train),first_signal=str(train.trade_date.min()),last_signal=str(train.trade_date.max()),latest_outcome=str(train.label_available_at.max()),decision_at=str(decision)))
        for group,cols in groups.items():
            for kind in ('ridge','lightgbm'):
                model=pipeline(kind).fit(train[cols],train.target)
                pred=f[['period','security_id','market_cap','entry_date','exit_date','target']].copy();pred['score']=model.predict(f[cols]);pred['variant']=f'{group}/{kind}';frames.append(pred)
                joblib.dump(dict(model=model,features=cols,training=windows[-1]),out/'models'/f'{period.replace("/","_")}_{group}_{kind}.joblib')
        pred=f[['period','security_id','market_cap','entry_date','exit_date','target']].copy();pred['score']=f.iv_gap.to_numpy();pred['variant']='iv_gap/rank';frames.append(pred)
    # The operational artifact uses all outcomes matured before the latest decision.
    latest=panel.sort_values('decision_at').iloc[-1].decision_at
    mature=panel[panel.label_available_at.lt(latest)&panel.target.notna()]
    model=pipeline('ridge').fit(mature[STOCK+OPTION],mature.target)
    joblib.dump(dict(model=model,features=STOCK+OPTION,decision_at=latest,training_weeks=int(mature.period.nunique())),out/'latest_model.joblib')
    dump(out/'training_windows.json',windows)
    return pd.concat(frames,ignore_index=True)

def paper_models(tables,out):
    panel,coverage=build_monthly_panel(tables,paper_sample_filters=True)
    sch=schedule(panel.trade_date.min(),panel.trade_date.max(),False)
    panel=panel.merge(sch[['signal_date','decision_at','entry_date','exit_date']],left_on='trade_date',right_on='signal_date',validate='many_to_one')
    panel=panel[panel.score_at.le(panel.decision_at)].copy();panel['period']=panel.period.astype(str)
    panel['iv_gap']=panel.avg_call_iv-panel.avg_put_iv
    desc={'class_counts':panel.label.value_counts().to_dict(),'class_feature_means':panel.groupby('label')[PAPER_OPTIONS].mean().replace({np.nan:None}).to_dict(), 'coverage':coverage}
    dump(out/'paper_descriptives.json',desc)
    groups={'iv_pair':['avg_call_iv','avg_put_iv'],'ten_options':PAPER_OPTIONS};frames=[];windows=[];unavailable=[]
    for period in pd.period_range('2025-05','2025-10',freq='M').astype(str):
        score=panel[panel.period.eq(period)].copy()
        if score.empty:unavailable.append(dict(period=period,reason='missing signal month'));continue
        train=panel[panel.label.notna()&panel.label_available_at.lt(score.decision_at.iloc[0])].copy()
        for group,cols in groups.items():
            combined=pd.concat([train,score],ignore_index=True)
            for col in cols:
                g=combined.groupby('period')[col];filled=combined[col].fillna(g.transform('median'))
                mean=filled.groupby(combined.period).transform('mean');sd=filled.groupby(combined.period).transform(lambda x:x.std(ddof=0)).replace(0,1)
                combined[col]=(filled-mean)/sd
            tr=combined.iloc[:len(train)];sc=combined.iloc[len(train):]
            for kind in ('logistic','lightgbm'):
                variant=f'{group}/{kind}';months=sorted(tr.period.unique());fit=tr
                if kind=='lightgbm':
                    split=max(1,int(.8*len(months)));fit=tr[tr.period.isin(months[:split])];val=tr[tr.period.isin(months[split:])]
                if set(fit.label)!={'crash','normal','jump'} or tr[cols].isna().any().any() or sc[cols].isna().any().any() or (kind=='lightgbm' and val.empty):
                    unavailable.append(dict(period=period,variant=variant,reason='not estimable: class/feature/validation coverage'));continue
                model=LogisticRegression(class_weight='balanced',max_iter=1000,random_state=SEED) if kind=='logistic' else LGBMClassifier(class_weight='balanced',n_jobs=1,random_state=SEED,verbosity=-1)
                if kind=='logistic':model.fit(tr[cols],tr.label)
                else:model.fit(fit[cols],fit.label,eval_set=[(val[cols],val.label)],callbacks=[early_stopping(5,verbose=False)])
                probs=model.predict_proba(sc[cols]);pred=sc[['period','security_id','market_cap','entry_date','exit_date','label']].copy()
                for label in ('jump','crash','normal'):pred[f'p_{label}']=probs[:,list(model.classes_).index(label)]
                pred['score']=pred.p_jump;pred['variant']=variant;frames.append(pred)
                windows.append(dict(period=period,variant=variant,training_months=len(months),class_counts=tr.label.value_counts().to_dict(),latest_training_period=str(tr.period.max()),latest_outcome=str(tr.label_available_at.max()),decision_at=str(score.decision_at.iloc[0])))
                joblib.dump(dict(model=model,features=cols,normalization='same-month median and z-score'),out/'models'/f'{period}_{group}_{kind}.joblib')
    dump(out/'training_windows.json',windows);dump(out/'not_estimable.json',unavailable)
    dump(out/'replication_deviations.json',dict(status='limited_paper_method_diagnostic',published_window='1996-2022',published_test='2006-2022',diagnostic_signals='2025-05 through 2025-10',missing_controls=['AGE','DTURN','EXRET','TANG','TSKEW','TVOL','RM12','SG','SIZE'],unavailable_table3_specifications=19,unverified_filters=['31-90 day expiry','positive option volume and bid','bid<ask','midpoint>=0.10','no-arbitrage','IV 0.01-5'],price_basis='last trade, not bid/ask midpoint',identity='ticker proxy',universe='covered tickers; historical common-stock and S&P membership unverified',training_limit='approximately two initial matured months; not ten years',label_schedule='matured outcomes only; published schedule ambiguity recorded'))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def monte_carlo(prepared,observed,iterations,freq):
    rng=np.random.default_rng(SEED);null=[];invalid=0
    for _ in range(iterations):
        try:null.append(float(ledger(prepared,rng=rng,capture=False)['periodic'].net_return.mean()))
        except InvalidPortfolio:invalid+=1
    r=observed['periodic'].net_return.to_numpy();block=4 if freq==52 else 3;boot=[]
    for _ in range(iterations):
        starts=rng.integers(0,len(r)-block+1,size=int(np.ceil(len(r)/block)))
        sample=np.concatenate([r[i:i+block] for i in starts])[:len(r)];sd=sample.std(ddof=1)
        boot.append([sample.mean(),np.prod(1+sample)**(freq/len(sample))-1, sample.mean()/sd*np.sqrt(freq) if sd else np.nan])
    a=np.array(boot)
    return dict(iterations=iterations,seed=SEED,invalid_null_draws=invalid,
        status='complete' if invalid==0 else 'inconclusive: random portfolios have missing prices or insolvency',
        p_raw=(1+int(np.sum(np.array(null)>=r.mean())))/(iterations+1) if invalid==0 else None,
        null_mean_return_95pct=np.quantile(null,[.025,.975]).tolist() if null else None,
        mean_return_95pct=np.quantile(a[:,0],[.025,.975]).tolist(),annualized_return_95pct=np.quantile(a[:,1],[.025,.975]).tolist(),
        sharpe_95pct=np.nanquantile(a[:,2],[.025,.975]).tolist(),block_periods=block,
        limitation='six monthly observations: uncertainty estimates extremely limited' if freq==12 else '26 weekly test observations; preliminary evidence')

def alpha_proxy(r,b):
    y=np.array(r);x=np.column_stack([np.ones(len(y)),b]);beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta;z=x*res[:,None];meat=z.T@z
    for lag in range(1,min(4,len(y)-1)+1):
        cross=z[lag:].T@z[:-lag];meat+=(1-lag/5)*(cross+cross.T)
    inv=np.linalg.pinv(x.T@x);cov=inv@meat@inv*len(y)/(len(y)-2);se=np.sqrt(max(0,cov[0,0]))
    return dict(alpha_weekly=float(beta[0]),alpha_annualized_linear=float(52*beta[0]),beta=float(beta[1]),newey_west_lags=4,t=float(beta[0]/se) if se else None,definition='internal-universe alpha proxy; not FF5 plus momentum alpha')

def evaluate(predictions,prices,out,iterations,weekly):
    frequency=52 if weekly else 12;reports={};sims={};periodic=[];daily=[];all_holdings=[];benchmark={}
    for variant,f in predictions.groupby('variant',sort=True):
        print(f'evaluating {variant}',flush=True)
        if f.period.nunique()!=(26 if weekly else 6):
            reports[variant]={'status':'not estimable for every test period'};continue
        prepared=prepare(f,prices)
        br=[]
        for p in prepared:
            relative=p['prices'][-1]/p['prices'][0]-1
            br.append(float(np.mean(relative)) if np.isfinite(relative).all() else np.nan)
        benchmark[variant]=br
        try:
            run=ledger(prepared);primary=metrics(run,frequency);reports[variant]={'primary':primary,'cases':{}}
            for bps,borrow,label in ((0,0,'gross'),(30,0,'net_zero_borrow'),(10,.03,'cost_10bps'),(50,.03,'cost_50bps')):
                case=ledger(prepared,bps=bps,borrow=borrow);reports[variant]['cases'][label]=metrics(case,frequency)
            for table,dest in ((run['periodic'],periodic),(run['daily'],daily),(run['holdings'],all_holdings)):
                table['variant']=variant;dest.append(table)
            sims[variant]=monte_carlo(prepared,run,iterations,frequency)
            if weekly:
                reports[variant]['rank_ic_by_week']={str(p):float(g[['score','target']].corr(method='spearman').iloc[0,1]) for p,g in f.groupby('period')}
                reports[variant]['alpha_proxy']=alpha_proxy(run['periodic'].net_return,br) if np.isfinite(br).all() else {'status':'unavailable: missing benchmark prices'}
            else:
                reports[variant]['classification']={}
                for label in ('jump','crash'):
                    labeled=f[f.label.notna()];truth=labeled.label.eq(label)
                    auc=float(roc_auc_score(truth,labeled[f'p_{label}'])) if truth.nunique()==2 else None
                    top=pd.concat([g.nlargest(max(1,int(np.ceil(.05*len(g)))),f'p_{label}') for _,g in labeled.groupby('period')])
                    reports[variant]['classification'][label]=dict(auc=auc,accuracy_ratio=2*auc-1 if auc is not None else None,top5_precision=float(top.label.eq(label).mean()))
        except InvalidPortfolio as exc:reports[variant]={'status':'invalid strategy','reason':str(exc)}
    family=7 if weekly else 4
    for v,s in sims.items():s['p_adjusted']=min(1,s['p_raw']*family) if s['p_raw'] is not None else None
    if periodic:
        ret=pd.concat(periodic);ret.to_parquet(out/'periodic_returns.parquet',index=False)
        pd.concat(daily).to_parquet(out/'daily_equity.parquet',index=False);pd.concat(all_holdings).to_parquet(out/'holdings.parquet',index=False)
        pd.DataFrame({k:pd.Series(v) for k,v in benchmark.items()}).to_parquet(out/'equal_weight_benchmark.parquet')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(11,8))
        for v,g in pd.concat(daily).groupby('variant'):
            axes[0].plot(g.date,g.equity,label=v);axes[1].plot(g.date,g.equity/g.equity.cummax()-1,label=v)
        axes[0].set_ylabel('Net equity');axes[1].set_ylabel('Drawdown');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'equity_drawdown.png');plt.close(fig)
        if weekly and {'combined/ridge','stock_only/ridge'}<=set(ret.variant):
            a=ret[ret.variant.eq('combined/ridge')].rename(columns={'net_return':'gross_return'});b=ret[ret.variant.eq('stock_only/ridge')].rename(columns={'net_return':'gross_return'})
            dump(out/'incremental_options_value.json',newey_west_paired_mean(a,b,lags=4))
    dump(out/'metrics.json',reports);dump(out/'monte_carlo.json',sims)
    status=('diagnostic_complete' if not weekly else 'strategy_not_supported') if sum('primary'in x for x in reports.values())==family else 'inconclusive_incomplete_variants'
    if weekly and 'primary' in reports.get('combined/ridge',{}) and 'primary' in reports.get('stock_only/ridge',{}):
        m=reports['combined/ridge']['primary'];s=sims['combined/ridge'];base=reports['stock_only/ridge']['primary']
        gates=dict(positive_net=m['cumulative_return']>0,sharpe_at_least_one=m['sharpe'] is not None and m['sharpe']>=1,drawdown_limit=m['max_drawdown']>=-.2,beats_stock_only=m['mean_periodic_return']>base['mean_periodic_return'],randomization=s['p_adjusted'] is not None and s['p_adjusted']<.05,bootstrap=s['mean_return_95pct'][0]>0)
        dump(out/'success_gates.json',gates)
        if all(gates.values()):status='promising_pilot'
    dump(out/'status.json',dict(status=status,complete_variants=sum('primary'in x for x in reports.values()),expected_variants=family,claims='limited paper-method diagnostic' if not weekly else 'preliminary weekly strategy on covered ticker universe'))
    (out/'REPORT.md').write_text('# '+('Weekly strategy' if weekly else 'Limited paper-method diagnostic')+'\n\nStatus: **'+status+'**.\n\n'+ '\n'.join(f'- {v}: '+(f"net cumulative {x['primary']['cumulative_return']:.2%}; Sharpe {x['primary']['sharpe']}; daily drawdown {x['primary']['max_drawdown']:.2%}" if 'primary'in x else str(x)) for v,x in reports.items())+'\n\nSee training_windows.json, metrics.json, monte_carlo.json, and dated Parquet outputs. Ticker identities, delistings, historical membership, last-trade IV, publication time and borrow availability remain limitations.\n')

def run(snapshot,out,mode,iterations,max_cores):
    _limit_cores(max_cores);snapshot=Path(snapshot);out=Path(out)
    if out.exists():raise FileExistsError('fresh output directory required')
    spec=verify_snapshot(snapshot);out.mkdir(parents=True)
    convert_ben_file(snapshot/'wide.parquet',out/'handoff',stocks_dir=snapshot/'stocks',allow_ticker_proxy=True)
    tables=load_handoff(out/'handoff')
    dump(out/'manifest.json',dict(snapshot=spec,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),max_cores=max_cores,seed=SEED,iterations=iterations,source_dates=[str(tables['options_features'].trade_date.min()),str(tables['options_features'].trade_date.max())],options_rows=len(tables['options_features']),tickers=tables['options_features'].security_id.nunique(),software={p:__import__('importlib.metadata',fromlist=['version']).version(p) for p in ('numpy','pandas','scikit-learn','lightgbm','exchange-calendars')},assumptions=['ticker proxy','last-trade option prices','11:30 ET assumed readiness','covered universe, not historical S&P','zero cash interest; financing and borrow availability unverified']))
    for track in (['paper_diagnostic','weekly'] if mode=='both' else [mode]):
        dest=out/track;(dest/'models').mkdir(parents=True)
        if track=='weekly':
            panel,sched=weekly_panel(tables);panel.attrs['price_last']=tables['stock_prices'].trade_date.max();pred=weekly_models(panel,sched,dest)
            panel.drop(columns=[],errors='ignore').to_parquet(dest/'weekly_panel.parquet',index=False)
            latest=panel[panel.decision_at.eq(panel.decision_at.max())].copy();artifact=joblib.load(dest/'latest_model.joblib');latest['score']=artifact['model'].predict(latest[artifact['features']]);latest[['security_id','trade_date','entry_date','score']].to_parquet(dest/'latest_scores.parquet',index=False)
            dump(dest/'feature_schema.json',dict(stock=STOCK,options=OPTION,primary='combined/ridge',target='next scheduled weekly entry-to-exit adjusted return'))
        else:pred=paper_models(tables,dest)
        if pred.empty:dump(dest/'status.json',dict(status='not_estimable'));continue
        pred.to_parquet(dest/'predictions.parquet',index=False);evaluate(pred,tables['stock_prices'],dest,iterations,track=='weekly')
    verify_snapshot(snapshot)
    (out/'REPORT.md').write_text('# Existing-data options experiments\n\nSee paper_diagnostic/REPORT.md and weekly/REPORT.md for results. The paper track is a limited method diagnostic; the weekly track is a different target. Source manifests, assumptions, and immutable hashes are in manifest.json.\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze');f.add_argument('--wide-input',required=True);f.add_argument('--stocks-dir',required=True);f.add_argument('--output-dir',required=True)
    r=sub.add_parser('run');r.add_argument('--snapshot',required=True);r.add_argument('--output-dir',required=True);r.add_argument('--mode',choices=['both','weekly','paper_diagnostic'],default='both');r.add_argument('--iterations',type=int,default=2000);r.add_argument('--max-cores',type=int,default=4)
    s=sub.add_parser('score');s.add_argument('--model',required=True);s.add_argument('--panel',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.wide_input,a.stocks_dir,a.output_dir)
    elif a.action=='run':run(a.snapshot,a.output_dir,a.mode,a.iterations,a.max_cores)
    else:
        artifact=joblib.load(a.model);panel=pd.read_parquet(a.panel);f=panel[panel.decision_at.eq(panel.decision_at.max())].copy()
        if pd.Timestamp(artifact['decision_at'])>f.decision_at.min():raise ValueError('model is newer than scoring decision')
        f['score']=artifact['model'].predict(f[artifact['features']]);f[['security_id','trade_date','entry_date','score']].to_parquet(a.output,index=False)
if __name__=='__main__':main()
