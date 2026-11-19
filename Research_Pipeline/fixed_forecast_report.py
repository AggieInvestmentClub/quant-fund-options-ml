"""Private tables, charts and plain-language fixed-universe research report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def load(path):return json.loads(Path(path).read_text())
def pct(x):return 'Unavailable' if x is None or not np.isfinite(x) else f'{100*x:.2f}%'
def number(x):return 'Unavailable' if x is None or not np.isfinite(x) else f'{x:.4f}'

def build_report(root):
    root=Path(root);fig=root/'figures';fig.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts=Path('/System/Library/Fonts/Supplemental')
    for p in fonts.glob('Times New Roman*.ttf'):font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':'Times New Roman' if fonts.exists() else 'DejaVu Serif',
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(name):
        plt.tight_layout();plt.savefig(fig/(name+'.png'),dpi=200,bbox_inches='tight',facecolor='white');plt.close()
    manifest=load(root/'manifest.json');inference=load(root/'historical/inference/statistics.json')
    rows=[]
    for window in ('development','historical'):
        accuracy=load(root/window/'accuracy.json');portfolios=load(root/window/'portfolios/metrics.json')
        for v,a in accuracy.items():
            p=portfolios[v]['net'];gross=portfolios[v]['gross']
            rows.append(dict(window=window,variant=v,**a,status=p['status'],net_return=p.get('cumulative_return'),
                gross_return=gross.get('cumulative_return'),profit=p.get('net_profit'),sharpe=p.get('sharpe'),
                max_drawdown=p.get('max_drawdown'),mean_cash=p.get('mean_cash_fraction'),
                turnover=p.get('mean_weekly_turnover'),costs=p.get('cost_components'),
                adjusted_forecast_p=inference[v]['mse_improvement']['p_adjusted'] if window=='historical' else None,
                adjusted_portfolio_p=inference[v]['net']['p_adjusted'] if window=='historical' else None))
    comparison=pd.DataFrame(rows);comparison.to_csv(root/'comparison.csv',index=False)
    current=comparison[comparison.window.eq('historical')].set_index('variant')
    colors=['#315a78','#20776b','#9b653c','#7c65a0','#587436','#a44242']
    fig1,ax=plt.subplots(figsize=(8,3))
    for (v,_),color in zip(current.iterrows(),colors):
        q=pd.read_csv(root/'historical/accuracy_by_decision.csv');q=q[q.variant.eq(v)]
        ax.plot(range(len(q)),q.mse_improvement,marker='o',label=v,color=color)
    ax.axhline(0,color='black',lw=.6);ax.set_ylabel('MSE improvement versus historical mean');ax.set_xlabel('Four-week decision');ax.legend(ncol=2,fontsize=8,frameon=False);save('forecast_skill')
    _,ax=plt.subplots(figsize=(8,3))
    for (v,_),color in zip(current.iterrows(),colors):
        p=root/'historical/portfolios'/v.replace('/','_')/'net_daily.parquet'
        if p.exists():
            q=pd.read_parquet(p);ax.plot(q.date,q.equity/1000,label=v,color=color)
    q=pd.read_parquet(root/'historical/portfolios/passive_25/net_daily.parquet');ax.plot(q.date,q.equity/1000,label='Passive 25',color='black',ls='--')
    ax.set_ylabel('Net equity in thousands of dollars');ax.tick_params(axis='x',rotation=15);ax.legend(ncol=2,fontsize=8,frameon=False);save('equity')
    _,ax=plt.subplots(figsize=(8,2.8))
    for v,color in zip(['full/ridge','pruned/ridge'],colors):
        p=root/'historical/portfolios'/v.replace('/','_')/'net_daily.parquet'
        if p.exists():
            q=pd.read_parquet(p);ax.plot(q.date,q.gross_exposure*100,label=v,color=color)
    ax.set_ylabel('Invested percent of equity');ax.tick_params(axis='x',rotation=15);ax.legend(frameon=False);save('exposure')
    pred=pd.read_parquet(root/'historical/predictions.parquet');q=pred[pred.variant.eq('full/ridge')&np.isfinite(pred.target)]
    _,axes=plt.subplots(1,2,figsize=(8,3))
    axes[0].scatter(q.score*100,q.target*100,s=14,alpha=.55);axes[0].set_xlabel('Predicted four-week return percent');axes[0].set_ylabel('Actual four-week return percent')
    ordered=q.sort_values(['score','security_id']).copy();ordered['bin']=np.repeat(np.arange(5),[len(x) for x in np.array_split(ordered,5)])
    calibration=ordered.groupby('bin').agg(predicted=('score','mean'),actual=('target','mean'));calibration.to_csv(root/'historical/calibration.csv')
    axes[1].plot(calibration.predicted*100,calibration.actual*100,'o-');axes[1].set_xlabel('Mean predicted return percent');axes[1].set_ylabel('Mean realized return percent');save('forecast_calibration')
    importance_root=root/'historical/feature_effects'
    candidates=sorted(importance_root.glob('*/pruned_ridge/importance.csv'))
    if candidates:
        q=pd.read_csv(candidates[-1]);q=q[q.kind.eq('individual')].nlargest(8,'mse_increase').iloc[::-1]
        _,ax=plt.subplots(figsize=(8,3));ax.barh(q.feature,q.mse_increase,color=colors[0]);ax.set_xlabel('Validation MSE increase after shuffling');save('importance')
        q=pd.read_csv(candidates[-1].with_name('local_effects.csv'));chosen=list(q.feature.unique())[:3]
        _,axes=plt.subplots(1,len(chosen),figsize=(8,2.6),squeeze=False)
        for ax,c in zip(axes[0],chosen):
            f=q[q.feature.eq(c)];ax.plot((f.low+f.high)/2,100*f.centered_effect,'o-');ax.set_title(c,fontsize=9)
        axes[0,0].set_ylabel('Centered predicted-return effect pp');save('local_effects')
    sensitivities=load(root/'historical/sensitivity/comparison.json') if (root/'historical/sensitivity/comparison.json').exists() else []
    if sensitivities:
        records=[]
        for s in sensitivities:
            p=s['portfolio']['net'];records.append(dict(case=s['case'],kind=s['kind'],mse=s.get('forecast',{}).get('mse'),
                 net_return=p.get('cumulative_return'),sharpe=p.get('sharpe'),status=p['status']))
        pd.DataFrame(records).to_csv(root/'sensitivity_comparison.csv',index=False)
        _,axes=plt.subplots(1,2,figsize=(8,3))
        for group,color in [('full',colors[0]),('pruned',colors[1])]:
            vals=[next(s['forecast']['mse'] for s in sensitivities if s['case']==f'{group}_ridge_alpha_1'),
                  current.loc[group+'/ridge','mse'],next(s['forecast']['mse'] for s in sensitivities if s['case']==f'{group}_ridge_alpha_100')]
            axes[0].plot([0,1,2],vals,'o-',label=group,color=color)
            vals=[]
            for t in (.80,.90,.95):
                vals.append(current.loc['pruned/ridge','mse'] if t==.90 else next(s['forecast']['mse'] for s in sensitivities if s['case']==f'pruned_ridge_pearson_{t}'))
        axes[1].plot([.8,.9,.95],vals,'o-',color=colors[1]);axes[1].set_xlabel('Pearson pruning threshold');axes[1].set_ylabel('MSE')
        axes[0].set_xticks([0,1,2],[1,10,100]);axes[0].set_xlabel('Ridge alpha');axes[0].set_ylabel('MSE');axes[0].legend(frameon=False);save('sensitivity')
    lines=['# Fixed-universe return forecasting and long-or-cash trading','',
      'Exploratory research on frozen inputs. No stock ranking, top-20 selection, new market data or live orders.',
      '', '## Purpose and method',
      'Predict actual adjusted stock returns over four weeks. Train on matured broader-universe observations; score the same configured 25 stocks. Each stock has a 4% post-cost equity slot: positive forecast means long; zero or negative means cash. No redistribution. Full Ridge is the predeclared primary.',
      'Development: June 2, 2025 to March 9, 2026. Historical comparison: May 11 to October 26, 2026. The latter has six decisions and 24 weekly valuations. Dates were previously inspected.',
      '', '## Experiment → Model → Features → Results',
      '| Model | Features | RMSE | Direction | MSE skill | Rank IC | Net return | Sharpe | Drawdown | Mean cash |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v,r in current.iterrows():
        g,m=v.split('/');lines.append(f'| {m} | {g} | {number(r.rmse)} | {pct(r.direction_accuracy)} | {pct(r.mse_skill)} | {number(r.rank_ic)} | {pct(r.net_return)} | {number(r.sharpe)} | {pct(r.max_drawdown)} | {pct(r.mean_cash)} |')
    passive=load(root/'historical/portfolios/metrics.json')['passive_25']['net']
    lines+=['',f"Passive 25-stock four-week control: {pct(passive.get('cumulative_return'))} net. SPY is a separate frozen market benchmark. Profits from passive holdings are not ML profits.",
      '', '## What the measurements mean',
      'RMSE/MAE measure percentage-return forecast errors in decimal-return units. Direction accuracy measures correct positive/nonpositive calls; balanced accuracy weights both realized directions equally. MSE skill = 1 − model MSE / historical-mean MSE; negative means worse than the simple forecast. Date means have equal weight.',
      'Portfolio net return includes modeled commissions, regulatory fees, 1 bp execution and final liquidation. Cash earns zero; no shorts or borrow. Sharpe uses weekly returns and √52. Drawdown uses daily equity. Alpha uses SPY weekly returns and four-lag Newey–West uncertainty, not the paper factor alpha.',
      '', '## Pearson pruning and sensitivity',
      'Each fit removes >50% missing/constant inputs, then prunes |Pearson| ≥ .90 with at least 200 paired training observations. Coverage/name ordering is deterministic. Before/after Pearson/Spearman matrices and removal reasons are saved. Full-feature controls remain. Correlation alone is not proof that a variable lacks value.',
      'Sensitivities vary Ridge alpha, LightGBM leaves/trees, Pearson thresholds .80/.90/.95, entry cutoffs 0/.5/1%, execution 0/1/5 bp and capital $10k/$100k/$1m. They are descriptive; the primary model is not replaced.',
      'Permutation importance uses chronological fixed-25 validation, grouped correlated inputs and MSE loss; local response plots use observed supported bins. Effects are not causal.',
      '', '## Monte Carlo and bootstrap',
      '2,000 fixed-seed within-date market-cap-decile forecast shuffles, identical donors across six main variants. Forecast error and gross/net ledgers are recomputed. Family sizes are six per statistic, not a correction for all prior project trials. Size groups contain only two or three stocks and the null conditions on active-slot counts; it does not independently establish aggregate market timing.',
      '2,000 paired two-hold (eight-week) block samples resample forecast metrics and the corresponding four weekly returns together. Missing draws remain invalid, with conservative p-value bounds. Missing observed prices invalidate the affected portfolio rather than changing holdings.',
      '', '| Variant | Adjusted forecast p | Adjusted net p | Invalid net draws |', '|---|---:|---:|---:|']
    for v,s in inference.items():lines.append(f"| {v} | {number(s['mse_improvement']['p_adjusted'])} | {number(s['net']['p_adjusted'])} | {s['net']['invalid_draws']} |")
    lines+=['','## Paper connection and limitations',
      'The reference paper predicts 12-month jump/normal/crash classes using logistic regression and LightGBM. This experiment instead predicts four-week percentage returns using Ridge/LightGBM. The separate limited monthly diagnostic remains unchanged; neither this adaptation nor a positive return reproduces the paper.',
      'Short history, chosen stocks, ticker proxies, uncertain source timing and unresolved price/corporate-action issues remain. The 25-stock list is not a verified historical S&P universe. Distinguish forecast quality, costed trading value and software completion.',
      '', '## Reproducibility',f"Code commit: {manifest['code_commit']}. Source file hashes, code hashes, software versions, fitted models, forecasts, ledgers and inference remain private. Configurations and source manifests reject incompatible resumes."]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return comparison
