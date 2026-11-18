"""Private plain-language horizon report; presentation export is optional."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load(path):return json.loads(Path(path).read_text())
def fmt(x,percent=False):return 'Unavailable' if x is None or not np.isfinite(x) else f'{x*100:.2f}%' if percent else f'{x:.3f}'


def write_report(root):
    root=Path(root);selection=load(root/'selection.json');rows=[]
    for window in ('development','historical'):
        metrics=load(root/window/'metrics.json')
        for variant,case in metrics.items():
            h,target,features,model=variant.split('/')
            rank=pd.read_parquet(root/window/(variant.replace('/','_')+'_ranking.parquet'))
            n=case['net'];g=case['gross']
            rows.append(dict(experiment='horizons',window=window,variant=variant,horizon=h,target=target,
                features=features,model=model,mean_ic=rank.rank_ic.mean(),known_outcomes=rank.known_outcomes.sum(),
                net_return=n.get('cumulative_return'),gross_return=g.get('cumulative_return'),
                sharpe=n.get('sharpe'),drawdown=n.get('max_drawdown'),turnover=n.get('mean_weekly_turnover'),
                status=n['status'],reason=n.get('reason','')))
    table=pd.DataFrame(rows);table.to_csv(root/'comparison.csv',index=False)
    stats=load(root/'inference/statistics.json') if (root/'inference/statistics.json').exists() else {}
    lines=['# Options signals holding periods and selective trading','',
      'All results are exploratory historical research. These dates have already been inspected. No live orders were placed.',
      '', '## What we tested','',
      'We compared one, two and four week holds, absolute stock returns and returns relative to the available stock universe. We then tested momentum confirmations and trading rules that allow stronger entries and longer retention. The original monthly paper diagnostic is unchanged.',
      '', '## Data and dates','',
      'The inputs are the existing frozen November 16 panel and stock histories. Development covers June 2 2025 to March 9 2026. The historical comparison covers May 11 to October 26 2026, giving 24 one-week, 12 two-week or six four-week holds. Both earlier April 2026 coverage failures remain documented; no values are invented.',
      '', '## How the models work','',
      'Ridge predicts a return using a weighted sum of inputs and discourages unusually large coefficients. Here we compare stock-only and combined stock/options inputs, with the same fixed regularization and training-only preprocessing.',
      '', 'LightGBM combines decision trees that learn thresholds and interactions. Here it predicts the same horizon-specific target with fixed model settings. Neither model classifies weekly jumps or crashes.',
      '', 'The relative target subtracts the equal-weight mean known stock return for that decision period. It asks which stocks will do better than their peers. Its zero is relative to that universe, rather than a prediction of zero absolute return.',
      '', 'Each fit uses only outcomes whose exits occurred before its decision. Training examples are weekly and may overlap for longer horizons; actual portfolios do not overlap.',
      '', '## Development selection','',
      selection['status']+'. Configuration: `'+selection['variant']+'`.',
      'Selection uses complete development portfolios with positive net return and positive mean rank IC, then highest weekly Sharpe. If none qualifies, four-week relative full Ridge is the predefined diagnostic fallback. Historical results cannot change this selection.',
      '', '## Experiment horizon targets models and results','',
      '| Horizon | Target | Features | Model | Mean IC | Gross return | Net return | Sharpe | Status |',
      '|---|---|---|---|---:|---:|---:|---:|---|']
    for r in table[table.window.eq('historical')].itertuples():
        lines.append(f'| {r.horizon} | {r.target} | {r.features} | {r.model} | {fmt(r.mean_ic)} | {fmt(r.gross_return,True)} | {fmt(r.net_return,True)} | {fmt(r.sharpe)} | {r.status} |')
    lines+=['','## Momentum and trading rule results','',
      'Momentum ranks the trailing 63-session stock return. Long candidates require positive momentum; short candidates require negative momentum. The gap filter requires stronger call-minus-put IV for longs and weaker values for shorts. The final filter requires lower put skew for longs and higher skew for shorts. Missing confirmations fail the filter, leaving unfilled exposure in cash.',
      '', 'Baseline entries use the highest and lowest 5 percent. Stricter entries use 2.5 percent. The holding buffer retains existing stocks within the outer 10 percent. Retention never bypasses a failed confirmation. These are rank thresholds, not calibrated confidence probabilities.',
      '', '| Configuration | Net return | Sharpe | Drawdown | Status |','|---|---:|---:|---:|---|']
    rulefile=root/'historical/rules/metrics.json'
    if rulefile.exists():
        rule=load(rulefile)
        for name,m in rule.items():
            n=m['net'];lines.append(f'| {name} | {fmt(n.get("cumulative_return"),True)} | {fmt(n.get("sharpe"))} | {fmt(n.get("max_drawdown"),True)} | {n["status"]} |')
    lines+=['','## What the measurements mean','',
      'Rank IC measures whether predicted rankings agree with realized rankings; positive is better. Net return includes commission, regulatory charges, spread/slippage and borrow. Sharpe compares average weekly return with its variability. Drawdown is the worst daily loss from an earlier equity peak. Turnover measures traded dollars relative to portfolio equity.',
      '', 'We use $100,000, equal weights, a 5 percent cap per stock, the previously frozen IBKR commission approximation, 1 basis point execution cost and 3 percent annual short borrow. Positions can drift between scheduled rebalances. Zero cash interest and the prior financing limitations remain. Less exposure or fewer trades can improve losses without improving stock selection, so exposure and costs must be read alongside returns.',
      '', 'Long and short contributions are separated. Standalone portfolios have independent cash, costs and equity. Alpha is only an internal-universe proxy when all benchmark prices are available; it is not the reference paper factor alpha.',
      '', '## Monte Carlo and stability','',
      'Each declared historical strategy receives 2,000 score shuffles within decision period and market-cap decile. Momentum and confirmation inputs move together. Holdings, retention and charges are replayed after each shuffle. Shared permutations pair gross/net and comparable strategies. Independent period shuffles are not turnover matched.',
      '', 'All declared portfolio cases count toward one Bonferroni family; ranking tests form a separate 24-configuration family. Missing-price draws remain invalid and contribute conservative p-value bounds. We never report valid-draw-only p-values.',
      '', 'The same 2,000 eight-calendar-week block resamples are used across strategies. Intervals describe historical uncertainty, with only 24 weeks and six four-week holds. They are not live-trading validation.',
      '', '| Case | Net adjusted p | Invalid net draws |','|---|---:|---:|']
    for key,m in stats.items():lines.append(f'| {key} | {fmt(m["net"]["p_adjusted"])} | {m["net"]["invalid_draws"]} |')
    primary=stats.get('stage2/ml_both/long_short',{})
    rules=load(rulefile) if rulefile.exists() else {}
    net=rules.get('ml_both/long_short',{}).get('net',{})
    ci=primary.get('net_mean_weekly_return_95pct');p=primary.get('net',{}).get('p_adjusted')
    passes=(net.get('status')=='complete' and net.get('cumulative_return',-1)>0 and
       net.get('sharpe') is not None and net['sharpe']>=1 and net.get('max_drawdown',-1)>=-.2 and
       p is not None and p<.05 and ci is not None and ci[0]>0 and selection['status'].startswith('qualified'))
    conclusion='Promising for further testing only; historical evidence remains exploratory.' if passes else 'The selected ML strategy is not supported by the predefined profitability screen in this test.'
    lines+=['','## Interpretation','',conclusion,
      'The screen requires positive net return, Sharpe at least one, drawdown no worse than minus 20 percent, adjusted randomization p below .05 and a positive lower bootstrap bound for mean return. An options-value claim additionally requires improvement over a matched stock-only model with identical trading rules.',
      '', 'Negative returns, unavailable alpha and invalid portfolios remain in the report. Missing prices, ticker identities, adjustment quality and assumed publication times limit conclusions. These experiments do not reproduce the reference paper performance.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    charts(root,table)
    return table


def charts(root,table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'charts';out.mkdir(exist_ok=True)
    h=table[table.window.eq('historical')]
    fig,ax=plt.subplots(figsize=(10,8));ax.barh(h.variant,h.mean_ic);ax.axvline(0,color='black',lw=.7)
    ax.set_xlabel('Mean holding-period rank IC');ax.set_title('Prediction quality by horizon target model and features')
    fig.tight_layout();fig.savefig(out/'ranking.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for name in ['ml_baseline_long_short','ml_both_long_short','momentum_plain_long_short','momentum_gap_skew_both_long_short']:
        p=root/'historical/rules'/name/'net_daily.parquet'
        if p.exists():
            d=pd.read_parquet(p);ax.plot(d.date,d.equity/100000,label=name)
    ax.legend(fontsize=7);ax.set_ylabel('Equity relative to starting capital');fig.tight_layout()
    fig.savefig(out/'equity.png',dpi=130);plt.close(fig)
    # Reuse existing training-only effect CSVs to keep interpretation consistent.
    for p in root.glob('*/fits/*/effects/importance.csv'):
        f=pd.read_csv(p);f=f[f.kind.eq('individual')].sort_values('mean_ic_decrease').tail(6)
        fig,ax=plt.subplots(figsize=(8,4));ax.barh(f.features,f.mean_ic_decrease);ax.axvline(0,color='black',lw=.7)
        ax.set_xlabel('Chronological validation IC loss after permutation');fig.tight_layout();fig.savefig(p.parent/'importance.png',dpi=100);plt.close(fig)
        e=pd.read_csv(p.parent/'local_effects.csv');fig,axes=plt.subplots(2,3,figsize=(10,6))
        for ax,c in zip(axes.flat,f.features):
            a=e[e.feature.eq(c)]
            if 'centered_effect' in a:ax.plot(a.high,a.centered_effect,marker='o',ms=2)
            ax.set_title(c,fontsize=8)
        for ax in list(axes.flat)[len(f):]:ax.set_visible(False)
        fig.tight_layout();fig.savefig(p.parent/'sensitivity.png',dpi=100);plt.close(fig)
