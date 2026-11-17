"""Private charts and plain-language report for the sensitivity experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load(path):return json.loads(Path(path).read_text())

def build_charts(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'charts';out.mkdir(exist_ok=True)
    h=load(root/'historical/metrics.json')
    labels=list(h);ic=[h[v]['mean_ic'] for v in labels]
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(labels,ic);ax.axvline(0,color='black',lw=.7)
    ax.set_xlabel('Mean weekly rank IC');ax.set_title('Historical prediction quality by model and feature set')
    fig.tight_layout();fig.savefig(out/'ranking.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));count=0
    for v in labels:
        p=root/'historical'/v.replace('/','_')/'net_daily.parquet'
        if p.exists():
            d=pd.read_parquet(p);ax.plot(d.date,d.equity/100000,label=v);count+=1
    if count:ax.legend(fontsize=7,ncol=2)
    ax.set_ylabel('Equity relative to starting capital');ax.set_title('Complete net portfolios only; invalid portfolios are unavailable')
    fig.tight_layout();fig.savefig(out/'equity.png',dpi=150);plt.close(fig)
    draws=pd.read_parquet(root/'historical/random_draws.parquet')
    fig,ax=plt.subplots(figsize=(8,4));v='pruned_extended/ridge'
    d=draws[draws.variant.eq(v)].mean_rank_ic.dropna();ax.hist(d,bins=40,color='#517c9b')
    ax.axvline(h[v]['mean_ic'],color='#ac3737',label='Observed challenger');ax.legend()
    ax.set_xlabel('Mean rank IC after shuffling');ax.set_title('Size-conditioned random-signal comparison')
    fig.tight_layout();fig.savefig(out/'randomization.png',dpi=150);plt.close(fig)
    sens=pd.read_csv(root/'sensitivity/ranking_by_week.csv')
    summary=sens.groupby(['window','variant']).rank_ic.mean().unstack('window')
    fig,ax=plt.subplots(figsize=(11,6));summary.plot.barh(ax=ax);ax.axvline(0,color='black',lw=.7)
    ax.set_title('Model setting sensitivity; descriptive comparisons');ax.set_xlabel('Mean weekly rank IC')
    fig.tight_layout();fig.savefig(out/'model_sensitivity.png',dpi=150);plt.close(fig)
    for p in sorted((root/'diagnostics').glob('*/*/permutation_importance.csv')):
        f=pd.read_csv(p);f=f[f.kind.eq('individual')].sort_values('mean_ic_decrease').tail(10)
        fig,ax=plt.subplots(figsize=(8,5));ax.barh(f.features,f.mean_ic_decrease);ax.axvline(0,color='black',lw=.7)
        ax.set_xlabel('Validation rank IC decrease after permutation');ax.set_title('Chronological feature importance')
        fig.tight_layout();fig.savefig(p.parent/'importance.png',dpi=120);plt.close(fig)
        effects=pd.read_csv(p.parent/'local_effects.csv')
        fig,axes=plt.subplots(2,3,figsize=(11,6));top=f.tail(6).features.tolist()
        for ax,c in zip(axes.flat,top):
            e=effects[effects.feature.eq(c)]
            if 'centered_effect' in e:ax.plot((e.low+e.high)/2,e.centered_effect,marker='o',ms=3)
            ax.set_title(c,fontsize=9);ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(top):]:ax.set_visible(False)
        fig.suptitle('Local feature response; observed ranges, descriptive effects');fig.tight_layout()
        fig.savefig(p.parent/'sensitivity.png',dpi=120);plt.close(fig)


def write_markdown(root):
    root=Path(root);a=load(root/'audit/data_changes.json');h=load(root/'historical/metrics.json')
    sims=load(root/'historical/monte_carlo.json');delta=load(root/'historical/primary_ic_difference.json')
    lines=['# Options signals data and sensitivity experiment','',
      'Exploratory historical research. These periods were previously inspected. Negative returns and invalid portfolios are retained.',
      '', '## Updated data', '',
      f"The frozen update has {a['new_rows']:,} stock-days and {a['new_columns']} columns through {a['last_date']}; {a['stocks']} ticker strings.",
      'Stock histories provide total-return accounting. Undocumented fallback columns and precomputed targets are excluded from predictors.',
      'Monthly paper-method results remain the original limited diagnostic, with short history and incomplete paper controls.',
      '', '## Models in simple language','',
      'Ridge learns a weighted sum of inputs and penalizes large coefficients. Here it predicts the signed return between weekly entry and exit closes. The highest and lowest predictions determine stock selection.',
      'LightGBM combines decision trees that learn thresholds and interactions. Here it predicts the same return on identical dates and eligible rows, allowing a fair comparison with Ridge.',
      '', '## Experiment Model Feature set Results','',
      '| Model and features | Mean rank IC | Net return | Net Sharpe | Adjusted ranking p |',
      '|---|---:|---:|---:|---:|']
    for v,m in h.items():
        net=m['portfolio']['net'];ret=f"{net['cumulative_return']:.2%}" if net['status']=='complete' else 'Unavailable'
        sr=f"{net['sharpe']:.2f}" if net.get('sharpe') is not None else 'Unavailable'
        lines.append(f"| {v} | {m['mean_ic']:.4f} | {ret} | {sr} | {sims[v]['ranking_p_adjusted']} |")
    lines+=['','Rank IC measures whether predicted stock rankings match realized return rankings. Positive values are better; zero indicates no rank relationship.',
      'Sharpe measures average periodic return relative to its variability. Drawdown measures the largest daily equity decline from a prior peak. Turnover measures traded dollars relative to equity.',
      'Alpha, when available, is an internal-universe regression intercept with Newey-West uncertainty; it is not the paper factor alpha.',
      '',f"The predeclared challenger IC improvement over full Ridge is {delta['observed']:.5f}; its paired 95% bootstrap interval is {delta['bootstrap_95pct']}.",
      '', '## Monte Carlo and limitations','',
      'Each of twelve main variants receives 2,000 score shuffles within week and market-cap decile. Shared donor permutations pair the comparisons. Four-week block bootstrap samples measure uncertainty.',
      'Ranking and portfolio p-values are adjusted separately across twelve variants. Missing selected prices invalidate portfolio draws; counts and conservative p-value bounds are reported instead of dropping failures.',
      'Independent weekly shuffling changes turnover and persistence. The null is not turnover matched. The earlier assessment retains two undercovered weeks with unavailable rank statistics.',
      '', '## Sensitivity and attribution','',
      'Feature response uses local effects within observed training quantiles. Importance uses separate chronological validation fits, including grouped correlated inputs. These plots describe model behavior, not causal effects.',
      'Model settings, costs, account size, selection fraction, position caps and exposure are changed one at a time. Full results are in sensitivity/comparison files and the portfolio replay outputs.',
      'Long-only and short-only runs are distinct from P&L contributions within the combined portfolio. Costs include commission, regulatory charges, execution allowance and short borrow.',
      '', '## Interpretation','',
      'We report every main variant and sensitivity without choosing a new primary from historical performance. Missing prices, ticker identities, publication assumptions and corporate actions limit investment conclusions. This experiment cannot establish paper replication.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
