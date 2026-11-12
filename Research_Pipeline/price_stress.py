"""Conditional missing-price sensitivity; never a repaired historical backtest.

Uses saved scores and frozen prices. No model fitting or market-data downloads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .daily_ledger import prepare, select_weights, ledger, metrics, InvalidPortfolio
from .run import _limit_cores
from .strategy_experiments import dump

SHOCKS=(-.25,-.10,0.,.10,.25)

def missing_cells(predictions,prices):
    cells=[]
    for _,f in predictions.groupby('variant',sort=True):
        for p in prepare(f,prices):
            bad=~np.isfinite(p['prices'])
            for di,si in zip(*np.where(bad)):
                cells.append((p['days'][di],p['ids'][si]))
    return pd.DataFrame(sorted(set(cells)),columns=['trade_date','security_id'])

def apply_shock(prices,cells,shock):
    """Each missing run jumps once from its last *observed* close, then stays flat.

    Observed prices are immutable. These are hypothetical marks/settlements;
    execution at an invented entry price is prohibited by the caller.
    """
    if shock<=-1:raise ValueError('shock must exceed -100%')
    observed=prices[['trade_date','security_id','adjusted_close']].copy()
    if observed.duplicated(['trade_date','security_id']).any():raise ValueError('duplicate price key')
    records=[]
    for sid,missing in cells.groupby('security_id',sort=True):
        source=observed[observed.security_id.eq(sid)&np.isfinite(observed.adjusted_close)].sort_values('trade_date')
        for day in sorted(missing.trade_date.unique()):
            prior=source[source.trade_date.lt(day)]
            if prior.empty or prior.adjusted_close.iloc[-1]<=0:raise ValueError(f'no valid observed anchor: {sid}/{day}')
            anchor=prior.iloc[-1]
            existing=observed[observed.security_id.eq(sid)&observed.trade_date.eq(day)]
            if not existing.empty and existing.adjusted_close.notna().any():raise ValueError('refusing to replace observed value')
            records.append(dict(trade_date=day,security_id=sid,adjusted_close=float(anchor.adjusted_close*(1+shock)),anchor_date=anchor.trade_date,anchor_price=float(anchor.adjusted_close),shock=shock))
    patches=pd.DataFrame(records)
    base=observed[observed.adjusted_close.notna()]
    result=pd.concat([base,patches[['trade_date','security_id','adjusted_close']]],ignore_index=True)
    return result,patches

def check_entries(prepared):
    for p in prepared:
        w=select_weights(p['score'],p['caps'],p['ids'],p['crash'])
        if not np.isfinite(p['prices'][0,w!=0]).all():
            raise InvalidPortfolio('selected entry close is missing; synthetic entry execution prohibited')

def run(results,prices_path,out):
    _limit_cores(4);root=Path(results);price_path=Path(prices_path);out=Path(out)
    if out.exists():raise FileExistsError('use a fresh output directory')
    expected=json.loads((root/'final_audit/artifact_hashes.json').read_text())['handoff/stock_prices.parquet']
    price_hash=hashlib.sha256(price_path.read_bytes()).hexdigest()
    if price_hash!=expected:raise ValueError('price file does not match frozen original-run hash')
    pred_path=root/'weekly/predictions.parquet';pred_hash=hashlib.sha256(pred_path.read_bytes()).hexdigest()
    predictions=pd.read_parquet(pred_path);prices=pd.read_parquet(price_path)
    cells=missing_cells(predictions,prices)
    if cells.empty:raise ValueError('no missing candidate prices to stress')
    out.mkdir(parents=True)
    cells.to_csv(out/'missing_price_cells.csv',index=False)
    original={v:prepare(f,prices) for v,f in predictions.groupby('variant',sort=True)}
    exposures=[]
    for variant,ps in original.items():
        for p in ps:
            w=select_weights(p['score'],p['caps'],p['ids'],p['crash'])
            for j in np.flatnonzero((w!=0)&(~np.isfinite(p['prices'])).any(axis=0)):
                exposures.append(dict(variant=variant,period=p['period'],security_id=p['ids'][j],weight=float(w[j]),side='long' if w[j]>0 else 'short',entry=p['days'][0],exit=p['days'][-1]))
    pd.DataFrame(exposures).to_csv(out/'affected_positions.csv',index=False)
    rows=[];equities=[];verifications={};original_metrics=json.loads((root/'weekly/metrics.json').read_text())
    for shock in SHOCKS:
        altered,patches=apply_shock(prices,cells,shock)
        patches.to_parquet(out/f'assumed_prices_{shock:+.2f}.parquet',index=False)
        for variant,f in predictions.groupby('variant',sort=True):
            label=f'{variant}/{shock:+.2f}'
            try:
                check_entries(original[variant])
                prepared=prepare(f,altered);net=ledger(prepared);gross=ledger(prepared,bps=0,borrow=0)
                measures=metrics(net,52);measures['gross_cumulative_return']=gross['final_equity']-1
                daily=net['daily'];assert np.allclose(daily.equity,daily.cash+daily.position_value,atol=1e-11)
                assert np.isclose(daily.transaction_cost.sum(),net['total_transaction_cost'])
                assert np.isclose(daily.borrow_cost.sum(),net['total_borrow'])
                if 'primary'in original_metrics[variant]:
                    for field in ('cumulative_return','sharpe','max_drawdown'):
                        assert np.isclose(measures[field],original_metrics[variant]['primary'][field],atol=1e-11)
                for name,table in [('daily',daily),('periodic',net['periodic']),('holdings',net['holdings'])]:
                    table.to_parquet(out/f"{variant.replace('/','_')}_{shock:+.2f}_{name}.parquet",index=False)
                rows.append(dict(variant=variant,shock=shock,status='assumed_outcome_only',**measures))
                e=daily[['date','equity']].copy();e['variant']=variant;e['shock']=shock;equities.append(e)
                verifications[label]=dict(cash_reconciles=True,fees_reconcile=True,original_complete_strategy_unchanged='primary'in original_metrics[variant])
            except InvalidPortfolio as exc:rows.append(dict(variant=variant,shock=shock,status='invalid',reason=str(exc)))
    table=pd.DataFrame(rows);flat=table[table.shock.eq(0)].set_index('variant').cumulative_return
    table['change_vs_flat_percentage_points']=[100*(r.cumulative_return-flat[r.variant]) if hasattr(r,'cumulative_return') and pd.notna(r.cumulative_return) else np.nan for r in table.itertuples()]
    table.to_csv(out/'scenario_metrics.csv',index=False)
    assert hashlib.sha256(price_path.read_bytes()).hexdigest()==price_hash
    assert hashlib.sha256(pred_path.read_bytes()).hexdigest()==pred_hash
    dump(out/'verification.json',verifications)
    dump(out/'manifest.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_prices_sha256=price_hash,predictions_sha256=pred_hash,shocks=list(SHOCKS),max_cores=4,source_results=str(root),path_assumption='jump on first missing session from last observed price; flat until observed price resumes or assumed exit',holdings='saved scores and weights unchanged; no retraining',execution_assumption='hypothetical exits allowed, missing selected entry execution prohibited',status='conditional stress scenarios, not recovered prices; no p-values or alpha inferred'))
    primary=table[table.variant.eq('combined/ridge')].sort_values('shock')
    lines=['# Missing-price stress test','', '**These are assumed outcomes, not repaired historical prices.** The original primary backtest remains invalid.', '',
           'Saved predictions and portfolio selection were held fixed. Each missing price run jumps once from its last observed adjusted close by the scenario amount and remains flat through the gap. Actual recorded prices are untouched. Any exit on an assumed price is a hypothetical settlement; a selected missing entry price remains invalid. Costs are 30 bp per traded notional plus 3% annual borrow.', '',
           '## Primary strategy: combined Ridge','', '| Assumed stock-price change | Net six-month return | Gross return | Sharpe | Daily worst drawdown | Change vs flat |', '|---:|---:|---:|---:|---:|---:|']
    for row in primary.itertuples():
        if row.status=='invalid':lines.append(f'| {row.shock:+.0%} | Invalid | — | — | — | — |');continue
        lines.append(f'| {row.shock:+.0%} | {row.cumulative_return:.2%} | {row.gross_cumulative_return:.2%} | {row.sharpe:.2f} | {row.max_drawdown:.2%} | {row.change_vs_flat_percentage_points:+.2f} pp |')
    good=primary[primary.status.ne('invalid')]
    lines.extend(['',('All five assumed primary outcomes lose money after costs; the tested missing-price range does not rescue the strategy.' if len(good)==5 and (good.cumulative_return<0).all() else 'The primary conclusion depends on the stated assumptions; inspect all scenarios.'),'',
                  '## Affected holdings','',pd.DataFrame(exposures).to_string(index=False),'',
                  '## Interpretation and limits','',
                  'Shocks are changes in the missing stock price, not percentage changes in the portfolio. A price rise benefits a long and hurts a short. Fills for unselected stocks do not affect the actual portfolio. All five previously complete strategies were verified unchanged across scenarios.', '',
                  'The grid is a sensitivity range, not a probability distribution, confidence interval, upper/lower bound on true outcomes, or evidence of alpha. Corporate-action consideration, delisting returns and actual execution remain unknown. Sharpe and daily drawdown depend on the assumed price path. No models were tuned, no stocks were removed, and no Monte Carlo significance was inferred from the invented marks. Original report and outputs remain intact.', '',
                  'scenario_metrics.csv contains all seven strategies. Per-scenario holdings, daily equity, periodic returns, assumed cells and verification records are retained privately.'])
    (out/'STRESS_REPORT.md').write_text('\n'.join(lines)+'\n')
    if equities:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5))
        all_e=pd.concat(equities)
        for shock,g in all_e[all_e.variant.eq('combined/ridge')].groupby('shock'):
            ax.plot(g.date,g.equity,label=f'{shock:+.0%} missing stock')
        ax.set_ylabel('Assumed net equity');ax.legend();fig.tight_layout();fig.savefig(out/'primary_stress_equity.png');plt.close(fig)
    print(primary[['shock','status','cumulative_return','sharpe','max_drawdown']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--prices',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.results,a.prices,a.output_dir)
