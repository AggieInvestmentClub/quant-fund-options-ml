"""Reconcile completed experiment artifacts and document unresolvable price gaps.

This does not change portfolios, fit models, impute outcomes, or rerun random draws.
"""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .daily_ledger import prepare,ledger,metrics,InvalidPortfolio
from .strategy_experiments import dump

def finalize(root,prices_path):
    root=Path(root);audit=root/'final_audit'
    if audit.exists():raise FileExistsError('final_audit already exists')
    audit.mkdir();prices=pd.read_parquet(prices_path);gaps=[];checks={};summary=[]
    manifest=json.loads((root/'manifest.json').read_text())
    summary.extend(['# Final options strategy report','',
      '**Decision: no supported weekly trading strategy has been established.** The primary combined Ridge backtest is invalid because a selected stock has missing valuation prices. Five other weekly strategies completed and lost money after costs. The monthly paper-method diagnostic completed, but its four adjusted randomization p-values exceed 0.05.','',
      '## Data and windows','',
      f"Frozen source: {manifest['source_dates'][0]} through {manifest['source_dates'][1]}; {manifest['options_rows']:,} daily options rows; {manifest['tickers']} ticker proxies. Input file hashes are in manifest.json. No additional market data was obtained.",
      '',f"Model run commit: `{manifest['commit']}`. Final audit commit: `{subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}`.",
      '', 'The primary weekly fit began with 108 matured training weeks. Test signal weeks were May 8–October 30, 2026, with holdings from May 11–November 9. The monthly diagnostic uses May–October 2025 signals, 12-month ±0.70 log-return labels, and approximately two initial matured training months. Exact per-fit dates and class counts are saved in each training_windows.json.',
      '', '## Results after 30 bp trading costs and 3% annual borrow','',
      '| Track / variant | Net cumulative return | Sharpe | Daily max drawdown | Adjusted randomization p | Status |',
      '|---|---:|---:|---:|---:|---|'])
    for track,freq in [('paper_diagnostic',12),('weekly',52)]:
        dest=root/track;pred=pd.read_parquet(dest/'predictions.parquet');original=json.loads((dest/'metrics.json').read_text());sims=json.loads((dest/'monte_carlo.json').read_text())
        corrected=[];inference={}
        for variant,f in pred.groupby('variant'):
            ps=prepare(f,prices)
            for p in ps:
                bad=~np.isfinite(p['prices'])|(p['prices']<0)
                for di,si in zip(*np.where(bad)):
                    gaps.append(dict(track=track,period=p['period'],security_id=p['ids'][si],date=p['days'][di],entry_date=p['days'][0],exit_date=p['days'][-1]))
            key=f'{track}/{variant}'
            try:
                run=ledger(ps);m=metrics(run,freq);old=original[variant]['primary']
                for field in ('cumulative_return','sharpe','max_drawdown','mean_periodic_return'):
                    if not np.isclose(m[field],old[field],rtol=1e-10,atol=1e-12):raise ValueError(f'metric changed: {key}/{field}')
                d=run['daily'];cash_error=float(np.max(np.abs(d.equity-d.cash-d.position_value)))
                assert cash_error<1e-10
                assert np.isclose(d.transaction_cost.sum(),run['total_transaction_cost'])
                assert np.isclose(d.borrow_cost.sum(),run['total_borrow'])
                checks[key]=dict(cash_ledger_max_error=cash_error,transaction_cost_reconciles=True,borrow_reconciles=True,metrics_unchanged=True)
                d['variant']=variant;corrected.append(d)
                sim=sims[variant];null=np.asarray(sim['null_mean_returns']);invalid=sim['invalid_null_draws'];n=sim['iterations'];hits=int((null>=m['mean_periodic_return']).sum());family=7 if freq==52 else 4
                bounds=[(1+hits)/(n+1),(1+hits+invalid)/(n+1)]
                inference[variant]=dict(attempted_draws=n,valid_draws=len(null),invalid_draws=invalid,p_bounds=bounds,adjusted_p_bounds=[min(1,x*family) for x in bounds],p_adjusted=sim.get('p_adjusted'),interpretation='incomplete randomization; interval covers all possible outcomes of invalid draws' if invalid else 'complete randomization')
                pvalue=sim.get('p_adjusted');text=f'{pvalue:.4f}' if pvalue is not None else 'Unavailable'
                summary.append(f"| {track}: {variant} | {m['cumulative_return']:.2%} | {m['sharpe']:.2f} | {m['max_drawdown']:.2%} | {text} | {'MC incomplete' if invalid else 'Complete'} |")
            except InvalidPortfolio as exc:
                checks[key]=dict(status='invalid',reason=str(exc));summary.append(f'| {track}: {variant} | — | — | — | — | Invalid: selected price missing |')
        if corrected:pd.concat(corrected).to_parquet(audit/f'{track}_daily_equity.parquet',index=False)
        dump(audit/f'{track}_inference_bounds.json',inference)
    gap=pd.DataFrame(gaps).drop_duplicates();gap.to_csv(audit/'price_gaps.csv',index=False)
    artifact=joblib.load(root/'weekly/latest_model.joblib');panel=pd.read_parquet(root/'weekly/weekly_panel.parquet');latest=panel[panel.decision_at.eq(panel.decision_at.max())]
    scores=artifact['model'].predict(latest[artifact['features']]);saved=pd.read_parquet(root/'weekly/latest_scores.parquet')
    roundtrip=latest[['security_id']].assign(score=scores).merge(saved[['security_id','score']],on='security_id',suffixes=('_model','_saved'),validate='one_to_one')
    assert len(roundtrip)==len(saved)==len(latest) and np.allclose(roundtrip.score_model,roundtrip.score_saved,rtol=0,atol=1e-12)
    checks['latest_model_roundtrip']=dict(rows=len(latest),matches_saved_predictions=True)
    dump(audit/'verification.json',checks)
    summary.extend(['','## Missing-price conclusion','',
      'The frozen CSVs and wide file were inspected. EQR lacks August 24, 2026 valuation; WBD has no prices after October 6 for the affected holding interval; KVUE has no November 2–9 prices. The wide file has no replacement values for these gaps. The reason for the terminations is not established from the supplied files. No corporate-action settlement or delisting outcome was inferred.',
      '', 'WBD was held by combined Ridge and the IV-gap baseline. Their 26-week returns, Sharpe, drawdown, and Monte Carlo p-values are unavailable. Dropping those stocks after observing the gaps or filling terminal values would bias the experiment. The remaining five weekly randomization exercises each attempted 2,000 draws, but 560–858 draws encountered missing valuation data; valid-draw-only p-values would be biased. Their uncertainty bounds are retained in final_audit/weekly_inference_bounds.json.',
      '', '## Success criteria','',
      'The primary strategy cannot pass the predeclared gates because its complete return series is unavailable. No positive-return, Sharpe >=1, drawdown >=-20%, stock-only improvement, adjusted p<0.05, or positive bootstrap-lower-bound claim is made for it. Completed challengers all failed the positive net-return gate. No model was substituted for the primary after seeing test performance.',
      '', '## Paper replication limits','',
      'Track A is a limited method diagnostic: monthly data, 12-month class labels, balanced logistic/LightGBM and top-5% monthly long/short portfolios are implemented. The original 1996–2022 sample, ten-year initial training, nine company controls, complete common-stock universe and quote-mid contract filters are unavailable. The 19 controlled Table 3 specifications are not claimed reproduced. The positive diagnostic logistic returns have adjusted p-values above 0.05, and six monthly observations provide very limited evidence.',
      '', '## Reproducibility and artifacts','',
      'The raw model-run artifacts remain intact. final_audit contains corrected daily fee-event aggregation, ledger reconciliation checks, exact missing-price cells, and conservative bounds for incomplete randomizations. The correction preserves all portfolio equity and performance values while retaining borrow fees paid on rebalance dates in the daily event totals.',
      '', 'Each track includes dated predictions, per-fit models and training windows, holdings, daily equity, periodic returns, cost sensitivity, metrics, bootstrap draws, random-signal distributions and charts. The primary latest-model artifact was reloaded and verified against its saved scores. It is a research artifact without a validated backtest. Alpha against the internal benchmark is unavailable where benchmark prices are missing.',
      '', 'The server retains the frozen inputs in private home storage. Code and instructions are on research/weekly-options-strategy; market data and fitted artifacts are excluded from the public repository.'])
    (root/'FINAL_REPORT.md').write_text('\n'.join(summary)+'\n')
    hashes={}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='artifact_hashes.json':hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(audit/'artifact_hashes.json',hashes)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--prices',required=True);a=p.parse_args();finalize(a.results,a.prices)
