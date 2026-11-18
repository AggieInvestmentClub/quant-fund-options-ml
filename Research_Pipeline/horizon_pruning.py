"""Matched full/pruned four-week relative Ridge holding-buffer experiment."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from .horizon_data import FEATURES, WINDOWS, horizon_panel, window_schedule, matured_training, prediction_columns
from .horizon_strategy import (fit_configuration, portfolio_case, weekly_boundaries, SEED, NET, POLICIES)
from .horizon_statistics import (simulate_case, shuffled_periods, bootstrap_indices, ic_series, bounded_p)
from .feature_diagnostics import prune, correlations, diagnostic_validation, permutation_importance, accumulated_effects
from .feature_experiments import prediction_metrics
from .strategy_experiments import pipeline
from .portfolio_ledger import prepare_replay
from .reevaluate import dump, sha
from .run import _limit_cores

VARIANTS = {'full': 'h4/relative/full/ridge', 'pruned': 'h4/relative/pruned_full/ridge'}


def fit_pruned(panel, schedule, out, hashes, min_weeks=52, diagnostics=True):
    """Choose columns anew from each strictly matured training window."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frames, windows = [], []
    for row in schedule.itertuples(index=False):
        tr = matured_training(panel, row.decision_at, 'relative', min_weeks)
        sc = panel[panel.period.eq(row.period)].copy()
        if sc.empty:
            raise ValueError('empty scheduled signal universe')
        columns, removed = prune(tr, FEATURES['full'])
        window = dict(period=row.period, decision_at=str(row.decision_at),
                      training_rows=len(tr), training_weeks=tr.period.nunique(),
                      latest_outcome=str(tr.label_available_at.max()),
                      first_signal=str(tr.trade_date.min()), last_signal=str(tr.trade_date.max()),
                      columns=columns, removed=removed)
        metadata = dict(variant=VARIANTS['pruned'], sources=hashes, window=window,
                        parameters=pipeline('ridge').get_params(),
                        pruning=dict(missing_threshold=.5, correlation_threshold=.90, min_pairs=200))
        signature = json.dumps(metadata, sort_keys=True, default=str)
        path = out/(row.period.replace('/', '_')+'.joblib')
        if path.exists():
            saved = joblib.load(path)
            if saved['signature'] != signature:
                raise ValueError('pruned fit cache mismatch')
            model = saved['model']
        else:
            model = pipeline('ridge').fit(tr[columns], tr.target)
            joblib.dump(dict(model=model, metadata=metadata, signature=signature), path)
        sc['score'] = model.predict(sc[columns]); sc['variant'] = VARIANTS['pruned']
        sc['target'] = sc.target_relative
        if not np.isfinite(sc.score).all():
            raise ValueError('nonfinite pruned predictions')
        frames.append(prediction_columns(sc)); windows.append(window)
        if diagnostics:
            d = out/'diagnostics'/row.period.replace('/', '_')
            correlations(tr, sc, FEATURES['full'], d/'before', window)
            correlations(tr, sc, columns, d/'after', window)
        print('fit pruned', row.period, len(columns), 'features', flush=True)
    pred = pd.concat(frames, ignore_index=True)
    pred.to_parquet(out/'predictions.parquet', index=False)
    dump(out/'training_windows.json', windows)
    return pred


def verify_control(actual, saved, tolerance=1e-8):
    keys = ['period', 'security_id']
    z = actual.merge(saved[keys+['score', 'target']], on=keys, how='outer',
                     validate='one_to_one', suffixes=('_new', '_old'), indicator=True)
    if not z._merge.eq('both').all():
        raise AssertionError('control prediction keys differ')
    if not np.allclose(z.target_new, z.target_old, atol=0, rtol=0, equal_nan=True):
        raise AssertionError('control targets differ')
    error = float((z.score_new-z.score_old).abs().max())
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError('control scores differ: '+str(error))
    return dict(rows=len(z), max_prediction_error=error, tolerance=tolerance)


def require_same_candidates(full, pruned):
    cols = ['period', 'security_id', 'entry_date', 'exit_date', 'target', 'market_cap']
    a = full[cols].sort_values(cols[:2]).reset_index(drop=True)
    b = pruned[cols].sort_values(cols[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def diagnostic_effects(panel, decision, group, out):
    tr = matured_training(panel, decision, 'relative')
    fitting, validation, status = diagnostic_validation(tr)
    out.mkdir(parents=True, exist_ok=True); dump(out/'validation.json', status)
    if not status['eligible']:
        return
    cols = prune(fitting, FEATURES['full'])[0] if group == 'pruned' else FEATURES['full']
    model = pipeline('ridge').fit(fitting[cols], fitting.target)
    permutation_importance(model, fitting, validation, cols).to_csv(out/'importance.csv', index=False)
    accumulated_effects(model, fitting, cols).to_csv(out/'local_effects.csv', index=False)
    dump(out/'schema.json', dict(columns=cols, fit_cutoff=str(validation.decision_at.min())))


def run(panel_path, prices_path, reference, out, cores=2, iterations=2000):
    _limit_cores(cores)
    out = Path(out).resolve(); reference = Path(reference).resolve()
    paths = dict(panel=Path(panel_path).resolve(), prices=Path(prices_path).resolve(),
                 reference_predictions=reference/'historical/fits/h4_relative_full_ridge/predictions.parquet',
                 reference_metrics=reference/'historical/rules/metrics.json',
                 reference_manifest=reference/'manifest.json')
    if out == reference or reference in out.parents or any(out == p or out in p.parents for p in paths.values()):
        raise ValueError('outputs must not overwrite reference inputs')
    hashes = {k:sha(p) for k,p in paths.items()}
    original = json.loads(paths['reference_manifest'].read_text())
    for k in ('panel', 'prices'):
        if hashes[k] != original['sources'][k]:
            raise ValueError('inputs do not match the saved horizon run')
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(schema=1, sources=hashes, paths={k:str(v) for k,v in paths.items()},
                variants=VARIANTS, windows=WINDOWS, seed=SEED, iterations=iterations, cores=cores,
                policy=asdict(POLICIES['buffer']), costs=asdict(NET), mode='long_only',
                initial_capital=100000, fit_target='relative four-week adjusted return',
                portfolio_family=52, ranking_family=26,
                family_note='conservative counts include 50 prior portfolio cases and 24 prior ranking variants plus these two cases; not all earlier research trials',
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                implementation_sha256=sha(Path(__file__)),
                software={k:version(k) for k in ('numpy','pandas','scikit-learn','lightgbm','exchange-calendars')},
                interpretation='exploratory previously inspected history; fixed matched contrast, no model selection')
    spec = json.loads(json.dumps(spec))
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != spec:
        raise ValueError('incompatible resume; use a fresh directory')
    dump(out/'manifest.json', spec)
    panel, prices = pd.read_parquet(paths['panel']), pd.read_parquet(paths['prices'])
    labeled, master = horizon_panel(panel, prices, 4)
    allpred, reports, ranking = {}, {}, {}
    for window, dates in WINDOWS.items():
        schedule = window_schedule(master, 4, *dates)
        folder = out/window; folder.mkdir(exist_ok=True)
        full = fit_configuration(labeled, schedule, VARIANTS['full'], folder/'fits/full', hashes, diagnostics=False)
        if window == 'historical':
            dump(out/'control_verification.json', verify_control(full, pd.read_parquet(paths['reference_predictions'])))
        pruned = fit_pruned(labeled, schedule, folder/'fits/pruned', hashes)
        require_same_candidates(full, pruned)
        reports[window], ranking[window], allpred[window] = {}, {}, dict(full=full, pruned=pruned)
        for group, pred in allpred[window].items():
            m, _, _ = portfolio_case(pred, prices, folder/group, 'buffer', 'long_only', weekly_boundaries(*dates))
            reports[window][group] = m
            rm = prediction_metrics(pred); rm.to_parquet(folder/(group+'_ranking.parquet'), index=False)
            ranking[window][group] = rm
            diagnostic_effects(labeled, schedule.decision_at.iloc[-1], group, folder/'effects'/group)
        dump(folder/'metrics.json', reports[window])
    control = json.loads(paths['reference_metrics'].read_text())['ml_buffer/long_only']
    for label in ('net','gross'):
        for key in ('cumulative_return','sharpe','max_drawdown','transaction_cost','borrow_cost'):
            if not np.isclose(reports['historical']['full'][label][key], control[label][key], atol=1e-8, rtol=0):
                raise AssertionError('saved portfolio control differs: '+label+'/'+key)
    print('Saved full-feature scores and portfolio control reproduced', flush=True)
    bounds = weekly_boundaries(*WINDOWS['historical'])
    indices = bootstrap_indices(len(bounds)-1, iterations)
    np.save(out/'paired_block_indices.npy', indices)
    jobs = [(g, dict(mode='long_only',policy='buffer'),p,prices,str(out/'inference'/g),
             iterations,52,bounds,indices) for g,p in allpred['historical'].items()]
    if cores == 1:
        list(map(simulate_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(cores,2)) as pool:
            list(pool.map(simulate_case, jobs))
    inference = {g:json.loads((out/'inference'/g/'statistics.json').read_text()) for g in VARIANTS}
    draws = {}; observed_ic = {}; random_ic = {}
    for group,pred in allpred['historical'].items():
        prepared = prepare_replay(pred, prices)
        observed_ic[group] = ic_series(pred, prepared)
        random_ic[group] = np.array([np.mean(ic_series(pred, shuffled_periods(prepared,i))) for i in range(iterations)])
        inference[group]['ranking'] = bounded_p(random_ic[group], float(observed_ic[group].mean()), iterations, 26)
        draws[group] = pd.read_parquet(out/'inference'/group/'random_draws.parquet')
    if not np.array_equal(np.isfinite(observed_ic['full']), np.isfinite(observed_ic['pruned'])):
        raise AssertionError('ranking outcome masks differ')
    # Signal IC is observed only at four-week decisions; use the same calendar blocks.
    positions = bounds[:-1].get_indexer(pd.DatetimeIndex(allpred['historical']['full'].groupby('period').entry_date.first()))
    differences = np.full(len(bounds)-1, np.nan)
    differences[positions] = observed_ic['pruned']-observed_ic['full']
    counts = np.isfinite(differences[indices]).sum(axis=1)
    paired_ic = np.divide(np.nansum(differences[indices],axis=1),counts,
                          out=np.full(iterations,np.nan),where=counts>0)
    contrasts = dict(mean_ic_difference=float(np.mean(observed_ic['pruned']-observed_ic['full'])),
                     ic_difference_95pct=np.nanquantile(paired_ic,[.025,.975]).tolist())
    for label in ('net','gross'):
        col=label+'_mean_weekly_return'
        bcol=label+'_mean_weekly_return'
        x=pd.read_parquet(out/'inference/pruned/bootstrap.parquet')
        y=pd.read_parquet(out/'inference/full/bootstrap.parquet')
        if bcol in x and bcol in y:
            boot=x[bcol]-y[bcol]
            contrasts[label+'_mean_return_difference_95pct']=np.quantile(boot,[.025,.975]).tolist()
            delta=reports['historical']['pruned'][label]['mean_weekly_return']-reports['historical']['full'][label]['mean_weekly_return']
            contrasts[label+'_mean_return_difference']=delta
            contrasts[label+'_paired_randomization']=bounded_p(draws['pruned'][col]-draws['full'][col],delta,iterations,52)
        else:
            contrasts[label+'_comparison_status']='unavailable: incomplete portfolio'
    dump(out/'inference/statistics.json', inference); dump(out/'paired_contrasts.json', contrasts)
    rows=[]
    for window in WINDOWS:
        for group in VARIANTS:
            net=reports[window][group]['net']; gross=reports[window][group]['gross']
            rows.append(dict(window=window,features=group,mean_rank_ic=ranking[window][group].rank_ic.mean(),
                             net_return=net.get('cumulative_return'),gross_return=gross.get('cumulative_return'),
                             sharpe=net.get('sharpe'),drawdown=net.get('max_drawdown'),status=net['status'],reason=net.get('reason')))
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    verified={k:sha(p)==hashes[k] for k,p in paths.items()}
    dump(out/'source_verification.json',verified)
    if not all(verified.values()):raise ValueError('source changed during run')
    dump(out/'complete.json',dict(status='complete',negative_and_invalid_results_retained=True))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','prices','reference-results','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-cores',type=int,default=2);p.add_argument('--iterations',type=int,default=2000)
    args=p.parse_args()
    if args.iterations < 2:raise ValueError('need at least two simulations')
    run(args.panel,args.prices,args.reference_results,args.output_dir,args.max_cores,args.iterations)


if __name__ == '__main__':main()
