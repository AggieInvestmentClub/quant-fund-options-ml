"""Verified, dated inputs for the data and feature sensitivity experiment."""
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc
from .ben_handoff import convert_ben_file
from .strategy_data import weekly_panel
from .feature_experiments import features, GROUPS as LEGACY_GROUPS
from .reevaluate import dump, sha

EXTRA = ['ret_63', 'ret_252', 'rv_60', 'rv_90', 'skew_60', 'volume_ratio_20', 'iv_rv_ratio']
GROUPS = {**LEGACY_GROUPS, 'pruned_full': LEGACY_GROUPS['full'],
          'pruned_extended': list(dict.fromkeys(LEGACY_GROUPS['full'] + LEGACY_GROUPS['compact_plus'] + EXTRA))}
DEFINITIONS = {
    'ret_63': 'adjusted_close[t]/adjusted_close[t-63 exchange sessions]-1',
    'ret_252': 'adjusted_close[t]/adjusted_close[t-252 exchange sessions]-1',
    'rv_60': 'sqrt(252)*sample std of last 60 daily adjusted returns',
    'rv_90': 'sqrt(252)*sample std of last 90 daily adjusted returns',
    'skew_60': 'pandas unbiased sample skew of last 60 daily adjusted returns',
    'volume_ratio_20': 'volume[t]/mean(volume[t-19:t]); 20 complete exchange sessions',
    'iv_rv_ratio': 'mean(avg_call_iv,avg_put_iv)/(sqrt(252)*vol_20); both IVs required',
}


def reject_future_columns(columns):
    forbidden = [c for c in columns if c.startswith(('forward_', 'target_', 'jump_', 'crash_',
                   'outperformance_', 'is_target_')) or c in ('is_actively_trading', 'is_options_investable')]
    if forbidden:
        raise ValueError('future or undocumented selection fields: ' + ', '.join(forbidden))


def strict_stock_features(prices):
    """Reindex to exchange sessions so gaps cannot masquerade as consecutive days."""
    if prices.duplicated(['security_id', 'trade_date']).any():
        raise ValueError('duplicate stock/date')
    cal = xc.get_calendar('XNYS', start=prices.trade_date.min()-pd.Timedelta(days=10),
                          end=prices.trade_date.max()+pd.Timedelta(days=10))
    sessions = cal.sessions.tz_localize(None) if cal.sessions.tz is not None else cal.sessions
    chunks = []
    for sid, f in prices.groupby('security_id', sort=True):
        f = f.set_index('trade_date').sort_index()
        dates = sessions[(sessions >= f.index.min()) & (sessions <= f.index.max())]
        x = f.reindex(dates); p = x.adjusted_close; r = p.pct_change(fill_method=None)
        for n in (63, 252):
            count = p.rolling(n+1, min_periods=n+1).count()
            x['ret_'+str(n)] = (p/p.shift(n)-1).where(count.eq(n+1))
        for n in (60, 90):
            x['rv_'+str(n)] = r.rolling(n, min_periods=n).std(ddof=1)*np.sqrt(252)
        x['skew_60'] = r.rolling(60, min_periods=60).skew()
        mean = x.volume.rolling(20, min_periods=20).mean()
        x['volume_ratio_20'] = x.volume/mean.where(mean.gt(0))
        x['security_id'] = sid; x.index.name = 'trade_date'
        chunks.append(x[['security_id', *EXTRA[:-1]]].reset_index())
    return pd.concat(chunks, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def legacy_slopes(raw):
    """Retain original complete-endpoint definitions; filled exporter slopes are quarantined."""
    raw = raw.copy()
    raw['atm_term_slope'] = raw.atm_iv_90d-raw.atm_iv_30d
    raw['rr_term_slope'] = raw.risk_reversal_25_delta_90d-raw.risk_reversal_25_delta_30d
    return raw


def build_panel(snapshot, out):
    snapshot, out = Path(snapshot), Path(out)
    out.mkdir(parents=True)
    report = convert_ben_file(snapshot/'wide.parquet', out/'handoff', allow_ticker_proxy=True,
                              stocks_dir=snapshot/'stocks')
    tables = {k: pd.read_parquet(out/'handoff'/(k+'.parquet')) for k in ('stock_prices', 'options_features')}
    dump(out/'price_quality.json', price_quality(tables['stock_prices']))
    raw = pd.read_parquet(snapshot/'wide.parquet')
    raw.trade_date = pd.to_datetime(raw.trade_date)
    slopes = legacy_slopes(raw)
    for col in ('atm_term_slope', 'rr_term_slope'):
        tables['options_features'][col] = slopes[col].to_numpy()
    panel, schedule = weekly_panel(tables)
    panel = features(panel)
    extra = strict_stock_features(tables['stock_prices'])
    panel = panel.merge(extra, on=['security_id', 'trade_date'], how='left', validate='one_to_one')
    panel['iv_rv_ratio'] = panel.iv_level/(np.sqrt(252)*panel.vol_20.where(panel.vol_20.gt(0)))
    panel[EXTRA] = panel[EXTRA].replace([np.inf, -np.inf], np.nan)
    reject_future_columns(list(dict.fromkeys(c for cols in GROUPS.values() for c in cols)))
    panel.to_parquet(out/'weekly_panel.parquet', index=False)
    tables['stock_prices'].to_parquet(out/'stock_prices.parquet', index=False)
    provenance = {c: {'definition': DEFINITIONS.get(c, 'existing legacy definition'),
                      'source': 'recomputed from frozen contemporaneous prices/options',
                      'missing_fraction': float(panel[c].isna().mean())} for c in GROUPS['pruned_extended']}
    dump(out/'feature_provenance.json', provenance)
    dump(out/'adapter_report.json', report)
    return panel, tables['stock_prices']


def price_quality(prices):
    """Flag stale marks and adjustment discontinuities without changing the sample."""
    rows=[]
    for sid,f in prices.sort_values(['security_id','trade_date']).groupby('security_id'):
        factor=f.adjusted_close/f.close.where(f.close.gt(0))
        shifts=factor.pct_change(fill_method=None).abs().gt(.2)
        zero=f.volume.eq(0)
        unchanged=f.close.eq(f.close.shift()) & zero
        if shifts.any() or zero.any():
            rows.append(dict(security_id=sid,last_date=f.trade_date.max(),
                zero_volume_rows=int(zero.sum()),unchanged_zero_volume_rows=int(unchanged.sum()),
                large_adjustment_factor_changes=f.loc[shifts,['trade_date','close','adjusted_close']].to_dict('records')))
    return dict(rows=len(prices),tickers=prices.security_id.nunique(),
        last_date_by_security=prices.groupby('security_id').trade_date.max().to_dict(),flags=rows,
        interpretation='flags require source/corporate-action reconciliation; no synthetic repairs or retrospective exclusions')


def data_changes(old_snapshot, new_snapshot, out):
    a = pd.read_parquet(Path(old_snapshot)/'wide.parquet')
    b = pd.read_parquet(Path(new_snapshot)/'wide.parquet')
    keys = ['trade_date', 'underlying']
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError('duplicate source key')
    x, y = a.set_index(keys), b.set_index(keys); common = x.index.intersection(y.index)
    changed = {}
    for c in a.columns.intersection(b.columns):
        if c in keys: continue
        v, w = x.loc[common, c], y.loc[common, c]
        equal = v.eq(w) | (v.isna() & w.isna())
        if not equal.all():
            changed[c] = dict(changed=int((~equal).sum()), filled=int((v.isna() & w.notna()).sum()),
                              lost=int((v.notna() & w.isna()).sum()))
    report = dict(old_rows=len(a), new_rows=len(b), old_columns=len(a.columns), new_columns=len(b.columns),
                  first_date=b.trade_date.min(), last_date=b.trade_date.max(), stocks=b.underlying.nunique(),
                  added_columns=sorted(set(b)-set(a)), added_keys=len(y.index.difference(x.index)),
                  removed_keys=len(x.index.difference(y.index)), historical_changes=changed,
                  coverage={c:float(b[c].isna().mean()) for c in b},
                  old_sha256=sha(Path(old_snapshot)/'wide.parquet'), new_sha256=sha(Path(new_snapshot)/'wide.parquet'))
    dump(Path(out)/'data_changes.json', report)
    return report
