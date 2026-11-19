"""Training-only pruning and chronological model diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from .reevaluate import dump
from .feature_experiments import ranked_correlation


def prune(frame, columns, missing_threshold=.5, correlation_threshold=.90, min_pairs=200):
    candidates = frame[columns]; missing = candidates.isna().mean()
    corr = candidates.corr(min_periods=min_pairs)
    retained, removed = [], {}
    for c in sorted(columns, key=lambda c: (missing[c], c)):
        if missing[c] > missing_threshold:
            removed[c] = 'more than 50% missing in training'; continue
        if candidates[c].nunique(dropna=True) <= 1:
            removed[c] = 'constant or all missing in training'; continue
        match = next((k for k in retained if pd.notna(corr.loc[c,k]) and abs(corr.loc[c,k]) >= correlation_threshold), None)
        if match is None: retained.append(c)
        else: removed[c] = f'absolute Pearson >= {correlation_threshold:.2f} with '+match
    if not retained: raise ValueError('no usable features after training-only pruning')
    return retained, removed


def heatmap(matrix, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 10))
    m = ax.imshow(matrix.to_numpy(float), vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=6)
    ax.set_title(title, fontsize=11); fig.colorbar(m, ax=ax, shrink=.7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def correlations(train, scoring, columns, out, window):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    x = train[columns]; finite = x.notna().astype(np.int64)
    dump(out/'coverage.json', dict(window=window, feature_order=columns,
         training_missing=x.isna().mean().to_dict(), scoring_missing=scoring[columns].isna().mean().to_dict(),
         pair_counts=(finite.T@finite).to_dict()))
    for method in ('pearson', 'spearman'):
        c = x.corr(method=method, min_periods=200)
        c.to_csv(out/(method+'.csv'))
        heatmap(c, out/(method+'.png'), method.title()+' correlation on matured training observations')


def weekly_ic(model, f, columns, min_known=200):
    if f.empty: return np.nan
    score = model.predict(f[columns]); vals = []
    for ix in f.groupby('period', sort=True).indices.values():
        vals.append(ranked_correlation(score[ix,None], f.target.to_numpy()[ix], min_known)[0])
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def correlation_groups(train, columns):
    c = train[columns].corr(min_periods=200).abs(); remaining = set(columns); groups=[]
    while remaining:
        seed=min(remaining); component={seed}; queue=[seed]; remaining.remove(seed)
        while queue:
            a=queue.pop()
            near={b for b in remaining if pd.notna(c.loc[a,b]) and c.loc[a,b]>=.9}
            remaining-=near; component|=near; queue.extend(sorted(near))
        if len(component)>1: groups.append(sorted(component))
    return groups


def permutation_importance(model, train, validation, columns, repetitions=20, seed=17):
    baseline=weekly_ic(model, validation, columns); rng=np.random.default_rng(seed); rows=[]
    groups=[([c], 'individual') for c in columns]+[(g,'correlated_group') for g in correlation_groups(train,columns)]
    for group,kind in groups:
        values=[]
        for _ in range(repetitions):
            changed=validation.copy()
            for ix in validation.groupby('period',sort=True).indices.values():
                donor=rng.permutation(ix)
                changed.iloc[ix,changed.columns.get_indexer(group)]=validation.iloc[donor][group].to_numpy()
            values.append(baseline-weekly_ic(model,changed,columns))
        rows.append(dict(features='|'.join(group),kind=kind,baseline_ic=baseline,
                         mean_ic_decrease=float(np.mean(values)),std_ic_decrease=float(np.std(values)),
                         repetitions=repetitions))
    return pd.DataFrame(rows)


def accumulated_effects(model, training, columns, bins=10, minimum=20, max_rows=3000):
    """Local effects use observed conditional bins; missing rows do not define support."""
    rows=[]
    for c in columns:
        values=training[c].dropna(); edges=np.unique(np.quantile(values,np.linspace(0,1,bins+1))) if len(values) else []
        pieces=[]
        for j in range(len(edges)-1):
            mask=training[c].ge(edges[j]) & (training[c].le(edges[j+1]) if j==len(edges)-2 else training[c].lt(edges[j+1]))
            f=training.loc[mask,columns]; count=len(f)
            if count>=minimum:
                f=f.sample(min(count,max_rows),random_state=17); lo=f.copy();hi=f.copy()
                lo[c]=edges[j];hi[c]=edges[j+1]
                effect=float(np.mean(model.predict(hi)-model.predict(lo)))
            else: effect=np.nan
            pieces.append(dict(feature=c,bin=j,low=float(edges[j]),high=float(edges[j+1]),support=count,
                               local_effect=effect,status='complete' if np.isfinite(effect) else 'sparse'))
        if pieces and all(np.isfinite(r['local_effect']) for r in pieces):
            cumulative=np.cumsum([r['local_effect'] for r in pieces]);weights=np.array([r['support'] for r in pieces])
            centered=cumulative-np.average(cumulative,weights=weights)
            for r,a in zip(pieces,centered):r['centered_effect']=a
        rows.extend(pieces)
    return pd.DataFrame(rows)


def diagnostic_validation(train):
    months=sorted(train.period.unique()); val_periods=months[-13:]
    validation=train[train.period.isin(val_periods)].copy()
    cutoff=validation.decision_at.min()
    fit=train[train.label_available_at.lt(cutoff)].copy()
    scorable=sum(np.isfinite(g.target).sum()>=200 for _,g in validation.groupby('period'))
    return fit, validation, dict(fitting_weeks=fit.period.nunique(),validation_weeks=len(val_periods),
        scorable_validation_weeks=scorable,cutoff=cutoff,eligible=fit.period.nunique()>=39 and scorable>=8)
