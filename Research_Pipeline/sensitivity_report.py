"""Private charts and plain-language report for the sensitivity experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load(path):return json.loads(Path(path).read_text())

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')

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
    legs=[];incremental=[]
    for v in labels:
        folder=root/'historical'/v.replace('/','_');tp=folder/'net_trades.parquet'
        if not tp.exists():continue
        t=pd.read_parquet(tp);n=h[v]['portfolio']['net'];c=n['costs_dollars']
        row=dict(variant=v,long_pnl=n['long_pnl_dollars'],short_pnl=n['short_pnl_dollars'],borrow=c['borrow'])
        row['long_cost']=float(t.long_transaction_cost.sum());row['short_cost']=float(t.short_transaction_cost.sum())
        row['long_after_allocated_costs']=row['long_pnl']-row['long_cost']
        row['short_after_allocated_costs']=row['short_pnl']-row['short_cost']-row['borrow'];legs.append(row)
        if not np.isclose(row['long_after_allocated_costs']+row['short_after_allocated_costs'],n['cumulative_return']*100000,atol=1e-5):
            raise AssertionError('leg attribution does not reconcile')
        baseline=root/'historical'/('stock_only_'+v.split('/')[1])/'net_periodic.parquet'
        if baseline.exists() and not v.startswith('stock_only/'):
            a=pd.read_parquet(folder/'net_periodic.parquet');b=pd.read_parquet(baseline)
            joined=a[['period','net_return']].merge(b[['period','net_return']],on='period',validate='one_to_one',suffixes=('_model','_stock'))
            diff=joined.net_return_model-joined.net_return_stock
            incremental.append(dict(variant=v,mean_paired_net_return_difference=float(diff.mean()),periods=len(diff)))
    pd.DataFrame(legs).to_csv(root/'historical/leg_attribution.csv',index=False)
    pd.DataFrame(incremental).to_csv(root/'historical/incremental_options_value.csv',index=False)
    if legs:
        x=pd.DataFrame(legs).set_index('variant')
        fig,ax=plt.subplots(figsize=(10,6));x[['long_after_allocated_costs','short_after_allocated_costs']].plot.barh(ax=ax)
        ax.set_xlabel('Dollar P and L contribution after allocated charges');ax.set_title('Long and short contributions to complete combined portfolios')
        fig.tight_layout();fig.savefig(out/'leg_attribution.png',dpi=150);plt.close(fig)
    for p in sorted((root/'diagnostics').glob('*/*/permutation_importance.csv')):
        f=pd.read_csv(p);f=f[f.kind.eq('individual')].sort_values('mean_ic_decrease').tail(10)
        fig,ax=plt.subplots(figsize=(8,5));ax.barh(f.features,f.mean_ic_decrease);ax.axvline(0,color='black',lw=.7)
        ax.set_xlabel('Validation rank IC decrease after permutation');ax.set_title('Chronological feature importance')
        fig.tight_layout();fig.savefig(p.parent/'importance.png',dpi=120);plt.close(fig)
        effects=pd.read_csv(p.parent/'local_effects.csv')
        fig,axes=plt.subplots(2,3,figsize=(11,6));top=f.tail(6).features.tolist()
        for ax,c in zip(axes.flat,top):
            e=effects[effects.feature.eq(c)]
            if 'centered_effect' in e:ax.plot(e.high,e.centered_effect,marker='o',ms=3)
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


def document_sections(root):
    """One content stream drives Word and TeX so displayed numbers cannot diverge."""
    root=Path(root);a=load(root/'audit/data_changes.json');h=load(root/'historical/metrics.json')
    earlier=load(root/'assessment/metrics.json');s=load(root/'historical/monte_carlo.json')
    delta=load(root/'historical/primary_ic_difference.json');adapter=load(root/'prepared/adapter_report.json')
    fmt=lambda x,d=4:'Unavailable' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
    pct=lambda x:'Unavailable' if x is None or not np.isfinite(x) else f'{x:.2%}'
    blocks=[]
    def heading(t):blocks.append(('heading',t))
    def paragraph(t):blocks.append(('paragraph',t))
    def table(headers,rows):blocks.append(('table',headers,rows))
    heading('Updated data feature diagnostics and sensitivity')
    paragraph('This experiment extends the weekly trading study using the November 2026 data update. All comparisons are exploratory because the historical period and earlier feature correlations were already examined. Original monthly and weekly experiments above remain immutable reference results.')
    table(['Data item','Updated snapshot'],[
        ['Stock days',f"{a['new_rows']:,}"],['Dates',f"{a['first_date']} through {a['last_date']}"],
        ['Columns',str(a['new_columns'])+' including '+str(len(a['added_columns']))+' added columns'],
        ['Ticker labels',str(a['stocks'])],['Matching stock histories',str(adapter['stock_history_tickers'])],
        ['Adjusted price disagreements',str(adapter['overlap_adjusted_close_mismatches'])+' rows; stock CSVs are the consistent valuation source'],
        ['Assessment','52 signals May 2 2025 through April 24 2026; two weeks unscorable'],
        ['Historical comparison','26 holding periods May 11 through November 9 2026']])
    paragraph('The update adds observations and stock variables, but does not supply ten years of options history or repair every missing exit price. EQR, WBD and KVUE still have incomplete histories. Filled term slopes and robust fallback features have unverified definitions; the experiment reconstructs original endpoint slopes and uses only verified input formulas. Forward returns, labels and target-validity flags never enter predictors or stock selection.')
    paragraph('CTVA has a revised adjusted-price history that removes the original September 24 one-day discontinuity. This is a different source vintage, so its effect is isolated in the updated-price comparison. The calculation of corporate-action adjustments remains subject to source verification; we do not infer cash settlements or synthetic exit prices.')
    heading('How data and modeling changes are separated')
    reproduction=load(root/'baseline/control_reproduction.json')
    paragraph('Four original stock-only/full controls were refitted on original inputs. Their largest absolute prediction difference is '+str(max(x['max_absolute_prediction_error'] for x in reproduction.values()))+'. Original daily equity, holdings and costs were also checked. Saved predictions are then replayed with updated prices, followed by unchanged models retrained on updated data. Only after these controls are the feature variants compared.')
    price=pd.read_csv(root/'baseline/price_change/comparison.csv')
    table(['Saved model','Updated price replay return','Status'],[[r.variant,pct(getattr(r,'cumulative_return',None)),r.status] for r in price.itertuples()])
    heading('Models and feature comparisons')
    paragraph('Ridge predicts return as a weighted sum of standardized features and penalizes large coefficients. LightGBM combines trees that learn thresholds and interactions. Both predict signed returns from the scheduled weekly entry close to the next rebalance close. They rank stocks rather than classify weekly jumps. Each weekly fit uses only matured outcomes known before its decision.')
    paragraph('Both models use stock-only, full combined, compact, compact plus alternatives, pruned full and pruned extended sets. Extended adds verified 63 and 252 session returns, 60 and 90 session annualized realized volatility, 60 session skewness, a 20 session volume ratio and an IV to realized-volatility ratio. Implied and realized volatility are distinct quantities.')
    paragraph('Pruning uses each training window only: remove features over 50 percent missing or constant, then keep better-covered representatives when absolute Pearson correlation reaches 0.90 with at least 200 paired observations. The full control is retained and every eligible stock-week remains in every model.')
    for kind,label in [('ridge','Ridge'),('lightgbm','LightGBM')]:
        heading('Weekly experiment '+label+' results by feature set')
        rows=[]
        for v in h:
            if not v.endswith('/'+kind):continue
            m=h[v];net=m['portfolio']['net']
            rows.append([v.split('/')[0],fmt(earlier[v]['mean_ic']),fmt(m['mean_ic']),
                         pct(net.get('cumulative_return')),fmt(net.get('sharpe'),2),fmt(s[v]['ranking_p_adjusted'],3)])
        table(['Features','Earlier IC','Historical IC','Net return','Sharpe','Adjusted p'],rows)
    blocks.append(('image','charts/ranking.png','Rank IC measures agreement between predicted and realized stock rankings. Higher positive values are better.'))
    heading('Long and short contributions and costs')
    rows=[]
    for v,m in h.items():
        n=m['portfolio']['net']
        if n['status']!='complete':rows.append([v,'Unavailable','Unavailable','Unavailable']);continue
        c=n['costs_dollars'];rows.append([v,f"${n['long_pnl_dollars']:,.0f}",f"${n['short_pnl_dollars']:,.0f}",f"${sum(c[k] for k in ['commission','regulatory','execution','flat','borrow']):,.0f}"])
    table(['Model and features','Long P and L','Short P and L','All charges'],rows)
    cost_rows=[];concentration=[]
    for v,m in h.items():
        n=m['portfolio']['net']
        if n['status']!='complete':continue
        c=n['costs_dollars']
        cost_rows.append([v,*[f"${c[k]:,.0f}" for k in ['commission','regulatory','execution','borrow']]])
        nv=n['nvda'];concentration.append([v,pct(n['largest_target_position']),pct(n['largest_daily_position']),
            str(nv['periods_held']),pct(nv['average_abs_target_weight']),f"${nv['pnl_dollars']:,.0f}"])
    table(['Model and features','Commission','Regulatory','Execution','Borrow'],cost_rows)
    table(['Model and features','Max target','Max daily','NVDA weeks','NVDA mean weight','NVDA P and L'],concentration)
    paragraph('Leg P and L contributions precede charges and belong to the combined portfolio. Standalone long-only and short-only portfolios are replayed separately with their own cash, equity and costs. Rebalance orders, reversals and liquidation use net traded dollars; borrow accrues only on short exposure. All complete ledgers reconcile cash plus positions, costs, stock attribution and compounded returns.')
    if (root/'charts/leg_attribution.png').exists():blocks.append(('image','charts/leg_attribution.png','Net leg contributions allocate net-order fees by long and short traded dollars, with borrow assigned to the short side.'))
    blocks.append(('image','charts/equity.png','Only portfolios with complete required prices are plotted. Missing selected prices invalidate a result.'))
    heading('Monte Carlo and bootstrap results')
    paragraph('Each of twelve main variants receives 2,000 within-week score shuffles inside market-cap deciles. Donor permutations are shared across models and gross/net portfolios. Ranking and portfolio p-values are adjusted separately for twelve comparisons. A paired four-week block bootstrap uses 2,000 samples to measure uncertainty. These simulations do not add market history or create independent evidence.')
    bounds=lambda values:'Unavailable' if values is None else '['+', '.join(fmt(x,4) for x in values)+']'
    paragraph('The predeclared pruned-extended Ridge challenger has a mean historical IC difference versus full Ridge of '+fmt(delta['observed'],5)+'. Its 95 percent paired bootstrap interval is '+bounds(delta['bootstrap_95pct'])+'. Original Combined Ridge remains the original primary strategy.')
    table(['Model and features','Adjusted rank p','Invalid net draws','Raw net p bounds','Adjusted net p bounds'],[[v,fmt(m['ranking_p_adjusted'],3),str(m['invalid_net_draws']),bounds(m['portfolio_p_bounds']),bounds(m['portfolio_adjusted_p_bounds'])] for v,m in s.items()])
    paragraph('Invalid portfolio draws remain in the inference through conservative bounds. No valid-draw-only p-value is reported. Week-by-week shuffling changes persistence and turnover, so the randomized portfolios are not a turnover-matched null. Earlier assessment block summaries retain two weeks with only one eligible stock; whole-window trading and complete 52-week ranking inference remain unavailable there.')
    blocks.append(('image','charts/randomization.png','The observed challenger ranking is compared with scores randomly reassigned among similarly sized stocks.'))
    heading('Model setting and portfolio sensitivity')
    paragraph('Model settings vary one at a time: Ridge alpha 1, 10 and 100; LightGBM leaves 7, 15 and 31; trees 50, 100 and 200. Other settings remain fixed. Full and pruned-extended models are assessed on both windows. All cases are reported; no best setting is promoted from these inspected periods.')
    x=pd.read_csv(root/'sensitivity/ranking_by_week.csv');x=x.groupby(['variant','window']).rank_ic.mean().unstack('window')
    table(['Setting variant','Earlier IC','Historical IC'],[[v,fmt(row.get('assessment')),fmt(row.get('historical'))] for v,row in x.iterrows()])
    blocks.append(('image','charts/model_sensitivity.png','Changes in mean rank IC across the fixed model setting grid.'))
    p=pd.read_csv(root/'sensitivity/portfolio/comparison.csv')
    table(['Model','Portfolio case','Net return','Sharpe'],[[r.variant,r.condition,pct(getattr(r,'cumulative_return',None)),fmt(getattr(r,'sharpe',None),2)] for r in p.itertuples()])
    paragraph('The main case uses 100000 dollars, equal weights, a 5 percent rebalance cap, 100 percent exposure per side, IBKR Fixed commission estimates, one basis point execution allowance and 3 percent annual borrow. Sensitivities cover selection fraction, caps, exposure, execution, borrow and account size. These are descriptive assumptions, not optimized trading instructions. The frozen current fee schedule is not evidence of historical broker charges.')
    heading('Correlations feature importance and local response')
    paragraph('Pearson and Spearman matrices, paired counts and missingness are saved for every fit. At six reporting endpoints, separate diagnostic models fit earlier matured labels and validate on the next 13 matured weeks. Twenty within-week permutations test individual inputs and correlated groups. Coefficients and tree gain supplement validation importance. Sensitivity curves show accumulated local effects over observed training quantiles; sparse bins and support counts are retained.')
    diag=sorted((root/'diagnostics').glob('*/full_ridge/importance.png'))
    if diag:
        folder=diag[-1].parent
        blocks.append(('image',str((folder/'importance.png').relative_to(root)),'Chronological validation importance for full Ridge at the last reporting endpoint. Negative values mean permutation did not hurt this fitted model.'))
        blocks.append(('image',str((folder/'sensitivity.png').relative_to(root)),'Local response curves vary one input within observed conditional bins. They describe model response and do not establish causality.'))
    corr=sorted((root/'diagnostics').glob('*/pearson.png'))
    if corr:blocks.append(('image',str(corr[-1].relative_to(root)),'Training-only Pearson correlations at the last historical fit. Pairwise counts and Spearman matrices accompany this chart.'))
    heading('Interpretation and reproducibility')
    paragraph('Completion establishes reproducible software comparisons, regardless of returns. Improvement in a feature variant or fee scenario does not validate live trading, resolve corporate actions or reproduce the published study. Rank IC and uncertainty take priority over choosing the largest historical profit. Alpha is reported only as an internal-universe proxy where benchmark coverage is complete. Sharpe uses a zero cash-rate assumption; daily drawdown includes price drift and costs.')
    manifest=load(root/'manifest.json')
    paragraph('The private run stores source hashes, frozen configuration, training dates, schemas, models, predictions, holdings, daily equity, component costs, importance, sensitivity plots, shared shuffle draws and bootstrap indices. The calculation code commit is '+manifest['code_commit']+'. Maximum CPU allocation is four cores, with one numerical thread per worker. Raw inputs and derived results remain private.')
    return blocks


def export_documents(root,output,base_docx,base_tex):
    """Called with the bundled document runtime; preserves the supplied reference reports."""
    from docx import Document
    from docx.shared import Inches,Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    import shutil
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    blocks=document_sections(root);doc=Document(base_docx)
    doc.paragraphs[0].text='Options Signals for Stock Jumps Crashes and Weekly Trading'
    note=doc.paragraphs[1].insert_paragraph_before('Updated November 16 2026. Original experiments are retained as reference evidence; the final section reports the updated data and sensitivity experiment.')
    note.style='Subtitle'
    # Introduce the two research parts in the retained presentation document.
    for para in list(doc.paragraphs):
        if para.text=='Experiment A Monthly diagnostic results':
            p=para.insert_paragraph_before('Part One Replicating the reference paper');p.style='Heading 1'
            p.paragraph_format.page_break_before=True;para.paragraph_format.page_break_before=False
        if para.text=='Experiment B Weekly strategy results':
            p=para.insert_paragraph_before('Part Two Weekly trading strategy');p.style='Heading 1'
            p.paragraph_format.page_break_before=True;para.paragraph_format.page_break_before=False
    doc.add_page_break()
    tex=[]
    def esc(s):
        return ''.join({'&':r'\&','%':r'\%','$':r'\$','_':r'\_','#':r'\#','{':r'\{','}':r'\}','\\':r'\textbackslash{}'}.get(c,c) for c in str(s))
    for block in blocks:
        kind=block[0]
        if kind=='heading':
            doc.add_heading(block[1],level=1 if not tex else 2)
            tex.append(r'\section{'+esc(block[1])+'}' if not tex else r'\subsection{'+esc(block[1])+'}')
        elif kind=='paragraph':
            doc.add_paragraph(block[1]);tex.append(esc(block[1])+'\n')
        elif kind=='table':
            headers,rows=block[1:];table=doc.add_table(rows=1,cols=len(headers));table.style='Light Shading Accent 1'
            table.autofit=False
            first=2.2 if len(headers)>4 else 2.5
            widths_in=[first]+[(6.8-first)/(len(headers)-1)]*(len(headers)-1)
            for column,width in zip(table.columns,widths_in):column.width=Inches(width)
            for cell,text in zip(table.rows[0].cells,headers):cell.text=str(text)
            repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
            for row in rows:
                cells=table.add_row().cells
                for c,text in zip(cells,row):c.text=str(text).replace('_',' ').replace('/',' / ')
            for row in table.rows:
                avoid=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(avoid)
                for c,width in zip(row.cells,widths_in):
                    c.width=Inches(width)
                    for p in c.paragraphs:
                        p.paragraph_format.space_after=Pt(4)
                        for r in p.runs:r.font.size=Pt(9)
            widths=[.34]+[(.66/(len(headers)-1))]*(len(headers)-1)
            spec=''.join(r'p{'+str(round(w*6.25,3))+'in}' for w in widths)
            tex.extend([r'{\small\setlength{\tabcolsep}{3pt}\begin{longtable}{'+spec+'}',r'\toprule',
                        ' & '.join(r'\textbf{'+esc(x)+'}' for x in headers)+r' \\ \midrule\endhead'])
            tex += [' & '.join(esc(x).replace(r'\_',r'\_\allowbreak{}').replace('/',r'/\allowbreak{}') for x in row)+r' \\' for row in rows]
            tex.extend([r'\bottomrule\end{longtable}}',''])
        elif kind=='image':
            image=root/block[1]
            if image.exists():
                doc.add_picture(str(image),width=Inches(6.8));p=doc.add_paragraph(block[2]);p.style='Caption'
                target=output/'figures'/image.name;target.parent.mkdir(exist_ok=True)
                # Prefix duplicate plot names by parent to preserve every included figure.
                target=target.parent/(image.parent.name+'_'+image.name);shutil.copy2(image,target)
                tex += [r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth,height=.7\textheight,keepaspectratio]{figures/'+target.name+'}',r'\caption{'+esc(block[2])+r'}\end{figure}']
    doc.save(output/'Options_Signals_Updated_Data_Report.docx')
    source=Path(base_tex).read_text();source=source.replace(r'\usepackage{enumitem,needspace,fancyhdr}',r'\usepackage{enumitem,needspace,fancyhdr,longtable,graphicx}')
    source=source.replace(r'\date{November 12, 2026}',r'\date{November 16, 2026}')
    source=source.replace('The weekly strategies show weak ranking evidence and negative returns in every complete main-condition backtest.',
                          'On the original snapshot, weekly strategies show weak ranking evidence and negative returns in every complete main-condition backtest. The updated-data comparisons in Part II retain weak ranking evidence and explicit missing-price failures.')
    source=source.replace('An unresolved example is CTVA on September 24, 2026:',
                          'In the original frozen snapshot, an unresolved example was CTVA on September 24, 2026:')
    source=source.replace('The anomaly can still affect targets, training and benchmarks.',
                          'That original-vintage anomaly can affect its targets, training and benchmarks. The updated snapshot revises the adjustment history and removes this one-day discontinuity, with source verification still required.')
    source=source.replace('Software validation comprises 57 passing tests,',
                          'Original software validation comprised 57 passing tests,')
    source=source.replace(r'\section{Limitations and conclusions across both parts}',
                          '\n'.join(tex)+'\n\\clearpage\n'+r'\section{Limitations and conclusions across both parts}')
    source=source.replace(r'\begin{abstract}',r'\begin{abstract}'+'\n'+esc('The original experiments are retained as reference results. The additional updated-data experiment in Part II reports the November 2026 snapshot, pruning, sensitivity analysis and paired inference.')+'\n')
    (output/'main.tex').write_text(source)
    digest=lambda p:__import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
    dump(output/'report_sources.json',dict(results_manifest_sha256=digest(root/'manifest.json'),
         base_docx=str(base_docx),base_tex=str(base_tex),shared_content_blocks=len(blocks),
         reference_document_hashes={str(p):digest(p) for p in [base_docx,base_tex]},
         displayed_result_hashes={str(p.relative_to(root)):digest(p) for p in
            [root/'historical/metrics.json',root/'historical/monte_carlo.json',root/'historical/primary_ic_difference.json']}))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True)
    p.add_argument('--output-dir');p.add_argument('--base-docx');p.add_argument('--base-tex')
    a=p.parse_args()
    if a.output_dir:
        if not a.base_docx or not a.base_tex:p.error('document export requires both reference documents')
        export_documents(a.results,a.output_dir,a.base_docx,a.base_tex)
    else:build_charts(a.results);write_markdown(a.results)

if __name__=='__main__':main()
