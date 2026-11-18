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
        attribution=[]
        for name,m in rule.items():
            n=m['net']
            if n['status']!='complete':continue
            folder=root/'historical/rules'/name.replace('/','_')
            daily=pd.read_parquet(folder/'net_daily.parquet')
            trades=pd.read_parquet(folder/'net_trades.parquet')
            cost=n['cost_components'];nv=n.get('nvda',{})
            attribution.append(dict(configuration=name,net_return=n['cumulative_return'],
                long_pnl=n['long_pnl_dollars'],short_pnl=n['short_pnl_dollars'],
                long_transaction_cost=float(trades.long_transaction_cost.sum()),short_transaction_cost=float(trades.short_transaction_cost.sum()),
                average_gross_exposure=float(daily.gross_exposure.mean()),average_net_exposure=float(daily.net_exposure.mean()),
                largest_position=n['largest_daily_position'],nvda_pnl=nv.get('pnl_dollars'),
                **{k:cost[k] for k in ['commission','regulatory','execution','borrow']}))
        pd.DataFrame(attribution).to_csv(root/'costs_exposure_attribution.csv',index=False)
    lines+=['','## What the measurements mean','',
      'Rank IC measures whether predicted rankings agree with realized rankings; positive is better. Net return includes commission, regulatory charges, spread/slippage and borrow. Sharpe compares average weekly return with its variability. Drawdown is the worst daily loss from an earlier equity peak. Turnover measures traded dollars relative to portfolio equity.',
      '', 'We use $100,000, equal weights, a 5 percent cap per stock, the previously frozen IBKR commission approximation, 1 basis point execution cost and 3 percent annual short borrow. Positions can drift between scheduled rebalances. Zero cash interest and the prior financing limitations remain. Less exposure or fewer trades can improve losses without improving stock selection, so exposure and costs must be read alongside returns.',
      '', 'Long and short contributions are separated. Standalone portfolios have independent cash, costs and equity. Alpha is only an internal-universe proxy when all benchmark prices are available; it is not the reference paper factor alpha.',
      '', '## Monte Carlo and stability','',
      'Each declared historical strategy receives 2,000 score shuffles within decision period and market-cap decile. Momentum and confirmation inputs move together. Holdings, retention and charges are replayed after each shuffle. Shared permutations pair gross/net and comparable strategies. Independent period shuffles are not turnover matched.',
      '', 'All declared portfolio cases count toward one Bonferroni family; ranking tests form a separate 24-configuration family. Missing-price draws remain invalid and contribute conservative p-value bounds. We never report valid-draw-only p-values.',
      '', 'The same 2,000 eight-calendar-week block resamples are used across strategies. Intervals describe historical uncertainty, with only 24 weeks and six four-week holds. They are not live-trading validation.',
      '', '| Case | Net adjusted p | Raw p bounds | Invalid net draws |','|---|---:|---|---:|']
    for key,m in stats.items():lines.append(f'| {key} | {fmt(m["net"]["p_adjusted"])} | {m["net"].get("p_bounds")} | {m["net"]["invalid_draws"]} |')
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
    aliases=load(root/'inference/aliases.json') if (root/'inference/aliases.json').exists() else {}
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    choices=['stage2/ml_both/long_short','stage2/ml_both/long_only',
             'stage2/momentum_plain/long_short','stage2/momentum_gap_skew_both/long_short']
    for ax,key in zip(axes.flat,choices):
        folder=root/'inference'/aliases.get(key,key).replace('/','_');p=folder/'random_draws.parquet'
        if p.exists():
            draws=pd.read_parquet(p);values=draws.net_mean_weekly_return.dropna()*100
            if len(values):ax.hist(values,bins=30,color='#4779a0',alpha=.8)
            else:ax.text(.5,.5,'No complete randomized portfolios',ha='center',transform=ax.transAxes,fontsize=8)
            actual=folder/'net_weekly.parquet'
            if actual.exists():ax.axvline(pd.read_parquet(actual)['return'].mean()*100,color='darkred',label='Observed');ax.legend(fontsize=7)
            else:ax.text(.03,.92,'Observed portfolio invalid',transform=ax.transAxes,fontsize=8)
        ax.set_title(key.removeprefix('stage2/'),fontsize=8);ax.set_xlabel('Mean net weekly return percent',fontsize=8)
    fig.tight_layout();fig.savefig(out/'monte_carlo.png',dpi=130);plt.close(fig)
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


def export_word(root,base_docx,output):
    """Append the new experiment to the retained report; use the document runtime."""
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    root=Path(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    doc=Document(base_docx)
    rule_metrics=load(root/'historical/rules/metrics.json')
    buffer_return=rule_metrics['ml_buffer/long_only']['net'].get('cumulative_return')
    for p in doc.paragraphs:
        if p.text.startswith('Updated November 16 2026.'):
            p.text='Prepared November 18 2026. This report includes the reference paper diagnostic, weekly feature studies, and the holding-period and selective-trading experiment.'
        elif p.text=='Feature experiments and portfolio comparisons':p.text='Paper methods options signals and trading experiments'
        elif 'Aggie Investment Club Options ML team | November 16, 2026' in p.text:
            p.text=p.text.replace('November 16, 2026','November 18, 2026')
        elif p.text.startswith('The updated-data experiment compares twelve main model'):
            p.text=('The latest experiment compares one, two and four week holds and absolute versus relative returns. Complete ML long-short portfolios lost money, while others remain invalid because selected prices are missing. A four-week long-only holding-buffer case returned '+fmt(buffer_return,True)+' after modeled costs, but its randomization checks do not establish an edge. Missing prices prevent a validated development selection; the predefined four-week Ridge fallback remains a diagnostic.')
    doc.add_page_break()
    def para(text):
        p=doc.add_paragraph(text)
        for run in p.runs:run.font.size=Pt(10);run.font.color.rgb=RGBColor(0,0,0)
        return p
    def heading(text,level=2):
        p=doc.add_heading(text,level)
        for run in p.runs:run.font.color.rgb=RGBColor(0,0,0)
    def table(headers,rows):
        t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False
        if headers[0]=='Configuration':widths=[2.3]+[(6.2-2.3)/(len(headers)-1)]*(len(headers)-1)
        elif headers[0]=='Combined strategy':widths=[2.6,1.,1.4,1.2] if len(headers)==4 else [3.6,1.3,1.3]
        elif len(headers)==7:widths=[.5,.9,1.,.8,1.,.9,1.1]
        else:widths=[6.2/len(headers)]*len(headers)
        for column,width in zip(t.columns,widths):column.width=Inches(width)
        for cell,text in zip(t.rows[0].cells,headers):cell.text=str(text)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in rows:
            for cell,value in zip(t.add_row().cells,row):cell.text=str(value)
        for ri,row in enumerate(t.rows):
            for cell,width in zip(row.cells,widths):
                cell.width=Inches(width)
                tcPr=cell._tc.get_or_add_tcPr()
                margins=OxmlElement('w:tcMar')
                for edge in ['top','left','bottom','right']:
                    node=OxmlElement('w:'+edge);node.set(qn('w:w'),'80');node.set(qn('w:type'),'dxa');margins.append(node)
                tcPr.append(margins)
                if ri==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DCE6F1');tcPr.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(3)
                    for r in p.runs:r.font.size=Pt(8);r.font.bold=ri==0;r.font.color.rgb=RGBColor(0,0,0)
        doc.add_paragraph()
    heading('Holding periods relative returns and selective trading',1)
    selection=load(root/'selection.json')
    para('This experiment uses the same frozen data to test longer holds, a relative-return target and more selective trading. All results are exploratory because the historical period was previously inspected. The earlier paper diagnostic and weekly experiments remain reference results.')
    para('Development covers June 2 2025 to March 9 2026. Historical comparison covers May 11 to October 26 2026. The common historical window contains 24 one-week holds, 12 two-week holds or six four-week holds. Portfolios rebalance only when a holding period ends; daily prices value positions in between.')
    heading('Models and targets')
    para('Ridge learns a weighted sum of features while penalizing unusually large coefficients. LightGBM combines trees that learn thresholds and interactions. Each predicts the percentage return for its own holding horizon. The relative target subtracts the equal-weight return of eligible stocks with known outcomes, so it predicts performance against peers. Neither model predicts weekly jump or crash classes here.')
    para('All preprocessing is fitted on matured training observations. Weekly training labels may overlap at longer horizons, but actual portfolios do not overlap. A label becomes usable only after its exit close. Stock-only and combined models share the same eligible stocks and calendar.')
    para('Development selection: '+selection['status']+'. Configuration '+selection['variant']+'. Only development outcomes choose the configuration; historical results cannot replace it.')
    heading('Horizon experiment results')
    frame=pd.read_csv(root/'comparison.csv');frame=frame[frame.window.eq('historical')]
    stats=load(root/'inference/statistics.json')
    for model in ['ridge','lightgbm']:
        heading('Ridge results' if model=='ridge' else 'LightGBM results',3)
        rows=[]
        for r in frame[frame.model.eq(model)].itertuples():
            p=stats.get('stage1/'+r.variant,{}).get('net',{}).get('p_adjusted')
            rows.append([r.horizon[1:],r.target,'Stock' if r.features=='stock_only' else 'Combined',fmt(r.mean_ic),fmt(r.net_return,True),fmt(r.sharpe),fmt(p)])
        table(['Weeks','Target','Inputs','Rank IC','Net return','Sharpe','Adj p'],rows)
    para('Unavailable returns indicate a failed portfolio accounting requirement, including missing selected prices. Such stocks are never removed retrospectively to create a complete backtest. Every configuration is retained in the private comparison files.')
    para('Adjusted p in these tables refers to the portfolio randomization test. Separate adjusted ranking tests and target-specific prediction errors are retained in the artifacts. Quintile outcomes use the fitted target: absolute returns or returns relative to peers; their top-minus-bottom spread compares the same cross section.')
    heading('Momentum confirmations and trading rules')
    para('Momentum ranks the trailing 63-session return. Longs require positive momentum and shorts negative momentum. The gap confirmation requires above-median call-minus-put IV for longs and below-median values for shorts. Skew adds below-median put skew for longs and above-median skew for shorts. Missing confirmations fail the filter, with unfilled exposure left in cash.')
    para('Baseline entry and retention use the highest and lowest 5 percent. Stricter entry uses 2.5 percent. The holding buffer retains incumbents within the outer 10 percent, provided all confirmations still pass. These thresholds measure ranks rather than calibrated prediction confidence.')
    rules=load(root/'historical/rules/metrics.json')
    for mode,label in [('long_short','Combined long and short'),('long_only','Long only'),('short_only','Short only')]:
        heading(label,3);rows=[]
        for key,m in rules.items():
            if not key.endswith('/'+mode):continue
            n=m['net'];rows.append([key.rsplit('/',1)[0],fmt(n.get('cumulative_return'),True),fmt(n.get('sharpe')),fmt(n.get('max_drawdown'),True)])
        table(['Configuration','Net return','Sharpe','Drawdown'],rows)
    para('All portfolios start with $100,000. Selected positions use equal weights and a 5 percent stock cap. Costs use the frozen IBKR approximation, 1 basis point execution allowance and 3 percent annual short borrow. Cash earns zero interest. Sparse portfolios can have lower gross exposure, so improved returns alone do not prove improved stock selection.')
    attr=pd.read_csv(root/'costs_exposure_attribution.csv')
    heading('Exposure costs and leg contributions',3)
    a=attr[attr.configuration.str.endswith('/long_short')]
    table(['Configuration','Avg gross','Long P and L','Short P and L','Trading fees','Borrow'],
        [[r.configuration.split('/')[0],fmt(r.average_gross_exposure,True),f'${r.long_pnl:,.0f}',
          f'${r.short_pnl:,.0f}',f'${r.commission+r.regulatory+r.execution:,.0f}',f'${r.borrow:,.0f}'] for r in a.itertuples()])
    para('Leg P and L precedes costs; standalone leg returns above include their own costs. Component commission, regulatory, execution and borrow charges, average net exposure, concentration and NVDA attribution are retained in the accompanying cost and exposure CSV.')
    heading('What the measurements tell us')
    para('Rank IC measures agreement between predicted and realized stock rankings; positive is better. Net return includes the modeled charges. Sharpe compares average weekly return with its variability. Drawdown measures the largest daily peak-to-trough decline. Turnover measures traded dollars relative to equity. Alpha is only an internal-universe regression proxy when benchmark prices are complete, not the reference paper factor alpha.')
    for name,caption in [('ranking.png','Prediction quality for all horizon configurations'),('equity.png','Selected trading rules and momentum portfolios')]:
        p=root/'charts'/name
        if p.exists():doc.add_picture(str(p),width=Inches(6));para(caption)
    chosen=selection['variant'].replace('/','_')
    for name in ['importance.png','sensitivity.png']:
        p=root/'historical/fits'/chosen/'effects'/name
        if p.exists():doc.add_picture(str(p),width=Inches(6))
    para('Feature importance uses earlier chronological validation, including correlated groups. Local sensitivity curves stay within observed training ranges. They describe model behavior and do not establish causal effects.')
    heading('Monte Carlo and bootstrap checks')
    para('Each declared strategy receives 2,000 market-cap-decile shuffles. Scores and confirmation inputs move together, and positions, retention and costs are rebuilt. Gross and net simulations share the same permutations. The portfolio family contains every declared case; ranking uses a separate 24-model family. Invalid draws remain visible through counts and conservative bounds. Independent period shuffles are not turnover matched.')
    if (root/'charts/monte_carlo.png').exists():doc.add_picture(str(root/'charts/monte_carlo.png'),width=Inches(6))
    rows=[]
    for key,m in stats.items():
        if not key.startswith('stage2/') or not key.endswith('/long_short'):continue
        bounds=m['net'].get('p_bounds')
        rows.append([key.split('/')[1],fmt(m['net']['p_adjusted']),
                     'Unavailable' if bounds is None else fmt(bounds[0])+' to '+fmt(bounds[1]),m['net']['invalid_draws']])
    table(['Combined strategy','Adjusted p','Raw p bounds','Invalid draws'],rows)
    primary=stats['stage2/ml_both/long_short']
    for label in ['mean_weekly_return','annualized_return','sharpe']:
        ci=primary.get('net_'+label+'_95pct')
        para('Primary strategy '+label.replace('_',' ')+' bootstrap 95 percent interval: '+(str(ci) if ci else 'Unavailable')+'.')
    para('Bootstrap checks use 2,000 shared eight-calendar-week block resamples. Only 24 historical weeks and six four-week holding periods exist, so uncertainty estimates have limited precision. Full random draws, gross results, costs, attribution and all confidence intervals accompany the private artifacts.')
    heading('Conclusion')
    text=(root/'REPORT.md').read_text();conclusion=text.split('## Interpretation\n\n',1)[1].split('\n',1)[0]
    para(conclusion)
    para('The screen requires positive net return, Sharpe at least one, daily drawdown no worse than minus 20 percent, adjusted randomization p below .05 and a positive lower bootstrap bound for mean return. Options must also improve a matched stock-only strategy before an incremental-value claim is made. This is preliminary research, not paper replication or live-trading validation.')
    para('Calculation commit '+load(root/'manifest.json')['code_commit']+'. Input hashes, configurations, training cutoffs, models, predictions, ledgers and raw simulations are retained privately. Source data and earlier experiment artifacts were not changed.')
    doc.save(output)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True)
    parser.add_argument('--base-docx');parser.add_argument('--output-docx')
    args=parser.parse_args()
    if bool(args.base_docx)!=bool(args.output_docx):parser.error('provide both --base-docx and --output-docx')
    write_report(args.results)
    if args.base_docx:export_word(args.results,args.base_docx,args.output_docx)
