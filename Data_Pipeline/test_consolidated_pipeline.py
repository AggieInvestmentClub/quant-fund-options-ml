"""Offline compatibility checks for the consolidated calculator and data handoff."""
from datetime import date
import gzip,io,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import flatfile_iv,utils

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('symbols',[None,{'A'},{'AAPL'},{'A','AAPL'},set()])
@pytest.mark.parametrize('dated',[False,True])
def test_ingestion_python_native_match_multimember_and_final_line(tmp_path,monkeypatch,symbols,dated):
    header=b'ticker,close,volume,transactions\r\n'
    rows=[b'O:A260320C00100000,3,10,2\r\n',b'O:AAPL260320P00100000,4,20,3\n',
          b'invalid,1,1,1\n',b'O:A260130C00100000,5,1,1\n',b'O:AAPL260320C00110000,1,2,3']
    path=tmp_path/'day.csv.gz'
    path.write_bytes(gzip.compress(header+rows[0]+rows[1])+gzip.compress(b''.join(rows[2:])))
    result=[]
    for backend in ('python','native'):
        monkeypatch.setenv('OPTIONS_OPRA_BACKEND',backend)
        monkeypatch.setenv('OPTIONS_CRR_BACKEND','python') # ingestion is independent
        result.append(flatfile_iv.stream_filter_opra_gzip(path,symbols,date(2026,1,30) if dated else None))
    assert result[0]==result[1]
    if symbols=={'A'}:assert b'O:AAPL' not in result[0]
    if symbols is None or 'AAPL' in symbols:assert b'C00110000' in result[0]

@pytest.mark.parametrize('contents',[b'garbage',gzip.compress(b'ticker,close\nO:A260320C00100000,1\n')[:-4]])
def test_corrupt_gzip_fails_for_explicit_backends(tmp_path,monkeypatch,contents):
    path=tmp_path/'bad.gz';path.write_bytes(contents)
    for backend in ('python','native'):
        monkeypatch.setenv('OPTIONS_OPRA_BACKEND',backend)
        with pytest.raises((ValueError,OSError,EOFError)):
            flatfile_iv.stream_filter_opra_gzip(path,None)


def test_explicit_native_cannot_silently_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv('OPTIONS_OPRA_BACKEND','native');monkeypatch.setattr(flatfile_iv,'_HAS_NATIVE_OPRA_STREAM',False)
    with pytest.raises(RuntimeError):flatfile_iv.stream_filter_opra_gzip(tmp_path/'missing.gz',None)


@pytest.mark.parametrize('model',['crr','lr'])
@pytest.mark.parametrize('backend',['python','native'])
def test_selection_matches_model_specific_price_solver_and_greeks(model,backend):
    code='''import json,numpy as np,utils
from pricing_settings import PRICING_MODEL
x=(100.,105.,60./365.25,.04,.25,"put",.01)
expected=utils.american_option_price_crr(*x) if PRICING_MODEL=="crr" else utils.american_option_price_lr(*x)
price=utils.american_option_price(*x)
iv=utils.compute_iv(price,x[0],x[1],x[2],x[3],x[5],x[6])
a=utils.batch_regular_greeks(*[np.array([v]) for v in (100.,105.,60./365.25,.04,.25,.01,0)])
print(json.dumps(dict(model=PRICING_MODEL,price=price,expected=expected,iv=iv,greeks=[float(z[0]) for z in a])))'''
    env={**os.environ,'PYTHONPATH':str(ROOT/'Data_Pipeline'),'OPTIONS_PRICING_MODEL':model,
         'OPTIONS_CRR_BACKEND':backend,'OPTIONS_LR_BACKEND':backend,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    r=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,text=True))
    assert r['model']==model and r['price']==pytest.approx(r['expected'],abs=1e-12)
    assert r['iv']==pytest.approx(.25,abs=1e-3 if model=='lr' else 1e-6)
    assert np.isfinite(r['greeks']).all()


def test_crr_batch_greeks_matches_scalar_and_keeps_invalid_rows():
    sys.path.insert(0,str(ROOT/'Default_Strike_Pipeline'))
    from flatfile_regular_greeks import compute_row_regular_greeks_task
    values=[(100.,105.,.25,.04,.25,.01,0),(100.,95.,.25,.04,.3,0.,1)]
    batch=utils.batch_regular_greeks(*[np.array([r[j] for r in values]) for j in range(7)])
    for i,(s,k,t,r,v,q,call) in enumerate(values):
        expected=compute_row_regular_greeks_task((v,s,k,t,r,q,'call' if call else 'put'))
        assert [a[i] for a in batch]==pytest.approx([expected[k] for k in ('delta','gamma','theta','vega')],abs=1e-10)
    bad=utils.batch_regular_greeks(*[np.array([x]) for x in (np.nan,100.,.25,.04,.25,.01,0)])
    assert all(np.isnan(a[0]) for a in bad)


def test_synthetic_atm_to_default_strike_to_parquet_to_research(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'Default_Strike_Pipeline'))
    import flatfile_regular_greeks as ds
    from regular_features import build_daily_features
    from Dataset_Prep.export_daily_parquet import build_single_day_dataset,export_partitioned_parquet_date
    from Dataset_Prep.dataset_schema import get_pyarrow_schema
    from Research_Pipeline.ben_handoff import convert_ben_file
    day=date(2026,1,30)
    inputs=[]
    for expiry in [date(2026,3,6),date(2026,4,10)]:
        tenor=(expiry-day).days/365.25
        for side in ['call','put']:
            for strike in [85.,95.,100.,105.,115.]:
                price=utils.american_option_price(100.,strike,tenor,.04,.3,side,.01)
                inputs.append(dict(ticker=f'O:A{expiry:%y%m%d}{side[0].upper()}{int(strike*1000):08}',close=price,volume=20,transactions=4))
    raw=tmp_path/'raw';raw.mkdir();source=raw/'2026-01-30.csv.gz';pd.DataFrame(inputs).to_csv(source,index=False,compression='gzip')
    monkeypatch.setattr(flatfile_iv,'fetch_underlying_closes',lambda *a,**k:{'A':100.})
    monkeypatch.setattr(flatfile_iv,'fetch_underlying_dividend_yields',lambda *a,**k:{'A':.01})
    atm=tmp_path/'atm';atm_file,contracts,_=flatfile_iv.process_file(source,raw,atm,True,{},False,'A',{'A'},1,{day:.04})
    assert pd.read_csv(atm_file).implied_volatility.notna().all()
    output,regular,_=ds.process_file(source,raw,tmp_path/'regular',True,{},False,'A',{'A'},1,atm,False,{day:.04})
    assert len(regular)==len(contracts)
    surface=build_daily_features(output);surface_file=tmp_path/'surface.csv';surface.to_csv(surface_file,index=False)
    avg=flatfile_iv.build_daily_feature_frame(pd.read_csv(atm_file));avg_file=tmp_path/'avg.csv';avg.to_csv(avg_file,index=False)
    stocks=pd.DataFrame([dict(trade_date=day.isoformat(),underlying='A',stock_close=100.,stock_adj_close=100.,stock_volume=1000000.,market_cap=1e10)])
    wide=build_single_day_dataset(day.isoformat(),surface_file,avg_file,stocks,[(day,{'A'})],tmp_path/'none')
    file=export_partitioned_parquet_date(wide,day.isoformat(),tmp_path/'parquet',get_pyarrow_schema())
    report=convert_ben_file(file,tmp_path/'handoff',allow_ticker_proxy=True)
    handoff=pd.read_parquet(tmp_path/'handoff/options_features.parquet')
    assert report['rows']==1 and handoff.avg_call_iv.notna().all() and handoff.avg_put_iv.notna().all()


def test_checkpoint_reuse_requires_matching_model_provenance(tmp_path,monkeypatch):
    import pricing_settings
    path=tmp_path/'contract.csv';pd.DataFrame({'iv':[.2]}).to_csv(path,index=False)
    assert not pricing_settings.pricing_output_matches(path)
    pricing_settings.write_pricing_stamp(path)
    assert pricing_settings.pricing_output_matches(path)
    monkeypatch.setattr(pricing_settings,'PRICING_MODEL','lr')
    assert not pricing_settings.pricing_output_matches(path)
    monkeypatch.setattr(pricing_settings,'PRICING_MODEL','crr')
    path.write_text('iv\n0.3\n')
    assert not pricing_settings.pricing_output_matches(path)
