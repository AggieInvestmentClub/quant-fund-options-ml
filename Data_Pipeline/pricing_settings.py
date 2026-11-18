"""Explicit process-wide financial model; inherited unchanged by worker processes."""
import os

PRICING_MODEL = os.environ.get('OPTIONS_PRICING_MODEL', 'crr').strip().lower()
if PRICING_MODEL not in ('crr', 'lr'):
    raise ValueError('OPTIONS_PRICING_MODEL must be crr or lr')
for name in ('OPTIONS_CRR_BACKEND', 'OPTIONS_LR_BACKEND', 'OPTIONS_OPRA_BACKEND'):
    if os.environ.get(name, 'auto').strip().lower() not in ('auto', 'python', 'native'):
        raise ValueError(name+' must be auto, python, or native')


def pricing_signature():
    return {'model':PRICING_MODEL,'definition':'crr-201-bisection-v1' if PRICING_MODEL=='crr' else 'lr-65-newton6-v1'}


def write_pricing_stamp(path):
    import json
    from pathlib import Path
    path=Path(path);st=path.stat();target=path.with_suffix(path.suffix+'.pricing.json')
    temp=target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps({**pricing_signature(),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})+'\n')
    temp.replace(target)


def pricing_output_matches(path):
    """An unversioned or different-model checkpoint cannot be reused silently."""
    import json
    from pathlib import Path
    path=Path(path)
    try:
        meta=json.loads(path.with_suffix(path.suffix+'.pricing.json').read_text());st=path.stat()
        return all(meta.get(k)==v for k,v in pricing_signature().items()) and meta.get('bytes')==st.st_size and meta.get('mtime_ns')==st.st_mtime_ns
    except (OSError,ValueError):
        return False
