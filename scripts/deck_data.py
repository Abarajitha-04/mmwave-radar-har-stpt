"""Collect every number the presentation needs into presentation/deck_data.json."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from analyze import load_all, agg, ORDER, NAMES

df = load_all()
D = {}


def tab(pfx):
    t = agg(df, [pfx + o for o in ORDER])
    return [dict(key=r.group[len(pfx):], name=NAMES[r.group[len(pfx):]], acc=round(r.acc_mean, 2), sd=round(r.acc_std, 2),
                 f1=round(r.f1_mean, 2), f1sd=round(r.f1_std, 2), prec=round(r.prec, 2), rec=round(r.rec, 2), top3=round(r.top3, 2),
                 params=int(r.params), mflops=round(float(r.mflops), 1), lat=round(r.lat_ms, 2), size=round(r.size_mb, 3),
                 int8=round(float(r.int8_acc), 2), int8mb=round(float(r.int8_mb), 3), n=int(r.n)) for _, r in t.iterrows()]


D['radhar'] = tab('rh_'); D['milipoint'] = tab('mp_')
G = {r.group: r for _, r in agg(df, sorted(set(df.group))).iterrows()}
g = lambda k: (round(G[k].acc_mean, 2), round(G[k].acc_std, 2)) if k in G else (None, None)
D['ablation'] = {k: dict(rh=g(f'rh_abl_{k}'), mp=g(f'mp_abl_{k}')) for k in ['noattn', 'nomotion', 'notemporal', 'xyz', 'nogate', 'augnone', 'augnojitter']}
D['ablation']['full'] = dict(rh=g('rh_stpt_lite'), mp=g('mp_stpt_lite'))
def paired(v, ref):
    a = df[df.group == v]
    if not len(a): return None
    f = df[(df.group == ref) & df.seed.isin(set(a.seed))]
    return round(float(a.acc.mean() - f.acc.mean()), 2), int(len(a))
for k in list(D['ablation']):
    if k != 'full':
        D['ablation'][k]['d_rh'] = paired(f'rh_abl_{k}', 'rh_stpt_lite'); D['ablation'][k]['d_mp'] = paired(f'mp_abl_{k}', 'mp_stpt_lite')
D['protocol'] = {m: dict(subject=g(f'mp_{m}'), official=g(f'mpo_official_{m}'), random=g(f'mpo_random_{m}')) for m in ['stpt_lite', 'pointnet', 'pointnet2']}
D['transfer'] = {f'{f}': dict(scratch=g('rh_abl_xyz' if f == 1.0 else f'rh_xyz_scratch_f{f}'), init=g(f'rh_xyz_mpinit_f{f}')) for f in [0.1, 0.25, 1.0]}
for f in ['results/stats.json', 'results/leakage_audit.json', 'results/eda_stats.json', 'results/xai.json']:
    if os.path.exists(f): D[os.path.basename(f)[:-5]] = json.load(open(f))
if os.path.exists('results/robustness_summary.csv'):
    R = pd.read_csv('results/robustness_summary.csv')
    D['robust'] = R.astype(object).where(pd.notna(R), None).to_dict(orient='records')
os.makedirs('presentation', exist_ok=True)
json.dump(D, open('presentation/deck_data.json', 'w'), indent=1, default=float)
print('ok', list(D))
