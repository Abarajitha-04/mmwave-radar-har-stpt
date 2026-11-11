"""MiliPoint action-classification windows (49 actions, 11 subjects, 24 FPS).

Raw frames are grouped into contiguous runs of one action inside one session
file.  A sample is a 2-s window (48 frames) inside a run, so a window never
mixes two actions.  Points are xyz only; axes are remapped to the RadHAR
convention (x forward, y lateral, z up) so both datasets share one frame for
cross-dataset transfer.  Velocity/intensity channels are zero.

Splits
- 'subject' (primary, leakage-free): test subjects 8, 9, 10; val subjects 6, 7;
  train the remaining six subjects.  No person appears in two splits.
- 'official': the MiliPoint code's protocol -- labelled frames concatenated in
  file order and cut 80/10/10 chronologically -- kept only to compare with the
  numbers published in the MiliPoint paper.
"""
import json, os
import numpy as np
import torch
from data_radhar import RadHARWindows
import safe_pickle

RAW = os.environ.get('MILIPOINT_RAW', 'ext/milipoint_raw')
ID_JSON = os.environ.get('MILIPOINT_IDS', 'ext/MiliPoint/data/raw/id.json')
CACHE = 'data/processed/milipoint_runs.npy'
WIN_MP = 48
SUBJ_SPLIT = dict(test=[8, 9, 10], val=[6, 7])


def build_runs(raw=RAW, id_json=ID_JSON, out=CACHE):
    lab = np.load(f'{raw}/action_label.npy')
    subj_of = {f: int(s) for s, fs in json.load(open(id_json)).items() for f in fs}
    runs, off, gidx = {}, 0, 0
    for fi in range(19):
        d = safe_pickle.load(f'{raw}/{fi}.pkl')
        L = lab[off:off + len(d)]
        gpos = np.arange(off, off + len(d))
        off += len(d)
        i = 0
        while i < len(d):
            if L[i] < 0:
                i += 1; continue
            j = i
            while j < len(d) and L[j] == L[i]:
                j += 1
            pts, fr = [], []
            for t in range(i, j):
                p = d[t]['x'].astype(np.float32)
                p = p[:, [1, 0, 2]]  # MiliPoint (lateral, forward, up) -> (forward, lateral, up)
                pts.append(np.concatenate([p, np.zeros((len(p), 2), np.float32)], 1))
                fr.append(np.full(len(p), t - i, np.int32))
            key = f'f{fi:02d}_r{gidx:04d}'
            runs[key] = dict(pts=np.concatenate(pts), frame=np.concatenate(fr), label=int(L[i]),
                             subject=subj_of[fi], file=fi, gstart=int(gpos[i]), gend=int(gpos[j - 1]) + 1)
            gidx += 1
            i = j
    np.save(out, runs, allow_pickle=True)
    return runs


def load_runs(path=CACHE):
    if not os.path.exists(path):
        return build_runs(out=path)
    return np.load(path, allow_pickle=True).item()


def split_runs(runs, protocol='subject'):
    if protocol == 'subject':
        te = sorted(k for k, v in runs.items() if v['subject'] in SUBJ_SPLIT['test'])
        va = sorted(k for k, v in runs.items() if v['subject'] in SUBJ_SPLIT['val'])
        tr = sorted(k for k in runs if k not in te and k not in va)
        return tr, va, te, runs
    # 'official': chronological 80/10/10 over labelled frames (runs cut at boundaries)
    keys = sorted(runs, key=lambda k: runs[k]['gstart'])
    n = sum(runs[k]['frame'].max() + 1 for k in keys)
    b1, b2 = 0.8 * n, 0.9 * n
    tr, va, te, new, c = [], [], [], {}, 0
    for k in keys:
        r = runs[k]; nf = r['frame'].max() + 1
        cuts = sorted({0, nf, *[int(b - c) for b in (b1, b2) if c < b < c + nf]})
        for a, b in zip(cuts[:-1], cuts[1:]):
            m = (r['frame'] >= a) & (r['frame'] < b)
            kk = f'{k}_{a}'
            new[kk] = dict(r, pts=r['pts'][m], frame=r['frame'][m] - a)
            mid = c + a
            (tr if mid < b1 else va if mid < b2 else te).append(kk)
        c += nf
    return tr, va, te, new


class _Sub(torch.utils.data.Subset):
    @property
    def labels(self):
        return self.dataset.labels[self.indices]


def get_milipoint(G=12, P=32, protocol='subject', seed=0, label_frac=1.0, train_stride=24, eval_stride=12, aug='std'):
    runs = load_runs()
    if protocol == 'random':
        # deliberately LEAKY protocol used by some prior work: overlapping windows from
        # all subjects are shuffled and split 80/10/10, so test windows share frames
        # with training windows.  Used only to quantify the inflation.
        keys = sorted(runs)
        kw = dict(G=G, P=P, feats='xyz', roi=False, center=True, win=WIN_MP, stride=eval_stride)
        dtr_all = RadHARWindows(runs, keys, train=True, seed=seed, aug=aug, **kw)
        dev_all = RadHARWindows(runs, keys, train=False, int_stats=dtr_all.int_stats, **kw)
        idx = np.random.RandomState(0).permutation(len(dtr_all))
        a, b = int(0.8 * len(idx)), int(0.9 * len(idx))
        tr = idx[:a][::max(1, train_stride // eval_stride)]  # same training-set size as other protocols
        return _Sub(dtr_all, tr), _Sub(dev_all, idx[a:b]), _Sub(dev_all, idx[b:])
    tr, va, te, runs = split_runs(runs, protocol)
    kw = dict(G=G, P=P, feats='xyz', roi=False, center=True, win=WIN_MP)
    dtr = RadHARWindows(runs, tr, train=True, stride=train_stride, seed=seed, label_frac=label_frac, aug=aug, **kw)
    dva = RadHARWindows(runs, va, train=False, stride=eval_stride, int_stats=dtr.int_stats, **kw)
    dte = RadHARWindows(runs, te, train=False, stride=eval_stride, int_stats=dtr.int_stats, **kw)
    return dtr, dva, dte
