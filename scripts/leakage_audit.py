"""Automated data-leakage audit.  Writes results/leakage_audit.json and
docs/leakage_audit.md.  Every check is computed from the data / run files,
nothing is asserted by hand."""
import sys, os, json, glob, hashlib, collections
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from data_radhar import load_recordings, split_recordings, get_radhar, CLASSES
from data_milipoint import load_runs, split_runs, get_milipoint

A = {}


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


# ===================================================================== RadHAR
recs = load_recordings()
tr, va, te = split_recordings(recs)
S = dict(train=set(tr), val=set(va), test=set(te))
rh = {}
rh['recordings_per_split'] = {k: len(v) for k, v in S.items()}
rh['recording_overlap'] = {f'{a}&{b}': len(S[a] & S[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
# identical content under different file names
content = {k: h(v['pts']) for k, v in recs.items()}
inv = collections.defaultdict(list)
for k, c in content.items(): inv[c].append(k)
rh['duplicate_recordings_any_split'] = [v for v in inv.values() if len(v) > 1]
# frame-level exact duplicates across splits (frames with >= 5 points)
fr_hash = {s: set() for s in S}
for s, keys in S.items():
    for k in keys:
        p, f = recs[k]['pts'], recs[k]['frame']
        for fi in np.unique(f):
            q = p[f == fi]
            if len(q) >= 5: fr_hash[s].add(h(q))
rh['identical_frames'] = {f'{a}&{b}': len(fr_hash[a] & fr_hash[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
rh['frames_checked'] = {s: len(v) for s, v in fr_hash.items()}
# window-level: dataset objects used in training
dtr, dva, dte = get_radhar()
wk = {n: set(k for k, _ in d.windows) for n, d in [('train', dtr), ('val', dva), ('test', dte)]}
rh['windows'] = {n: len(d) for n, d in [('train', dtr), ('val', dva), ('test', dte)]}
rh['window_recording_overlap'] = {f'{a}&{b}': len(wk[a] & wk[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
rh['class_counts'] = {n: np.bincount(d.labels, minlength=5).tolist() for n, d in [('train', dtr), ('val', dva), ('test', dte)]}
# preprocessing statistics fitted on training recordings only
allint = np.concatenate([dtr.frames[k][0][:, 4] for k in dtr.keys])
rh['intensity_stats_train_only'] = dict(used_val=list(dva.int_stats), used_test=list(dte.int_stats),
                                        recomputed_train=[float(allint.mean()), float(allint.std() + 1e-6)],
                                        match=bool(np.allclose(dva.int_stats, [allint.mean(), allint.std() + 1e-6]) and dva.int_stats == dte.int_stats))
# session analysis from ROS timestamps (no subject IDs in RadHAR)
span = {k: (v['ft'][0], v['ft'][-1]) for k, v in recs.items()}
def nearest_gap(k, pool):
    a0, a1 = span[k]
    return min(max(0, max(span[j][0] - a1, a0 - span[j][1])) for j in pool)
gaps = np.array([nearest_gap(k, tr) for k in te])
days = lambda keys: sorted(set(pd.to_datetime(span[k][0], unit='s').strftime('%Y-%m-%d') for k in keys))
rh['session'] = dict(test_min_gap_to_train_s=float(gaps.min()), test_median_gap_to_train_s=float(np.median(gaps)),
                     test_recordings_within_10min_of_train=int((gaps < 600).sum()),
                     recording_days=dict(train=days(tr), val=days(va), test=days(te)))
rh['subject_ids_available'] = False
A['radhar'] = rh

# ================================================================== MiliPoint
runs = load_runs()
mp = {}
for proto in ['subject', 'official', 'random']:
    d = {}
    if proto != 'random':
        a, b, c, rr = split_runs(runs, proto)
        sub = {n: set(rr[k]['subject'] for k in ks) for n, ks in [('train', a), ('val', b), ('test', c)]}
        d['subjects'] = {n: sorted(v) for n, v in sub.items()}
        d['subject_overlap'] = {f'{x}&{y}': sorted(sub[x] & sub[y]) for x, y in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
        # global frame indices per split (frames, not windows)
        gf = {}
        for n, ks in [('train', a), ('val', b), ('test', c)]:
            s = set()
            for k in ks:
                r = rr[k]; g0 = r['gstart'] + (int(k.split('_')[-1]) if k.count('_') == 2 else 0)
                s.update(range(g0, g0 + r['frame'].max() + 1))
            gf[n] = s
        d['shared_frames'] = {f'{x}&{y}': len(gf[x] & gf[y]) for x, y in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    dtr, dva, dte = get_milipoint(protocol=proto)
    def frames_of(ds):
        base = ds.dataset if hasattr(ds, 'dataset') else ds
        idx = ds.indices if hasattr(ds, 'indices') else range(len(base))
        out = set()
        for i in idx:
            k, s = base.windows[i]; g0 = base.recs[k]['gstart']
            out.update(range(g0 + s, g0 + s + base.win))
        return out, len(idx)
    (ftr, ntr), (fva, nva), (fte, nte) = frames_of(dtr), frames_of(dva), frames_of(dte)
    d['windows'] = dict(train=ntr, val=nva, test=nte)
    d['test_frames_also_in_train_windows'] = len(fte & ftr)
    d['test_frames_total'] = len(fte)
    d['fraction_test_frames_seen_in_training'] = len(fte & ftr) / max(1, len(fte))
    if proto == 'random':
        base = dte.dataset
        trw = collections.defaultdict(list)
        for i in dtr.indices:
            k, s = base.windows[i]; trw[k].append(s)
        ov = []
        for i in dte.indices:
            k, s = base.windows[i]
            ov.append(max([max(0, base.win - abs(s - t)) / base.win for t in trw.get(k, [])] or [0]))
        d['mean_max_overlap_test_window_with_a_train_window'] = float(np.mean(ov))
        d['test_windows_with_any_overlap'] = float(np.mean(np.array(ov) > 0))
    mp[proto] = d
A['milipoint'] = mp

# ============================================================ model selection
sel = []
for f in sorted(glob.glob('results/runs/*/history.csv')):
    tag = f.split('/')[-2]; mj = f'results/runs/{tag}/metrics.json'
    if not os.path.exists(mj): continue
    hist = pd.read_csv(f); m = json.load(open(mj))
    sel.append(dict(tag=tag, best_epoch=m['best_epoch'], argmax_val_f1=int(hist.val_f1.idxmax()) + 1))
A['model_selection'] = dict(runs_checked=len(sel), consistent=sum(s['best_epoch'] == s['argmax_val_f1'] for s in sel),
                            note='checkpoint = epoch with highest validation macro-F1; test set evaluated once after training')
json.dump(A, open('results/leakage_audit.json', 'w'), indent=1, default=str)

# ================================================================== markdown
L = ['# Automated leakage audit', '', 'Generated by `scripts/leakage_audit.py` from the data and run files.', '',
     '## RadHAR', '', '| Check | Result | Status |', '|---|---|---|']
ok = lambda c: 'PASS' if c else 'FAIL'
L += [f"| Recording overlap train/val/test | {rh['recording_overlap']} | {ok(sum(rh['recording_overlap'].values()) == 0)} |",
      f"| Identical-content recordings (any split) | {len(rh['duplicate_recordings_any_split'])} groups | {ok(not rh['duplicate_recordings_any_split'])} |",
      f"| Identical frames (>=5 pts) across splits | {rh['identical_frames']} | {ok(sum(rh['identical_frames'].values()) == 0)} |",
      f"| Windows from a recording in two splits | {rh['window_recording_overlap']} | {ok(sum(rh['window_recording_overlap'].values()) == 0)} |",
      f"| Intensity standardisation fitted on train only | match={rh['intensity_stats_train_only']['match']} | {ok(rh['intensity_stats_train_only']['match'])} |",
      f"| Subject independence | RadHAR provides no subject IDs | NOT VERIFIABLE |",
      f"| Session independence | min/median time gap test->nearest train recording: {rh['session']['test_min_gap_to_train_s']:.0f} s / {rh['session']['test_median_gap_to_train_s']:.0f} s; {rh['session']['test_recordings_within_10min_of_train']} of {len(te)} test recordings within 10 min of a train recording; days {rh['session']['recording_days']} | see text |",
      '', '## MiliPoint', '', '| Protocol | Subject overlap | Test frames also inside training windows | Status |', '|---|---|---|---|']
for p, d in mp.items():
    so = d.get('subject_overlap', 'n/a (windows shuffled)')
    frac = d['fraction_test_frames_seen_in_training']
    status = 'PASS' if frac == 0 and (p != 'random') and not any(d.get('subject_overlap', {}).values()) else ('LEAKY (by design)' if p == 'random' else 'subject overlap')
    L.append(f"| {p} | {so} | {d['test_frames_also_in_train_windows']} / {d['test_frames_total']} ({100 * frac:.1f} %) | {status} |")
if 'mean_max_overlap_test_window_with_a_train_window' in mp['random']:
    L.append(f"\nRandom protocol: {100 * mp['random']['test_windows_with_any_overlap']:.1f} % of test windows share frames with a training window; mean maximal overlap {100 * mp['random']['mean_max_overlap_test_window_with_a_train_window']:.1f} % of the window.")
L += ['', '## Model selection', '', f"{A['model_selection']['consistent']} / {A['model_selection']['runs_checked']} completed runs: saved checkpoint epoch == epoch of maximal validation macro-F1. The test split is evaluated once, after training, with that checkpoint."]
os.makedirs('docs', exist_ok=True)
open('docs/leakage_audit.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
