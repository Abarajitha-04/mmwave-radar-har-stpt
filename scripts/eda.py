"""Exploratory data analysis + pre-processing figures for both datasets."""
import sys, json, collections
sys.path.insert(0, 'src')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_radhar import load_recordings, split_recordings, CLASSES, RadHARWindows, roi_filter
from data_milipoint import load_runs, split_runs, SUBJ_SPLIT

plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 200})
C5 = ['#2f6690', '#3a7d44', '#d1495b', '#edae49', '#7b5ea7']
F = 'figures/'
stats = {}

# ------------------------------------------------------------- RadHAR
recs = load_recordings()
tr, va, te = split_recordings(recs)
ppf = {c: [] for c in range(5)}; vel = {c: [] for c in range(5)}; xs = []
roi_removed = []
for k, v in recs.items():
    n = np.bincount(v['frame']); ppf[v['label']] += list(n)
    vel[v['label']].append(v['pts'][:, 3]); xs.append(v['pts'][:, :3])
    roi_removed.append(1 - roi_filter(v['pts']).mean())
xs = np.concatenate(xs)
stats['radhar'] = dict(
    recordings={s: len(x) for s, x in zip(['train', 'val', 'test'], [tr, va, te])},
    pts_per_frame_mean=float(np.mean(sum(ppf.values(), []))),
    pts_per_frame_median=float(np.median(sum(ppf.values(), []))),
    roi_removed_frac=float(np.mean(roi_removed)),
    frame_rate_hz=float(np.median([1 / np.median(np.diff(v['ft'])) * 1 for v in list(recs.values())[:20]])),
    xyz_min=xs.min(0).tolist(), xyz_max=xs.max(0).tolist())

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
for c in range(5):
    ax[0].hist(ppf[c], bins=np.arange(0, 45, 2), histtype='step', color=C5[c], label=CLASSES[c], lw=1.2, density=True)
    v = np.concatenate(vel[c]); v = v[v != 0]
    ax[1].hist(v, bins=np.linspace(-1.3, 1.3, 40), histtype='step', color=C5[c], lw=1.2, density=True)
ax[0].set_xlabel('points per frame'); ax[0].set_ylabel('density'); ax[0].legend(frameon=False, fontsize=6)
ax[1].set_xlabel('radial velocity (m/s, non-zero)')
from matplotlib.colors import LogNorm
ax[2].hist2d(xs[:, 0], xs[:, 2], bins=80, range=[[0, 5], [-2, 2.5]], cmap='Blues', norm=LogNorm())
ax[2].set_xlabel('x / range (m)'); ax[2].set_ylabel('z / height (m)')
ax[0].set_title('(a) sparsity', fontsize=8); ax[1].set_title('(b) Doppler', fontsize=8); ax[2].set_title('(c) spatial density', fontsize=8)
fig.tight_layout(); fig.savefig(F + 'eda_radhar.pdf'); fig.savefig(F + 'eda_radhar.png'); plt.close(fig)

# example windows, one per class, coloured by time
ds = RadHARWindows(recs, te, train=False)
fig = plt.figure(figsize=(7.2, 1.8))
for c in range(5):
    i = int(np.where(ds.labels == c)[0][len(np.where(ds.labels == c)[0]) // 2])
    x, m, _ = ds[i]; x = x.numpy().reshape(-1, 6); m = m.numpy().reshape(-1) > 0; x = x[m]
    a = fig.add_subplot(1, 5, c + 1, projection='3d')
    a.scatter(x[:, 1], x[:, 0], x[:, 2], c=x[:, 5], cmap='viridis', s=1.5)
    a.set_title(CLASSES[c], fontsize=8); a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
    a.view_init(15, -60)
fig.tight_layout(); fig.savefig(F + 'radhar_examples.pdf'); fig.savefig(F + 'radhar_examples.png'); plt.close(fig)

# pre-processing before / after for one window
ds_g = RadHARWindows(recs, te, train=False, gate=1.2)
wi = len(ds_g.windows) // 3
k, s0 = ds_g.windows[wi]
raw = recs[k]['pts'][(recs[k]['frame'] >= s0) & (recs[k]['frame'] < s0 + 60)]
roi = roi_filter(raw); kept = raw[roi]
ctr = ds_g.person_centre(ds_g.raw_window(wi))
gated = np.linalg.norm(kept[:, :2] - ctr[:2], axis=1) <= 1.2
x, m, _ = ds_g[wi]; x = x.numpy().reshape(-1, 6)[m.numpy().reshape(-1) > 0]
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.4))
ax[0].scatter(raw[~roi, 0], raw[~roi, 2], s=5, c='#d1495b', label='antenna self-return (ROI)')
ax[0].scatter(kept[~gated, 0], kept[~gated, 2], s=3, c='#bbbbbb', label='static clutter (gated)')
ax[0].scatter(kept[gated, 0], kept[gated, 2], s=3, c='#2f6690', label='person returns')
mv = kept[np.abs(kept[:, 3]) > 0]
ax[0].scatter(mv[:, 0], mv[:, 2], s=6, facecolors='none', edgecolors='k', lw=0.4, label='moving (Doppler \u2260 0)')
ax[0].axvline(ctr[0], ls='--', c='k', lw=0.6)
ax[0].set_title(f'raw 2-s window ({len(raw)} returns)', fontsize=8); ax[0].set_xlabel('x / range (m)'); ax[0].set_ylabel('z (m)')
ax[0].legend(frameon=False, fontsize=5.5, loc='lower right')
sc = ax[1].scatter(x[:, 0], x[:, 2], s=3, c=x[:, 5], cmap='viridis')
ax[1].set_title('model input: 12 segments x 32 pts, person-centred', fontsize=8); ax[1].set_xlabel('x (m)')
plt.colorbar(sc, ax=ax[1], label='time in window')
fig.tight_layout(); fig.savefig(F + 'preproc_before_after.pdf'); fig.savefig(F + 'preproc_before_after.png'); plt.close(fig)
stats['radhar']['example_window'] = dict(raw=int(len(raw)), roi_removed=int((~roi).sum()), gated_removed=int((~gated).sum()), kept=int(gated.sum()))
gate_frac = []
for wi2 in range(0, len(ds_g.windows), 15):
    sg = ds_g.raw_window(wi2); c2 = ds_g.person_centre(sg)
    pa = np.concatenate([p for p, _ in sg])
    gate_frac.append((np.linalg.norm(pa[:, :2] - c2[:2], axis=1) > 1.2).mean())
stats['radhar']['gate_removed_frac_after_roi'] = float(np.mean(gate_frac))

# split sizes
dtr = RadHARWindows(recs, tr, train=False, stride=20); dva = RadHARWindows(recs, va); dte = ds
stats['radhar']['windows'] = {n: np.bincount(d.labels, minlength=5).tolist() for n, d in
                              zip(['train(stride20)', 'val', 'test'], [dtr, dva, dte])}

# ------------------------------------------------------------- MiliPoint
runs = load_runs()
mtr, mva, mte, _ = split_runs(runs, 'subject')
ppf_mp = np.concatenate([np.bincount(v['frame']) for v in runs.values()])
lab_frames = collections.Counter()
subj_frames = collections.Counter()
for v in runs.values():
    nf = v['frame'].max() + 1; lab_frames[v['label']] += nf; subj_frames[v['subject']] += nf
stats['milipoint'] = dict(runs=len(runs), labelled_frames=int(sum(lab_frames.values())),
                          pts_per_frame_mean=float(ppf_mp.mean()), pts_per_frame_median=float(np.median(ppf_mp)),
                          frames_per_subject={int(k): int(v) for k, v in sorted(subj_frames.items())},
                          split_runs=dict(train=len(mtr), val=len(mva), test=len(mte)), subject_split=SUBJ_SPLIT)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1))
ax[0].hist(ppf_mp, bins=np.arange(0, 25, 1), color='#2f6690'); ax[0].set_xlabel('points per frame'); ax[0].set_ylabel('frames')
ax[0].set_title('(a) sparsity', fontsize=8)
ks = sorted(lab_frames); ax[1].bar(ks, [lab_frames[k] / 24 for k in ks], color='#3a7d44', width=0.8)
ax[1].set_xlabel('action id'); ax[1].set_ylabel('seconds'); ax[1].set_title('(b) labelled time per action', fontsize=8)
ss = sorted(subj_frames)
col = ['#d1495b' if s in SUBJ_SPLIT['test'] else '#edae49' if s in SUBJ_SPLIT['val'] else '#2f6690' for s in ss]
ax[2].bar(ss, [subj_frames[s] / 24 / 60 for s in ss], color=col); ax[2].set_xlabel('subject'); ax[2].set_ylabel('minutes')
ax[2].set_title('(c) subject split: train/val/test', fontsize=8)
fig.tight_layout(); fig.savefig(F + 'eda_milipoint.pdf'); fig.savefig(F + 'eda_milipoint.png'); plt.close(fig)

json.dump(stats, open('results/eda_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))
