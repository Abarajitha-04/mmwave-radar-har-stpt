"""RadHAR windowing and leakage-free splits.

A sample is a 2-s window (60 radar frames, as in the original RadHAR protocol)
taken with a 10-frame stride from ONE recording.  The window is cut into G
temporal segments; from each segment we sample P points (with replacement when
the segment has fewer than P points), giving a tensor [G, P, C].

Pre-processing: (1) ROI -- drop antenna self-returns (x < 0.3 m);
(2) person centring -- median of moving (non-zero Doppler) points;
(3) spatial gating -- drop returns > 1.2 m (horizontal) from the person, which
removes static multipath/clutter clusters (about half of all returns).

Point features (C = 6):  x, y, z (person-centred), radial velocity (Doppler),
intensity (train-standardised), normalised time inside the window.

Splits
- test: the official RadHAR test folder (separate recording sessions).
- val:  20 % of TRAIN RECORDINGS per class (fixed seed), not 20 % of windows;
  capped at 55 windows per recording so each recording weighs equally.
  Windows overlap by 50/60 frames, so a window-level split (as in the original
  code) leaks near-duplicate samples into validation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ['boxing', 'jack', 'jump', 'squats', 'walk']
WIN, STRIDE = 60, 10


def load_recordings(path='data/processed/radhar_frames.npy'):
    return np.load(path, allow_pickle=True).item()


def split_recordings(recs, val_frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for c in range(len(CLASSES)):
        keys = sorted(k for k, v in recs.items() if v['split'] == 'Train' and v['label'] == c)
        rng.shuffle(keys)
        nv = int(round(val_frac * len(keys)))
        val += keys[:nv]; train += keys[nv:]
    test = sorted(k for k, v in recs.items() if v['split'] == 'Test')
    return sorted(train), sorted(val), test


def roi_filter(pts):
    """Remove radar self-returns very close to the antenna (< 0.3 m in front).
    Kept deliberately mild; the effect is evaluated in the ablation."""
    return pts[:, 0] > 0.3


def make_windows(recs, keys, win=WIN, stride=STRIDE, max_per_rec=None):
    out = []  # (key, start_frame)
    for k in keys:
        nf = recs[k]['frame'].max() + 1
        starts = list(range(0, nf - win + 1, stride))
        if max_per_rec and len(starts) > max_per_rec:  # evenly spaced subset
            starts = [starts[int(i)] for i in np.linspace(0, len(starts) - 1, max_per_rec)]
        out += [(k, s) for s in starts]
    return out


class RadHARWindows(Dataset):
    def __init__(self, recs, keys, G=12, P=32, train=False, feats='full', roi=True,
                 center=True, int_stats=None, win=WIN, stride=STRIDE, seed=0, label_frac=1.0, gate=None,
                 max_per_rec=None, aug='std'):
        self.recs, self.G, self.P, self.train, self.gate, self.aug = recs, G, P, train, gate, aug
        self.feats, self.center, self.win = feats, center, win
        if label_frac < 1.0:  # label-efficiency study: keep a fraction of recordings per class
            rng = np.random.RandomState(1000 + seed)
            kk = []
            for c in range(len(CLASSES)):
                ck = [k for k in keys if recs[k]['label'] == c]
                rng.shuffle(ck); kk += ck[:max(1, int(round(label_frac * len(ck))))]
            keys = sorted(kk)
        self.keys = keys
        # per-recording point arrays grouped by frame
        self.frames = {}
        for k in keys:
            r = recs[k]
            pts, fr = r['pts'], r['frame']
            if roi:
                m = roi_filter(pts); pts, fr = pts[m], fr[m]
            nf = r['frame'].max() + 1
            order = np.argsort(fr, kind='stable')
            pts, fr = pts[order], fr[order]
            bounds = np.searchsorted(fr, np.arange(nf + 1))
            self.frames[k] = (pts, bounds)
        self.windows = make_windows(recs, keys, win, stride, max_per_rec)
        self.labels = np.array([recs[k]['label'] for k, _ in self.windows])
        if int_stats is None:
            allint = np.concatenate([self.frames[k][0][:, 4] for k in keys])
            int_stats = (float(allint.mean()), float(allint.std() + 1e-6))
        self.int_stats = int_stats
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.windows)

    def raw_window(self, i):
        k, s = self.windows[i]
        pts, b = self.frames[k]
        seg_len = self.win / self.G
        segs = []
        for g in range(self.G):
            f0, f1 = s + int(round(g * seg_len)), s + int(round((g + 1) * seg_len))
            p = pts[b[f0]:b[f1]]
            fidx = np.concatenate([np.full(b[f + 1] - b[f], f - s) for f in range(f0, f1)]) if len(p) else np.zeros(0)
            segs.append((p, fidx))
        return segs

    def person_centre(self, segs):
        """Robust person location for the window: median of the moving points
        (non-zero Doppler -- static clutter such as walls and furniture has zero
        radial velocity), falling back to the median of all points."""
        allp = np.concatenate([p for p, _ in segs]) if any(len(p) for p, _ in segs) else np.zeros((0, 5), np.float32)
        if len(allp) == 0:
            return None
        mov = allp[np.abs(allp[:, 3]) > 0]
        return np.median(mov[:, :3] if len(mov) else allp[:, :3], 0).astype(np.float32)

    def __getitem__(self, i):
        segs = self.raw_window(i)
        G, P = self.G, self.P
        ctr = self.person_centre(segs) if (self.center or self.gate) else None
        if self.gate and ctr is not None:  # spatial gating: keep returns within `gate` m (horizontal) of the person
            segs = [(p[np.linalg.norm(p[:, :2] - ctr[:2], axis=1) <= self.gate],
                     f[np.linalg.norm(p[:, :2] - ctr[:2], axis=1) <= self.gate]) if len(p) else (p, f)
                    for p, f in segs]
        out = np.zeros((G, P, 6), np.float32)
        mask = np.zeros((G, P), np.float32)
        rng = self.rng if self.train else np.random.RandomState(i)
        for g, (p, fidx) in enumerate(segs):
            n = len(p)
            if n == 0:
                continue
            idx = rng.choice(n, P, replace=n < P)
            q = p[idx]
            out[g, :, :5] = q
            out[g, :, 5] = fidx[idx] / (self.win - 1)
            mask[g] = 1.0
        valid = mask > 0
        if self.center and ctr is not None:
            out[..., :3] -= ctr
        out[..., 4] = (out[..., 4] - self.int_stats[0]) / self.int_stats[1]
        if self.train and self.aug != 'none':
            # geometric augmentation: small rotation about the vertical axis + scaling,
            # and (aug='std') 1-cm coordinate jitter.  Point re-sampling every epoch is
            # always on (it is how a window is turned into a fixed-size tensor).
            th = rng.uniform(-np.pi / 12, np.pi / 12)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
            out[..., :2] = out[..., :2] @ R.T
            out[..., :3] *= rng.uniform(0.9, 1.1)
            if self.aug == 'std':
                out[..., :3] += rng.normal(0, 0.01, out[..., :3].shape).astype(np.float32)
        out[~valid] = 0
        if self.feats == 'xyz':
            out[..., 3:5] = 0
        return torch.from_numpy(out), torch.from_numpy(mask), int(self.labels[i])


def get_radhar(G=12, P=32, feats='full', roi=True, center=True, seed=0, label_frac=1.0,
               path='data/processed/radhar_frames.npy', train_stride=20, gate=1.2, aug='std'):
    recs = load_recordings(path)
    tr, va, te = split_recordings(recs)
    dtr = RadHARWindows(recs, tr, G, P, True, feats, roi, center, seed=seed, label_frac=label_frac,
                        stride=train_stride, gate=gate, aug=aug)
    kw = dict(G=G, P=P, train=False, feats=feats, roi=roi, center=center, int_stats=dtr.int_stats, gate=gate)
    # validation: at most 55 windows (= one standard 20-s recording) per recording, so
    # the few long (120-190 s) recordings cannot dominate model selection
    return dtr, RadHARWindows(recs, va, max_per_rec=55, **kw), RadHARWindows(recs, te, **kw)
