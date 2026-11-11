"""Reproducible inference: load an exported model bundle and classify raw input.

A bundle directory (created by scripts/export_model.py) contains
  model.pt       -- state_dict
  bundle.json    -- model arguments, class names, pre-processing settings and the
                    training-set intensity statistics (fitted on training data only)
  metrics.json   -- the evaluation record of the exported run
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from models import build
from data_radhar import RadHARWindows
from radhar_parse import parse_text
import noise


class Bundle:
    def __init__(self, path):
        self.path = path
        self.cfg = json.load(open(os.path.join(path, 'bundle.json')))
        a = self.cfg['model_args']
        self.model = build(a['model'], len(self.cfg['classes']), **a.get('kwargs', {}))
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location='cpu'))
        self.model.eval()
        self.classes = self.cfg['classes']
        self.pre = self.cfg['preprocessing']

    # ---------------------------------------------------------------- inputs
    def windows_from_frames(self, pts, frame, stride=None):
        """pts [N,5] (x,y,z,Doppler,intensity), frame [N] -> (x [W,G,P,6], mask, starts)."""
        if pts.ndim != 2 or pts.shape[1] != 5:
            raise ValueError(f'expected points of shape [N,5], got {pts.shape}')
        if not np.isfinite(pts).all():
            raise ValueError('points contain NaN/inf')
        nf = int(frame.max()) + 1
        win = self.pre['win']
        if nf < win:
            raise ValueError(f'recording has {nf} frames; at least {win} ({self.pre["win_s"]} s) are needed')
        rec = {'input': dict(pts=pts.astype(np.float32), frame=frame.astype(np.int32), label=-1)}
        ds = RadHARWindows(rec, ['input'], G=self.pre['G'], P=self.pre['P'], train=False,
                           feats=self.pre['feats'], roi=self.pre['roi'], center=self.pre['center'],
                           int_stats=tuple(self.pre['int_stats']), win=win, stride=stride or self.pre['stride'],
                           gate=self.pre['gate'])
        xs, ms = zip(*[ds[i][:2] for i in range(len(ds))])
        return torch.stack(xs), torch.stack(ms), [s for _, s in ds.windows], ds

    def windows_from_radhar_text(self, txt, stride=None):
        pts, frame, _ = parse_text(txt)
        return self.windows_from_frames(pts, frame, stride)

    # ------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict(self, x, m):
        t = time.perf_counter()
        p = F.softmax(self.model(x, m), -1).numpy()
        return p, (time.perf_counter() - t) * 1000

    def predict_with_noise(self, x, m, condition, level, seed=0):
        g = torch.Generator().manual_seed(seed)
        xn, mn = noise.perturb(x, m, condition, level, g)
        p, ms = self.predict(xn, mn)
        return p, ms, xn, mn


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Classify a raw RadHAR point-cloud recording (.txt)')
    ap.add_argument('--bundle', default='checkpoints/stpt_lite_kd_radhar')
    ap.add_argument('--input', required=True)
    ap.add_argument('--stride', type=int, default=None)
    a = ap.parse_args()
    b = Bundle(a.bundle)
    x, m, starts, _ = b.windows_from_radhar_text(open(a.input).read(), a.stride)
    p, ms = b.predict(x, m)
    for s, pp in zip(starts, p):
        print(f'frames {s:5d}-{s + b.pre["win"] - 1:5d}: {b.classes[pp.argmax()]:8s} ({100 * pp.max():.1f} %)')
    vote = np.bincount(p.argmax(1), minlength=len(b.classes))
    print(f'\nrecording-level (majority of {len(p)} windows): {b.classes[vote.argmax()]}; '
          f'{ms / len(p):.2f} ms per window (batch of {len(p)})')


if __name__ == '__main__':
    main()
