"""Controlled radar-noise perturbations on model inputs.

All functions act on a batch  x [B,G,P,C] (person-centred points, C = 6:
x, y, z, Doppler, intensity, time) and mask [B,G,P], take an explicit
torch.Generator, and return NEW tensors, so the same seed gives identical
perturbed test samples for every model (paired comparison).

Used in two places:
  * evaluation  -- `perturb(x, m, condition, level, g)` (robustness study, demo)
  * training    -- `robust_augment(x, m, g)` (noise-robust training variant)
"""
import torch

CONDITIONS = {
    # name: (levels, unit)
    'clean': ([0], ''),
    'jitter': ([0.02, 0.05, 0.10], 'sigma (m)'),           # random coordinate displacement
    'dropout': ([0.05, 0.15, 0.30], 'fraction dropped'),    # missing points (slots become empty)
    'sparsity': ([0.50, 0.25, 0.125], 'fraction kept'),     # fewer unique returns, re-sampled to P
    'outliers': ([0.05, 0.10, 0.20], 'fraction replaced'),  # false detections / clutter
    'density': ([0.5, 0.25, 0.1], 'min keep fraction'),     # variable density per segment
    'frameloss': ([0.25, 0.50], 'fraction of segments lost'),
    'combined': ([1, 2, 3], 'severity'),
}


def _rand(shape, g):
    return torch.rand(shape, generator=g)


def jitter(x, m, sigma, g):
    x = x.clone()
    x[..., :3] += torch.randn(x[..., :3].shape, generator=g) * sigma
    x[m == 0] = 0
    return x, m.clone()


def dropout(x, m, frac, g):
    """Remove a fraction of returns: their slots are masked out (variable count)."""
    keep = (_rand(m.shape, g) >= frac).float()
    m2 = m * keep
    # keep at least one point per non-empty segment
    empty = (m2.sum(-1) == 0) & (m.sum(-1) > 0)
    m2[..., 0][empty] = m[..., 0][empty]
    x2 = x.clone(); x2[m2 == 0] = 0
    return x2, m2


def _resample(x, m, keep_frac, g):
    """Keep `keep_frac` (scalar or [B,G]) of the P sampled returns and re-sample
    them (with replacement) back to P slots: fewer UNIQUE returns, same tensor size."""
    B, G, P, C = x.shape
    kf = keep_frac if torch.is_tensor(keep_frac) else torch.full((B, G), float(keep_frac))
    k = (kf * P).round().clamp(min=1).long()                     # [B,G]
    order = _rand((B, G, P), g).argsort(-1)                        # random subset order
    pick = (_rand((B, G, P), g) * k.unsqueeze(-1)).long()          # index into first k of order
    idx = order.gather(-1, pick)
    x2 = x.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, C))
    return x2, m.clone()


def sparsity(x, m, keep, g):
    return _resample(x, m, keep, g)


def density(x, m, min_keep, g):
    B, G = m.shape[:2]
    kf = min_keep + (1 - min_keep) * _rand((B, G), g)
    return _resample(x, m, kf, g)


def outliers(x, m, frac, g, extent=1.5):
    """Replace a fraction of returns by false detections uniformly distributed in a
    +-extent m box around the person (zero Doppler, random intensity/time)."""
    x2, m2 = x.clone(), m.clone()
    sel = (_rand(m.shape, g) < frac) & (m > 0)
    n = int(sel.sum())
    if n:
        fake = torch.zeros(n, x.shape[-1])
        fake[:, :3] = (_rand((n, 3), g) * 2 - 1) * extent
        fake[:, 4] = torch.randn(n, generator=g)        # standardised intensity
        fake[:, 5] = _rand((n,), g)
        if (x[..., 3:5] == 0).all():                   # xyz-only input keeps zeros
            fake[:, 3:5] = 0
        x2[sel] = fake
    return x2, m2


def frameloss(x, m, frac, g):
    B, G = m.shape[:2]
    k = int(round(frac * G))
    if k == 0:
        return x.clone(), m.clone()
    drop = _rand((B, G), g).argsort(-1)[:, :k]
    dm = torch.zeros(B, G).scatter_(1, drop, 1).bool()
    x2, m2 = x.clone(), m.clone()
    x2[dm] = 0; m2[dm] = 0
    return x2, m2


def combined(x, m, sev, g):
    s = {1: (0.02, 0.05, 0.05, 0.5), 2: (0.05, 0.15, 0.10, 0.25), 3: (0.10, 0.30, 0.20, 0.1)}[int(sev)]
    x, m = density(x, m, s[3], g)
    x, m = jitter(x, m, s[0], g)
    x, m = outliers(x, m, s[2], g)
    x, m = dropout(x, m, s[1], g)
    return x, m


FN = dict(jitter=jitter, dropout=dropout, sparsity=sparsity, outliers=outliers, density=density,
          frameloss=frameloss, combined=combined)


def perturb(x, m, condition, level, g):
    if condition == 'clean':
        return x.clone(), m.clone()
    return FN[condition](x, m, level, g)


def robust_augment(x, m, g, p=0.5):
    """Training-time noise augmentation (applied per batch, each component with
    probability p): variable density, point dropout, false-detection injection and
    segment loss.  Levels are drawn from ranges that only partly overlap the test
    severities, and the test perturbation seeds are never used for training."""
    if _rand((1,), g).item() < p:
        x, m = density(x, m, 0.3, g)
    if _rand((1,), g).item() < p:
        x, m = dropout(x, m, 0.2 * _rand((1,), g).item(), g)
    if _rand((1,), g).item() < p:
        x, m = outliers(x, m, 0.1 * _rand((1,), g).item(), g)
    if _rand((1,), g).item() < 0.25:
        x, m = frameloss(x, m, 1 / 12, g)
    return x, m
