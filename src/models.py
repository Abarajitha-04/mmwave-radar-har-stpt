"""Models for mmWave point-cloud HAR.  Every model takes
    x    : [B, G, P, C]  points grouped into G temporal segments (C = 6)
    mask : [B, G, P]     1 for real points, 0 for empty segments
and returns class logits [B, K].  Static point-set baselines flatten the
segments into one set (time is kept as the 6th feature).

Baselines (re-implemented in pure PyTorch, no CUDA ops):
  PointNet [Qi+17], PointNet++ SSG [Qi+17b], DGCNN [Wang+19],
  Point Transformer [Zhao+21], PointNet+BiLSTM (frame-level temporal model,
  as used in mmWave HAR, e.g. [Ding+21]), and the voxel TD-CNN+BiLSTM of
  RadHAR [Singh+19] (computationally reduced, see VoxelTDCNN).
Proposed:  STPT-Lite (see class docstring).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, last_act=True, bn=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            if bn:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def flat(x, mask):
    B, G, P, C = x.shape
    return x.reshape(B, G * P, C), mask.reshape(B, G * P)


def masked_max(f, m, dim):
    return f.masked_fill(m.unsqueeze(-1) == 0, -1e4).max(dim)[0]


# --------------------------------------------------------------------- PointNet
class PointNet(nn.Module):
    def __init__(self, in_ch=6, k=5, width=1.0):
        super().__init__()
        w = lambda c: int(c * width)
        self.feat = mlp([in_ch, w(64), w(128), w(256)])
        self.head = nn.Sequential(nn.Linear(w(256), w(128)), nn.GELU(), nn.Dropout(0.3), nn.Linear(w(128), k))

    def embed(self, x, mask):
        x, m = flat(x, mask)
        return masked_max(self.feat(x), m, 1)

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


# ------------------------------------------------------------------ PointNet++
def fps(xyz, m, n):
    """Farthest point sampling; xyz [B,N,3], m [B,N] -> idx [B,n]."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, n, dtype=torch.long)
    dist = torch.full((B, N), 1e10).masked_fill(m == 0, -1)
    far = torch.argmax(m, 1)  # first valid point
    bi = torch.arange(B)
    for i in range(n):
        idx[:, i] = far
        c = xyz[bi, far].unsqueeze(1)
        d = ((xyz - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d.masked_fill(m == 0, -1))
        far = torch.argmax(dist, 1)
    return idx


def gather(x, idx):
    B = x.shape[0]
    bi = torch.arange(B).view(B, *([1] * (idx.dim() - 1)))
    return x[bi, idx]


def ball_group(xyz, new_xyz, m, radius, k):
    d = torch.cdist(new_xyz, xyz) ** 2
    d = d.masked_fill(m.unsqueeze(1) == 0, 1e10)
    dk, idx = d.topk(k, dim=-1, largest=False)
    first = idx[..., :1].expand_as(idx)
    idx = torch.where(dk > radius ** 2, first, idx)
    return idx


class SA(nn.Module):
    def __init__(self, npoint, radius, k, in_ch, dims):
        super().__init__()
        self.npoint, self.radius, self.k = npoint, radius, k
        self.mlp = mlp([in_ch + 3] + dims)

    def forward(self, xyz, feat, m):
        idx = fps(xyz, m, self.npoint)
        nxyz = gather(xyz, idx)
        g = ball_group(xyz, nxyz, m, self.radius, self.k)
        gxyz = gather(xyz, g) - nxyz.unsqueeze(2)
        gf = torch.cat([gxyz / self.radius, gather(feat, g)], -1)
        return nxyz, self.mlp(gf).max(2)[0], torch.ones(nxyz.shape[:2])


class PointNet2(nn.Module):
    def __init__(self, in_ch=6, k=5):
        super().__init__()
        self.sa1 = SA(128, 0.3, 16, in_ch, [64, 64, 128])
        self.sa2 = SA(32, 0.6, 16, 128, [128, 128, 256])
        self.glob = mlp([256 + 3, 256, 512])
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256, k))

    def embed(self, x, mask):
        x, m = flat(x, mask)
        xyz = x[..., :3]
        xyz1, f1, m1 = self.sa1(xyz, x, m)
        xyz2, f2, _ = self.sa2(xyz1, f1, m1)
        return self.glob(torch.cat([xyz2, f2], -1)).max(1)[0]

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


# ------------------------------------------------------------------------ DGCNN
def knn_idx(x, k, m=None):
    d = torch.cdist(x, x)
    if m is not None:
        d = d.masked_fill(m.unsqueeze(1) == 0, 1e10)
    return d.topk(k, -1, largest=False)[1]


class EdgeConv(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.k = k
        self.mlp = mlp([2 * i, o])

    def forward(self, f, m, idx=None):
        if idx is None:
            idx = knn_idx(f, self.k, m)
        nb = gather(f, idx)
        c = f.unsqueeze(2).expand_as(nb)
        return self.mlp(torch.cat([nb - c, c], -1)).max(2)[0]


class DGCNN(nn.Module):
    def __init__(self, in_ch=6, k=5, kn=16, npts=128):
        super().__init__()
        self.npts = npts
        self.e1, self.e2, self.e3 = EdgeConv(in_ch, 64, kn), EdgeConv(64, 64, kn), EdgeConv(64, 128, kn)
        self.fuse = mlp([256, 512])
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, k))

    def embed(self, x, mask):
        x, m = flat(x, mask)
        if x.shape[1] > self.npts:  # random subset for O(N^2) kNN cost
            sel = torch.randperm(x.shape[1])[:self.npts] if self.training else torch.arange(0, x.shape[1], x.shape[1] // self.npts)[:self.npts]
            x, m = x[:, sel], m[:, sel]
        f1 = self.e1(x, m, knn_idx(x[..., :3], self.e1.k, m))
        f2 = self.e2(f1, m)
        f3 = self.e3(f2, m)
        f = self.fuse(torch.cat([f1, f2, f3], -1))
        return torch.cat([masked_max(f, m, 1), (f * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)], -1)

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


# ------------------------------------------------------------ Point Transformer
class PTBlock(nn.Module):
    """Point Transformer layer [Zhao+21]: vector self-attention over kNN with
    relative positional encoding."""
    def __init__(self, d, k=16):
        super().__init__()
        self.k = k
        self.q, self.kk, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.pos = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, xyz, f, m):
        idx = knn_idx(xyz, self.k, m)
        q = self.q(f).unsqueeze(2)
        kf, vf = gather(self.kk(f), idx), gather(self.v(f), idx)
        rel = self.pos(xyz.unsqueeze(2) - gather(xyz, idx))
        w = torch.softmax(self.att(q - kf + rel), 2)
        return self.norm(f + (w * (vf + rel)).sum(2))


class PointTransformer(nn.Module):
    def __init__(self, in_ch=6, k=5, d=96, npts=128):
        super().__init__()
        self.npts = npts
        self.inp = mlp([in_ch, d, d])
        self.b1 = PTBlock(d)
        self.down = mlp([d + 3, 2 * d])
        self.b2 = PTBlock(2 * d)
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(0.4), nn.Linear(128, k))

    def embed(self, x, mask):
        x, m = flat(x, mask)
        if x.shape[1] > self.npts:
            sel = torch.randperm(x.shape[1])[:self.npts] if self.training else torch.arange(0, x.shape[1], x.shape[1] // self.npts)[:self.npts]
            x, m = x[:, sel], m[:, sel]
        xyz = x[..., :3]
        f = self.b1(xyz, self.inp(x), m)
        idx = fps(xyz, m, 32)
        nxyz = gather(xyz, idx)
        g = knn_idx_between(nxyz, xyz, 16, m)
        gf = torch.cat([gather(xyz, g) - nxyz.unsqueeze(2), gather(f, g)], -1)
        f2 = self.down(gf).max(2)[0]
        f2 = self.b2(nxyz, f2, torch.ones(nxyz.shape[:2]))
        return f2.max(1)[0]

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


def knn_idx_between(q, x, k, m):
    d = torch.cdist(q, x).masked_fill(m.unsqueeze(1) == 0, 1e10)
    return d.topk(k, -1, largest=False)[1]


# ------------------------------------------------------------- PointNet+BiLSTM
class PointNetLSTM(nn.Module):
    """Frame-level temporal baseline: shared PointNet per segment, BiLSTM over time."""
    def __init__(self, in_ch=6, k=5):
        super().__init__()
        self.feat = mlp([in_ch, 64, 128])
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, k))

    def embed(self, x, mask):
        B, G, P, C = x.shape
        f = masked_max(self.feat(x), mask, 2)  # [B,G,128]
        f = f.masked_fill(mask.max(2)[0].unsqueeze(-1) == 0, 0)
        o, _ = self.lstm(f)
        return o.mean(1)

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


# ------------------------------------------------------------ Voxel TD-CNN-LSTM
def voxelize(x, mask, D=10, H=32, W=32, extent=((-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5))):
    """Occupancy-count voxels per segment from (window-centred) points:
    [B,G,P,C] -> [B,G,1,D,H,W].  Same 10x32x32 grid as RadHAR, but fixed metric
    extent instead of per-window min/max (avoids scale leakage)."""
    B, G, P, _ = x.shape
    (x0, x1), (y0, y1), (z0, z1) = extent
    ix = ((x[..., 0] - x0) / (x1 - x0) * H).long().clamp(0, H - 1)
    iy = ((x[..., 1] - y0) / (y1 - y0) * W).long().clamp(0, W - 1)
    iz = ((x[..., 2] - z0) / (z1 - z0) * D).long().clamp(0, D - 1)
    flat_i = ((iz * H + ix) * W + iy)
    vox = torch.zeros(B, G, D * H * W)
    vox.scatter_add_(2, flat_i.view(B, G, P), mask)
    return vox.view(B, G, 1, D, H, W)


class VoxelTDCNN(nn.Module):
    """RadHAR TD-CNN+BiLSTM [Singh+19] on 10x32x32 voxels.  Reduced for CPU
    training: per-segment voxels (G=12 time steps of 5 frames instead of 60
    single frames) and 8/16/16 instead of 32 filters per stage."""
    def __init__(self, in_ch=6, k=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d((1, 4, 4)), nn.Flatten())
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(256, 16, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(32, k))

    def embed(self, x, mask):
        v = voxelize(x, mask)
        B, G = v.shape[:2]
        f = self.cnn(v.flatten(0, 1)).view(B, G, -1)
        o, _ = self.lstm(self.drop(f))
        return o[:, -1]

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


# ----------------------------------------------------------------- STPT-Lite
class SegmentTokenizer(nn.Module):
    """Intra-segment point attention: one full self-attention block over the P
    points of a segment (P is small, so this is cheap), with a relative-position
    bias from an MLP of pairwise 3-D offsets, followed by attentive + max pooling."""
    def __init__(self, in_ch, d, heads=4, use_attn=True):
        super().__init__()
        self.use_attn, self.h = use_attn, heads
        self.emb = mlp([in_ch, d, d])
        if use_attn:
            self.qkv = nn.Linear(d, 3 * d)
            self.rel = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads))
            self.proj = nn.Linear(d, d)
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.pool_q = nn.Linear(d, 1)
        self.out = nn.Linear(2 * d, d)

    def forward(self, x, m):  # x [N,P,C], m [N,P]
        f = self.emb(x)
        if self.use_attn:
            N, P, d = f.shape
            q, k, v = self.qkv(self.n1(f)).view(N, P, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
            bias = self.rel(x[:, :, None, :3] - x[:, None, :, :3]).permute(0, 3, 1, 2)
            a = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h) + bias
            a = a.masked_fill(m[:, None, None, :] == 0, -1e4).softmax(-1)
            f = f + self.proj((a @ v).transpose(1, 2).reshape(N, P, d))
            f = f + self.ff(self.n2(f))
        w = self.pool_q(f).squeeze(-1).masked_fill(m == 0, -1e4).softmax(-1)
        att = (w.unsqueeze(-1) * f).sum(1)
        mx = masked_max(f, m, 1)
        return self.out(torch.cat([att, mx], -1)), w


def motion_descriptor(x, mask):
    """Per-segment kinematic summary of the radar return (8 values):
    centroid xyz, centroid displacement to the previous segment (3),
    mean |Doppler|, log point count.  Explicitly gives the temporal model the
    body-motion cues that radar measures directly."""
    m = mask.unsqueeze(-1)
    n = m.sum(2).clamp(min=1)
    c = (x[..., :3] * m).sum(2) / n
    dc = torch.cat([torch.zeros_like(c[:, :1]), c[:, 1:] - c[:, :-1]], 1)
    dop = (x[..., 3:4].abs() * m).sum(2) / n
    cnt = torch.log1p(mask.sum(2, keepdim=True))
    return torch.cat([c, dc, dop, cnt], -1)


class TBlock(nn.Module):
    """Pre-norm Transformer encoder block (plain Linear layers so the whole block
    can be INT8-quantised); keeps attention weights for explainability."""
    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)
        self.attn_w = None

    def forward(self, z, pad):
        B, T, d = z.shape
        q, k, v = self.qkv(self.n1(z)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h)
        a = a.masked_fill(pad[:, None, None, :], -1e4).softmax(-1)
        self.attn_w = a.detach()
        z = z + self.drop(self.proj((self.drop(a) @ v).transpose(1, 2).reshape(B, T, d)))
        return z + self.drop(self.ff(self.n2(z)))


class STPTLite(nn.Module):
    """Spatio-Temporal Point Transformer, Lite (proposed).

    Factorised attention: (1) a SegmentTokenizer attends among the P points of
    each short temporal segment and compresses it into one token; (2) a small
    Transformer encoder attends across the G segment tokens.  Cost is
    O(G*P^2 + G^2) instead of O((G*P)^2) for joint attention over all points,
    and no kNN / FPS is needed, which keeps edge latency low.  (3) A
    motion-descriptor branch injects explicit radar kinematics (centroid
    trajectory, Doppler energy, point count) into every token.
    """
    def __init__(self, in_ch=6, k=5, d=64, depth=2, heads=4, G=12, use_attn=True,
                 use_motion=True, use_temporal=True):
        super().__init__()
        self.use_motion, self.use_temporal = use_motion, use_temporal
        self.tok = SegmentTokenizer(in_ch, d, heads, use_attn)
        if use_motion:
            self.mot = mlp([8, d // 2, d], last_act=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, G + 1, d) * 0.02)
        if use_temporal:
            self.temporal = nn.ModuleList([TBlock(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, k))
        self.last_point_w = None

    def embed(self, x, mask):
        B, G, P, C = x.shape
        t, w = self.tok(x.reshape(B * G, P, C), mask.reshape(B * G, P))
        self.last_point_w = w.view(B, G, P)
        t = t.view(B, G, -1)
        if self.use_motion:
            t = t + self.mot(motion_descriptor(x, mask))
        seg_valid = mask.max(2)[0]
        if self.use_temporal:
            z = torch.cat([self.cls.expand(B, -1, -1), t], 1) + self.pos[:, :G + 1]
            pad = torch.cat([torch.zeros(B, 1), 1 - seg_valid], 1).bool()
            for blk in self.temporal:
                z = blk(z, pad)
            self.last_temporal_attn = self.temporal[-1].attn_w  # [B, heads, G+1, G+1]
            return self.norm(z[:, 0])
        return self.norm((t * seg_valid.unsqueeze(-1)).sum(1) / seg_valid.sum(1, keepdim=True).clamp(min=1))

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


MODELS = {
    'pointnet': PointNet, 'pointnet2': PointNet2, 'dgcnn': DGCNN,
    'pointtransformer': PointTransformer, 'pointnet_lstm': PointNetLSTM,
    'voxel_tdcnn': VoxelTDCNN, 'stpt_lite': STPTLite,
}


def build(name, k, **kw):
    if name == 'stpt_teacher':
        return STPTLite(k=k, d=128, depth=4, heads=8, **kw)
    if name == 'stpt_tiny':
        return STPTLite(k=k, d=32, depth=2, heads=4, **kw)
    if name.startswith('stpt'):
        return STPTLite(k=k, **kw)
    return MODELS[name](k=k)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
