"""Train / evaluate one configuration.  Writes results/runs/<tag>/:
metrics.json, history.csv, {val,test}_probs.npy, test_labels.npy, model.pt
Defaults come from configs/default.yaml; command-line flags override them.

Model selection uses validation macro-F1 only; the test set is touched once,
with the selected checkpoint.
"""
import argparse, json, os, sys, time, io, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from models import build, n_params
from noise import robust_augment


def get_data(a):
    if a.data == 'radhar':
        from data_radhar import get_radhar
        return get_radhar(a.G, a.P, a.feats, a.roi, a.center, a.seed, a.label_frac,
                          gate=a.gate if a.gate > 0 else None, aug=a.aug)
    from data_milipoint import get_milipoint
    return get_milipoint(a.G, a.P, a.data.split('_')[1], a.seed, a.label_frac, aug=a.aug)


@torch.no_grad()
def predict(model, dl):
    model.eval(); P, Y = [], []
    for x, m, y in dl:
        P.append(F.softmax(model(x, m), -1)); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def scores(p, y, K=None):
    pr = p.argmax(1)
    labels = list(range(K if K else p.shape[1]))
    top3 = np.mean([yy in np.argsort(-pp)[:3] for pp, yy in zip(p, y)])
    P_, R_, F_, _ = precision_recall_fscore_support(y, pr, labels=labels, average='macro', zero_division=0)
    nll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))
    return dict(acc=float(accuracy_score(y, pr)), macro_f1=float(F_), macro_precision=float(P_),
                macro_recall=float(R_), top3=float(top3), nll=nll)


def latency_ms(model, G, P, reps=200):
    torch.set_num_threads(1)
    model.eval(); x = torch.randn(1, G, P, 6); m = torch.ones(1, G, P)
    with torch.no_grad():
        for _ in range(20): model(x, m)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); model(x, m); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def mflops(model, G, P):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval(); x = torch.randn(1, G, P, 6); m = torch.ones(1, G, P)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x, m)
    return fc.get_total_flops() / 1e6


def quantise_int8(model):
    """Weight-only INT8: every Linear/Conv weight is quantised per output channel
    (symmetric, scale = max|w|/127) and de-quantised for inference.  Returns the
    model and its storage size (int8 weights + fp32 scales + fp32 other params)."""
    import copy
    q = copy.deepcopy(model).eval(); nbytes = 0; quantised = set()
    for mod in q.modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv3d)):
            w = mod.weight.data
            sc = w.abs().flatten(1).max(1)[0].clamp(min=1e-8) / 127
            wq = torch.round(w / sc.view(-1, *[1] * (w.dim() - 1))).clamp(-127, 127)
            mod.weight.data = wq * sc.view(-1, *[1] * (w.dim() - 1))
            nbytes += w.numel() + 4 * sc.numel(); quantised.add(id(mod.weight))
    nbytes += sum(4 * p.numel() for p in q.parameters() if id(p) not in quantised)
    return q, nbytes


def size_mb(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return b.tell() / 1e6


def model_kwargs(a):
    kw = {}
    if a.model.startswith('stpt'):
        kw = dict(G=a.G, use_attn=not a.no_attn, use_motion=not a.no_motion, use_temporal=not a.no_temporal)
    return kw


def main():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='radhar')  # radhar | mp_subject | mp_official
    ap.add_argument('--model', default='stpt_lite')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=cfg['epochs'])
    ap.add_argument('--bs', type=int, default=cfg['optimiser']['bs'])
    ap.add_argument('--lr', type=float, default=cfg['optimiser']['lr'])
    ap.add_argument('--wd', type=float, default=cfg['optimiser']['wd'])
    ap.add_argument('--G', type=int, default=cfg['G'])
    ap.add_argument('--P', type=int, default=cfg['P'])
    ap.add_argument('--feats', default=cfg['feats'])
    ap.add_argument('--roi', type=int, default=1)
    ap.add_argument('--center', type=int, default=1)
    ap.add_argument('--gate', type=float, default=cfg['gate'])
    ap.add_argument('--no_attn', action='store_true')
    ap.add_argument('--no_motion', action='store_true')
    ap.add_argument('--no_temporal', action='store_true')
    ap.add_argument('--teacher', default='')       # run dir of a trained teacher -> knowledge distillation
    ap.add_argument('--kd_alpha', type=float, default=cfg['kd_alpha'])
    ap.add_argument('--kd_T', type=float, default=cfg['kd_T'])
    ap.add_argument('--init', default='')          # run dir of a pretrained model (cross-dataset)
    ap.add_argument('--label_frac', type=float, default=1.0)
    ap.add_argument('--patience', type=int, default=cfg['patience'])
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--aug', default=cfg['aug'])                 # std | nojitter | none
    ap.add_argument('--robust', type=int, default=cfg['robust'])  # noise-robust training augmentation
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = f'results/runs/{a.tag}'; os.makedirs(out, exist_ok=True)
    dtr, dva, dte = get_data(a)
    K = 5 if a.data == 'radhar' else 49
    g = torch.Generator(); g.manual_seed(a.seed)
    tl = DataLoader(dtr, a.bs, shuffle=True, drop_last=True, generator=g)
    ebs = 32 if a.model == 'voxel_tdcnn' else 256   # voxel grids are memory-hungry at evaluation
    vl, tel = DataLoader(dva, ebs), DataLoader(dte, ebs)
    model = build(a.model, K, **model_kwargs(a))
    gaug = torch.Generator(); gaug.manual_seed(10_000 + a.seed)  # training-noise stream (never used at test)

    if a.init:  # cross-dataset initialisation: copy every matching tensor except the classifier
        src = torch.load(f'{a.init}/model.pt', map_location='cpu')
        own = model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and v.shape == own[k].shape and not k.startswith('head')}
        own.update(keep); model.load_state_dict(own)
        print(f'init: loaded {len(keep)}/{len(own)} tensors from {a.init}')

    teacher = None
    if a.teacher:
        targs = json.load(open(f'{a.teacher}/metrics.json'))['args']
        tns = argparse.Namespace(**targs)
        teacher = build(tns.model, K, **model_kwargs(tns))
        teacher.load_state_dict(torch.load(f'{a.teacher}/model.pt', map_location='cpu')); teacher.eval()

    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.wd)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    hist, best, best_ep, bad = [], -1, -1, 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = n = corr = 0
        for x, m, y in tl:
            if a.robust:
                x, m = robust_augment(x, m, gaug)
            logits = model(x, m)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg['label_smoothing'])
            if teacher is not None:
                with torch.no_grad():
                    tlog = teacher(x, m)
                kd = F.kl_div(F.log_softmax(logits / a.kd_T, -1), F.softmax(tlog / a.kd_T, -1),
                              reduction='batchmean') * a.kd_T ** 2
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            lr_now = opt.param_groups[0]['lr']
            opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y); corr += (logits.argmax(1) == y).sum().item()
        pv, yv = predict(model, vl); sv = scores(pv, yv, K)
        hist.append(dict(epoch=ep + 1, train_loss=tot / n, train_acc=corr / n, val_loss=sv['nll'],
                         val_acc=sv['acc'], val_f1=sv['macro_f1'], lr=lr_now, time=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
        if sv['macro_f1'] > best:
            best, best_ep, bad = sv['macro_f1'], ep + 1, 0
            torch.save(model.state_dict(), f'{out}/model.pt')
        else:
            bad += 1
            if bad >= a.patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(torch.load(f'{out}/model.pt'))
    pv, yv = predict(model, vl); pt, yt = predict(model, tel)
    np.save(f'{out}/val_probs.npy', pv.astype(np.float32)); np.save(f'{out}/test_probs.npy', pt.astype(np.float32))
    np.save(f'{out}/test_labels.npy', yt)
    res = dict(args=vars(a), params=n_params(model), size_mb=size_mb(model), best_epoch=best_ep,
               epochs_run=len(hist), train_time_s=train_time, val=scores(pv, yv, K), test=scores(pt, yt, K),
               test_cm=confusion_matrix(yt, pt.argmax(1), labels=range(K)).tolist(),
               test_per_class=dict(zip(['precision', 'recall', 'f1', 'support'], [v.tolist() for v in
                   precision_recall_fscore_support(yt, pt.argmax(1), labels=list(range(K)), zero_division=0)])),
               config=cfg, torch_version=torch.__version__,
               latency_ms_cpu1=latency_ms(model, a.G, a.P))
    res['mflops'] = mflops(model, a.G, a.P)
    # post-training per-channel symmetric INT8 weight quantisation (edge storage)
    q, qbytes = quantise_int8(model)
    pq, _ = predict(q, tel)
    res['int8'] = dict(test=scores(pq, yt, K), size_mb=qbytes / 1e6)
    np.save(f'{out}/test_probs_int8.npy', pq.astype(np.float32))
    json.dump(res, open(f'{out}/metrics.json', 'w'), indent=1)
    import csv
    with open(f'{out}/history.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, hist[0].keys()); w.writeheader(); w.writerows(hist)
    print('TEST', json.dumps(res['test']), 'params', res['params'], 'lat', res['latency_ms_cpu1'])


if __name__ == '__main__':
    main()
