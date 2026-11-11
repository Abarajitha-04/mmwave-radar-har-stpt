"""Explainability + robustness analysis on trained seed-0 models.
- Robustness: test accuracy when (a) only a fraction of the radar returns
  survives in every segment (sparser sensor / lower SNR) and (b) whole
  segments are lost (dropped frames).
- XAI for STPT-Lite+KD: CLS->segment attention, segment-occlusion importance,
  per-point gradient saliency, t-SNE of embeddings."""
import sys, os, json, argparse
sys.path.insert(0, 'src')
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import build
from train import model_kwargs
from data_radhar import get_radhar, CLASSES
from data_milipoint import get_milipoint
import noise

torch.set_num_threads(int(os.environ.get('THREADS', 1)))
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 200})
R = 'results/runs'


def load(tag, K):
    a = argparse.Namespace(**json.load(open(f'{R}/{tag}/metrics.json'))['args'])
    m = build(a.model, K, **model_kwargs(a)); m.load_state_dict(torch.load(f'{R}/{tag}/model.pt')); m.eval()
    return m


def batches(ds, bs=128):
    return DataLoader(ds, bs)


@torch.no_grad()
def eval_conditions(model, ds, conds, K, bs=256):
    """Evaluate one model on every (condition, level).  The perturbation generator is
    re-seeded per condition and the loader order is fixed, so every model sees
    exactly the same perturbed test samples."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    out = {}; clean_pred = None
    for cond, level in conds:
        g = torch.Generator().manual_seed(12345)
        P, Y = [], []
        for x, m, y in batches(ds, bs):   # same batch size for every model -> identical perturbed samples
            xn, mn = noise.perturb(x, m, cond, level, g)
            P.append(model(xn, mn).argmax(1)); Y.append(y)
        pr, yt = torch.cat(P).numpy(), torch.cat(Y).numpy()
        if cond == 'clean': clean_pred = pr
        _, rec, f1, _ = precision_recall_fscore_support(yt, pr, labels=list(range(K)), zero_division=0)
        out[(cond, level)] = dict(acc=100 * accuracy_score(yt, pr), f1=100 * f1.mean(), macro_recall=100 * rec.mean(),
                                  stability=100 * float(np.mean(pr == clean_pred)), per_class_recall=(100 * rec).round(2).tolist())
    return out


def robustness(which='radhar'):
    import pandas as pd
    conds = [('clean', 0)] + [(c, l) for c, (lv, _) in noise.CONDITIONS.items() if c != 'clean' for l in lv]
    if which == 'radhar':
        _, _, te = get_radhar(); K = 5; pfx = 'rh_'
        groups = {'stpt_lite_kd': 3, 'stpt_lite_kd_robust': 3, 'pointnet': 3, 'pointnet_robust': 3, 'pointnet_lstm': 1,
                  'pointnet2': 1, 'dgcnn': 1, 'pointtransformer': 1, 'voxel_tdcnn': 1, 'stpt_tiny_kd': 1}
    else:
        _, _, te = get_milipoint(); K = 49; pfx = 'mp_'
        groups = {'stpt_lite_kd': 3, 'stpt_lite_kd_robust': 3, 'pointnet': 3, 'pointnet_robust': 3, 'pointnet2': 1}
    rows = []
    for grp, ns in groups.items():
        for sd in range(ns):
            t = f'{pfx}{grp}_s{sd}'
            if not os.path.exists(f'{R}/{t}/model.pt'): print('missing', t); continue
            res = eval_conditions(load(t, K), te, conds, K, bs=64 if which == 'radhar' else 256)
            for (c, l), r in res.items():
                rows.append(dict(dataset=which, model=grp, seed=sd, condition=c, level=l, **{k: v for k, v in r.items() if k != 'per_class_recall'},
                                 per_class_recall=json.dumps(r['per_class_recall'])))
            print(t, 'clean', round(res[('clean', 0)]['acc'], 2), 'combined3', round(res[('combined', 3)]['acc'], 2), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f'results/robustness_{which}.csv', index=False)


def robustness_figures():
    import pandas as pd
    NM = {'stpt_lite_kd': 'STPT-Lite+KD', 'stpt_lite_kd_robust': 'STPT-Lite+KD+robust', 'pointnet': 'PointNet',
          'pointnet_robust': 'PointNet+robust', 'pointnet_lstm': 'PointNet+BiLSTM', 'pointnet2': 'PointNet++',
          'dgcnn': 'DGCNN', 'pointtransformer': 'Point Transformer', 'voxel_tdcnn': 'Voxel TD-CNN', 'stpt_tiny_kd': 'STPT-Tiny+KD'}
    ST = {'stpt_lite_kd_robust': dict(color='#d7301f', lw=2.0), 'stpt_lite_kd': dict(color='#fc8d59', lw=1.4, ls='--'),
          'pointnet': dict(color='#4292c6', lw=1.2), 'pointnet_robust': dict(color='#4292c6', lw=1.2, ls='--')}
    summ = []
    for which in ['radhar', 'milipoint']:
        f = f'results/robustness_{which}.csv'
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        agg = df.groupby(['model', 'condition', 'level']).agg(acc=('acc', 'mean'), acc_sd=('acc', 'std'), f1=('f1', 'mean'),
                                                               macro_recall=('macro_recall', 'mean'), stability=('stability', 'mean'),
                                                               n=('seed', 'count')).reset_index()
        clean = agg[agg.condition == 'clean'].set_index('model').acc
        agg['drop'] = agg.apply(lambda r: clean[r.model] - r.acc, axis=1)
        agg.insert(0, 'dataset', which); summ.append(agg)
        conds = [c for c in noise.CONDITIONS if c != 'clean']
        fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6)); ax = ax.ravel()
        for j, c in enumerate(conds):
            for mdl in agg.model.unique():
                d = agg[(agg.model == mdl) & (agg.condition == c)].sort_values('level', ascending=c in ('sparsity', 'density'))
                if c in ('sparsity', 'density'): d = d.iloc[::-1]
                xs = [0] + list(range(1, len(d) + 1)); ys = [clean[mdl]] + list(d.acc)
                ax[j].plot(xs, ys, marker='o', ms=2, label=NM[mdl], **ST.get(mdl, dict(lw=0.8, alpha=0.7)))
            lv = noise.CONDITIONS[c][0]
            ax[j].set_xticks(range(len(lv) + 1)); ax[j].set_xticklabels(['0'] + [str(v) for v in lv], fontsize=5.5)
            ax[j].set_title(c, fontsize=7); ax[j].tick_params(labelsize=6)
        ax[7].axis('off'); h, l = ax[0].get_legend_handles_labels(); ax[7].legend(h, l, fontsize=5.5, frameon=False, loc='center')
        fig.supylabel('test accuracy (%)', fontsize=7); fig.suptitle(f"Robustness, {'RadHAR' if which == 'radhar' else 'MiliPoint (unseen subjects)'}", fontsize=8)
        fig.tight_layout(); fig.savefig(f'figures/robustness_{which}.pdf'); fig.savefig(f'figures/robustness_{which}.png'); plt.close(fig)
    if summ:
        S = pd.concat(summ); S.round(2).to_csv('results/robustness_summary.csv', index=False)
        print(S[(S.condition.isin(['clean', 'combined']))].round(2).to_string())


def xai():
    _, _, te = get_radhar()
    m = load('rh_stpt_lite_kd_robust_s0', 5)
    G = 12
    att = np.zeros((5, G)); occ = np.zeros((5, G)); cnt = np.zeros(5)
    embs, ys = [], []
    for x, mk, y in batches(te):
        with torch.no_grad():
            p = torch.softmax(m(x, mk), -1); e = m.embed(x, mk)
            a = m.last_temporal_attn.mean(1)[:, 0, 1:].numpy()  # CLS -> segments, last layer, mean of heads
            base = p[torch.arange(len(y)), y]
            oc = []
            for g in range(G):
                m2 = mk.clone(); m2[:, g] = 0; x2 = x.clone(); x2[:, g] = 0
                oc.append((base - torch.softmax(m(x2, m2), -1)[torch.arange(len(y)), y]).numpy())
            oc = np.stack(oc, 1)
        for c in range(5):
            s = (y == c).numpy(); att[c] += a[s].sum(0); occ[c] += oc[s].sum(0); cnt[c] += s.sum()
        embs.append(e.numpy()); ys.append(y.numpy())
    att /= cnt[:, None]; occ /= cnt[:, None]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 1.9))
    for j, (M, t) in enumerate([(att, '(a) CLS→segment attention (last layer)'), (occ, '(b) occlusion: drop in true-class prob.')]):
        im = ax[j].imshow(M, aspect='auto', cmap='magma'); ax[j].set_yticks(range(5)); ax[j].set_yticklabels(CLASSES)
        ax[j].set_xticks(range(G)); ax[j].set_xticklabels([f'{(g + 0.5) / 6:.1f}' for g in range(G)], fontsize=5.5)
        ax[j].set_xlabel('segment centre (s)'); ax[j].set_title(t, fontsize=8); plt.colorbar(im, ax=ax[j], fraction=0.04)
    fig.tight_layout(); fig.savefig('figures/xai_temporal.pdf'); fig.savefig('figures/xai_temporal.png'); plt.close(fig)
    corr = float(np.corrcoef(att.ravel(), occ.ravel())[0, 1])

    # per-point gradient saliency on one window per class
    fig = plt.figure(figsize=(7.2, 1.9))
    for c in range(5):
        i = int(np.where(te.labels == c)[0][len(np.where(te.labels == c)[0]) // 2])
        x, mk, _ = te[i]; x = x.unsqueeze(0).requires_grad_(True); mk = mk.unsqueeze(0)
        m(x, mk)[0, c].backward()
        sal = x.grad[0, ..., :3].norm(dim=-1).reshape(-1).numpy(); pts = x[0].detach().reshape(-1, 6).numpy()
        v = mk.reshape(-1).numpy() > 0
        a = fig.add_subplot(1, 5, c + 1, projection='3d')
        from scipy.stats import rankdata
        r_ = rankdata(sal[v]) / v.sum()                      # rank-normalised saliency in (0,1]
        a.scatter(pts[v, 1], pts[v, 0], pts[v, 2], c=r_, cmap='viridis', s=5, vmin=0, vmax=1)
        a.set_title(CLASSES[c], fontsize=8); a.set_xticks([]); a.set_yticks([]); a.set_zticks([]); a.view_init(12, -60)
    fig.suptitle('Gradient saliency per return (rank-normalised; yellow = most influential)', fontsize=8)
    fig.tight_layout(); fig.savefig('figures/xai_saliency.pdf'); fig.savefig('figures/xai_saliency.png'); plt.close(fig)

    # t-SNE of embeddings: STPT-Lite+KD vs PointNet
    from sklearn.manifold import TSNE
    ys = np.concatenate(ys); sel = np.random.RandomState(0).choice(len(ys), 1500, replace=False)
    pn = load('rh_pointnet_s0', 5); epn = []
    with torch.no_grad():
        for x, mk, y in batches(te): epn.append(pn.embed(x, mk).numpy())
    from sklearn.metrics import silhouette_score
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.4)); sil = {}
    for j, (E, t) in enumerate([(np.concatenate(epn), 'PointNet'), (np.concatenate(embs), 'STPT-Lite+KD+robust')]):
        Z = TSNE(2, random_state=0, init='pca', perplexity=30).fit_transform(E[sel])
        sil[t] = float(silhouette_score(E[sel], ys[sel]))
        for c in range(5):
            s = ys[sel] == c; ax[j].scatter(Z[s, 0], Z[s, 1], s=1.5, label=CLASSES[c])
        ax[j].set_title(f'{t} (silhouette {sil[t]:.2f})', fontsize=8); ax[j].set_xticks([]); ax[j].set_yticks([])
    ax[1].legend(fontsize=6, frameon=False, markerscale=4, loc='best')
    fig.tight_layout(); fig.savefig('figures/tsne.pdf'); fig.savefig('figures/tsne.png'); plt.close(fig)
    json.dump(dict(att_occ_corr=corr, silhouette=sil, att=att.tolist(), occ=occ.tolist()), open('results/xai.json', 'w'), indent=1)
    print('att/occ corr', corr, 'silhouette', sil)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'robust':
        robustness(sys.argv[2] if len(sys.argv) > 2 else 'radhar')
    elif cmd == 'robustfig':
        robustness_figures()
    else:
        xai()
