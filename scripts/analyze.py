"""Aggregate all runs into tables, statistics and figures.
Outputs: results/summary_*.csv, results/stats.json, figures/*.pdf|png"""
import json, glob, os, re, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import binomtest

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 200})
R = 'results/runs'
NAMES = {'voxel_tdcnn': 'Voxel TD-CNN+BiLSTM [RadHAR]', 'pointnet': 'PointNet', 'pointnet_lstm': 'PointNet+BiLSTM',
         'pointnet2': 'PointNet++ (SSG)', 'dgcnn': 'DGCNN', 'pointtransformer': 'Point Transformer',
         'stpt_teacher': 'STPT-Large (teacher)', 'stpt_lite': 'STPT-Lite', 'stpt_lite_kd': 'STPT-Lite + KD',
         'stpt_tiny_kd': 'STPT-Tiny + KD', 'stpt_lite_kd_robust': 'STPT-Lite + KD + robust (proposed)',
         'pointnet_robust': 'PointNet + noise-robust training'}
ORDER = ['voxel_tdcnn', 'pointnet', 'pointnet_lstm', 'pointnet2', 'dgcnn', 'pointtransformer', 'stpt_teacher',
         'stpt_lite', 'stpt_lite_kd', 'stpt_tiny_kd', 'stpt_lite_kd_robust', 'pointnet_robust']


def load_all():
    rows = []
    for f in glob.glob(f'{R}/*/metrics.json'):
        tag = f.split('/')[-2]; r = json.load(open(f))
        m = re.match(r'(.*)_s(\d)$', tag)
        rows.append(dict(tag=tag, group=m.group(1), seed=int(m.group(2)), acc=100 * r['test']['acc'],
                         f1=100 * r['test']['macro_f1'], top3=100 * r['test']['top3'], params=r['params'],
                         prec=100 * r['test'].get('macro_precision', np.nan), rec=100 * r['test'].get('macro_recall', np.nan),
                         size_mb=r['size_mb'], lat=r['latency_ms_cpu1'], mflops=r.get('mflops', np.nan),
                         int8_acc=100 * r['int8']['test']['acc'] if 'int8' in r else np.nan,
                         int8_mb=r['int8']['size_mb'] if 'int8' in r else np.nan,
                         best_epoch=r['best_epoch'], train_min=r['train_time_s'] / 60,
                         val_f1=100 * r['val']['macro_f1']))
    return pd.DataFrame(rows)


def agg(df, groups):
    out = []
    for g in groups:
        d = df[df.group == g]
        if len(d) == 0: continue
        out.append(dict(group=g, n=len(d), acc_mean=d.acc.mean(), acc_std=d.acc.std(ddof=1) if len(d) > 1 else 0,
                        f1_mean=d.f1.mean(), f1_std=d.f1.std(ddof=1) if len(d) > 1 else 0, top3=d.top3.mean(),
                        params=d.params.iloc[0], size_mb=d.size_mb.iloc[0], mflops=d.mflops.iloc[0],
                        lat_ms=d.lat.median(), int8_acc=d.int8_acc.mean(), int8_mb=d.int8_mb.iloc[0],
                        prec=d.prec.mean(), rec=d.rec.mean(),
                        best_epoch=d.best_epoch.mean(), train_min=d.train_min.mean()))
    return pd.DataFrame(out)


def bootstrap_ci(y, p, B=2000, seed=0):
    rng = np.random.RandomState(seed); n = len(y); c = (p == y).astype(float)
    accs = [c[rng.randint(0, n, n)].mean() for _ in range(B)]
    return 100 * np.percentile(accs, 2.5), 100 * np.percentile(accs, 97.5)


def mcnemar(y, pa, pb):
    a, b = (pa == y), (pb == y)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, p


def holm(ps):
    idx = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(idx):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def stats_block(prefix, ref, others, seeds=(0, 1, 2)):
    """Paired tests of the reference model against each baseline on the same test
    windows: exact McNemar per seed, Holm-corrected within each seed, reported
    with the median adjusted p and the number of seeds significant at 0.05."""
    out = {}
    for s in seeds:
        fr = f'{R}/{prefix}{ref}_s{s}'
        if not os.path.exists(f'{fr}/test_probs.npy'): continue
        y = np.load(f'{fr}/test_labels.npy'); pr = np.load(f'{fr}/test_probs.npy').argmax(1)
        lo, hi = bootstrap_ci(y, pr)
        res = {'ref_acc': 100 * (pr == y).mean(), 'ref_ci': [lo, hi], 'tests': {}}
        ps, keys = [], []
        for o in others:
            fo = f'{R}/{prefix}{o}_s{s}'
            if not os.path.exists(f'{fo}/test_probs.npy'): continue
            po = np.load(f'{fo}/test_probs.npy').argmax(1)
            n01, n10, p = mcnemar(y, pr, po)
            l2, h2 = bootstrap_ci(y, po)
            res['tests'][o] = dict(acc=100 * (po == y).mean(), ci=[l2, h2], ref_only=n01, other_only=n10, p=p)
            ps.append(p); keys.append(o)
        for k, pa in zip(keys, holm(np.array(ps)) if ps else []):
            res['tests'][k]['p_holm'] = float(pa)
        out[s] = res
    return out


def fmt(m, s):
    return f'{m:.2f} ± {s:.2f}'


def main():
    df = load_all(); df.to_csv('results/all_runs.csv', index=False)
    S = {}
    for pfx, name in [('rh_', 'radhar'), ('mp_', 'milipoint')]:
        t = agg(df, [pfx + o for o in ORDER]); t['model'] = [NAMES[g[len(pfx):]] for g in t.group]
        t.to_csv(f'results/summary_{name}.csv', index=False)
        print(f'\n=== {name} ==='); print(t[['model', 'n', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std', 'top3', 'params', 'mflops', 'lat_ms', 'int8_acc', 'best_epoch', 'train_min']].round(2).to_string())
        S[name] = stats_block(pfx, 'stpt_lite_kd_robust', [o for o in ORDER if o != 'stpt_lite_kd_robust'])
    abl = agg(df, sorted(set(g for g in df.group if '_abl_' in g)) + ['rh_stpt_lite', 'mp_stpt_lite'])
    abl.to_csv('results/summary_ablation.csv', index=False); print('\n=== ablation ==='); print(abl[['group', 'n', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std', 'params']].round(2).to_string())
    tr = agg(df, sorted(set(g for g in df.group if 'scratch' in g or 'init' in g)) + ['rh_abl_xyz', 'mp_stpt_lite'])
    tr.to_csv('results/summary_transfer.csv', index=False); print('\n=== transfer ==='); print(tr[['group', 'n', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std']].round(2).to_string())
    pr = agg(df, sorted(set(g for g in df.group if g.startswith('mpo_'))) + ['mp_stpt_lite', 'mp_pointnet', 'mp_pointnet2'])
    pr.to_csv('results/summary_protocol.csv', index=False); print('\n=== protocol ==='); print(pr[['group', 'n', 'acc_mean', 'acc_std', 'f1_mean', 'top3']].round(2).to_string())
    json.dump(S, open('results/stats.json', 'w'), indent=1, default=float)
    for name in S:
        for s, r in S[name].items():
            print(f'\n{name} seed {s}: ref {r["ref_acc"]:.2f} CI {np.round(r["ref_ci"], 2)}')
            for o, v in r['tests'].items():
                print(f'   vs {o:18s} acc {v["acc"]:.2f} b/c {v["ref_only"]}/{v["other_only"]} p_holm {v.get("p_holm", np.nan):.2e}')
    figures(df)


def figures(df):
    C = {'stpt_lite_kd_robust': '#d7301f', 'pointnet_robust': '#c6dbef', 'voxel_tdcnn': '#8c8c8c', 'pointnet': '#9ecae1', 'pointnet_lstm': '#4292c6', 'pointnet2': '#2171b5',
         'dgcnn': '#a1d99b', 'pointtransformer': '#31a354', 'stpt_teacher': '#fdae6b', 'stpt_lite': '#f16913',
         'stpt_lite_kd': '#fc8d59', 'stpt_tiny_kd': '#7f2704'}
    # accuracy vs parameters / latency (both datasets)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for j, (pfx, title) in enumerate([('rh_', 'RadHAR (5 classes)'), ('mp_', 'MiliPoint, subject-wise (49)')]):
        t = agg(df, [pfx + o for o in ORDER])
        for _, r in t.iterrows():
            o = r.group[len(pfx):]
            ax[j].errorbar(r.params / 1e3, r.acc_mean, yerr=r.acc_std, fmt='o', ms=4, color=C[o], capsize=2,
                           label=NAMES[o] if j == 0 else None)
        ax[j].set_xscale('log'); ax[j].set_xlabel('parameters (K, log scale)'); ax[j].set_title(title, fontsize=8)
        ax[j].set_xticks([30, 100, 300, 1000]); ax[j].set_xticklabels(['30', '100', '300', '1000'])
        ax[j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_ylabel('test accuracy (%)')
    t = agg(df, ['rh_' + o for o in ORDER])
    for _, r in t.iterrows():
        o = r.group[3:]; ax[2].scatter(r.lat_ms, r.acc_mean, color=C[o], s=18)
    ax[2].set_xscale('log'); ax[2].set_xlabel('latency (ms, 1 CPU thread)'); ax[2].set_title('RadHAR: accuracy vs latency', fontsize=8)
    ax[2].set_xticks([1, 2, 5, 10]); ax[2].set_xticklabels(['1', '2', '5', '10']); ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=6)
    fig.set_size_inches(7.2, 3.0)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig('figures/efficiency.pdf'); fig.savefig('figures/efficiency.png'); plt.close(fig)

    # confusion matrices of the proposed model (seed 0)
    from sklearn.metrics import confusion_matrix
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1, 1.6]))
    for j, pfx in enumerate(['rh_', 'mp_']):
        f = f'{R}/{pfx}stpt_lite_kd_robust_s0'
        if not os.path.exists(f + '/test_probs.npy'): continue
        y = np.load(f + '/test_labels.npy'); p = np.load(f + '/test_probs.npy').argmax(1)
        K = 5 if pfx == 'rh_' else 49
        cm = confusion_matrix(y, p, labels=range(K), normalize='true')
        im = ax[j].imshow(cm, cmap='Blues', vmin=0, vmax=1)
        if K == 5:
            names = ['boxing', 'jack', 'jump', 'squats', 'walk']
            ax[j].set_xticks(range(5)); ax[j].set_xticklabels(names, rotation=30); ax[j].set_yticks(range(5)); ax[j].set_yticklabels(names)
            for a in range(5):
                for b in range(5):
                    ax[j].text(b, a, f'{100 * cm[a, b]:.0f}', ha='center', va='center', fontsize=6, color='w' if cm[a, b] > 0.5 else 'k')
            ax[j].set_title('(a) RadHAR', fontsize=8)
        else:
            ax[j].set_title('(b) MiliPoint (49 actions, unseen subjects)', fontsize=8)
        ax[j].set_xlabel('predicted'); ax[j].set_ylabel('true')
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig('figures/confusion.pdf'); fig.savefig('figures/confusion.png'); plt.close(fig)

    # training curves (validation macro-F1, seed 0)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for j, pfx in enumerate(['rh_', 'mp_']):
        for o in ORDER:
            f = f'{R}/{pfx}{o}_s0/history.csv'
            if os.path.exists(f):
                h = pd.read_csv(f); ax[j].plot(h.epoch, 100 * h.val_f1, color=C[o], lw=1, label=NAMES[o])
        ax[j].set_xlabel('epoch'); ax[j].set_ylabel('val macro-F1 (%)'); ax[j].set_title(['RadHAR', 'MiliPoint (unseen subjects)'][j], fontsize=8)
    hs, ls = [], []
    for a_ in ax:
        for h_, l_ in zip(*a_.get_legend_handles_labels()):
            if l_ not in ls: hs.append(h_); ls.append(l_)
    fig.legend(hs, ls, loc='lower center', ncol=4, frameon=False, fontsize=6)
    fig.set_size_inches(7.2, 3.0)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig('figures/curves.pdf'); fig.savefig('figures/curves.png'); plt.close(fig)

    # label efficiency / transfer
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for j, (sc, ini, full_sc, title) in enumerate([('rh_xyz_scratch_f', 'rh_xyz_mpinit_f', 'rh_abl_xyz', 'MiliPoint → RadHAR (xyz)'),
                                                   ('mp_scratch_f', 'mp_rhinit_f', 'mp_stpt_lite', 'RadHAR → MiliPoint')]):
        fr = [0.1, 0.25, 1.0] if j == 0 else [0.25, 1.0]
        for lab, pre, col in [('from scratch', sc, '#4292c6'), ('pre-trained on other dataset', ini, '#d7301f')]:
            ms, ss, xs = [], [], []
            for f in fr:
                g = full_sc if (pre == sc and f == 1.0) else f'{pre}{f}'
                d = df[df.group == g]
                if len(d): xs.append(100 * f); ms.append(d.acc.mean()); ss.append(d.acc.std(ddof=1) if len(d) > 1 else 0)
            if xs: ax[j].errorbar(xs, ms, yerr=ss, marker='o', ms=3, capsize=2, color=col, label=lab)
        ax[j].set_xscale('log'); ax[j].set_xlabel('% of target training recordings'); ax[j].set_ylabel('test accuracy (%)')
        ax[j].set_title(title, fontsize=8); ax[j].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig('figures/transfer.pdf'); fig.savefig('figures/transfer.png'); plt.close(fig)

    # evaluation-protocol study
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    labs = ['subject-wise\n(ours)', 'chronological\n(MiliPoint code)', 'random windows\n(leaky)']
    for k, (m, col) in enumerate([('stpt_lite', '#d7301f'), ('pointnet', '#9ecae1'), ('pointnet2', '#2171b5')]):
        vals, errs = [], []
        for g in [f'mp_{m}', f'mpo_official_{m}', f'mpo_random_{m}']:
            d = df[df.group == g]; vals.append(d.acc.mean() if len(d) else np.nan); errs.append(d.acc.std(ddof=1) if len(d) > 1 else 0)
        ax.bar(np.arange(3) + (k - 1) * 0.26, vals, 0.26, yerr=errs, color=col, capsize=2,
               label={'stpt_lite': 'STPT-Lite', 'pointnet': 'PointNet', 'pointnet2': 'PointNet++'}[m])
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=6.5); ax.set_ylabel('test accuracy (%)')
    ax.legend(fontsize=6, frameon=False); ax.set_title('MiliPoint: same models, different splits', fontsize=8)
    fig.tight_layout(); fig.savefig('figures/protocol.pdf'); fig.savefig('figures/protocol.png'); plt.close(fig)


if __name__ == '__main__':
    main()
