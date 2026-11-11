"""Generate LaTeX tables and number macros for the paper straight from results/.
Every number in the paper text comes from here, so text and tables stay
consistent with the run logs."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from analyze import load_all, agg, ORDER, NAMES

OUT = 'paper/generated'; os.makedirs(OUT, exist_ok=True)
df = load_all()
S = json.load(open('results/stats.json'))
macros = {}


def pm(m, s, d=2):
    return f'{m:.{d}f}\\,$\\pm$\\,{s:.{d}f}'


def best_mask(vals, higher=True):
    v = np.array(vals, float); b = np.nanmax(v) if higher else np.nanmin(v)
    return [abs(x - b) < 1e-9 for x in v]


# ------------------------------------------------------------ main tables
def main_table(pfx, name, K):
    t = agg(df, [pfx + o for o in ORDER]).reset_index(drop=True)
    st = S.get(name, {}).get('0', {}).get('tests', {})
    st_all = {s: S.get(name, {}).get(s, {}).get('tests', {}) for s in ['0', '1', '2']}
    bacc, bf1 = best_mask(t.acc_mean), best_mask(t.f1_mean)
    rows = []
    for i, r in t.iterrows():
        o = r.group[len(pfx):]
        acc = pm(r.acc_mean, r.acc_std); f1 = pm(r.f1_mean, r.f1_std)
        if bacc[i]: acc = f'\\textbf{{{acc}}}'
        if bf1[i]: f1 = f'\\textbf{{{f1}}}'
        sig = ''
        if o != 'stpt_lite_kd_robust':
            nsig = sum(1 for s in st_all.values() if o in s and s[o].get('p_holm', 1) < 0.05)
            better = np.mean([s[o]['ref_only'] > s[o]['other_only'] for s in st_all.values() if o in s]) if any(o in s for s in st_all.values()) else np.nan
            if not np.isnan(better):
                sig = f'{nsig}/3' + ('$\\uparrow$' if better >= 0.5 else '$\\downarrow$')
        else:
            sig = '--'
        name_s = NAMES[o].replace('(proposed)', '').strip()
        if o == 'stpt_lite_kd_robust': name_s = f'\\textbf{{{name_s} (ours)}}'
        extra = f' & {r.top3:.1f}' if K > 5 else ''
        rows.append(f'{name_s} & {r.params / 1e3:.1f} & {r.mflops:.1f} & {r.lat_ms:.2f} & {acc} & {f1}{extra} & {sig} \\\\')
        if o == 'pointtransformer': rows.append('\\midrule')
    return t, rows


for pfx, name, K, cap, lab in [('rh_', 'radhar', 5, 'RadHAR official test recordings (3{,}565 windows, 5 classes). Mean\\,$\\pm$\\,std over 3 seeds. Sig.: number of seeds in which the Holm-corrected McNemar test against the proposed model gives $p<0.05$; $\\uparrow$ ours better / $\\downarrow$ ours worse. Latency: median, batch 1, one CPU thread.', 'tab:radhar'),
                               ('mp_', 'milipoint', 49, 'MiliPoint, subject-disjoint test (subjects 8--10, 3{,}569 windows, 49 actions). Same conventions as Table~\\ref{tab:radhar}.', 'tab:milipoint')]:
    t, rows = main_table(pfx, name, K)
    top3h = ' & Top-3' if K > 5 else ''
    cols = 'lrrrcc' + ('r' if K > 5 else '') + 'c'
    tex = ['\\begin{table*}[t]', '\\centering', f'\\caption{{{cap}}}', f'\\label{{{lab}}}', '\\setlength{\\tabcolsep}{5pt}',
           f'\\begin{{tabular}}{{{cols}}}', '\\toprule',
           f'Model & Params (K) & MFLOPs & Lat. (ms) & Accuracy (\\%) & Macro-F1 (\\%){top3h} & Sig. \\\\', '\\midrule',
           *rows, '\\bottomrule', '\\end{tabular}', '\\end{table*}']
    open(f'{OUT}/tab_{name}.tex', 'w').write('\n'.join(tex))
    for _, r in t.iterrows():
        key = (name[:2] + r.group[len(pfx):]).replace('_', '')
        macros[key + 'Acc'] = f'{r.acc_mean:.2f}'; macros[key + 'Std'] = f'{r.acc_std:.2f}'; macros[key + 'Fone'] = f'{r.f1_mean:.2f}'
        macros[key + 'Topthree'] = f'{r.top3:.2f}'

# ------------------------------------------------------------ ablation table
abl = agg(df, sorted(set(g for g in df.group if '_abl_' in g)) + ['rh_stpt_lite', 'mp_stpt_lite'])
A = {r.group: r for _, r in abl.iterrows()}
desc = [('', 'STPT-Lite (full, no KD)'), ('noattn', '-- intra-segment attention (max+attentive pool only)'),
        ('nomotion', '-- motion descriptor'), ('notemporal', '-- temporal Transformer (mean over segments)'),
        ('xyz', '-- Doppler \\& intensity ($xyz$ only)'), ('nogate', '-- spatial gating'),
        ('nocenter', '-- person centring \\& gating'), ('noroi', '-- ROI filter'),
        ('augnone', '-- all training augmentation'), ('augnojitter', '-- coordinate jitter augmentation'),
        ('G6', '$G{=}6$ segments $\\times$ 64 pts'), ('G20', '$G{=}20$ segments $\\times$ 20 pts')]
_nab = lambda p: '/'.join(map(str, sorted(set(int(A[g].n) for g in A if g.startswith(p))))) or '0'
rows = []
for k, d in desc:
    rg, mg = ('rh_stpt_lite', 'mp_stpt_lite') if k == '' else (f'rh_abl_{k}', f'mp_abl_{k}')
    def cell(g, ref):
        if g not in A: return '--'
        r = A[g]; s = pm(r.acc_mean, r.acc_std)
        if g != ref and ref in A:
            dlt = r.acc_mean - A[ref].acc_mean; s += f' ({dlt:+.2f})'
        return s
    rows.append(f'{d} & {cell(rg, "rh_stpt_lite")} & {cell(mg, "mp_stpt_lite")} \\\\')
    if k == '': rows.append('\\midrule')
tex = ['\\begin{table}[t]', '\\centering', f'\\caption{{Ablation of STPT-Lite (no distillation): test accuracy (\\%), mean\\,$\\pm$\\,std over seeds (variants: {_nab("rh_abl_")} on RadHAR, {_nab("mp_abl_")} on MiliPoint; full model: 3), change w.r.t.\\ the full model in brackets.}}',
       '\\label{tab:ablation}', '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{lcc}', '\\toprule',
       'Variant & RadHAR & MiliPoint (subj.) \\\\', '\\midrule', *rows, '\\bottomrule', '\\end{tabular}}', '\\end{table}']
open(f'{OUT}/tab_ablation.tex', 'w').write('\n'.join(tex))
for k, _ in desc[1:]:
    if f'rh_abl_{k}' in A and 'rh_stpt_lite' in A:
        macros['ablDrop' + k.capitalize()] = f'{A["rh_stpt_lite"].acc_mean - A[f"rh_abl_{k}"].acc_mean:.2f}'
    if f'mp_abl_{k}' in A and 'mp_stpt_lite' in A:
        macros['ablMpDrop' + k.capitalize()] = f'{A["mp_stpt_lite"].acc_mean - A[f"mp_abl_{k}"].acc_mean:.2f}'

ab_seeds = sorted(set(int(t[-1]) for t in df.tag if '_abl_' in t))
macros['ablSeeds'] = {1: 'one', 2: 'two', 3: 'three'}.get(len(ab_seeds), str(len(ab_seeds)))
# ------------------------------------------------------------ transfer table
G = {g: r for g, r in ((r.group, r) for _, r in agg(df, sorted(set(df.group))).iterrows())}
def gv(g):
    return pm(G[g].acc_mean, G[g].acc_std) if g in G else '--'
rows = []
for f, lab in [(0.1, '10\\%'), (0.25, '25\\%'), (1.0, '100\\%')]:
    sc = 'rh_abl_xyz' if f == 1.0 else f'rh_xyz_scratch_f{f}'
    rows.append(f'MiliPoint $\\rightarrow$ RadHAR & {lab} & {gv(sc)} & {gv(f"rh_xyz_mpinit_f{f}")} \\\\')
rows.append('\\midrule')
for f, lab in [(0.25, '25\\%'), (1.0, '100\\%')]:
    sc = 'mp_stpt_lite' if f == 1.0 else f'mp_scratch_f{f}'
    rows.append(f'RadHAR $\\rightarrow$ MiliPoint & {lab} & {gv(sc)} & {gv(f"mp_rhinit_f{f}")} \\\\')
tex = ['\\begin{table}[t]', '\\centering', '\\caption{Cross-dataset transfer (STPT-Lite, $xyz$ input): test accuracy (\\%) when fine-tuning on a fraction of target training recordings/runs, from scratch vs.\\ initialised from the model trained on the other dataset (mean\\,$\\pm$\\,std, 3 seeds).}',
       '\\label{tab:transfer}', '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{llcc}', '\\toprule',
       'Direction & Target data & Scratch & Pre-trained \\\\', '\\midrule', *rows, '\\bottomrule', '\\end{tabular}}', '\\end{table}']
open(f'{OUT}/tab_transfer.tex', 'w').write('\n'.join(tex))
for g in ['rh_xyz_scratch_f0.1', 'rh_xyz_mpinit_f0.1', 'rh_xyz_scratch_f0.25', 'rh_xyz_mpinit_f0.25', 'rh_xyz_mpinit_f1.0',
          'mp_scratch_f0.25', 'mp_rhinit_f0.25', 'mp_rhinit_f1.0', 'rh_abl_xyz']:
    if g in G: macros['tr' + g.replace('_', '').replace('.', 'p').replace('0p1', 'Ten').replace('0p25', 'Quarter').replace('1p0', 'Full')] = f'{G[g].acc_mean:.2f}'

# ------------------------------------------------------------ protocol table
rows = []
for m, lab in [('stpt_lite', 'STPT-Lite'), ('pointnet', 'PointNet'), ('pointnet2', 'PointNet++')]:
    rows.append(f'{lab} & {gv("mp_" + m)} & {gv("mpo_official_" + m)} & {gv("mpo_random_" + m)} \\\\')
tex = ['\\begin{table}[t]', '\\centering', '\\caption{MiliPoint (49 actions) top-1 accuracy (\\%) of the same models under three splits. Published numbers: 34.45 (PointNet++, \\cite{cui2023milipoint}) and 98.25 (MAK-GCN, random 80/10/10, \\cite{zakka2025makgcn}).}',
       '\\label{tab:protocol}', '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{lccc}', '\\toprule',
       'Model & Subject-disjoint & Chronological & Random windows \\\\', ' & (ours, primary) & (MiliPoint code) & (leaky) \\\\', '\\midrule', *rows,
       '\\bottomrule', '\\end{tabular}}', '\\end{table}']
open(f'{OUT}/tab_protocol.tex', 'w').write('\n'.join(tex))
for g, k in [('mpo_random_stpt_lite', 'mpRandomAcc'), ('mpo_official_stpt_lite', 'mpOfficialAcc'), ('mpo_random_pointnet', 'mpRandomPnAcc'),
             ('mpo_official_pointnet2', 'mpOfficialPntwoAcc'), ('mpo_random_pointnet2', 'mpRandomPntwoAcc'), ('mp_pointnet2', 'mpSubjPntwoAcc')]:
    if g in G: macros[k] = f'{G[g].acc_mean:.2f}'

# ------------------------------------------------------------ headline macros
if 'rh_stpt_lite_kd_robust' in G:
    r = G['rh_stpt_lite_kd_robust']
    macros.update(rhOursAcc=f'{r.acc_mean:.2f}', rhOursStd=f'{r.acc_std:.2f}', rhOursFone=f'{r.f1_mean:.2f}',
                  oursParamsK=f'{r.params / 1e3:.1f}', oursMflops=f'{r.mflops:.1f}', oursLat=f'{r.lat_ms:.2f}',
                  oursIntMB=f'{r.int8_mb:.2f}', oursFpMB=f'{r.size_mb:.2f}', rhOursIntAcc=f'{r.int8_acc:.2f}')
if 'mp_stpt_lite_kd_robust' in G:
    r = G['mp_stpt_lite_kd_robust']
    macros.update(mpOursAcc=f'{r.acc_mean:.2f}', mpOursStd=f'{r.acc_std:.2f}', mpOursFone=f'{r.f1_mean:.2f}', mpOursTopthree=f'{r.top3:.2f}', mpOursIntAcc=f'{r.int8_acc:.2f}')
for name in ['radhar', 'milipoint']:
    s0 = S.get(name, {}).get('0')
    if s0:
        macros[name[:2] + 'CiLo'] = f'{s0["ref_ci"][0]:.2f}'; macros[name[:2] + 'CiHi'] = f'{s0["ref_ci"][1]:.2f}'
if os.path.exists('results/robustness.json'):
    rb = json.load(open('results/robustness.json'))
    for n, r in rb.items():
        k = ''.join(ch for ch in n if ch.isalpha())[:10]
        macros['rbKeepEighth' + k] = f'{r["keep"]["0.125"]:.2f}'; macros['rbKeepQuarter' + k] = f'{r["keep"]["0.25"]:.2f}'
        macros['rbSegNine' + k] = f'{r["segdrop"]["9"]:.2f}'; macros['rbSegSix' + k] = f'{r["segdrop"]["6"]:.2f}'
if os.path.exists('results/xai.json'):
    x = json.load(open('results/xai.json'))
    macros['xaiCorr'] = f'{x["att_occ_corr"]:.2f}'
    for k, v in x['silhouette'].items():
        macros['sil' + ''.join(ch for ch in k if ch.isalpha())] = f'{v:.2f}'

# ------------------------------------------------------------ leakage audit
if os.path.exists('results/leakage_audit.json'):
    au = json.load(open('results/leakage_audit.json')); rh = au['radhar']; mpa = au['milipoint']
    def esc(x): return str(x).replace('&', '\\&').replace('_', '\\_').replace('{', '').replace('}', '').replace("'", '')
    ok = lambda c: 'PASS' if c else 'FAIL'
    rows = [f"RadHAR recording overlap (train/val, train/test, val/test) & {esc(list(rh['recording_overlap'].values()))} & {ok(sum(rh['recording_overlap'].values()) == 0)} \\\\",
            f"RadHAR identical-content recordings & {len(rh['duplicate_recordings_any_split'])} groups & {ok(not rh['duplicate_recordings_any_split'])} \\\\",
            f"RadHAR identical frames ($\\geq$5 pts) across splits & {esc(list(rh['identical_frames'].values()))} of {sum(rh['frames_checked'].values()):,} frames & {ok(sum(rh['identical_frames'].values()) == 0)} \\\\",
            f"RadHAR windows of one recording in two splits & {esc(list(rh['window_recording_overlap'].values()))} & {ok(sum(rh['window_recording_overlap'].values()) == 0)} \\\\",
            f"Intensity statistics fitted on train only & recomputed = used: {rh['intensity_stats_train_only']['match']} & {ok(rh['intensity_stats_train_only']['match'])} \\\\",
            f"RadHAR subject independence & no subject IDs in dataset & NOT VERIFIABLE \\\\",
            f"RadHAR session independence & test-to-train gap min {rh['session']['test_min_gap_to_train_s']:.0f}\\,s, median {rh['session']['test_median_gap_to_train_s']:.0f}\\,s; same 4 days & NOT INDEPENDENT \\\\"]
    for pname, lab in [('subject', 'MiliPoint subject-disjoint'), ('official', 'MiliPoint chronological'), ('random', 'MiliPoint random windows')]:
        d = mpa[pname]; so = d.get('subject_overlap')
        sov = 'n/a' if so is None else ('none' if not any(so.values()) else 'train/test: ' + esc(so['train&test']))
        frac = 100 * d['fraction_test_frames_seen_in_training']
        st = 'PASS' if (frac == 0 and so is not None and not any(so.values())) else ('LEAKY' if pname == 'random' else 'SUBJECT OVERLAP')
        rows.append(f"{lab}: subject overlap; test frames inside training windows & {sov}; {d['test_frames_also_in_train_windows']:,} / {d['test_frames_total']:,} ({frac:.1f}\\%) & {st} \\\\")
    ms = au['model_selection']
    rows.append(f"Checkpoint = argmax validation macro-F1 & {ms['consistent']} / {ms['runs_checked']} runs & {ok(ms['consistent'] == ms['runs_checked'] and ms['runs_checked'] > 0)} \\\\")
    open(f'{OUT}/audit_rows.tex', 'w').write('\n'.join(rows))
    r = mpa['random']
    macros.update(mpAuditLeakFrames=f"{100 * r['fraction_test_frames_seen_in_training']:.1f}",
                  mpAuditLeakWins=f"{100 * r['test_windows_with_any_overlap']:.1f}",
                  mpAuditLeakOverlap=f"{100 * r['mean_max_overlap_test_window_with_a_train_window']:.1f}",
                  auditSelConsistent=str(ms['consistent']), auditSelRuns=str(ms['runs_checked']))

with open(f'{OUT}/numbers.tex', 'w') as f:
    for k, v in sorted(macros.items()):
        k = k.replace('2', 'Two').replace('1', 'One').replace('0', 'Zero').replace('.', '')
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(len(macros), 'macros'); print('\n'.join(f'{k}={v}' for k, v in sorted(macros.items())))

# ------------------------------------------------------------ development log -> LaTeX longtable rows
def devlog_rows(path='docs/DEVLOG.md'):
    rows = []
    for line in open(path):
        if not line.startswith('| ') or line.startswith('| #') or line.startswith('|---'):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) < 8 or not c[0].isdigit():
            continue
        def tx(t):
            t = t.replace('\\', '/').replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')
            t = t.replace('->', '$\\rightarrow$').replace('<=', '$\\leq$').replace('>=', '$\\geq$').replace('~', '$\\sim$')
            import re
            t = re.sub(r'`([^`]*)`', r'\\texttt{\1}', t)
            return t.replace('!=', '$\\neq$').replace('^2', '$^2$')
        rows.append(f'{c[0]} & {tx(c[2])} & {tx(c[3])} & {tx(c[4])} & {tx(c[5])} & {tx(c[6])} & {tx(c[7])} \\\\')
    open(f'{OUT}/devlog_rows.tex', 'w').write('\n'.join(rows))
    return len(rows)
print('devlog rows', devlog_rows())

# ------------------------------------------------------------ report-specific tables
for name, lab in [('radhar', 'tab:radharreport'), ('milipoint', 'tab:mpreport')]:
    t = open(f'{OUT}/tab_{name}.tex').read()
    t = t.replace('\\begin{table*}[t]', '\\begin{table}[h]').replace('\\end{table*}', '\\end{table}')
    t = t.replace(f'\\label{{tab:{name}}}', f'\\label{{{lab}}}').replace('Table~\\ref{tab:radhar}', 'Table~\\ref{tab:radharreport}')
    t = t.replace('\\begin{tabular}', '\\resizebox{\\textwidth}{!}{\\begin{tabular}').replace('\\end{tabular}', '\\end{tabular}}')
    open(f'{OUT}/tab_{name}_report.tex', 'w').write(t)

# per-class precision / recall for the proposed model (seed-mean)
import glob as _g
rows = []
cls = ['boxing', 'jumping jacks', 'jump', 'squats', 'walk']
ms = [json.load(open(f)) for f in sorted(_g.glob('results/runs/rh_stpt_lite_kd_robust_s[0-9]/metrics.json'))]
if ms:
    P = np.mean([m['test_per_class']['precision'] for m in ms], 0); Rr = np.mean([m['test_per_class']['recall'] for m in ms], 0)
    F_ = np.mean([m['test_per_class']['f1'] for m in ms], 0); Sp = ms[0]['test_per_class']['support']
    rows = [f'{c} & {100 * p:.2f} & {100 * r:.2f} & {100 * f:.2f} & {int(s)} \\\\' for c, p, r, f, s in zip(cls, P, Rr, F_, Sp)]
mp_ms = [json.load(open(f)) for f in sorted(_g.glob('results/runs/mp_stpt_lite_kd_robust_s[0-9]/metrics.json'))]
mp_line = ''
if mp_ms:
    R49 = np.mean([m['test_per_class']['recall'] for m in mp_ms], 0)
    o = np.argsort(R49)
    mp_line = (f"On MiliPoint the per-action recall of the proposed model (mean over seeds) ranges from {100 * R49[o[0]]:.1f}\\% "
               f"(action {o[0]}) to {100 * R49[o[-1]]:.1f}\\% (action {o[-1]}); {int((R49 >= 0.5).sum())} of 49 actions are recalled at $\\geq$50\\% "
               f"and {int((R49 < 0.2).sum())} below 20\\%. The weakest actions are confused with kinematically similar ones (off-diagonal clusters in Fig.~\\ref{{fig:confreport}}).")
open(f'{OUT}/perclass_report.tex', 'w').write('\n'.join([
    '\\begin{table}[h]', '\\centering', '\\caption{Per-class test precision, recall and F1 (\\%) of the proposed model (STPT-Lite+KD+robust) on RadHAR (mean over seeds).}',
    '\\label{tab:perclass}', '\\begin{tabular}{lrrrr}', '\\toprule', 'Class & Precision & Recall & F1 & Windows \\\\', '\\midrule', *rows,
    '\\bottomrule', '\\end{tabular}', '\\end{table}', mp_line]))

# robustness summary table (accuracy, macro recall, stability) for the report
if os.path.exists('results/robustness_summary.csv'):
    R = pd.read_csv('results/robustness_summary.csv')
    show = [('clean', 0), ('jitter', 0.05), ('dropout', 0.3), ('sparsity', 0.25), ('outliers', 0.1), ('density', 0.25), ('frameloss', 0.5), ('combined', 2)]
    for ds_ in ['radhar', 'milipoint']:
        d = R[R.dataset == ds_]
        models = [m for m in ['pointnet', 'pointnet_robust', 'pointnet2', 'stpt_lite_kd', 'stpt_lite_kd_robust'] if m in set(d.model)]
        lines = []
        for c, l in show:
            cells = []
            for m in models:
                r = d[(d.model == m) & (d.condition == c) & (np.isclose(d.level, l))]
                cells.append(f'{r.acc.iloc[0]:.1f} / {r.stability.iloc[0]:.0f}' if len(r) else '--')
            lines.append(f'{c} ({l}) & ' + ' & '.join(cells) + ' \\\\')
        nm = {'pointnet': 'PointNet', 'pointnet_robust': 'PointNet+rob.', 'pointnet2': 'PointNet++', 'stpt_lite_kd': 'STPT+KD', 'stpt_lite_kd_robust': 'STPT+KD+rob.'}
        tex = ['\\begin{table}[h]', '\\centering', f"\\caption{{Robustness on {'RadHAR' if ds_ == 'radhar' else 'MiliPoint (unseen subjects)'}: test accuracy (\\%) / prediction stability (\\% of predictions unchanged w.r.t.\\ clean), mean over seeds.}}",
               f'\\label{{tab:rob{ds_}}}', '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{l' + 'c' * len(models) + '}', '\\toprule',
               'Condition (level) & ' + ' & '.join(nm[m] for m in models) + ' \\\\', '\\midrule', *lines, '\\bottomrule', '\\end{tabular}}', '\\end{table}']
        open(f'{OUT}/tab_robust_{ds_}.tex', 'w').write('\n'.join(tex))
    open(f'{OUT}/tab_robust_report.tex', 'w').write('\\input{../paper/generated/tab_robust_milipoint.tex}\n\\input{../paper/generated/tab_robust_radhar.tex}\n')

# demo screenshots for the report
figs = [('demo_input', 'Tab 1: raw window with antenna returns, gated clutter and kept person returns; pre-processed model input.'),
        ('demo_recognition', 'Tab 2: live recording-level prediction, latency and per-window class probabilities.'),
        ('demo_noise', 'Tab 3: controlled noise injection; clean vs.\\ noisy point cloud and predictions.'),
        ('demo_model', 'Tab 4: model card with held-out test metrics and confusion matrix.')]
open(f'{OUT}/demo_figs.tex', 'w').write('\n'.join(
    f'\\begin{{figure}}[h]\\centering\\includegraphics[width=0.92\\textwidth]{{demo/{f}.png}}\\caption{{Demonstration (real screenshot), {c}}}\\label{{fig:{f}}}\\end{{figure}}'
    for f, c in figs if os.path.exists(f'figures/demo/{f}.png')))
print('report tables written')
