"""Export a trained run as a self-contained inference bundle in checkpoints/.
The seed to export is chosen by VALIDATION macro-F1 (never by test score)."""
import json, os, shutil, sys, argparse
sys.path.insert(0, 'src')
from data_radhar import CLASSES, get_radhar
from train import model_kwargs

def export(group, out, dataset):
    cands = [f'results/runs/{group}_s{s}' for s in range(3) if os.path.exists(f'results/runs/{group}_s{s}/metrics.json')]
    best = max(cands, key=lambda r: json.load(open(f'{r}/metrics.json'))['val']['macro_f1'])
    m = json.load(open(f'{best}/metrics.json')); a = argparse.Namespace(**m['args'])
    if dataset == 'radhar':
        dtr, _, _ = get_radhar(a.G, a.P, a.feats, a.roi, a.center, a.seed, gate=a.gate if a.gate > 0 else None)
        pre = dict(dataset='radhar', win=60, win_s=2.0, stride=10, G=a.G, P=a.P, feats=a.feats, roi=bool(a.roi),
                   center=bool(a.center), gate=a.gate if a.gate > 0 else None, int_stats=list(dtr.int_stats), fps=30)
        classes = CLASSES
    else:
        pre = dict(dataset='milipoint', win=48, win_s=2.0, stride=12, G=a.G, P=a.P, feats='xyz', roi=False,
                   center=True, gate=None, int_stats=[0.0, 1e-6], fps=24)
        classes = [f'action_{i:02d}' for i in range(49)]
    os.makedirs(out, exist_ok=True)
    shutil.copy(f'{best}/model.pt', f'{out}/model.pt'); shutil.copy(f'{best}/metrics.json', f'{out}/metrics.json')
    json.dump(dict(source_run=best, selected_by='validation macro-F1 among seeds 0-2', classes=classes,
                   model_args=dict(model=a.model, kwargs=model_kwargs(a)), preprocessing=pre,
                   test_accuracy=m['test']['acc'], params=m['params']), open(f'{out}/bundle.json', 'w'), indent=1)
    print('exported', best, '->', out)

if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2], sys.argv[3])
