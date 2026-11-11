"""Headless functional test of the Streamlit demo (streamlit.testing AppTest) -> results/apptest.json.
Exercises: both bundles, every bundled sample, every noise condition. Records exceptions and errors shown."""
import json, time
from streamlit.testing.v1 import AppTest

checks, t0 = [], time.time()


def run(at, label):
    at.run(timeout=600)
    checks.append(dict(step=label, exceptions=len(at.exception), errors=len(at.error),
                       detail=[str(e.value)[:200] for e in at.exception] + [str(e.value)[:200] for e in at.error]))
    return at


at = run(AppTest.from_file('../demo/app.py', default_timeout=600), 'start (default bundle + sample)')
models = at.sidebar.selectbox[0].options
for b in models:
    at.sidebar.selectbox[0].select(b); run(at, f'bundle {b}')
    for s in at.sidebar.selectbox[1].options:
        at.sidebar.selectbox[1].select(s); run(at, f'{b} / sample {s}')
    conds = at.selectbox(key=None) if False else [w for w in at.selectbox if w.label == 'Perturbation'][0].options
    for c in conds:
        [w for w in at.selectbox if w.label == 'Perturbation'][0].select(c); run(at, f'{b} / noise {c}')
res = dict(n_steps=len(checks), n_failed=sum(c['exceptions'] + c['errors'] > 0 for c in checks),
           seconds=round(time.time() - t0, 1), models=list(models), checks=checks)
json.dump(res, open('results/apptest.json', 'w'), indent=1)
print('steps', res['n_steps'], 'failed', res['n_failed'], 'time', res['seconds'])
