"""Streamlit demonstration: privacy-preserving HAR from mmWave radar point clouds.

Every prediction shown here is computed live by the exported, trained STPT-Lite
model (checkpoints/), on the actual dataset formats:
  * RadHAR raw ROS point-cloud text dumps (.txt)  -> 5 activities
  * MiliPoint frames (.npz with pts [N,3], frame [N]) -> 49 actions
Run:  streamlit run demo/app.py
"""
import os, sys, json, io, time
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from inference import Bundle
from data_radhar import roi_filter
import noise

st.set_page_config(page_title='mmWave Radar HAR', layout='wide')
CK = os.path.join(ROOT, 'checkpoints')
SAMPLES = os.path.join(ROOT, 'demo', 'samples')
NOISE_HELP = {
    'clean': 'unmodified input', 'jitter': 'Gaussian coordinate noise, sigma in metres',
    'dropout': 'fraction of returns removed', 'sparsity': 'fraction of unique returns kept',
    'outliers': 'fraction of returns replaced by false detections', 'density': 'minimum per-segment keep fraction',
    'frameloss': 'fraction of temporal segments lost', 'combined': 'severity 1-3 of all of the above'}


@st.cache_resource
def load_bundle(name):
    return Bundle(os.path.join(CK, name))


def available_bundles():
    if not os.path.isdir(CK):
        return []
    b = [d for d in os.listdir(CK) if os.path.exists(os.path.join(CK, d, 'bundle.json'))]
    return sorted(b, key=lambda d: (0 if 'radhar' in d else 1, d))   # RadHAR model first (richest visualisation)


def read_input(b, uploaded, sample_name):
    """Returns pts [N,5], frame [N], ground-truth label (or None), source description."""
    meta = json.load(open(os.path.join(SAMPLES, 'samples.json')))
    if uploaded is not None:
        name, raw = uploaded.name, uploaded.getvalue()
        label = None
    else:
        name, raw = sample_name, open(os.path.join(SAMPLES, sample_name), 'rb').read()
        label = meta[sample_name]['label']
    if name.endswith('.txt'):
        if b.pre['dataset'] != 'radhar':
            raise ValueError('a .txt RadHAR recording needs the RadHAR model')
        from radhar_parse import parse_text
        pts, frame, _ = parse_text(raw.decode('utf-8', errors='replace'))
    elif name.endswith('.npz'):
        if b.pre['dataset'] != 'milipoint':
            raise ValueError('a MiliPoint .npz sample needs the MiliPoint model')
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        if 'pts' not in z or 'frame' not in z:
            raise ValueError('npz must contain arrays "pts" [N,3] and "frame" [N]')
        p3 = z['pts'].astype(np.float32)
        if p3.ndim != 2 or p3.shape[1] != 3 or len(p3) != len(z['frame']):
            raise ValueError(f'invalid shapes pts {p3.shape}, frame {z["frame"].shape}')
        pts = np.concatenate([p3, np.zeros((len(p3), 2), np.float32)], 1)
        frame = z['frame'].astype(np.int32)
    else:
        raise ValueError('unsupported file type (use RadHAR .txt or MiliPoint .npz)')
    return pts, frame, label, name


def scatter3d(p, color, title, cmap='Viridis', size=2.5, extra=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=p[:, 1], y=p[:, 0], z=p[:, 2], mode='markers',
                               marker=dict(size=size, color=color, colorscale=cmap, showscale=False), name='points'))
    for tr in extra or []:
        fig.add_trace(tr)
    fig.update_layout(title=title, height=380, margin=dict(l=0, r=0, t=40, b=0), showlegend=bool(extra),
                      scene=dict(xaxis_title='y lateral (m)', yaxis_title='x range (m)', zaxis_title='z height (m)',
                                 aspectmode='data'), legend=dict(orientation='h'))
    return fig


def prob_bar(p, classes, top=10):
    o = np.argsort(-p)[:top]
    fig = go.Figure(go.Bar(x=[100 * p[i] for i in o][::-1], y=[classes[i] for i in o][::-1], orientation='h',
                           marker_color='#d7301f'))
    fig.update_layout(height=60 + 28 * len(o), margin=dict(l=0, r=0, t=10, b=0), xaxis_title='softmax probability (%)')
    return fig


# =============================================================== sidebar
st.sidebar.title('mmWave Radar HAR')
bundles = available_bundles()
if not bundles:
    st.error('No exported model found in checkpoints/. Run scripts/export_model.py first.')
    st.stop()
bname = st.sidebar.selectbox('Trained model', bundles)
b = load_bundle(bname)
ds_name = b.pre['dataset']
meta = json.load(open(os.path.join(SAMPLES, 'samples.json')))
own = [k for k in meta if k.startswith(ds_name)]
sample = st.sidebar.selectbox('Bundled test sample (ground truth known)', own)
up = st.sidebar.file_uploader('...or upload a recording', type=['txt', 'npz'],
                              help='RadHAR raw ROS text dump (.txt) or MiliPoint npz with pts [N,3] and frame [N]')
st.sidebar.caption('Bundled samples come from the held-out TEST split '
                   '(RadHAR official test recordings / MiliPoint test subjects 8-10).')

st.title('Privacy-preserving activity recognition from mmWave radar point clouds')
st.caption(f'Model: **{bname}** - live inference with the saved checkpoint `{os.path.relpath(b.path, ROOT)}/model.pt`')

try:
    pts, frame, label, src = read_input(b, up, sample)
    x, m, starts, ds = b.windows_from_frames(pts, frame)
except Exception as e:  # invalid uploads are reported, never crash the app
    st.error(f'Input rejected: {e}')
    st.stop()

nwin = len(starts)
tabs = st.tabs(['1  Input & pre-processing', '2  Recognition', '3  Noise robustness', '4  Model & results'])

# ================================================== tab 1: input + pre-processing
with tabs[0]:
    c0, c1, c2 = st.columns(3)
    c0.metric('Frames', int(frame.max()) + 1); c1.metric('Radar returns', len(pts)); c2.metric('2-s windows', nwin)
    wi = st.slider('Window', 0, nwin - 1, min(nwin // 2, nwin - 1), key='w1')
    s0 = starts[wi]
    sel = (frame >= s0) & (frame < s0 + b.pre['win'])
    raw = pts[sel]
    col_a, col_b = st.columns(2)
    if b.pre['roi']:
        r = roi_filter(raw)
        segs = ds.raw_window(wi); ctr = ds.person_centre(segs)
        kept = raw[r]
        gated = np.linalg.norm(kept[:, :2] - ctr[:2], axis=1) <= (b.pre['gate'] or 1e9)
        cat = np.zeros(len(raw)); cat[~r] = 0; idx = np.where(r)[0]; cat[idx[~gated]] = 1; cat[idx[gated]] = 2
        fig = go.Figure()
        for v, nm, col in [(0, 'antenna self-return (ROI removed)', '#d1495b'), (1, 'static clutter (gated)', '#aaaaaa'),
                           (2, 'person (kept)', '#2f6690')]:
            q = raw[cat == v]
            fig.add_trace(go.Scatter3d(x=q[:, 1], y=q[:, 0], z=q[:, 2], mode='markers', marker=dict(size=2.5, color=col), name=nm))
        fig.update_layout(title=f'Raw window: {len(raw)} returns', height=380, margin=dict(l=0, r=0, t=40, b=0),
                          legend=dict(orientation='h'), scene=dict(aspectmode='data', xaxis_title='y (m)', yaxis_title='x (m)', zaxis_title='z (m)'))
        col_a.plotly_chart(fig, width='stretch', key='chart143')
        col_a.caption(f'ROI removed {int((~r).sum())}, gating removed {int((~gated).sum())}, kept {int(gated.sum())} returns. '
                      f'Person centre = median of moving (Doppler != 0) returns.')
    else:
        col_a.plotly_chart(scatter3d(raw, raw[:, 2], f'Raw window: {len(raw)} returns'), width='stretch', key='chart147')
    xi = x[wi].reshape(-1, 6).numpy(); mi = m[wi].reshape(-1).numpy() > 0
    col_b.plotly_chart(scatter3d(xi[mi], xi[mi, 5], f'Model input: {b.pre["G"]} segments x {b.pre["P"]} points (colour = time)'),
                       width='stretch')
    col_b.caption('Coordinates are person-centred; returns are sampled per temporal segment.')

# ============================================================ tab 2: recognition
with tabs[1]:
    b.predict(x[:1], m[:1])                       # warm-up (first call allocates)
    p, ms = b.predict(x, m)
    ms1 = float(np.median([b.predict(x[:1], m[:1])[1] for _ in range(20)]))
    vote = np.bincount(p.argmax(1), minlength=len(b.classes))
    pred = b.classes[vote.argmax()]
    c0, c1, c2, c3 = st.columns(4)
    c0.metric('Recording-level prediction', pred)
    c1.metric('Ground truth', label if label else 'unknown (upload)')
    c2.metric('Latency, 1 window', f'{ms1:.1f} ms')
    c3.metric(f'Latency, {nwin} windows', f'{ms:.1f} ms')
    if label:
        acc = np.mean([b.classes[i] == label for i in p.argmax(1)])
        st.write(f'Window-level agreement with the ground truth on this recording: **{100 * acc:.1f} %** ({nwin} windows).')
    wi2 = st.slider('Inspect window', 0, nwin - 1, min(nwin // 2, nwin - 1), key='w2')
    ca, cb = st.columns([1, 1.4])
    ca.plotly_chart(prob_bar(p[wi2], b.classes), width='stretch', key='chart169')
    timeline = go.Figure(go.Heatmap(z=p.T * 100, x=[s / b.pre['fps'] for s in starts], y=b.classes, colorscale='Blues',
                                    colorbar=dict(title='%')))
    timeline.update_layout(height=320 if len(b.classes) <= 5 else 520, margin=dict(l=0, r=0, t=30, b=0),
                           title='Class probabilities over time', xaxis_title='window start (s)')
    cb.plotly_chart(timeline, width='stretch', key='chart174')
    st.info('Probabilities are the model\'s softmax outputs. They express the model\'s relative preference '
            'and are not calibrated measures of correctness; verified accuracy is the test-set figure in tab 4.')

# ======================================================== tab 3: noise robustness
with tabs[2]:
    c0, c1, c2 = st.columns(3)
    cond = c0.selectbox('Perturbation', list(noise.CONDITIONS)[1:], help='See src/noise.py')
    lv = noise.CONDITIONS[cond][0]
    level = c1.select_slider(f'Level ({noise.CONDITIONS[cond][1]})', options=lv, value=lv[len(lv) // 2])
    seed = c2.number_input('Noise seed', 0, 9999, 0)
    st.caption(NOISE_HELP[cond])
    p0, _ = b.predict(x, m)
    pn, msn, xn, mn = b.predict_with_noise(x, m, cond, level, int(seed))
    same = np.mean(p0.argmax(1) == pn.argmax(1))
    cols = st.columns(4)
    cols[0].metric('Prediction (clean)', b.classes[np.bincount(p0.argmax(1), minlength=len(b.classes)).argmax()])
    cols[1].metric('Prediction (noisy)', b.classes[np.bincount(pn.argmax(1), minlength=len(b.classes)).argmax()])
    cols[2].metric('Window predictions unchanged', f'{100 * same:.1f} %')
    if label:
        a0 = np.mean([b.classes[i] == label for i in p0.argmax(1)]); a1 = np.mean([b.classes[i] == label for i in pn.argmax(1)])
        cols[3].metric('Window accuracy clean -> noisy', f'{100 * a1:.1f} %', f'{100 * (a1 - a0):+.1f} pts')
    wi3 = st.slider('Window', 0, nwin - 1, min(nwin // 2, nwin - 1), key='w3')
    ca, cb = st.columns(2)
    xc = x[wi3].reshape(-1, 6).numpy(); mc = m[wi3].reshape(-1).numpy() > 0
    xnn = xn[wi3].reshape(-1, 6).numpy(); mnn = mn[wi3].reshape(-1).numpy() > 0
    ca.plotly_chart(scatter3d(xc[mc], xc[mc, 5], 'Clean model input'), width='stretch', key='chart200')
    cb.plotly_chart(scatter3d(xnn[mnn], xnn[mnn, 5], f'{cond} = {level}'), width='stretch', key='chart201')
    ca.plotly_chart(prob_bar(p0[wi3], b.classes, 5), width='stretch', key='chart202')
    cb.plotly_chart(prob_bar(pn[wi3], b.classes, 5), width='stretch', key='chart203')
    rb = os.path.join(ROOT, 'results', 'robustness_summary.csv')
    if os.path.exists(rb):
        import pandas as pd
        st.subheader('Test-set robustness (all test windows, 3 seeds)')
        st.dataframe(pd.read_csv(rb), width='stretch')

# ======================================================== tab 4: model + results
with tabs[3]:
    mt = json.load(open(os.path.join(b.path, 'metrics.json')))
    bj = b.cfg
    c = st.columns(5)
    c[0].metric('Parameters', f"{mt['params'] / 1e3:.1f} K"); c[1].metric('FP32 size', f"{mt['size_mb']:.2f} MB")
    c[2].metric('INT8 size', f"{mt['int8']['size_mb']:.2f} MB"); c[3].metric('MFLOPs / window', f"{mt['mflops']:.1f}")
    c[4].metric('CPU latency (1 thread)', f"{mt['latency_ms_cpu1']:.2f} ms")
    t = mt['test']
    st.subheader('Held-out test performance of this checkpoint')
    c = st.columns(5)
    for i, (k, n) in enumerate([('acc', 'Accuracy'), ('macro_f1', 'Macro-F1'), ('macro_precision', 'Macro precision'),
                                ('macro_recall', 'Macro recall'), ('top3', 'Top-3')]):
        c[i].metric(n, f'{100 * t[k]:.2f} %')
    st.caption(f"Exported run: `{bj['source_run']}` (selected by {bj['selected_by']}). "
               f"Test split: {'RadHAR official test recordings' if ds_name == 'radhar' else 'MiliPoint subjects 8-10 (never seen in training)'}.")
    cm = np.array(mt['test_cm']); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(z=cmn * 100, x=b.classes, y=b.classes, colorscale='Blues', colorbar=dict(title='%'),
                               text=cm if len(b.classes) <= 5 else None, texttemplate='%{text}' if len(b.classes) <= 5 else None))
    fig.update_layout(height=420 if len(b.classes) <= 5 else 700, xaxis_title='predicted', yaxis_title='true',
                      yaxis_autorange='reversed', margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch', key='chart231')
    with st.expander('Architecture and training configuration'):
        st.markdown('**STPT-Lite (KD + noise-robust training)**: per-point MLP (6->64) -> intra-segment multi-head self-attention with '
                    'relative-position bias -> attentive + max pooling -> + motion descriptor -> [CLS] + 12 segment tokens '
                    '-> 2 Transformer blocks (d=64, 4 heads) -> LayerNorm -> linear classifier. '
                    'Trained with knowledge distillation from STPT-Large and noise-robust augmentation.')
        st.json({'model': bj['model_args'], 'preprocessing': b.pre, 'training': mt['args'], 'config': mt.get('config')})
