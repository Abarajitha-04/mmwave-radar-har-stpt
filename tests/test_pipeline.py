"""Unit / integration tests: python -m pytest -q tests"""
import glob, json, os, pickle, io
import numpy as np, pytest, torch

SAMPLE = 'demo/samples/radhar_test_jump.txt'
HAVE_RADHAR = os.path.exists('data/processed/radhar_frames.npy')
HAVE_MP = os.path.exists('data/processed/milipoint_runs.npy')


# ------------------------------------------------------------------ parsing
def test_parse_sample_recording():
    from radhar_parse import parse_text
    pts, frame, ft = parse_text(open(SAMPLE).read())
    assert pts.shape[1] == 5 and len(pts) == len(frame)
    assert frame.min() == 0 and frame.max() + 1 >= 359
    assert np.all(np.diff(frame) >= 0)            # frames in order
    assert np.isfinite(pts).all()


def test_parse_rejects_garbage():
    from radhar_parse import parse_text
    with pytest.raises(ValueError):
        parse_text('this is not a radar file')


def test_safe_pickle_blocks_code():
    import safe_pickle, tempfile
    class Evil:
        def __reduce__(self):
            return (os.system, ('echo pwned',))
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl'); pickle.dump(Evil(), f); f.close()
    with pytest.raises(pickle.UnpicklingError):
        safe_pickle.load(f.name)


# ------------------------------------------------------------------ data
@pytest.mark.skipif(not HAVE_RADHAR, reason='processed RadHAR not available')
def test_radhar_splits_disjoint_and_shapes():
    from data_radhar import get_radhar, load_recordings, split_recordings
    tr, va, te = split_recordings(load_recordings())
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    dtr, dva, dte = get_radhar()
    x, m, y = dte[0]
    assert x.shape == (12, 32, 6) and m.shape == (12, 32) and 0 <= y < 5
    assert torch.isfinite(x).all()
    # evaluation samples are deterministic (no augmentation outside training)
    assert torch.equal(dte[5][0], dte[5][0]) and torch.equal(dva[3][0], dva[3][0])
    # standardisation statistics come from the training split
    assert dva.int_stats == dtr.int_stats == dte.int_stats


@pytest.mark.skipif(not HAVE_MP, reason='processed MiliPoint not available')
def test_milipoint_subject_disjoint():
    from data_milipoint import load_runs, split_runs
    runs = load_runs(); tr, va, te, rr = split_runs(runs, 'subject')
    s = lambda ks: {rr[k]['subject'] for k in ks}
    assert not (s(tr) & s(te)) and not (s(tr) & s(va)) and not (s(va) & s(te))
    assert s(te) == {8, 9, 10}


# ------------------------------------------------------------------ noise
def _batch(B=4):
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, 12, 32, 6, generator=g), torch.ones(B, 12, 32)


@pytest.mark.parametrize('cond', ['jitter', 'dropout', 'sparsity', 'outliers', 'density', 'frameloss', 'combined'])
def test_noise_shapes_and_determinism(cond):
    import noise
    x, m = _batch(); lv = noise.CONDITIONS[cond][0][-1]
    a = noise.perturb(x, m, cond, lv, torch.Generator().manual_seed(1))
    b = noise.perturb(x, m, cond, lv, torch.Generator().manual_seed(1))
    assert a[0].shape == x.shape and a[1].shape == m.shape
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(x, _batch()[0])              # input not modified in place


def test_noise_effects():
    import noise
    x, m = _batch(64); g = torch.Generator().manual_seed(0)
    _, md = noise.dropout(x, m, 0.3, g); assert 0.6 < md.mean() < 0.8
    _, mf = noise.frameloss(x, m, 0.5, g); assert (mf.max(-1)[0] == 0).sum() == 64 * 6
    xs, _ = noise.sparsity(x, m, 0.125, g)
    uniq = torch.tensor([len(torch.unique(xs[0, 0, :, 0]))]); assert uniq.item() <= 4


# ------------------------------------------------------------------ models
@pytest.mark.parametrize('name', ['pointnet', 'pointnet_lstm', 'pointnet2', 'dgcnn', 'pointtransformer',
                                  'voxel_tdcnn', 'stpt_lite', 'stpt_tiny', 'stpt_teacher'])
def test_model_forward_with_empty_segments(name):
    from models import build
    net = build(name, 5).eval(); x, m = _batch(2); m[0, 4:] = 0; x[0, 4:] = 0
    with torch.no_grad():
        out = net(x, m)
    assert out.shape == (2, 5) and torch.isfinite(out).all()


# ------------------------------------------------------------------ inference
BUNDLES = [d for d in glob.glob('checkpoints/*') if os.path.exists(f'{d}/bundle.json')]


@pytest.mark.skipif(not BUNDLES, reason='no exported model')
def test_bundle_inference_on_real_sample():
    from inference import Bundle
    b = Bundle([d for d in BUNDLES if 'radhar' in d][0])
    x, m, starts, _ = b.windows_from_radhar_text(open(SAMPLE).read())
    p, ms = b.predict(x, m)
    assert p.shape == (len(starts), 5) and np.allclose(p.sum(1), 1, atol=1e-4)
    p2, _ = b.predict(x, m); assert np.allclose(p, p2)       # deterministic
    with pytest.raises(ValueError):
        b.windows_from_frames(np.zeros((10, 5), np.float32), np.zeros(10, np.int32))   # too short
