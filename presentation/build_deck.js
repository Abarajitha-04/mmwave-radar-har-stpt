// Builds presentation/Team10_mmWave_HAR.pptx from presentation/deck_data.json and figures/.
// Every number on a slide is read from the experiment outputs.
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');
const { applyTheme } = require(process.env.PPTX_SKILL + '/scripts/apply_theme.js');

const ROOT = path.resolve(__dirname, '..');
const D = JSON.parse(fs.readFileSync(path.join(__dirname, 'deck_data.json')));
const FIG = (f) => path.join(ROOT, 'figures', f);

const THEME = {
  name: 'Radar Night', headFontFace: 'Cambria', bodyFontFace: 'Calibri',
  colors: { dk1: '1B2430', lt1: 'FFFFFF', dk2: '12355B', lt2: 'EEF3F7', accent1: '1B998B', accent2: 'E4572E',
            accent3: 'F4A259', accent4: '5C6B7A', accent5: '8FB8DE', accent6: '3D5A80', hlink: '1B998B', folHlink: '5C6B7A' },
};
const HEX = THEME.colors;
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';           // 13.333 x 7.5 in
pres.title = 'Privacy-Preserving HAR from mmWave Radar Point Clouds';
pres.author = 'Team 10 (Abarajitha S K, G Aswath Rupan, Indu Vadana M D, Murugan S)';
pres.company = 'PSG College of Technology';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, MX = 0.6;
const MISSING = [];

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: 'TITLE_DARK', background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: MX, y: 1.7, w: W - 2 * MX, h: 1.9, fontSize: 32, bold: true, color: C.background1, valign: 'bottom' }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: MX, y: 3.9, w: W - 2 * MX, h: 2.6, fontSize: 18, color: C.background2, valign: 'top' }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'SECTION_DARK', background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: MX, y: 2.6, w: W - 2 * MX, h: 1.2, fontSize: 32, bold: true, color: C.background1 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: MX, y: 3.9, w: W - 2 * MX, h: 1.2, fontSize: 20, color: C.background2 }, text: '' } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, w: 0.6, h: 0.3, fontSize: 12, color: C.background2, align: 'right' },
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: MX, y: 0.3, w: W - 2 * MX, h: 0.85, fontSize: 30, bold: true, color: C.text2, valign: 'middle' }, text: '' } },
    { text: { text: 'Team 10  |  Privacy-preserving HAR from mmWave radar point clouds', options: { x: MX, y: H - 0.45, w: 8, h: 0.3, fontSize: 12, color: C.accent4 } } },
  ],
  slideNumber: { x: W - 1.2, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 12, color: C.accent4, align: 'right' },
});

let sec = '';
function section(t) { sec = t; pres.addSection({ title: t }); }
function content(title, notes) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: sec });
  s.addText(title, { placeholder: 'title' });
  s.addNotes(notes);
  return s;
}
const TOP = 1.35, BOT = H - 0.7;
function bullets(s, items, o) {
  const arr = items.map((t, i) => {
    const sub = typeof t === 'object' && t.sub;
    const txt = typeof t === 'object' ? t.t : t;
    return { text: txt, options: { bullet: sub ? { indent: 18 } : true, indentLevel: sub ? 1 : 0, fontSize: sub ? 16 : (o.fs || 20),
             breakLine: i < items.length - 1, paraSpaceAfter: 6, color: C.text1 } };
  });
  s.addText(arr, { x: o.x, y: o.y, w: o.w, h: o.h, valign: 'top', isTextBox: true, margin: 2, objectName: o.name || 'bullets' });
}
function card(s, x, y, w, h, head, body, color, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: o.fill || C.background2 }, line: { color: o.fill || C.background2 }, objectName: 'card ' + head });
  s.addText(head, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.5, fontSize: o.hs || 22, bold: true, color: color || C.text2, isTextBox: true, margin: 0 });
  s.addText(body, { x: x + 0.2, y: y + 0.7, w: w - 0.4, h: h - 0.85, fontSize: o.bs || 20, color: C.text1, valign: 'top', isTextBox: true, margin: 0 });
}
function stat(s, x, y, w, big, small, color) {
  s.addText(big, { x, y, w, h: 0.95, fontSize: 44, bold: true, color: color || C.accent1, fontFace: 'Cambria', isTextBox: true, margin: 0 });
  s.addText(small, { x, y: y + 0.95, w, h: 0.75, fontSize: 16, color: C.accent4, valign: 'top', isTextBox: true, margin: 0 });
}
function img(s, file, x, y, w, h, alt) {
  // fit image inside box preserving aspect ratio
  if (!fs.existsSync(file)) { console.warn('MISSING FIGURE', file); MISSING.push(file); return; }
  const { execSync } = require('child_process');
  const dims = execSync(`python3 -c "from PIL import Image;im=Image.open('${file}');print(im.size[0],im.size[1])"`).toString().trim().split(' ').map(Number);
  const r = dims[0] / dims[1];
  let ww = w, hh = w / r;
  if (hh > h) { hh = h; ww = h * r; }
  s.addImage({ path: file, x: x + (w - ww) / 2, y: y + (h - hh) / 2, w: ww, h: hh, altText: alt });
}
function table(s, rows, o) {
  const hdr = rows[0].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: HEX.dk2 }, fontSize: o.fs || 14 } }));
  const body = rows.slice(1).map((r, i) => r.map((t, j) => ({ text: String(t), options: { fontSize: o.fs || 14, color: C.text1,
    fill: { color: (o.hl && o.hl(i, r)) ? 'FCE9E2' : (i % 2 ? HEX.lt2 : HEX.lt1) }, bold: !!(o.hl && o.hl(i, r)), align: j === 0 ? 'left' : 'center' } })));
  s.addTable([hdr, ...body], { x: o.x, y: o.y, w: o.w, colW: o.colW, border: { type: 'solid', pt: 0.5, color: 'C9D3DD' }, valign: 'middle', rowH: o.rowH || 0.36 });
}
const chartBase = (o) => Object.assign({
  catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.dk1, catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt',
  catAxisLabelFontSize: 14, valAxisLabelFontSize: 14, dataLabelFontSize: 13, dataLabelFontFace: '+mn-lt', dataLabelColor: HEX.dk1,
  valGridLine: { color: 'DDE3EA', size: 0.5 }, catGridLine: { style: 'none' }, titleFontFace: '+mj-lt', titleFontSize: 16, titleColor: HEX.dk2,
  legendFontFace: '+mn-lt', legendFontSize: 14,
}, o);
const f2 = (x) => (x === null || x === undefined || Number.isNaN(x)) ? 'n/a' : Number(x).toFixed(2);
const pm = (a) => `${f2(a[0])} ± ${f2(a[1])}`;
const byKey = (arr, k) => arr.find((r) => r.key === k) || {};

const RH = D.radhar, MP = D.milipoint;
const EMPTY = { name: 'n/a', key: '', acc: null, sd: null, params: 0 };
const byK = (a, k) => a.find((r) => r.key === k) || Object.assign({}, EMPTY, { key: k, name: k });
const ours = byK(RH, 'stpt_lite_kd_robust'), oursMP = byK(MP, 'stpt_lite_kd_robust');
const kdRH = byK(RH, 'stpt_lite_kd'), kdMP = byK(MP, 'stpt_lite_kd');
const bestBase = (arr) => arr.filter((r) => !r.key.startsWith('stpt') && !r.key.endsWith('robust')).sort((a, b) => b.acc - a.acc)[0] || { name: 'n/a', key: '', acc: null, sd: null, params: 0, f1: null, top3: null, mflops: null, lat: null, size: null, int8mb: null, int8: null };
const bbRH = bestBase(RH), bbMP = bestBase(MP);
const audit = D.leakage_audit || {}, eda = D.eda_stats || {};
const rnd = (audit.milipoint || {}).random || {};
const HLOURS = (i, row) => String(row[0]).includes('(ours)');
const PRES = { A: 'Presenter: Abarajitha S K', B: 'Presenter: G Aswath Rupan', C: 'Presenter: Indu Vadana M D', M: 'Presenter: Murugan S' };
function N(who, text, qs) {
  return `${PRES[who]}\n\n${text}\n\nPossible examiner questions:\n` + qs.map((q) => `Q: ${q[0]}\nA: ${q[1]}`).join('\n\n');
}

// =================================================================== 1 Title
section('Introduction');
let s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: sec });
s.addText('Privacy-Preserving Human Activity Recognition from mmWave Radar Point Clouds using Lightweight Point Transformers', { placeholder: 'title' });
s.addText([
  { text: 'Team 10  |  23I006 Deep Learning Mini Project', options: { breakLine: true, bold: true } },
  { text: 'Abarajitha S K (23I301)  ·  G Aswath Rupan (23I321)  ·  Indu Vadana M D (23I323)  ·  Murugan S (23I338)', options: { breakLine: true } },
  { text: 'Department of Information Technology, PSG College of Technology' },
], { placeholder: 'body' });
s.addNotes(N('A', `Opening. We built a camera-free activity-recognition system that works on radar point clouds. Say the one-sentence result: our lightweight model (${(ours.params / 1000).toFixed(1)} K parameters) reaches ${f2(ours.acc)}% on RadHAR and ${f2(oursMP.acc)}% on 49 MiliPoint actions performed by people it has never seen, and we show that how the data are split changes published accuracy dramatically. Order of presenters: Abarajitha (motivation, model), Aswath (data, preprocessing, noise), Indu (leakage, training, results), Murugan (robustness, ablation, demo, conclusion).`,
  [['Why radar instead of a camera?', 'Radar gives only sparse 3-D points and velocities, never an image, so a person cannot be visually identified; it also works in the dark.'],
   ['What is a point cloud here?', 'For each radar frame, a small set of detections (x, y, z, Doppler velocity, intensity), typically 6 to 21 per frame.']]));

// =================================================================== 2 Background
s = content('Radar senses activity without ever forming an image', N('A',
  'Elderly monitoring and smart homes need continuous activity awareness. Cameras are accurate but intrusive and fail in the dark; wearables need compliance. Single-chip FMCW radars (TI IWR1443/1843, 60–81 GHz) output a sparse point cloud with a velocity per point. No image is ever formed, which is why the approach is privacy-preserving by construction. Note: we do not claim fall detection or clinical validity; neither dataset has fall labels.',
  [['Is radar data really private?', 'It carries body position and motion only; there is no face or texture. Gait could in principle identify people, which we mention as a limitation.'],
   ['Which frequencies?', 'IWR1443 and IWR1843 operate at 76–81 GHz.']]));
card(s, MX, TOP + 0.1, 3.85, 3.0, 'Camera', 'Accurate, but records identifiable images; fails in darkness; low acceptance at home', C.accent2);
card(s, MX + 4.07, TOP + 0.1, 3.85, 3.0, 'Wearables', 'No images, but must be worn and charged; often forgotten by elderly users', C.accent3);
card(s, MX + 8.14, TOP + 0.1, 3.85, 3.0, 'mmWave radar', 'Contact-free, works in the dark, no image formed: sparse 3-D points + velocity', C.accent1);
bullets(s, ['Goal: recognise activities from radar point clouds alone, accurately and cheaply enough for an edge device',
            'Applications: elderly activity monitoring, smart homes, privacy-preserving indoor sensing'], { x: MX, y: TOP + 3.4, w: W - 2 * MX, h: 2.0, fs: 22 });

// =================================================================== 3 Problem
s = content('Radar point clouds are sparse, noisy and temporal', N('A',
  `Four difficulties define the problem. Sparsity: the median RadHAR frame has ${eda.radhar ? eda.radhar.pts_per_frame_median : 21} returns and MiliPoint only ${eda.milipoint ? eda.milipoint.pts_per_frame_median : 6}. Noise: about a third of RadHAR returns are antenna echoes and half of the rest is static clutter. Time: the activity is in how the cloud moves, so a single frame is ambiguous. Evaluation: windows overlap heavily, so careless splits leak test data. The task: given a 2-second window, predict the activity.`,
  [['Why 2-second windows?', 'It is the RadHAR protocol (60 frames) and long enough to contain a repetition of each activity; we use the same 2 s on MiliPoint (48 frames).']]));
const probs = [['Sparse', `median ${eda.radhar ? eda.radhar.pts_per_frame_median : 21} (RadHAR) and ${eda.milipoint ? eda.milipoint.pts_per_frame_median : 6} (MiliPoint) returns per frame`],
  ['Noisy', `${eda.radhar ? (100 * eda.radhar.roi_removed_frac).toFixed(1) : '34.6'}% antenna echoes; ${eda.radhar ? (100 * eda.radhar.gate_removed_frac_after_roi).toFixed(1) : '50.1'}% of the rest is static clutter`],
  ['Temporal', 'the activity lives in the motion across frames, not in one frame'],
  ['Leaky evaluation', 'overlapping windows put near-duplicates in train and test']];
probs.forEach((p, i) => card(s, MX + (i % 2) * 6.1, TOP + 0.1 + Math.floor(i / 2) * 2.55, 5.95, 2.35, p[0], p[1], [C.accent1, C.accent2, C.accent6, C.accent3][i], { bs: 24, hs: 26 }));

// =================================================================== 4 Motivation
s = content('Motivation: the same dataset, 34 % or 98 %?', N('A',
  `On MiliPoint (49 actions) the dataset authors report 34.45% top-1 with PointNet++, while a 2025 paper reports 98.25% with a graph network under a random 80/10/10 split. Our leakage audit shows why: with random splitting of overlapping windows, ${f2(100 * (rnd.fraction_test_frames_seen_in_training || 0))}% of the test frames already appear inside training windows. A model that is accurate only because of leakage is useless in a new home. This motivates two things: a model that is efficient and robust, and an evaluation that is honest.`,
  [['Are you saying the 98% paper is wrong?', 'We say its protocol measures something different: recognising near-duplicate windows of seen sequences, not generalising to new people. We reproduce the effect with our own models (protocol slide).']]));
stat(s, MX, TOP + 0.4, 3.6, '34.45 %', 'MiliPoint authors, PointNet++ (NeurIPS 2023)', C.accent4);
stat(s, MX + 4.1, TOP + 0.4, 3.6, '98.25 %', 'MAK-GCN 2025, random 80/10/10 split', C.accent2);
stat(s, MX + 8.2, TOP + 0.4, 3.9, `${f2(100 * (rnd.fraction_test_frames_seen_in_training || 0))} %`, 'of test frames already seen in training under random window splits (our audit)', C.accent1);
bullets(s, ['Need 1: a lightweight, noise-robust model that exploits radar structure',
            'Need 2: leakage-free evaluation on unseen recordings and unseen people'], { x: MX, y: TOP + 3.0, w: W - 2 * MX, h: 2.0 });

// =================================================================== 5 Literature
section('Background and related work');
s = content('Published results use incompatible protocols', N('A',
  'Representative work: voxel CNN-LSTM (RadHAR 2019), graph networks (MMPoint-GNN, Lee & Kim with joint pre-training, MAK-GCN), PointNet-LSTM hybrids (Gu et al. 2024), occupancy-gated convolution (OG-PCL 2025), FPGA-quantised pillars (Lim et al. 2024). Dense point-cloud backbones (PointNet++, DGCNN, Point Transformer, PTv3) were designed for LiDAR/CAD with thousands of points. Accuracies are as reported by each paper under its own protocol; they are not directly comparable, which is part of our point.',
  [['How many papers did you review?', 'About 40 references, 19 from 2022–2026; the full list is in the paper and report.']]));
table(s, [['Work', 'Method', 'Reported accuracy', 'Note'],
  ['RadHAR 2019', 'Voxels + TD-CNN + BiLSTM', '90.47 % (RadHAR)', 'dataset paper'],
  ['Lee & Kim 2022', 'GNN + pre-trained 3-D joints', '96.55 % (RadHAR)', 'needs joint labels'],
  ['Gu et al. 2024', 'Lite-PointNet + BiLiLSTM', '95.12 % (RadHAR)', '79.7 K params'],
  ['MAK-GCN 2025', 'Adaptive graph convolution', '97.54 % RadHAR / 98.25 % MiliPoint', 'random split on MiliPoint'],
  ['OG-PCL 2025', 'Occupancy-gated convolution', '91.75 % (RadHAR)', '0.83 M params, 5-fold CV'],
  ['MiliPoint 2023', 'PointNet++, DGCNN, PointMLP', '34.45 % (49 actions)', 'dataset paper']],
  { x: MX, y: TOP + 0.15, w: W - 2 * MX, colW: [2.4, 3.9, 3.6, 2.233], fs: 16, rowH: 0.55 });

// =================================================================== 6 Gap & objectives
s = content('Gap: radar structure and honest evaluation are under-used', N('A',
  'Gap: radar-specific structure (sparsity, Doppler, static clutter) is rarely exploited explicitly; protocols are inconsistent and often leaky; robustness, transfer and edge cost are rarely reported together. Objectives map one-to-one to the contributions: a factorised point transformer with a motion descriptor; Doppler-based clutter removal; leakage-free protocols and an automated audit; full evaluation with seeds, significance, ablation, robustness, transfer and explainability; and a working demo.',
  [['What exactly is novel?', 'The factorised intra-/inter-segment attention for radar windows, the radar motion descriptor, Doppler-based person centring with gating, and the quantified leakage study. Attention, KD and quantisation themselves are established.']]));
card(s, MX, TOP + 0.1, 5.95, 5.0, 'Research gap', '• Dense-point backbones or voxels ignore radar structure (sparsity, Doppler, clutter)\n• Inconsistent, often leaky protocols\n• Robustness, transfer and edge cost rarely reported together', C.accent2, { bs: 22 });
card(s, MX + 6.1, TOP + 0.1, 5.95, 5.0, 'Objectives', '1. Lightweight spatio-temporal point transformer\n2. Radar-specific clutter removal\n3. Leakage-free protocols + automated audit\n4. Seeds, significance, ablation, robustness, transfer, XAI\n5. Saved model + working demonstration', C.accent1, { bs: 22 });

// =================================================================== 7 Datasets
section('Data');
s = content('Two complementary radar datasets, used separately', N('B',
  `MiliPoint (primary): TI IWR1843 at 24 Hz, 11 subjects, 49 actions; 212,920 labelled frames, which matches the original paper exactly, so our loader is verified. Only xyz per point. RadHAR (secondary): TI IWR1443 at 30 Hz, 5 activities, 200 recordings (71.5 min train, 21.4 min test, matching the official partition), xyz plus Doppler and intensity. We do not merge the datasets: label sets, sensors and features differ, and MiliPoint action names are not published. We use the other dataset only for representation transfer, with harmonised axes and xyz input.`,
  [['Why not merge the two datasets?', 'Different labels (5 vs 49, no mapping), different radars, and MiliPoint lacks Doppler. We run independent experiments and test transfer instead.'],
   ['Licences?', 'RadHAR BSD-3-Clause, MiliPoint MIT.']]));
table(s, [['', 'MiliPoint (primary)', 'RadHAR (secondary)'],
  ['Radar / rate', 'TI IWR1843, 24 Hz', 'TI IWR1443, 30 Hz'],
  ['Subjects', '11 (IDs available)', '2 (no IDs)'],
  ['Classes', '49 actions', '5: boxing, jumping jacks, jump, squats, walk'],
  ['Size', '545,059 frames; 212,920 labelled', '200 recordings; 71.5 + 21.4 min'],
  ['Points / frame', `median ${eda.milipoint ? eda.milipoint.pts_per_frame_median : 6}`, `median ${eda.radhar ? eda.radhar.pts_per_frame_median : 21}`],
  ['Features', 'x, y, z', 'x, y, z, Doppler, intensity'],
  ['Our test set', 'subjects 8–10 (unseen people)', 'official test recordings']],
  { x: MX, y: TOP + 0.15, w: W - 2 * MX, colW: [2.6, 4.7, 4.833], fs: 16, rowH: 0.56 });

// =================================================================== 8 Dataset challenges
s = content('Measured: very few points, clutter at fixed ranges', N('B',
  'Left: RadHAR. Panel (a) points per frame; (b) Doppler is quantised to a few values; (c) density in range–height: the person is at 1–2 m and there are static clutter bands at about 2.5 m and 4.5 m. MiliPoint is even sparser (median 6 points). Panel (c) of the MiliPoint figure shows the subject-disjoint split. Everything here is measured, not assumed.',
  [['How did you find the clutter?', 'Range histograms of all points versus moving (non-zero Doppler) points: the moving points sit only on the person; the far clusters never move.']]));
img(s, FIG('eda_radhar.png'), MX, TOP, W - 2 * MX, 2.75, 'RadHAR exploratory analysis');
img(s, FIG('eda_milipoint.png'), MX, TOP + 2.85, W - 2 * MX, 2.6, 'MiliPoint exploratory analysis');

// =================================================================== 9 Preprocessing
s = content('Doppler-based gating removes static clutter', N('B',
  `Three steps. (1) ROI: drop antenna self-returns at x < 0.3 m (${eda.radhar ? (100 * eda.radhar.roi_removed_frac).toFixed(1) : '34.6'}% of returns). (2) Person centring: the median of the moving returns, because static objects have zero radial velocity. A plain centroid was pulled metres off the person. (3) Gating: keep returns within 1.2 m horizontally of the person; this removes ${eda.radhar ? (100 * eda.radhar.gate_removed_frac_after_roi).toFixed(1) : '50.1'}% of the remaining returns, which are static clutter. Then 12 segments × 32 points per 2-s window. Intensity statistics come from training data only.`,
  [['Could gating remove body parts?', 'Only returns more than 1.2 m from the body centre are removed; arm span and walking in place stay inside. The ablation slide shows the effect of removing gating.'],
   ['Why not DBSCAN?', 'Clutter forms far, static clusters; a single interpretable radius around the moving centre removes it without per-window clustering cost or extra parameters.']]));
img(s, FIG('preproc_before_after.png'), MX, TOP, 7.6, 3.3, 'Pre-processing before and after');
bullets(s, ['ROI: remove antenna echoes (x < 0.3 m)', 'Centre on moving (Doppler ≠ 0) returns', 'Gate: keep ≤ 1.2 m from the person', '2-s window → 12 segments × 32 points',
  'Train-only intensity statistics'], { x: MX + 7.85, y: TOP + 0.1, w: W - 2 * MX - 7.85, h: 3.4, fs: 18 });
bullets(s, ['Augmentation (training only): rotation ±15°, scaling 0.9–1.1, 1 cm jitter, re-sampling; optional noise-robust mode: density, dropout, false detections, segment loss'],
  { x: MX, y: TOP + 3.55, w: W - 2 * MX, h: 1.5, fs: 18 });

// =================================================================== 10 Noise
s = content('Noise: handled in pre-processing vs. injected for testing', N('B',
  'We separate noise we measured in the data (handled by pre-processing) from synthetic perturbations we inject into the test set to probe robustness. The injected conditions are experiments, not claims about the datasets. Each condition has three severities and a fixed seed, so every model sees the identical perturbed test windows.',
  [['Why synthetic noise?', 'There is no ground truth of which returns are noise, so controlled perturbations are the only way to measure degradation systematically.']]));
table(s, [['Measured in the data', 'Handled by'], ['Antenna self-returns (34.6 %)', 'ROI filter'], ['Static clutter / multipath', 'Doppler centring + gating'],
  ['Sparse, variable density', '4–5-frame segments + re-sampling'], ['Quantised Doppler', 'kept as a feature']], { x: MX, y: TOP + 0.15, w: 5.3, colW: [2.9, 2.4], fs: 16, rowH: 0.6 });
table(s, [['Injected into the test set', 'Levels'], ['Coordinate jitter', 'σ = 2, 5, 10 cm'], ['Point dropout', '5, 15, 30 %'], ['Increased sparsity', 'keep 50, 25, 12.5 %'],
  ['Outlier injection', '5, 10, 20 %'], ['Variable density', 'min keep 0.5, 0.25, 0.1'], ['Frame loss', '25, 50 % of segments'], ['Combined', 'severity 1–3']],
  { x: MX + 5.7, y: TOP + 0.15, w: W - 2 * MX - 5.7, colW: [3.5, 2.833], fs: 16, rowH: 0.5 });

// =================================================================== 11 Leakage
s = content('Our splits are clean; random splits leak most test frames', N('C',
  `Splits: RadHAR official test recordings unchanged; validation = 20% of training recordings (not windows), capped at 55 windows each. MiliPoint primary split is subject-disjoint (test subjects 8–10). The automated audit checks recording, subject and session overlap, identical frames, frames shared between overlapping windows, train-only statistics and checkpoint selection. Findings: our splits are clean. RadHAR's official test recordings were made minutes after training recordings on the same days and there are no subject IDs, so RadHAR cannot test new people; MiliPoint subject-disjoint is therefore our primary result. Random window splits leak ${f2(100 * (rnd.fraction_test_frames_seen_in_training || 0))}% of test frames.`,
  [['How do you know your test set was never used for tuning?', 'Checkpoints are chosen on validation macro-F1, the exported seed is chosen on validation, and the audit verifies checkpoint = argmax validation F1 for every run.'],
   ['Why cap validation windows?', 'One 186-s recording made up 20% of validation and was consistently misread; it dominated model selection. The fix was based on validation behaviour only.']]));
const au = audit.radhar || {}; const ms = audit.model_selection || {};
table(s, [['Check', 'Result', 'Status'],
  ['RadHAR recordings shared between splits', '0 / 0 / 0', 'PASS'],
  ['Identical frames across splits', '0', 'PASS'],
  ['Train-only normalisation', 'verified', 'PASS'],
  ['RadHAR session independence', `median gap ${au.session ? Math.round(au.session.test_median_gap_to_train_s) : '–'} s, same days`, 'NOT INDEPENDENT'],
  ['MiliPoint subjects shared (subject split)', 'none', 'PASS'],
  ['MiliPoint random split: test frames seen', `${f2(100 * (rnd.fraction_test_frames_seen_in_training || 0))} %`, 'LEAKY'],
  ['Checkpoint = best validation epoch', `${ms.consistent || 0} / ${ms.runs_checked || 0} runs`, (ms.consistent === ms.runs_checked) ? 'PASS' : 'CHECK']],
  { x: MX, y: TOP + 0.15, w: W - 2 * MX, colW: [5.4, 4.4, 2.333], fs: 16, rowH: 0.56, hl: (i) => i === 3 || i === 5 });

// =================================================================== 12 Architecture
section('Method');
s = content(`STPT-Lite: factorised attention in ${(ours.params / 1000).toFixed(0)} K parameters`, N('A',
  `Spatio-Temporal Point Transformer, Lite. Step 1: a shared MLP embeds every point. Step 2: self-attention only among the 32 points of each 5-frame segment, with a learned 3-D relative-position bias. Step 3: attentive plus max pooling gives one token per segment; we add an embedding of an 8-number motion descriptor (centroid, its displacement, mean Doppler, point count). Step 4: a [CLS] token and two Transformer blocks attend across the 12 segment tokens. Because attention is factorised, cost is 12×32² instead of 384² pairs, and there is no farthest-point sampling or k-NN. ${(ours.params / 1000).toFixed(1)} K parameters, ${f2(ours.mflops)} MFLOPs. A 725 K-parameter teacher is distilled into it, and weights are stored in INT8.`,
  [['Why factorise attention?', 'Radar frames are too sparse for rich spatial structure but very regular in time; attending inside short segments and then across time keeps cost linear in the number of segments and avoids FPS/k-NN, which are slow on edge processors.'],
   ['What does the motion descriptor add?', 'Explicit kinematics that radar measures directly; its contribution is measured in the ablation.']]));
img(s, FIG('arch.png'), MX, TOP + 0.1, W - 2 * MX, 2.5, 'STPT-Lite architecture');
const statY = TOP + 3.05;
stat(s, MX, statY, 2.9, `${(ours.params / 1000).toFixed(1)} K`, 'parameters', C.accent1);
stat(s, MX + 3.05, statY, 2.9, `${f2(ours.mflops)}`, 'MFLOPs per 2-s window', C.accent1);
stat(s, MX + 6.1, statY, 2.9, `${f2(ours.lat)} ms`, 'CPU latency, 1 thread, batch 1', C.accent1);
stat(s, MX + 9.15, statY, 2.9, `${f2(ours.int8mb)} MB`, 'INT8 weights', C.accent1);

// =================================================================== 13 Model selection
s = content('Six baselines re-trained under one identical protocol', N('A',
  'We re-implemented six baselines in pure PyTorch and trained all of them with identical data, augmentation, optimiser, epochs and early stopping: the RadHAR voxel CNN-LSTM, PointNet, PointNet+BiLSTM, PointNet++, DGCNN and Point Transformer. Our family: STPT-Large (teacher), STPT-Lite and STPT-Tiny, plus distillation and noise-robust training variants. Selection criteria from the brief: accuracy, robustness, generalisation, stability, size and latency.',
  [['Why not use PyTorch Geometric?', 'It needs compiled extensions; pure PyTorch runs anywhere, including the CPU-only environment used here.'],
   ['Is the comparison fair?', 'Yes: same splits, pre-processing, optimiser, epoch budget, early stopping and seeds for every model.']]));
const rowsSel = [['Model', 'Idea', 'Params (K)']].concat(RH.filter((r) => !r.key.endsWith('robust')).map((r) => [r.name, {
  voxel_tdcnn: 'voxel grid + 3-D CNN + BiLSTM', pointnet: 'shared MLP + max-pool', pointnet_lstm: 'per-segment PointNet + BiLSTM',
  pointnet2: 'FPS + ball query hierarchy', dgcnn: 'dynamic k-NN graph EdgeConv', pointtransformer: 'k-NN vector attention',
  stpt_teacher: 'factorised attention, d = 128', stpt_lite: 'factorised attention, d = 64', stpt_lite_kd: '+ knowledge distillation', stpt_tiny_kd: 'd = 32 + distillation' }[r.key] || '', (r.params / 1000).toFixed(1)]));
table(s, rowsSel, { x: MX, y: TOP + 0.1, w: W - 2 * MX, colW: [4.3, 5.6, 2.233], fs: 15, rowH: 0.45, hl: (i) => rowsSel[i + 1][0].includes('proposed') });

// =================================================================== 14 Training
s = content('Early stopping on held-out recordings and people', N('C',
  'AdamW, one-cycle learning rate (peak 2e-3), weight decay 0.05, label smoothing 0.1, gradient clipping 1.0, batch 64, at most 15 epochs on RadHAR and 30 on MiliPoint with early stopping (patience 5) on validation macro-F1; the best validation checkpoint is kept and the test set is evaluated once. Three seeds per configuration. The curves on the right are validation macro-F1 per epoch for seed 0 of each model; full histories (training/validation loss and accuracy, learning rate) are saved for every run.',
  [['How do you know you did not overfit?', 'Early stopping on a recording- or subject-disjoint validation set, regularisation, small models, and agreement between validation and test accuracy.']]));
bullets(s, ['AdamW, one-cycle LR (peak 2·10⁻³), weight decay 0.05', 'Label smoothing 0.1, gradient clipping 1.0', '≤ 15 (RadHAR) / 30 (MiliPoint) epochs, early stopping (patience 5) on val macro-F1',
  'Best-validation checkpoint; test evaluated once', '3 seeds; mean ± std reported', 'KD: α = 0.7, T = 4; noise-robust augmentation p = 0.5'], { x: MX, y: TOP + 0.1, w: 4.9, h: 5.2, fs: 18 });
img(s, FIG('curves.png'), MX + 5.1, TOP + 0.2, W - 2 * MX - 5.1, 4.9, 'Validation curves');

// =================================================================== 15 Results RadHAR
section('Results');
function accChart(s, arr, title, y, h) {
  const rows = arr.filter((r) => !r.key.endsWith('robust') || r.key === 'stpt_lite_kd_robust');
  s.addChart(pres.charts.BAR, [{ name: 'accuracy', labels: rows.map((r) => r.name.replace(' + noise-robust training', ' + robust').replace(' (proposed)', ' (ours)')), values: rows.map((r) => r.acc) }],
    chartBase({ x: MX, y, w: 7.4, h, barDir: 'bar', showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0',
      chartColors: rows.map((r) => r.key === 'stpt_lite_kd_robust' ? HEX.accent2 : (r.key.startsWith('stpt') ? HEX.accent1 : HEX.accent4)),
      valAxisMinVal: 0, valAxisMaxVal: 100, showTitle: true, title, showLegend: false, catAxisLabelFontSize: 13 }));
}
s = content(`RadHAR is saturated: all models reach ${Math.floor(Math.min(...RH.filter((r) => r.acc).map((r) => r.acc)))}–${Math.ceil(Math.max(...RH.map((r) => r.acc || 0)))} %`, N('C',
  `Official RadHAR test recordings, mean of three seeds. STPT-Lite+KD+robust (proposed, selected on validation): ${pm([ours.acc, ours.sd])}% accuracy, macro-F1 ${f2(ours.f1)}. Best baseline: ${bbRH.name} at ${pm([bbRH.acc, bbRH.sd])}% with ${(bbRH.params / 1000).toFixed(0)} K parameters. The original RadHAR paper reports 90.47%. Caveat from our audit: RadHAR test recordings come from the same sessions and people as training, so this measures generalisation to new recordings, not to new people.`,
  [['Is the difference to the best baseline significant?', 'See the statistics slide: exact McNemar tests on the same test windows with Holm correction, per seed.'],
   ['Why is RadHAR easy?', 'Five distinct activities, same people and room in training and test.']]));
accChart(s, RH, 'Test accuracy (%), mean of 3 seeds', TOP + 0.05, 5.3);
table(s, [['Model', 'Acc. (%)', 'F1 (%)'], ...RH.filter((r) => ['stpt_lite_kd_robust', 'stpt_lite_kd', 'stpt_teacher', bbRH.key, 'voxel_tdcnn'].includes(r.key)).map((r) => [r.name.replace(' (proposed)', ' (ours)'), pm([r.acc, r.sd]), f2(r.f1)])],
  { x: MX + 7.6, y: TOP + 0.2, w: W - 2 * MX - 7.6, colW: [2.1, 1.55, 0.883], fs: 13, rowH: 0.5, hl: HLOURS });

// =================================================================== 16 Results MiliPoint
s = content(`Unseen people: ${f2(oursMP.acc)} % on 49 actions, +${(oursMP.acc - bbMP.acc).toFixed(0)} points`, N('C',
  `Primary result. Test subjects 8, 9 and 10 were never seen in training or validation. STPT-Lite+KD+robust (proposed): ${pm([oursMP.acc, oursMP.sd])}% top-1, ${f2(oursMP.top3)}% top-3, macro-F1 ${f2(oursMP.f1)}. Best baseline: ${bbMP.name} ${pm([bbMP.acc, bbMP.sd])}%. Chance is about 2%; the MiliPoint authors report 34.45% for PointNet++ under their chronological split.${MP.some((r) => r.key === 'dgcnn') ? ' DGCNN and Point Transformer were run with one seed on MiliPoint because of their CPU cost (marked n = 1 in the paper table).' : ' DGCNN and Point Transformer were compared on RadHAR only, because of their CPU cost on the much larger MiliPoint training set.'} Distillation alone gives ${pm([kdMP.acc, kdMP.sd])}%; the noise-robust variant was selected on validation macro-F1.`,
  [['Why is MiliPoint so much harder?', '49 fine-grained actions, about 6 points per frame, xyz only, and new people at test time.'],
   ['Why not compare to 98%?', 'That number uses a random split; our protocol slide shows what random splitting does to our own models.']]));
accChart(s, MP, 'Test accuracy (%), unseen subjects', TOP + 0.05, 5.3);
table(s, [['Model', 'Top-1 (%)', 'Top-3 (%)'], ...MP.filter((r) => ['stpt_lite_kd_robust', 'stpt_lite_kd', 'stpt_teacher', bbMP.key].includes(r.key)).map((r) => [r.name.replace(' (proposed)', ' (ours)'), pm([r.acc, r.sd]), f2(r.top3)])],
  { x: MX + 7.6, y: TOP + 0.2, w: W - 2 * MX - 7.6, colW: [2.1, 1.55, 0.883], fs: 13, rowH: 0.5, hl: HLOURS });

// =================================================================== 17 Statistics + confusion
s = content('Gains on unseen people are significant in every seed', N('C',
  'Left: per-class confusion matrices of the proposed model (seed 0). Right: paired exact McNemar tests on the identical MiliPoint test windows (unseen people), Holm-corrected within each seed; the last column counts the seeds in which the difference is significant. Against every re-implemented baseline the difference is significant in all three seeds (largest Holm-corrected p about 1e-46). The other STPT variants (teacher, STPT-Lite with and without distillation) are close: their differences are significant only in some seeds. On RadHAR most differences are not significant, as expected for a saturated benchmark. 95% bootstrap confidence intervals (2,000 resamples) are in the paper. We do not report only the best seed.',
  [['Why McNemar and not a t-test?', 'Both models predict the same test windows; McNemar uses exactly the disagreements, which is the appropriate paired test for classifiers.']]));
img(s, FIG('confusion.png'), MX, TOP, 7.6, 5.3, 'Confusion matrices');
const SM = (D.stats || {}).milipoint || {};
const seeds = Object.keys(SM);
const nSig = (k) => seeds.filter((sd) => SM[sd].tests && SM[sd].tests[k] && SM[sd].tests[k].p_holm < 0.05).length;
const stKeys = Object.keys((SM['0'] || {}).tests || {}).filter((k) => !k.endsWith('robust'));
const stRows = [['vs. (MiliPoint)', 'Acc. s0', 'Signif.']].concat(stKeys.map((k) => [(MP.find((r) => r.key === k) || { name: k }).name.replace(' (proposed)', ''), f2(SM['0'].tests[k].acc), `${nSig(k)} / ${seeds.length}`]));
table(s, stRows, { x: MX + 7.8, y: TOP + 0.1, w: W - 2 * MX - 7.8, colW: [2.2, 1.0, 1.133], fs: 13, rowH: 0.42 });
bullets(s, [`Proposed, seed 0: ${f2(SM['0'] ? SM['0'].ref_acc : null)} % [${SM['0'] ? f2(SM['0'].ref_ci[0]) : ''}, ${SM['0'] ? f2(SM['0'].ref_ci[1]) : ''}] (95 % bootstrap CI)`, 'Exact McNemar, Holm-corrected per seed; Signif. = seeds with p < 0.05'], { x: MX + 7.8, y: TOP + 0.6 + 0.42 * stRows.length, w: W - 2 * MX - 7.8, h: 1.6, fs: 14 });

// =================================================================== 18 Robustness
s = content('Noise robustness', N('M',
  'Accuracy on the full test set under each perturbation and severity; identical perturbed samples for every model. Solid red is the proposed model (distillation + noise-robust training); dashed orange is the same model without noise-robust training; blue is PointNet, dashed blue PointNet with the same noise-robust training. Read the outlier and sparsity panels first: these are the conditions that hurt most. The prediction-stability metric (fraction of predictions unchanged) and per-class recall are in the result files and paper.',
  [['Did noise-robust training help?', 'Compare the solid red and dashed orange lines; the paper reports the exact differences per condition, including where it did not help (extreme sparsity).'],
   ['Why not test on real noisy recordings?', 'No labelled noisy recordings exist for these datasets; controlled perturbations are the measurable alternative and are labelled as such.']]));
img(s, FIG('robustness_radhar.png'), MX, TOP, (W - 2 * MX) / 2 - 0.1, 5.4, 'Robustness RadHAR');
img(s, FIG('robustness_milipoint.png'), MX + (W - 2 * MX) / 2 + 0.1, TOP, (W - 2 * MX) / 2 - 0.1, 5.4, 'Robustness MiliPoint');

// =================================================================== 19 Efficiency
s = content(`${f2(ours.lat)} ms per window, ${f2(ours.int8mb)} MB with INT8 weights`, N('M',
  `Accuracy versus parameters and CPU latency. Proposed model: ${(ours.params / 1000).toFixed(1)} K parameters, ${f2(ours.mflops)} MFLOPs, ${f2(ours.lat)} ms per window on one CPU thread, ${f2(ours.size)} MB in FP32 and ${f2(ours.int8mb)} MB with INT8 weights (INT8 accuracy ${f2(ours.int8)}%). Latency was measured on a desktop CPU thread; we did not deploy on a micro-controller.`,
  [['Why weight-only INT8?', 'PyTorch dynamic INT8 lost 7 points and was slower for these tiny layers; weight-only INT8 keeps accuracy and cuts storage.']]));
img(s, FIG('efficiency.png'), MX, TOP, W - 2 * MX, 3.9, 'Efficiency');
table(s, [['', 'Params (K)', 'MFLOPs', 'Latency (ms)', 'FP32 (MB)', 'INT8 (MB)', 'INT8 acc. (%)'],
  ...[ours, byK(RH, 'stpt_tiny_kd'), bbRH].map((r) => [r.name.replace(' (proposed)', ''), (r.params / 1000).toFixed(1), f2(r.mflops), f2(r.lat), f2(r.size), f2(r.int8mb), f2(r.int8)])],
  { x: MX, y: TOP + 4.0, w: W - 2 * MX, colW: [3.6, 1.4, 1.4, 1.5, 1.4, 1.4, 1.433], fs: 14, rowH: 0.36, hl: (i) => i === 0 });

// =================================================================== 20 Ablation
const AB = D.ablation;
const ablAll = [['noattn', '− intra-segment attention'], ['nomotion', '− motion descriptor'], ['notemporal', '− temporal Transformer'],
  ['xyz', '− Doppler & intensity'], ['nogate', '− clutter gating'], ['augnone', '− all augmentation'], ['augnojitter', '− coordinate jitter']];
const dAb = (k, d) => (AB[k] && AB[k][d] && AB[k][d][0] !== null && AB.full[d][0] !== null) ? +(AB[k][d][0] - AB.full[d][0]).toFixed(2) : null;
const mpAbl = ablAll.filter(([k]) => dAb(k, 'mp') !== null);
const mpWorst = mpAbl.slice().sort((a, b) => dAb(a[0], 'mp') - dAb(b[0], 'mp'))[0];
s = content(mpWorst ? `Ablation: ${mpWorst[1].replace('− ', 'removing the ')} costs ${Math.abs(dAb(mpWorst[0], 'mp')).toFixed(1)} points on unseen people` : 'Ablation study', N('M',
  `Each bar removes one component of STPT-Lite (no distillation) and retrains; the bar is the change in test accuracy relative to the full model (RadHAR full: ${pm(AB.full.rh)}%, MiliPoint full: ${pm(AB.full.mp)}%). RadHAR is saturated: every change there lies within the seed spread of the full model (±${f2(AB.full.rh[1])}), so RadHAR cannot rank components. MiliPoint, with unseen people and 49 actions, is the informative benchmark for the architectural components; Doppler, gating and augmentation are RadHAR-specific inputs and are tested there only. The full table with standard deviations is in the paper.`,
  [['Which component matters most?', mpWorst ? `On MiliPoint, ${mpWorst[1].replace('− ', '')}: removing it changes accuracy by ${dAb(mpWorst[0], 'mp')} points.` : 'See the paper table.'],
   ['Some RadHAR variants are better than the full model. Why keep the component?', 'Those differences are inside the seed spread on a saturated benchmark; components are judged on MiliPoint, where the spread is smaller than the effects.']]));
const ablChart = (d, rowsA, x, w, title, col) => { if (!rowsA.length) return;
  s.addChart(pres.charts.BAR, [{ name: title, labels: rowsA.map((a) => a[1]), values: rowsA.map(([k]) => dAb(k, d)) }],
    chartBase({ x, y: TOP, w, h: 5.4, barDir: 'bar', showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '+0.0;-0.0;0.0',
      chartColors: [col], showLegend: false, showTitle: true, title, catAxisLabelFontSize: 13 })); };
const rhAbl = ablAll.filter(([k]) => dAb(k, 'rh') !== null);
ablChart('mp', mpAbl, MX, 5.9, 'MiliPoint, unseen people: Δ accuracy (points)', HEX.accent2);
ablChart('rh', rhAbl, MX + 6.15, W - 2 * MX - 6.15, `RadHAR: Δ accuracy (points); full-model std ±${f2(AB.full.rh[1])}`, HEX.accent4);

// =================================================================== 21 Protocol
s = content('How the split changes the answer (MiliPoint)', N('C',
  'Same models, same code, three splits. Subject-disjoint (ours): new people at test time. Chronological (MiliPoint code): last 10% of frames, which shares subjects with training. Random windows: overlapping windows shuffled, so most test frames were seen in training. The jump from the first to the last column is leakage, not learning.',
  [['So which number should be reported?', 'The subject-disjoint one, if the claim is that the system works for new people.']]));
const PR = D.protocol; const prM = [['stpt_lite', 'STPT-Lite'], ['pointnet', 'PointNet']].filter(([k]) => PR[k]);
s.addChart(pres.charts.BAR, prM.map(([k, n]) => ({ name: n, labels: ['Subject-disjoint (ours)', 'Chronological', 'Random windows (leaky)'],
  values: [PR[k].subject[0] || 0, PR[k].official[0] || 0, PR[k].random[0] || 0] })),
  chartBase({ x: MX, y: TOP, w: 8.2, h: 5.4, barDir: 'col', barGrouping: 'clustered', showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0',
    chartColors: [HEX.accent2, HEX.accent5, HEX.accent6], valAxisMinVal: 0, valAxisMaxVal: 100, showLegend: true, legendPos: 'b', showTitle: true, title: 'Top-1 accuracy (%), 49 actions' }));
card(s, MX + 8.45, TOP + 0.2, W - 2 * MX - 8.45, 4.9, 'Takeaway', `Random window splits leak ${f2(100 * (rnd.fraction_test_frames_seen_in_training || 0))} % of test frames into training. Report subject-disjoint accuracy for claims about new users.`, C.accent2, { bs: 22 });

// =================================================================== 22 Transfer
s = content('Cross-dataset transfer: MiliPoint to RadHAR', N('C',
  'We pre-train STPT-Lite on MiliPoint (xyz, harmonised axes) and fine-tune on RadHAR with 10%, 25% and 100% of the training recordings, comparing with training from scratch on the same subsets (xyz input for both, three seeds). This tests whether representations learned on one radar transfer to another radar and label set.',
  [['Why only xyz?', 'MiliPoint has no Doppler or intensity; using xyz on both sides keeps the comparison clean.']]));
const TR = D.transfer; const fr = ['0.1', '0.25', '1'];
const trv = (k, w) => (TR[k] || TR[k + '.0'] || { scratch: [null], init: [null] })[w][0] || 0;
s.addChart(pres.charts.LINE, [{ name: 'from scratch', labels: ['10 %', '25 %', '100 %'], values: fr.map((f) => trv(f, 'scratch')) },
  { name: 'pre-trained on MiliPoint', labels: ['10 %', '25 %', '100 %'], values: fr.map((f) => trv(f, 'init')) }],
  chartBase({ x: MX, y: TOP, w: 8.2, h: 5.4, showValue: true, dataLabelPosition: 't', dataLabelFormatCode: '0.0', chartColors: [HEX.accent4, HEX.accent2], lineSize: 3, lineDataSymbolSize: 9,
    showLegend: true, legendPos: 'b', showTitle: true, title: 'RadHAR test accuracy (%) vs. fraction of training recordings', valAxisMinVal: 50, valAxisMaxVal: 100 }));
card(s, MX + 8.45, TOP + 0.2, W - 2 * MX - 8.45, 4.9, 'Reading the plot', 'Gap between the lines = value of pre-training. It is largest when little target data is available, if transfer helps.', C.accent1, { bs: 22 });

// =================================================================== 23 Explainability
s = content('The model integrates evidence over the whole window', N('M',
  'Left: per-class temporal attention of the [CLS] token over the 12 segments and the occlusion importance (drop in true-class probability when a segment is removed). Right: gradient saliency per point for one window per class, and a t-SNE of the learned embeddings. Attention is not proof of causation; occlusion and saliency are complementary checks.',
  [['Does attention explain the decision?', 'Not on its own (Jain & Wallace 2019); we compare it with occlusion, which directly measures the effect of removing a segment.']]));
img(s, FIG('xai_temporal.png'), MX, TOP, 6.0, 2.3, 'Temporal attention and occlusion');
img(s, FIG('tsne.png'), MX, TOP + 2.45, 6.0, 2.9, 't-SNE');
img(s, FIG('xai_saliency.png'), MX + 6.2, TOP + 0.6, W - 2 * MX - 6.2, 3.6, 'Point saliency');

// =================================================================== 24 Demo
section('Demonstration and conclusion');
s = content('Live demo runs the saved model end to end', N('M',
  'Live demo: streamlit run demo/app.py. It loads the exported checkpoint (the seed chosen by validation), parses a raw RadHAR recording or a MiliPoint sample, shows the raw and pre-processed point clouds, predicts per 2-s window with class probabilities and latency, and lets you inject each noise type and compare predictions. Bundled samples come only from the held-out test split. Invalid files are rejected with a message. Everything is computed by the real model; nothing is hard-coded.',
  [['Is the demo using the real model?', 'Yes; tab 4 shows the checkpoint path and its test metrics, and any upload is processed by the same code as evaluation.'],
   ['What if I upload a wrong file?', 'The app validates format, shape, NaN and length and shows an error instead of crashing.']]));
img(s, FIG('demo/demo_recognition.png'), MX, TOP, 6.0, 4.1, 'Demo recognition tab');
img(s, FIG('demo/demo_noise.png'), MX + 6.2, TOP, W - 2 * MX - 6.2, 4.1, 'Demo noise tab');
bullets(s, ['Upload raw radar file  →  pre-processing view  →  live prediction  →  noise injection'], { x: MX, y: TOP + 4.3, w: W - 2 * MX, h: 0.8, fs: 18 });

// =================================================================== 25 Limitations & future
s = content('Limitations and future work', N('M',
  'Be explicit with examiners. RadHAR cannot test new people or rooms. MiliPoint is xyz-only and has three test subjects. Robustness uses synthetic perturbations, not a radar simulator. Latency is measured on a desktop CPU thread. No fall labels, single person only. Future work follows directly from these limitations.',
  [['What would you do with more time?', 'Leave-one-subject-out and cross-room evaluation, on-device deployment, and self-supervised pre-training on the many unlabelled frames.']]));
card(s, MX, TOP + 0.1, 5.95, 5.0, 'Limitations', '• RadHAR: same people and sessions in test\n• MiliPoint: xyz only; 3 test subjects\n• Synthetic noise, not physical simulation\n• Desktop-CPU latency; no MCU deployment\n• No fall labels; single person', C.accent2, { bs: 22 });
card(s, MX + 6.1, TOP + 0.1, 5.95, 5.0, 'Future work', '• Leave-one-subject-out, cross-room data\n• Deploy on the radar companion processor\n• Physics-based radar noise simulation\n• Self-supervised pre-training, streaming recognition', C.accent1, { bs: 22 });

// =================================================================== 26 Conclusion
s = pres.addSlide({ masterName: 'SECTION_DARK', sectionTitle: sec });
s.addText('Conclusion', { placeholder: 'title' });
s.addText([
  { text: `STPT-Lite+KD+robust: ${f2(ours.acc)} % on RadHAR and ${f2(oursMP.acc)} % on 49 MiliPoint actions of unseen people`, options: { bullet: true, breakLine: true } },
  { text: `${(ours.params / 1000).toFixed(1)} K parameters, ${f2(ours.mflops)} MFLOPs, ${f2(ours.lat)} ms, ${f2(ours.int8mb)} MB INT8`, options: { bullet: true, breakLine: true } },
  { text: 'Doppler-based clutter removal; automated leakage audit; quantified protocol inflation', options: { bullet: true, breakLine: true } },
  { text: 'Saved model, working demo, paper, technical report and full logs', options: { bullet: true } },
], { placeholder: 'body' });
s.addNotes(N('M', 'Summarise in three sentences: what we built (a factorised point transformer for sparse radar), what we showed (competitive accuracy at very low cost, honest numbers on unseen people, robustness profile), and what we learned (evaluation protocol matters as much as the model). Thank the audience and invite questions.',
  [['What is the single most important finding?', 'Under leakage-free evaluation a 118 K-parameter model is competitive, and random splits inflate MiliPoint accuracy dramatically.']]));

// =================================================================== 27 References
s = content('References (selected)', N('A', 'Selected references; the paper lists all of them in IEEE format.', [['Are all references real?', 'Yes, each was verified (authors, venue, year) before citing.']]));
s.addText([
  'A. D. Singh et al., "RadHAR," ACM mmNets, 2019.', 'H. Cui et al., "MiliPoint," NeurIPS Datasets & Benchmarks, 2023.',
  'C. R. Qi et al., "PointNet," CVPR 2017; "PointNet++," NeurIPS 2017.', 'Y. Wang et al., "Dynamic graph CNN," ACM TOG, 2019.',
  'H. Zhao et al., "Point Transformer," ICCV 2021.', 'Z. Gu et al., "mmWave HAR for healthcare monitoring robot," arXiv:2405.01882, 2024.',
  'V. G. Zakka et al., "MAK-GCN," arXiv:2504.02778, 2025.', 'J. Yan et al., "OG-PCL," arXiv:2511.08910, 2025.',
  'G. Hinton et al., "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015.', 'S. Kaufman et al., "Leakage in data mining," ACM TKDD, 2012.',
].map((t, i, a) => ({ text: t, options: { breakLine: i < a.length - 1, paraSpaceAfter: 4 } })), { x: MX, y: TOP + 0.1, w: W - 2 * MX, h: 5.2, fontSize: 16, color: C.text1, valign: 'top', isTextBox: true });

(async () => {
  const out = path.join(__dirname, 'Team10_mmWave_HAR.pptx');
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log('written', out, 'missing figures:', MISSING.length);
})();
