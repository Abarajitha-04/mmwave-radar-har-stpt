"""Record a narrated-by-captions screen video of the real Streamlit demo (Playwright) -> figures/demo/demo.mp4"""
import subprocess, time, sys, os, glob, urllib.request
from playwright.sync_api import sync_playwright
OUT = 'figures/demo'; os.makedirs(OUT, exist_ok=True)
VID = os.path.join(OUT, '_video'); os.makedirs(VID, exist_ok=True)
proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'demo/app.py', '--server.headless', 'true',
                         '--server.port', '8598', '--browser.gatherUsageStats', 'false'],
                        stdout=open('results/streamlit_video.log', 'w'), stderr=subprocess.STDOUT)
CAP = """(t) => { let d = document.getElementById('cap'); if (!d) { d = document.createElement('div'); d.id = 'cap';
  d.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);background:rgba(18,53,91,.92);color:#fff;'
  + 'font:600 20px Arial;padding:12px 22px;border-radius:10px;z-index:99999;max-width:80%;text-align:center';
  document.body.appendChild(d); } d.textContent = t; }"""
try:
    for _ in range(90):
        try:
            urllib.request.urlopen('http://localhost:8598/_stcore/health', timeout=2); break
        except Exception:
            time.sleep(1)
    with sync_playwright() as p:
        br = p.chromium.launch()
        ctx = br.new_context(viewport={'width': 1440, 'height': 900}, record_video_dir=VID, record_video_size={'width': 1440, 'height': 900})
        pg = ctx.new_page()
        pg.goto('http://localhost:8598', wait_until='networkidle')
        pg.wait_for_selector('text=Frames', timeout=180000); time.sleep(3)
        cap = lambda t, s: (pg.evaluate(CAP, t), time.sleep(s))
        cap('Privacy-preserving activity recognition from mmWave radar point clouds (live model)', 4)
        cap('Input: a raw RadHAR test recording (no camera, only radar points)', 4)
        pg.mouse.wheel(0, 500); cap('Pre-processing: antenna echoes (red) and static clutter (grey) removed; person kept (blue)', 6)
        pg.mouse.wheel(0, -500)
        tabs = pg.get_by_role('tab')
        tabs.nth(1).click(); time.sleep(3); cap('Recognition: live prediction per 2-s window, probabilities and latency', 6)
        pg.mouse.wheel(0, 500); cap('Class probabilities over time for the whole recording', 5); pg.mouse.wheel(0, -500)
        tabs.nth(2).click(); time.sleep(3); cap('Noise robustness: inject controlled radar noise and compare predictions', 5)
        sb = pg.get_by_role('combobox').first
        cur = 'jitter'
        for cond in ['outliers', 'sparsity', 'combined']:
            try:
                pg.get_by_role('combobox', name='Perturbation').click(); time.sleep(1)
                pg.get_by_role('option', name=cond, exact=True).click(); time.sleep(7)
                cur = cond
                cap(f'Perturbation: {cond} - prediction stability and accuracy update live', 5)
            except Exception as e:
                print('select failed', cond, e)
        pg.mouse.wheel(0, 600); time.sleep(3); pg.mouse.wheel(0, -600)
        tabs.nth(3).click(); time.sleep(3); cap('Model card: 118 K parameters, held-out test metrics, confusion matrix', 6)
        pg.mouse.wheel(0, 700); time.sleep(4)
        cap('All outputs are computed by the saved checkpoint - nothing is hard-coded', 4)
        ctx.close(); br.close()
    webm = sorted(glob.glob(f'{VID}/*.webm'))[-1]
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', webm, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', f'{OUT}/demo.mp4'], check=True)
    print('video', os.path.getsize(f'{OUT}/demo.mp4') / 1e6, 'MB')
finally:
    proc.terminate()
