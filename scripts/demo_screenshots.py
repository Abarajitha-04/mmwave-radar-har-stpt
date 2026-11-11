"""Start the Streamlit demo headless and capture real screenshots of every tab."""
import subprocess, time, sys, os, urllib.request
from playwright.sync_api import sync_playwright
OUT = 'figures/demo'; os.makedirs(OUT, exist_ok=True)
proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'demo/app.py', '--server.headless', 'true',
                         '--server.port', '8599', '--browser.gatherUsageStats', 'false'],
                        stdout=open('results/streamlit.log', 'w'), stderr=subprocess.STDOUT)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen('http://localhost:8599/_stcore/health', timeout=2); break
        except Exception:
            time.sleep(1)
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={'width': 1500, 'height': 1500}, device_scale_factor=2)
        pg.goto('http://localhost:8599', wait_until='networkidle')
        pg.wait_for_selector('text=Frames', timeout=180000)
        pg.screenshot(path=f'{OUT}/_debug.png')
        time.sleep(4)
        tabs = pg.get_by_role('tab')
        print('tabs:', tabs.count(), flush=True)
        names = ['input', 'recognition', 'noise', 'model']
        for i, n in enumerate(names):
            tabs.nth(i).click(); time.sleep(4)
            if n == 'noise':
                pass
            pg.screenshot(path=f'{OUT}/demo_{n}.png', full_page=True)
            print('saved', n, flush=True)
        br.close()
finally:
    proc.terminate()
