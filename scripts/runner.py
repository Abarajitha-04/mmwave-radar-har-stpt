"""Tiny job queue: runs jobs from a text file with N parallel workers.
Line format:  tag | train.py args | optional dependency tag
Skips jobs whose results/runs/<tag>/metrics.json exists; defers jobs whose
dependency has not finished yet."""
import subprocess, sys, time, os
jobs_file, N = sys.argv[1], int(sys.argv[2])
max_start_min = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9   # stop launching new jobs after this many minutes
T0 = time.time()
done = lambda t: os.path.exists(f'results/runs/{t}/metrics.json')
# a job whose log changed in the last 15 min is being run by another runner instance
busy = lambda t: os.path.exists(f'results/logs/{t}.log') and time.time() - os.path.getmtime(f'results/logs/{t}.log') < 900
def load():
    out = []
    for l in open(jobs_file):
        l = l.strip()
        if not l or l.startswith('#'): continue
        parts = [p.strip() for p in l.split('|')]
        out.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ''))
    return out
running = {}
os.makedirs('results/logs', exist_ok=True)
while True:
    for t, p in list(running.items()):
        if p.poll() is not None:
            print(time.strftime('%H:%M:%S'), 'finished', t, p.returncode, flush=True); del running[t]
    pending = [j for j in load() if not done(j[0]) and j[0] not in running and not busy(j[0])]
    if not pending and not running: break
    if time.time() - T0 > 60 * max_start_min:
        if not running:
            print('TIME BUDGET reached, exiting cleanly', flush=True); break
        time.sleep(10); continue
    for t, args, dep in pending:
        if len(running) >= N: break
        if dep and not done(dep): continue
        log = open(f'results/logs/{t}.log', 'w')
        running[t] = subprocess.Popen(f'python3 src/train.py --tag {t} --threads 1 {args}', shell=True, stdout=log, stderr=subprocess.STDOUT)
        print(time.strftime('%H:%M:%S'), 'start', t, flush=True)
    time.sleep(10)
print('ALL DONE', flush=True)
