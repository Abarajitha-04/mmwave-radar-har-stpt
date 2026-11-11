"""Parse raw RadHAR ROS-text point-cloud dumps into per-recording frame arrays.

Each raw file is a sequence of ROS PointCloud messages, one point per message.
A new radar frame starts whenever point_id resets to 0.  For every recording we
store a flat float32 array of points [x, y, z, velocity, intensity] and an int
array giving the frame index of every point, plus the frame timestamps.
"""
import re, sys, glob, os, numpy as np

CLASSES = ['boxing', 'jack', 'jump', 'squats', 'walk']
PAT = re.compile(
    r"secs:\s*(\d+)\s*nsecs:\s*(\d+).*?point_id:\s*(\d+)\s*x:\s*(\S+)\s*y:\s*(\S+)\s*z:\s*(\S+)"
    r"\s*range:\s*\S+\s*velocity:\s*(\S+)\s*doppler_bin:\s*\S+\s*bearing:\s*\S+\s*intensity:\s*(\S+)", re.S)

def parse_file(path):
    return parse_text(open(path).read())


def parse_text(txt):
    """Parse the text of one RadHAR/ROS point-cloud dump.  Raises ValueError if
    the text does not contain any point records."""
    rows = PAT.findall(txt)
    if not rows:
        raise ValueError('no RadHAR point records found (expected ROS PointCloud text dump)')
    a = np.array(rows, dtype=np.float64)
    t = a[:, 0] + a[:, 1] * 1e-9
    pid = a[:, 2].astype(int)
    frame = np.cumsum(pid == 0) - 1
    if frame[0] < 0:  # file starts mid-frame: drop the partial frame
        keep = frame >= 0; a, t, frame = a[keep], t[keep], frame[keep]
    pts = a[:, 3:8].astype(np.float32)  # x y z vel intensity
    n_frames = frame.max() + 1
    ft = np.array([t[frame == f][0] for f in range(n_frames)])
    return pts, frame.astype(np.int32), ft

def main(raw_root, out):
    recs = {}
    for split in ['Train', 'Test']:
        for ci, c in enumerate(CLASSES):
            for f in sorted(glob.glob(f'{raw_root}/{split}/{c}/*.txt')):
                pts, fr, ft = parse_file(f)
                key = f'{split}/{c}/{os.path.basename(f)}'
                recs[key] = dict(pts=pts, frame=fr, ft=ft, label=ci, split=split)
                print(key, len(pts), fr.max() + 1, f'{ft[-1]-ft[0]:.1f}s', flush=True)
    np.save(out, recs, allow_pickle=True)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
