"""edgediff.py - numeric difference list for the side gate: for every photo column (or row), the top / bottom silhouette
crossings of the photo and of the camera-fitted model (silcmp output), matched pairwise; prints per region the mean and max
offset of each matched edge (px, + = model edge lower / further right) so fixes are measured, not eyeballed.
usage: .venv/bin/python guns/ak74/edgediff.py tag=p1b k=R [min=0.6]"""
import os, sys, json
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import ak_ref as R
import camfit as CF

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
TAG, K = ARGS.get("tag", "p1b"), ARGS.get("k", "R")
MIN = float(ARGS.get("min", 0.6))


def model_alpha_photo():
    alpha = np.load(os.path.join(HERE, "work", f"sil_{TAG}_{K}.npy"))
    mp = json.load(open(os.path.join(HERE, "work", f"map_{K}.json")))
    g = CF.grid(K)
    Hm = np.array(mp["photo_to_b"]).reshape(3, 3)
    H, W = R.mask(K).shape
    xs, rs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dx, dy = xs - mp["cx"], rs - mp["cy"]; f = 1 + mp["k1"] * (dx * dx + dy * dy)
    q = Hm @ np.stack([(mp["cx"] + dx * f).ravel(), (mp["cy"] + dy * f).ravel(), np.ones(xs.size)])
    u, v = CF.b2r(g, q[0] / q[2], q[1] / q[2])
    return ndimage.map_coordinates(alpha, [v - 0.5, u - 0.5], order=1).reshape(H, W)


def crossings(col):
    s = col > 0.5
    i = np.nonzero(s[1:] != s[:-1])[0]
    out = []
    for j in i:
        a, b = col[j], col[j + 1]
        out.append(j + 0.5 + (0.5 - a) / (b - a) if b != a else j + 1.0)
    return out


def main():
    ma = model_alpha_photo()
    cov = R.coverage(K) * ndimage.binary_dilation(R.mask(K), iterations=2) + 0.0
    cov = np.maximum(cov, ndimage.binary_erosion(R.mask(K)).astype(np.float32))
    g = CF.grid(K)
    mp = json.load(open(os.path.join(HERE, "work", f"map_{K}.json")))
    Hi = np.linalg.inv(np.array(mp["photo_to_b"]).reshape(3, 3))
    rows = []
    for name, reg in CF.REGIONS:
        y0, y1, z0, z1 = reg
        cs = [Hi @ np.array([yy, zz, 1.0]) for yy in (y0, y1) for zz in (z0, z1)]
        bx = (int(min(c[0] / c[2] for c in cs)), int(max(c[0] / c[2] for c in cs)))
        by = (int(min(c[1] / c[2] for c in cs)), int(max(c[1] / c[2] for c in cs)))
        offs = []
        for x in range(bx[0], bx[1]):
            pc = crossings(cov[by[0]:by[1], x]); mc = crossings(ma[by[0]:by[1], x])
            if len(pc) == len(mc) and pc:
                for kk, (a, b) in enumerate(zip(pc, mc)):
                    offs.append((x, a + by[0], b - a, kk))
            elif pc or mc:
                offs.append((x, (pc or mc)[0] + by[0], np.nan, -1))
        o = np.array([v for v in offs if not np.isnan(v[2])]) if offs else np.zeros((0, 4))
        bad = [v for v in offs if np.isnan(v[2])]
        print(f"== {name}: columns with a different edge count: {len(bad)}")
        if len(o):
            big = o[np.abs(o[:, 2]) >= MIN]
            # group runs of big offsets by edge index and contiguous x
            for kk in sorted(set(big[:, 3].astype(int))):
                b = big[big[:, 3] == kk]
                runs = np.split(b, np.nonzero(np.diff(b[:, 0]) > 2)[0] + 1)
                for r in runs:
                    if len(r) >= 2:
                        print(f"   edge {kk}: x {int(r[0, 0])}..{int(r[-1, 0])} row ~{r[:, 1].mean():.1f}  model-photo mean {r[:, 2].mean():+.2f} max {r[np.argmax(np.abs(r[:, 2])), 2]:+.2f} px")
        if bad:
            xs = np.array([v[0] for v in bad]); runs = np.split(xs, np.nonzero(np.diff(xs) > 2)[0] + 1)
            print("   edge-count mismatch at x " + ", ".join(f"{int(r[0])}..{int(r[-1])}" for r in runs if len(r)))


if __name__ == "__main__":
    main()
