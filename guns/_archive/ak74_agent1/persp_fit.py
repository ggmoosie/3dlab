"""persp_fit.py - pinhole camera fit for a profile photo on the exact SDF silhouette (no homography): the left product shot
was taken close and off-square (the photos disagree with each other by IoU 0.9635 after the best homography, so a planar
warp cannot explain it). Powell on (rotation, position, focal) minimising the blurred silhouette difference at 1x photo
resolution, then the exact IoU at 2x (silcmp persp mode). usage: .venv/bin/python guns/ak74/persp_fit.py k=L [iters=300]"""
import os, sys, json, time
import numpy as np
from scipy import ndimage, optimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
K = ARGS.get("k", "L")
import camfit as CF
CF.SS = 1
import silcmp as SC
SC.STEP = 0.8
import ak_ref as R

m = R.mask(K).astype(np.float32)
ys, xs = np.nonzero(m)
box = (xs.min() - 15, ys.min() - 15, xs.max() + 15, ys.max() + 15)
ms = ndimage.gaussian_filter(m, 1.0)[box[1]:box[3], box[0]:box[2]]
def from_homography(k):
    """pinhole camera (match34 convention) whose image of the plane x = 0 equals the fitted photo->B homography"""
    import match34 as M, ak_spec as P
    mp = json.load(open(os.path.join(HERE, "work", f"map_{k}.json")))
    G = np.linalg.inv(np.array(mp["photo_to_b"]).reshape(3, 3))          # B (y, z, 1) -> photo px
    o = np.array(P.ORIGIN_B)
    A = np.array([[-1000.0, 0, o[1]], [0, 1000.0, o[2]], [0, 0, 1]])        # Blender (y, z, 1) -> B (y, z, 1)
    Gp = G @ A
    Hh, Ww = R.mask(k).shape
    best = None
    for asp in np.linspace(0.94, 1.06, 61):
        for f in ([float(ARGS["fmm"]) / 36.0 * Ww] if "fmm" in ARGS else np.geomspace(800, 200000, 400)):
            Kc = np.array([[f, 0, Ww / 2], [0, f * asp, Hh / 2], [0, 0, 1]])
            Mx = np.linalg.solve(Kc, Gp)
            a, b = Mx[:, 0], Mx[:, 1]
            err = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) + abs(np.linalg.norm(a) / np.linalg.norm(b) - 1)
            if best is None or err < best[0]:
                best = (err, f, Mx, asp)
    err, f, Mx, ASP = best
    print(f"[persp] pixel aspect fy/fx {ASP:.4f}", flush=True)
    s_ = 2.0 / (np.linalg.norm(Mx[:, 0]) + np.linalg.norm(Mx[:, 1]))
    ry, rz, t = Mx[:, 0] * s_, Mx[:, 1] * s_, Mx[:, 2] * s_
    if t[2] < 0:
        ry, rz, t = -ry, -rz, -t
    rx = np.cross(ry, rz)
    Rcv = np.stack([rx, ry, rz], 1)
    U, _, Vt = np.linalg.svd(Rcv); Rcv = U @ Vt
    C = -Rcv.T @ t
    Rm = np.stack([Rcv[0], -Rcv[1], -Rcv[2]], 1)
    ang = np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    rv = v / np.linalg.norm(v) * ang
    print(f"[persp] homography -> pinhole: f {f:.0f} px ({f / Ww * 36:.0f} mm), orthogonality error {err:.4f}, centre {C}", flush=True)
    return np.r_[rv, C, f / Ww * 36.0, ASP]


x0 = from_homography(K) if ARGS.get("init", "h") == "h" else np.array(json.load(open(os.path.join(HERE, "work", f"persp_{K}.json")))["x"])
best = [1e9, x0.copy()]
n = [0]


def cost(x):
    t = time.time()
    D, pxmm = SC.model_d2_persp(K, x)
    a = np.clip(0.5 - D / pxmm, 0, 1)[box[1]:box[3], box[0]:box[2]]
    c = float(((ndimage.gaussian_filter(a, 1.0) - ms) ** 2).sum())
    s = a > 0.5; mm = m[box[1]:box[3], box[0]:box[2]] > 0.5
    iou = (s & mm).sum() / (s | mm).sum()
    n[0] += 1
    if c < best[0]:
        best[0], best[1] = c, x.copy()
        json.dump(dict(x=x.tolist(), iou_1x=float(iou)), open(os.path.join(HERE, "work", f"persp_{K}.json"), "w"))
    print(f"[persp] eval {n[0]} cost {c:.1f} IoU(1x) {iou:.4f} {time.time() - t:.1f}s", flush=True)
    return c


import builtins
_p = builtins.print
def quiet(*a, **k):
    if a and isinstance(a[0], str) and a[0].startswith("[silp]"):
        return
    _p(*a, **k)
builtins.print = quiet
if len(x0) == 7:
    x0 = np.r_[x0, 1.0]
scale = np.array([0.003, 0.003, 0.003, 0.004, 0.004, 0.004, 2.0, 0.003])
optimize.minimize(lambda q: cost(x0 + q * scale), np.zeros(8), method="Powell",
                  options=dict(maxfev=int(ARGS.get("iters", 300)), xtol=0.05, ftol=1e-4))
print("PERSP_DONE", best[0])
