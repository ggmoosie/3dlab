"""silfit.py - refine the 3/4 camera on the silhouette AND measure the gun's true width: maximise IoU between the photo
mask and the projected high-poly (work/high.npz, B frame) with the model's X (width) scaled by k about the bore plane.
Starts from camera_match's solution (result json key x0). Same camera model as camera_match.py / Glock silcam.py.
usage: .venv/bin/python guns/scar17s/silfit.py work/match_left34.json review/cam_left34.json [out.json]"""
import sys, json, math, os
import numpy as np
from PIL import Image
from scipy import ndimage, optimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_spec as P

J = json.load(open(sys.argv[1])); C = json.load(open(sys.argv[2]))
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "work", "silfit.json")
ref = np.asarray(Image.open(J["reference"]).convert("RGB")).astype(np.float32); H0, W0 = ref.shape[:2]
SC = 0.5; W, H = int(W0 * SC), int(H0 * SC)
i0 = max(2, int(0.015 * min(H0, W0))); i1 = i0 + 6
border = np.concatenate([ref[i0:i1].reshape(-1, 3), ref[-i1:-i0].reshape(-1, 3), ref[:, i0:i1].reshape(-1, 3), ref[:, -i1:-i0].reshape(-1, 3)])
dist = np.linalg.norm(ref - np.median(border, 0), axis=2)
pm = ndimage.median_filter(dist > 30.0, 5)
pm = np.asarray(Image.fromarray(pm.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 127
D = np.load(os.path.join(HERE, "work", "high.npz"))
names = sorted({k.split("__")[0] for k in D.files} - {"round", "bolt"})
V = np.concatenate([D[n + "__V"][::3] for n in names]).astype(np.float64)
o = np.array(P.ORIGIN_B)
BL = np.c_[V[:, 0] - o[0], -(V[:, 1] - o[1]), V[:, 2] - o[2]] * 0.001
SENSOR = 36.0


def rod(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def project(x, Pt, w, h):
    R = rod(x[:3]); t = x[3:6]; f = x[6] / SENSOR * w; pc = (Pt - t) @ R; z = -pc[:, 2]
    return np.stack([w / 2 + f * pc[:, 0] / z, h / 2 - f * pc[:, 1] / z], 1)


def mask(x, k):
    Pt = BL * np.array([k, 1, 1])
    uv = project(x, Pt, W, H); u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H); m = np.zeros((H, W), bool); m[v[ok], u[ok]] = True
    return ndimage.binary_closing(m, iterations=2)


def iou(p):
    m = mask(p[:7], p[7]); return (m & pm).sum() / max(1, (m | pm).sum())


x0 = np.array(C["x0"] + [1.0])
print("start IoU %.4f" % iou(x0))
best = x0
for it in range(2):
    r = optimize.minimize(lambda p: 1 - iou(p), best, method="Powell", options=dict(maxiter=3000, xtol=1e-5, ftol=1e-6))
    best = r.x; print("pass", it, "IoU %.4f  width scale k %.3f  focal %.1f" % (1 - r.fun, best[7], best[6]))
scan = {round(k, 2): float(iou(np.r_[best[:7], k])) for k in np.arange(0.6, 1.21, 0.05)}
print("k scan", scan)
json.dump(dict(x=best[:7].tolist(), k=float(best[7]), iou=float(iou(best)), scan=scan), open(out, "w"), indent=1)
print("SILFIT IoU %.4f k %.3f" % (iou(best), best[7]))
