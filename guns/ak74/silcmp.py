"""silcmp.py - light (no Blender) side-silhouette gate for iteration: the model silhouette is computed straight from the
part SDFs (min over x of f(x, y, z) = the 2-D signed distance of the projection; alpha = clip(0.5 - d / px)), on the same
orthographic camera grid compare.py renders, then the same per-photo camera fit (homography + radial) and IoU against
the photo masks. Peak memory stays < 1.5 GB (points evaluated in chunks).
usage: .venv/bin/python guns/ak74/silcmp.py [tag=p1] [photos=R,L] [parts=a,b] [step=0.45] [sheet=1]"""
import os, sys, json, math, time
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import ak_ref as R
import camfit as CF

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
TAG = ARGS.get("tag", "p1")
PHOTOS = ARGS.get("photos", "R,L").split(",")
STEP = float(ARGS.get("step", 0.45))
SKIP = {"rivets_rec", "rivets_front"}


def model_d2(k, only=None, chunk=600_000):
    """2-D signed distance (mm) of the model's side projection on camera k's render grid (SS x photo px)"""
    import parts as PT
    g = CF.grid(k)
    H, W = g["H"] * CF.SS, g["W"] * CF.SS
    D = np.full((H, W), 50.0, np.float32)
    for name, (fn, lo, hi, h, mat) in PT.table().items():
        if name in SKIP or (only and name not in only):
            continue
        t0 = time.time()
        # render-px box of this part
        u0, v0 = CF.b2r(g, lo[1], hi[2]); u1, v1 = CF.b2r(g, hi[1], lo[2])
        ua, ub = int(max(0, min(u0, u1) - 3)), int(min(W, max(u0, u1) + 3))
        va, vb = int(max(0, min(v0, v1) - 3)), int(min(H, max(v0, v1) + 3))
        if ub <= ua or vb <= va:
            continue
        uu, vv = np.meshgrid(np.arange(ua, ub), np.arange(va, vb))
        yy, zz = CF.r2b(g, uu + 0.5, vv + 0.5)
        yy, zz = yy.ravel(), zz.ravel()
        xs = np.arange(lo[0], hi[0] + 1e-6, STEP)
        best = np.full(len(yy), 50.0, np.float32)
        n = max(1, chunk // len(xs))
        for i in range(0, len(yy), n):
            Y = np.repeat(yy[i:i + n][:, None], len(xs), 1); Z = np.repeat(zz[i:i + n][:, None], len(xs), 1)
            X = np.broadcast_to(xs[None], Y.shape)
            f = fn(X.ravel(), Y.ravel(), Z.ravel()).reshape(Y.shape)
            best[i:i + n] = f.min(1)
        sub = D[va:vb, ua:ub]
        np.minimum(sub, best.reshape(vb - va, ub - ua), out=sub)
        print(f"[sil] {k} {name:10s} {ub - ua}x{vb - va} px  {time.time() - t0:.1f}s", flush=True)
    return D


def main():
    res = {}
    only = ARGS["parts"].split(",") if ARGS.get("parts") else None
    for k in PHOTOS:
        g = CF.grid(k)
        D = model_d2(k, only)
        pxmm = g["mmpx"] / CF.SS
        alpha = np.clip(0.5 - D / pxmm, 0, 1).astype(np.float32)
        np.save(os.path.join(HERE, "work", f"sil_{TAG}_{k}.npy"), alpha)
        C = CF.Cam(CF.initial_h(k, g), R.mask(k).shape)
        i0, _ = CF.iou_of(alpha, C, k)
        CF.fit_cam(alpha, k, C)
        iou, sil = CF.iou_of(alpha, C, k)
        aff = CF.describe(C)
        CF.save_map(k, g, C)
        print(f"SIL {k} IoU calibration-only {i0:.4f} -> camera-fitted {iou:.4f}  {aff}", flush=True)
        res[k] = dict(iou=iou, iou_calibration_only=i0, camera=aff, regions=CF.region_ious(k, sil, C, g))
        if ARGS.get("sheet", "1") == "1":
            CF.diff_sheets(k, sil, iou, aff, C, g, TAG, "sdf")
    json.dump(res, open(os.path.join(HERE, "review", f"iou_{TAG}_sdf.json"), "w"), indent=1)
    print("SIL VERDICT", "PASS" if all(v["iou"] >= 0.98 for v in res.values()) else "FAIL",
          {k: round(v["iou"], 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
