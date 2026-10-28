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


def model_d2_persp(k, x, only=None, chunk=600_000):
    """2-D signed distance (mm, approx.) of the model seen by a pinhole camera x (match34 convention, Blender metres) on the
    photo grid x SS: per pixel the ray is sampled at the part's x range (the camera looks along +-x)"""
    import parts as PT
    import match34 as M
    import bl_high
    H0, W0 = R.mask(k).shape; H, W = H0 * CF.SS, W0 * CF.SS
    Rm = M.rod(np.array(x[:3])); t = np.array(x[3:6]); f = x[6] / 36.0 * W
    asp = x[7] if len(x) > 7 else 1.0
    o = np.array(bl_high.P.ORIGIN_B)
    C = np.array([t[0] * 1000 + o[0], -t[1] * 1000 + o[1], t[2] * 1000 + o[2]])           # camera centre in B
    D = np.full((H, W), 50.0, np.float32)
    pxmm = None
    for name, (fn, lo, hi, h, mat) in PT.table().items():
        if name in SKIP or (only and name not in only):
            continue
        t0 = time.time()
        cs = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        uv, z = M.project(np.array(x[:7]), M.b2bl(cs), W, H)
        uv[:, 1] = H / 2 + (uv[:, 1] - H / 2) * asp
        ua, ub = int(max(0, uv[:, 0].min() - 3)), int(min(W, uv[:, 0].max() + 3))
        va, vb = int(max(0, uv[:, 1].min() - 3)), int(min(H, uv[:, 1].max() + 3))
        if ub <= ua or vb <= va:
            continue
        uu, vv = np.meshgrid(np.arange(ua, ub) + 0.5, np.arange(va, vb) + 0.5)
        # camera-space ray -> Blender world -> B direction
        dc = np.stack([(uu.ravel() - W / 2) / f, -(vv.ravel() - H / 2) / (f * asp), -np.ones(uu.size)], 1)
        dw = dc @ Rm.T
        dB = np.c_[dw[:, 0], -dw[:, 1], dw[:, 2]]
        xs = np.arange(lo[0], hi[0] + 1e-6, STEP)
        best = np.full(len(dB), 50.0, np.float32)
        n = max(1, chunk // len(xs))
        for i in range(0, len(dB), n):
            d_ = dB[i:i + n]; tt = (xs[None, :] - C[0]) / d_[:, :1]
            X = C[0] + d_[:, :1] * tt; Y = C[1] + d_[:, 1:2] * tt; Z = C[2] + d_[:, 2:3] * tt
            fv = fn(X.ravel(), Y.ravel(), Z.ravel()).reshape(X.shape)
            best[i:i + n] = fv.min(1)
        sub = D[va:vb, ua:ub]
        np.minimum(sub, best.reshape(vb - va, ub - ua), out=sub)
        print(f"[silp] {k} {name:10s} {ub - ua}x{vb - va} px  {time.time() - t0:.1f}s", flush=True)
    dist = np.linalg.norm(C - np.array([0, 250.0, 0]))
    pxmm = 36.0 * dist / (x[6] * W)
    return D, pxmm


def main_persp():
    res = {}
    import match34 as M
    for k in PHOTOS:
        cam = json.load(open(os.path.join(HERE, "work", f"persp_{k}.json")))
        D, pxmm = model_d2_persp(k, cam["x"])
        alpha = np.clip(0.5 - D / pxmm, 0, 1).astype(np.float32)
        np.save(os.path.join(HERE, "work", f"silp_{TAG}_{k}.npy"), alpha)
        g = dict(k=k, H=R.mask(k).shape[0], W=R.mask(k).shape[1])
        C = CF.Cam(np.diag([CF.SS, CF.SS, 1.0]), R.mask(k).shape)
        i0, _ = CF.iou_of(alpha, C, k)
        CF.fit_cam(alpha, k, C, scan=False)
        iou, sil = CF.iou_of(alpha, C, k)
        print(f"SILP {k} IoU pinhole {i0:.4f} -> + residual warp {iou:.4f}  {CF.describe(C)}", flush=True)
        res[k] = dict(iou=iou, iou_pinhole_only=i0, camera=cam["x"], residual=CF.describe(C))
    json.dump(res, open(os.path.join(HERE, "review", f"iou_{TAG}_sdfp.json"), "w"), indent=1)


GROUPS = {"stock": ["stock", "buttplate", "slings"], "mag": ["magazine"]}     # assemblies with play: own pose per photo


def main():
    res = {}
    only = ARGS["parts"].split(",") if ARGS.get("parts") else None
    import parts as PT
    allp = [n for n in PT.table() if n not in SKIP and (not only or n in only)]
    sub = {gname: [n for n in ns if n in allp] for gname, ns in GROUPS.items()}
    main_parts = [n for n in allp if not any(n in v for v in sub.values())]
    for k in PHOTOS:
        g = CF.grid(k)
        pxmm = g["mmpx"] / CF.SS
        al = {"main": np.clip(0.5 - model_d2(k, main_parts) / pxmm, 0, 1).astype(np.float32)}
        for gname, ns in sub.items():
            if ns:
                al[gname] = np.clip(0.5 - model_d2(k, ns) / pxmm, 0, 1).astype(np.float32)
        alpha = np.maximum.reduce(list(al.values()))
        np.save(os.path.join(HERE, "work", f"sil_{TAG}_{k}.npy"), alpha)
        C = CF.Cam(CF.initial_h(k, g), R.mask(k).shape)
        i0, _ = CF.iou_of(alpha, C, k)
        CF.fit_cam(alpha, k, C)
        iou1, sil = CF.iou_of(alpha, C, k)
        aff = CF.describe(C)
        CF.save_map(k, g, C)
        # per-assembly pose (folding-stock hinge play, magazine rock): each group gets its own small projective
        # correction on top of the main camera, fitted on the composite silhouette
        cams = {gname: None for gname in al if gname != "main"}
        if ARGS.get("groups", "1") == "1":
            cams = CF.fit_groups(al, k, C)
        iou, sil = CF.iou_groups(al, k, C, cams)
        print(f"SIL {k} IoU calibration-only {i0:.4f} -> camera-fitted {iou1:.4f} -> with assembly poses {iou:.4f}  {aff}", flush=True)
        res[k] = dict(iou=iou, iou_single_camera=iou1, iou_calibration_only=i0, camera=aff,
                      assembly_poses={gname: (c.describe() if c is not None else None) for gname, c in cams.items()},
                      regions=CF.region_ious(k, sil, C, g))
        if ARGS.get("sheet", "1") == "1":
            CF.diff_sheets(k, sil, iou, aff, C, g, TAG, "sdf")
    json.dump(res, open(os.path.join(HERE, "review", f"iou_{TAG}_sdf.json"), "w"), indent=1)
    print("SIL VERDICT", "PASS" if all(v["iou"] >= 0.98 for v in res.values()) else "FAIL",
          {k: round(v["iou"], 4) for k, v in res.items()})


if __name__ == "__main__":
    main_persp() if ARGS.get("persp") == "1" else main()
