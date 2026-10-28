"""match34.py - camera solves for the two 3/4 reference shots (same camera model as tools/camera_match.py: rodrigues
rotation, position, focal on a 36 mm horizontal sensor), with the model posed as photographed:
  H = 163300  high 3/4 of the RIGHT side, the rifle lying on its left side, magazine OUT (the loose magazine lying across the
              gun is masked out of the silhouette test)
  F = 163304  3/4 from the left-rear, stock FOLDED along the left side, magazine in
1. landmark solve (Levenberg-Marquardt) on trusted landmarks -> rms (gate <= 1 % of the width), check points reported
2. silhouette refinement: projected high-poly vertices (work/high.npz) vs the photo mask, camera + width scale k
   (silfit idea from the SCAR): the k scan says whether the cross-sections are too wide / narrow
3. writes work/match_<k>.json (tools/camera_match.py config, landmarks as world points in Blender metres) and
   review/cam_<k>.json (verdict, rms, per-point errors, silhouette IoU, k scan) and the region crop sheet inputs.
usage: .venv/bin/python guns/ak74/match34.py [shots=H,F] [npz=work/high.npz] [refine=1]"""
import os, sys, json, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import ak_ref as R
import ak_spec as P

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
SENSOR = 36.0
bp, lp = R.bp, R.lp2b


def yb(x):
    return bp(x, 285.98 + R.BORE_SLOPE * (x - 1030.0))[0]


def stock_fold_b(V, deg=180.0):
    """rotate B-frame points of the stock about the vertical hinge axis (x, y) = P.HINGE (folds to the left)"""
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    x, y = V[:, 0] - P.HINGE[0], V[:, 1] - P.HINGE[1]
    return np.c_[P.HINGE[0] + x * c - y * s, P.HINGE[1] + x * s + y * c, V[:, 2]]


def fold_pt(p):
    return tuple(stock_fold_b(np.array([p], float))[0])


ZTOP = bp(1086.0, 232.6)[1]
GRIPBOT = bp(477.0, 426.0)
# landmarks: name -> (photo px, B point mm, solve|check)
LM = {
    "H": {
        "ear_right_tip": ((927.5, 186.0), (-7.2, yb(1085.5), ZTOP), True),
        "ear_left_tip": ((930.0, 213.5), (7.2, yb(1085.5), ZTOP), True),
        "ch_paddle_end": ((580.5, 244.0), (-P.HW_REC - 20.6, *P.CH_POS), True),
        "latch_plate_hole": ((262.0, 466.0), (-P.HW_REC - 0.5, *bp(475.0, 296.0)), True),
        "grip_screw": ((619.0, 603.0), (0.0, GRIPBOT[0], GRIPBOT[1] - 0.5), True),
        "selector_nut": ((320.0, 420.0), (-P.HW_REC - 3.4, *P.SELECTOR_AXIS), True),
        "lhg_front_right_bottom": ((600.0, 296.0), (-P.HW_LHG + 4.0, yb(884.0), bp(884.0, 306.0)[1]), False),
    },
    "F": {
        "brake_face": ((224.0, 268.0), (0.0, P.MUZZLE_Y, 0.0), True),
        "ear_tip_a": ((377.0, 237.0), (7.2, yb(1085.5), ZTOP), True),
        "ear_tip_b": ((394.0, 257.0), (-7.2, yb(1085.5), ZTOP), True),
        "grip_screw": ((857.0, 655.0), (0.0, GRIPBOT[0], GRIPBOT[1] - 0.5), True),
        "mag_floor_front": ((388.0, 535.0), (6.0, *bp(731.0, 460.0)), True),
        "mag_floor_rear": ((400.0, 652.0), (6.0, *bp(678.0, 498.5)), True),
        "sling_loop": ((1050.0, 360.0), fold_pt((-P.HW_STOCK_WRIST - 2.0, *bp(437.0, 312.0))), False),
    },
}
for k in LM:
    for n, (px, b, s) in list(LM[k].items()):
        LM[k][n] = (px, tuple(float(v) for v in b[-3:]), s)


def b2bl(V):
    o = np.array(P.ORIGIN_B)
    W = np.asarray(V, float) - o
    return np.c_[W[..., 0], -W[..., 1], W[..., 2]] * 0.001


def rod(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def project(x, Pt, W, H):
    Rm = rod(x[:3]); t = x[3:6]; f = x[6] / SENSOR * W
    pc = (Pt - t) @ Rm; z = -pc[:, 2]; z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([W / 2 + f * pc[:, 0] / z, H / 2 - f * pc[:, 1] / z], 1), z


def look_rot(d):
    d = np.array(d, float); d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([0, 1.0, 0])
    zc = -d; xc = np.cross(up, zc); xc /= np.linalg.norm(xc); yc = np.cross(zc, xc)
    Rm = np.stack([xc, yc, zc], 1)
    ang = math.acos(max(-1, min(1, (np.trace(Rm) - 1) / 2)))
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return v / np.linalg.norm(v) * ang if np.linalg.norm(v) > 1e-9 else np.zeros(3)


def solve(k, init_dirs):
    W, H = Image.open(R.PHOTO[k]).size
    pts = LM[k]
    use = [n for n, v in pts.items() if v[2]]
    Pw = b2bl(np.array([pts[n][1] for n in use])); UV = np.array([pts[n][0] for n in use], float)
    best = None
    for d in init_dirs:
        for fmm in (50.0, 85.0, 135.0):
            c = Pw.mean(0); ext = np.linalg.norm(Pw.max(0) - Pw.min(0))
            x0 = np.concatenate([look_rot(d), c - np.array(d) / np.linalg.norm(d) * ext / (0.6 * SENSOR / fmm), [fmm]])
            def res(x):
                uv, z = project(x, Pw, W, H)
                return np.concatenate([(uv - UV).ravel(), 1e3 * np.minimum(z, 0)])
            s = optimize.least_squares(res, x0, x_scale=np.r_[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10.0], max_nfev=4000,
                                       bounds=(np.r_[[-np.inf] * 6, 15.0], np.r_[[np.inf] * 6, 400.0]))
            if best is None or s.cost < best.cost:
                best = s
    x = best.x
    allp = list(pts)
    uv, _ = project(x, b2bl(np.array([pts[n][1] for n in allp])), W, H)
    err = {n: float(np.linalg.norm(uv[i] - pts[n][0])) for i, n in enumerate(allp)}
    rms = float(np.sqrt(np.mean([err[n] ** 2 for n in use])))
    return x, rms, err, (W, H)


def photo_mask(k):
    a = np.asarray(Image.open(R.PHOTO[k]).convert("RGB")).astype(np.float32)
    L = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    m = ndimage.binary_opening(L < 200, iterations=1)
    lab, n = ndimage.label(m); size = ndimage.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, 1 + np.nonzero(size > 2000)[0])
    m = ndimage.binary_fill_holes(m) | m
    return m


# pixels excluded from the 3/4 silhouette test: the loose magazine in H, the dealer sticker in F
IGNORE = {"H": [(430, 300), (1150, 330), (1150, 600), (980, 590), (720, 470), (440, 380)],
          "F": [(780, 214), (850, 214), (850, 262), (780, 262)]}


def ignore_mask(k, shape):
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon(IGNORE[k], fill=255)
    return np.asarray(im) > 127


def model_points(k, npz):
    D = np.load(npz)
    out = []
    for n in sorted({f.split("__")[0] for f in D.files}):
        if n in ("rivets_rec", "rivets_front") or (k == "H" and n == "magazine"):
            continue
        V = D[n + "__V"][::4].astype(np.float64)
        if k == "F" and n in ("stock", "buttplate", "slings"):
            V = stock_fold_b(V)
        out.append(V)
    return np.concatenate(out)


def refine(k, x, npz, size):
    W, H = size
    m = photo_mask(k); ign = ignore_mask(k, m.shape)
    sc = 0.5; w2, h2 = int(W * sc), int(H * sc)
    pm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w2, h2), Image.BILINEAR)) > 127
    ig = np.asarray(Image.fromarray(ign.astype(np.uint8) * 255).resize((w2, h2), Image.NEAREST)) > 127
    Vb = model_points(k, npz)

    def mmask(p, kw):
        V = Vb.copy(); V[:, 0] *= kw
        uv, _ = project(p, b2bl(V), w2, h2)
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < w2) & (v >= 0) & (v < h2); mm = np.zeros((h2, w2), bool); mm[v[ok], u[ok]] = True
        return ndimage.binary_closing(ndimage.binary_dilation(mm, iterations=1), iterations=2)

    def iou(p, kw=1.0):
        mm = mmask(p, kw)
        a, b = mm & ~ig, pm & ~ig
        return (a & b).sum() / max(1, (a | b).sum())
    i0 = iou(x)
    r = optimize.minimize(lambda q: 1 - iou(q), x, method="Powell", options=dict(maxiter=2500, xtol=1e-5, ftol=1e-6))
    xr = r.x
    scan = {round(kw, 2): round(float(iou(xr, kw)), 4) for kw in np.arange(0.8, 1.21, 0.05)}
    return xr, i0, float(iou(xr)), scan


def main():
    shots = ARGS.get("shots", "H,F").split(",")
    npz = ARGS.get("npz", os.path.join(HERE, "work", "high.npz"))
    dirs = {"H": [(-0.6, 0.55, -0.6), (-0.8, 0.3, -0.5), (-0.5, 0.7, -0.5)],     # looking at the right side & belly, from behind
            "F": [(-0.7, -0.6, -0.4), (-0.6, -0.4, -0.7), (-0.8, -0.5, -0.3)]}
    for k in shots:
        x, rms, err, size = solve(k, dirs[k])
        W, H = size
        out = dict(shot=k, photo=os.path.basename(R.PHOTO[k]), rms_px=rms, rms_pct=100 * rms / W, focal_mm=float(x[6]),
                   verdict="PASS" if rms <= 0.01 * W else "FAIL", points=err, x0=[float(v) for v in x])
        print(f"CAM {k} rms {rms:.2f} px ({100 * rms / W:.2f} % of width) focal {x[6]:.1f}  {out['verdict']}")
        for n, e in err.items():
            print(f"   {n:24s} {'solve' if LM[k][n][2] else 'CHECK'} {e:6.1f} px")
        if ARGS.get("refine", "1") == "1" and os.path.exists(npz):
            xr, i0, i1, scan = refine(k, x, npz, size)
            Pw = b2bl(np.array([v[1] for v in LM[k].values()]))
            uv, _ = project(xr, Pw, W, H)
            errr = {n: float(np.linalg.norm(uv[i] - v[0])) for i, (n, v) in enumerate(LM[k].items())}
            rr = float(np.sqrt(np.mean([errr[n] ** 2 for n, v in LM[k].items() if v[2]])))
            out.update(sil_iou_landmark_cam=i0, sil_iou_refined=i1, width_scan=scan, x_refined=[float(v) for v in xr],
                       rms_refined_px=rr, rms_refined_pct=100 * rr / W)
            print(f"SIL {k} IoU {i0:.4f} -> {i1:.4f} (refined); landmark rms on the refined camera {rr:.1f} px ({100 * rr / W:.2f} %)")
            print(f"SIL {k} width scan {scan}")
            x = xr
        json.dump(out, open(os.path.join(HERE, "review", f"cam_{k}.json"), "w"), indent=1)
        fwd = -rod(np.array(x[:3]))[:, 2]
        cfg = {"reference": R.PHOTO[k], "camera": f"REFCAM_{k}", "fix_focal": False, "focal_range": [15, 400],
               "init": {"focal_mm": float(x[6]), "view_dir": fwd.tolist()},
               "x_solved": [float(v) for v in x],
               "points": [{"name": n, "px": list(v[0]), "world": b2bl(np.array([v[1]]))[0].tolist(), "check": not v[2]}
                          for n, v in LM[k].items()],
               "max_rms_pct": 1.0, "result": os.path.join(HERE, "review", f"cam_{k}_blender.json")}
        json.dump(cfg, open(os.path.join(HERE, "work", f"match_{k}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
