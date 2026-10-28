"""align_refs.py - line the two profile photos up: mirror the RIGHT-side photo (163254) and solve a similarity + 2-term
perspective map onto the LEFT profile master (163245) by maximising silhouette IoU. Writes (private, work/):
  work/ref_R_aligned.png   the right photo, mirrored + warped into the left photo's frame
  work/ref_pair.jpg        left | right-aligned | silhouette overlap (red left only, cyan right only, grey both)
and review/ref_align.json (map parameters + IoU before/after).
usage: .venv/bin/python guns/ak74m/align_refs.py"""
import os, sys, json, math
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ak_ref as R
from skimage.transform import warp, ProjectiveTransform

W_, H_ = Image.open(R.PHOTO).size


def H(p, src_shape):
    """left-frame px -> mirrored-right px (inverse map for skimage.warp)"""
    roll, sc, tx, ty, px, py = p
    a = math.radians(roll); c, s = math.cos(a) / sc, math.sin(a) / sc
    cx, cy = (W_ - 1) / 2, (H_ - 1) / 2
    hs, ws = src_shape
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    A = np.array([[c, -s, -tx], [s, c, -ty], [px * 1e-4, py * 1e-4, 1]], float)
    T2 = np.array([[1, 0, (ws - 1) / 2], [0, 1, (hs - 1) / 2], [0, 0, 1]], float)
    return T2 @ A @ T1


def main():
    fgL, aL = R.masks(R.PHOTO)
    fgR, aR = R.masks(R.PHOTO_R)
    fgR, aR = fgR[:, ::-1], aR[:, ::-1]                                           # mirror: muzzle left, like the master
    small = lambda m: m[::2, ::2]
    fL = small(fgL)

    def wmask(p, m=fgR, shape=(H_, W_)):
        return warp(m.astype(np.float32), ProjectiveTransform(H(p, m.shape)), output_shape=shape, order=1) > 0.5

    def f(p):
        w = wmask(p)[::2, ::2]
        return -float((w & fL).sum() / max((w | fL).sum(), 1))
    iou0 = -f([0, 1, 0, 0, 0, 0])
    best = min((f([0, sc, tx, ty, 0, 0]), [0, sc, tx, ty, 0, 0])
               for sc in (0.98, 1.0, 1.02) for tx in range(-40, 41, 8) for ty in range(-20, 21, 5))[1]
    x = best
    for scale in (1.0, 0.3):
        simp = [x] + [list(np.array(x) + d) for d in np.eye(6) * (np.array([0.3, 0.01, 3, 3, 0.5, 0.5]) * scale)]
        x = list(minimize(f, x, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=1500, initial_simplex=simp)).x)
    iou = -f(x)
    print(f"ALIGN IoU before {iou0:.4f} -> after {iou:.4f}  params {np.round(x, 4).tolist()}")
    Hm = ProjectiveTransform(H(x, fgR.shape))
    aligned = np.stack([warp(aR[..., i], Hm, output_shape=(H_, W_), order=1, cval=255, preserve_range=True) for i in range(3)], -1)
    Image.fromarray(aligned.clip(0, 255).astype(np.uint8)).save(os.path.join(HERE, "work", "ref_R_aligned.png"))
    wm = wmask(x)
    ov = np.full((H_, W_, 3), 245, np.uint8); ov[fgL & wm] = 150; ov[fgL & ~wm] = (230, 40, 40); ov[wm & ~fgL] = (40, 200, 230)
    ys, xs = np.nonzero(fgL | wm); box = (xs.min() - 15, ys.min() - 15, xs.max() + 15, ys.max() + 15)
    tiles = [Image.fromarray(t).crop(box) for t in (aL.astype(np.uint8), aligned.clip(0, 255).astype(np.uint8), ov)]
    w, h = tiles[0].size
    sh = Image.new("RGB", (w, h * 3 + 66), (30, 30, 30)); d = ImageDraw.Draw(sh)
    for i, (t, lab) in enumerate(zip(tiles, ("left profile (master)", "right profile, mirrored + aligned",
                                             f"overlap: IoU {iou0:.3f} raw -> {iou:.3f} aligned (red left only, cyan right only)"))):
        sh.paste(t, (0, i * (h + 22) + 22)); d.text((6, i * (h + 22) + 5), lab, fill="white")
    sh.save(os.path.join(HERE, "work", "ref_pair.jpg"), quality=90)
    json.dump(dict(iou_raw=round(iou0, 4), iou_aligned=round(iou, 4), params=dict(zip(
        ("roll_deg", "scale", "tx_px", "ty_px", "persp_x", "persp_y"), [round(float(v), 4) for v in x]))),
        open(os.path.join(HERE, "review", "ref_align.json"), "w"), indent=1)
    np.save(os.path.join(HERE, "work", "fg_R_aligned.npy"), wm)


if __name__ == "__main__":
    main()
