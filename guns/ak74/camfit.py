"""camfit.py - the side-profile cameras shared by silcmp.py (SDF silhouettes, no Blender) and compare.py (Cycles renders).
Camera k (R = right master, L = left): orthographic, square render pixels of mmpx / SS mm, axis-aligned with the B frame,
looking +x (R, right side) or -x (L). Photo px -> render px = radial term k1 about the photo centre, then a homography,
fitted per photo (the product shots are not exactly square to the gun: the left one has ~5 % perspective along the gun,
both are rotated ~0.4-0.55 deg). No bpy here."""
import os, math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, optimize
import ak_ref as R

HERE = os.path.dirname(os.path.abspath(__file__))
SS = 2
FONT = os.path.join(HERE, "../../fonts/WorkSans-Bold.ttf")


def grid(k):
    H, W = R.mask(k).shape
    if k == "R":
        mmpx = R.S_MM
        yc, zc = R.px2b(W / 2, H / 2); sgn = 1
    else:
        mmpx = R.L_SH
        yc = R.MUZZLE_Y - (W / 2 - R.L_X_MUZ) * R.L_SH
        zc = (R.L_ROW_MUZ + R.L_SLOPE * (W / 2 - R.L_X_MUZ) - H / 2) * R.L_SH; sgn = -1
    return dict(k=k, H=H, W=W, mmpx=float(mmpx), yc=float(yc), zc=float(zc), sgn=sgn)


def b2r(g, y, z):
    u = g["sgn"] * (np.asarray(y) - g["yc"]) / g["mmpx"] * SS + g["W"] * SS / 2
    v = -(np.asarray(z) - g["zc"]) / g["mmpx"] * SS + g["H"] * SS / 2
    return u, v


def r2b(g, u, v):
    y = g["yc"] + g["sgn"] * (np.asarray(u) - g["W"] * SS / 2) * g["mmpx"] / SS
    z = g["zc"] - (np.asarray(v) - g["H"] * SS / 2) * g["mmpx"] / SS
    return y, z


def b2photo(k, y, z):
    if k == "R":
        return R.b2px(y, z)
    x = R.L_X_MUZ + (R.MUZZLE_Y - np.asarray(y, float)) / R.L_SH
    row = R.L_ROW_MUZ + R.L_SLOPE * (x - R.L_X_MUZ) - np.asarray(z, float) / R.L_SV
    return x, row


def initial_h(k, g):
    pts = [(0.0, 0.0), (600.0, 0.0), (0.0, 80.0)]
    src = np.array([b2photo(k, y, z) for y, z in pts], float)
    dst = np.array([b2r(g, y, z) for y, z in pts], float)
    A = np.linalg.solve(np.c_[src, np.ones(3)], dst).T
    return np.vstack([A, [0, 0, 1.0]])


class Cam:
    def __init__(self, H0, shape):
        self.H0 = H0; self.h, self.w = shape; self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.p = np.zeros(9)

    def mats(self, p=None):
        p = self.p if p is None else p
        Hd = np.array([[1 + p[0] * 1e-3, p[1] * 1e-3, p[2]], [p[3] * 1e-3, 1 + p[4] * 1e-3, p[5]],
                       [p[6] * 1e-6, p[7] * 1e-6, 1.0]])
        return self.H0 @ Hd, p[8] * 1e-8

    def map(self, x, r, p=None):
        Hm, k1 = self.mats(p)
        dx, dy = x - self.cx, r - self.cy; f = 1 + k1 * (dx * dx + dy * dy)
        xu, yu = self.cx + dx * f, self.cy + dy * f
        q2 = Hm[2, 0] * xu + Hm[2, 1] * yu + Hm[2, 2]
        return (Hm[0, 0] * xu + Hm[0, 1] * yu + Hm[0, 2]) / q2, (Hm[1, 0] * xu + Hm[1, 1] * yu + Hm[1, 2]) / q2


def warp(img, cam, shape, p=None, box=None):
    H, W = shape
    box = box or (0, 0, W, H)
    xs, rs = np.meshgrid(np.arange(box[0], box[2]) + 0.5, np.arange(box[1], box[3]) + 0.5)
    u, v = cam.map(xs, rs, p)
    out = ndimage.map_coordinates(img, [v - 0.5, u - 0.5], order=1, mode="constant", cval=0.0)
    if box == (0, 0, W, H):
        return out
    full = np.zeros((H, W), np.float32); full[box[1]:box[3], box[0]:box[2]] = out
    return full


def fit_cam(alpha, k, cam):
    m = R.mask(k)
    ys, xs = np.nonzero(m)
    box = (max(0, xs.min() - 25), max(0, ys.min() - 25), min(m.shape[1], xs.max() + 25), min(m.shape[0], ys.max() + 25))
    ms = ndimage.gaussian_filter(m.astype(np.float32), 1.2)[box[1]:box[3], box[0]:box[2]][::2, ::2]

    def res(p):
        w = warp(alpha, cam, m.shape, p, box)[box[1]:box[3], box[0]:box[2]]
        return (ndimage.gaussian_filter(w, 1.2)[::2, ::2] - ms).ravel()
    for free in ([0, 1, 2, 3, 4, 5], list(range(9))):
        def r2(q):
            p = cam.p.copy(); p[free] = q
            return res(p)
        sol = optimize.least_squares(r2, cam.p[free], diff_step=1e-3, max_nfev=200)
        cam.p[free] = sol.x
    return cam


def iou_of(alpha, cam, k):
    w = warp(alpha, cam, R.mask(k).shape) > 0.5
    m = R.mask(k)
    return float((w & m).sum() / (w | m).sum()), w


def describe(cam):
    Hm, k1 = cam.mats()
    Rm = np.linalg.solve(cam.H0[:2, :2], Hm[:2, :2])
    sx, sy = np.linalg.norm(Rm[:, 0]), np.linalg.norm(Rm[:, 1])
    return dict(scale_x=round(1 / sx, 5), scale_y=round(1 / sy, 5), rot_deg=round(math.degrees(math.atan2(Rm[1, 0], Rm[0, 0])), 4),
                persp=[round(float(Hm[2, 0]), 8), round(float(Hm[2, 1]), 8)], k1=float(k1))


def photo_to_b(g, cam):
    pts = [(0.0, 0.0), (600.0, 0.0), (0.0, 80.0)]
    src = np.array([b2r(g, y, z) for y, z in pts], float); dst = np.array(pts, float)
    A = np.linalg.solve(np.c_[src, np.ones(3)], dst).T
    Hm, _ = cam.mats()
    return np.vstack([A, [0, 0, 1.0]]) @ Hm


def save_map(k, g, cam):
    Hm, k1 = cam.mats()
    json.dump(dict(photo_to_b=photo_to_b(g, cam).ravel().tolist(), cx=cam.cx, cy=cam.cy, k1=k1),
              open(os.path.join(HERE, "work", f"map_{k}.json"), "w"))


# front-to-back regions in the B frame (y0, y1, z0, z1) mm, from master px boxes
def _reg(x0, r0, x1, r1):
    cs = [R.bp(x, r) for x in (x0, x1) for r in (r0, r1)]
    return (min(c[0] for c in cs), max(c[0] for c in cs), min(c[1] for c in cs), max(c[1] for c in cs))


REGIONS = [("muzzle brake", _reg(1086, 266, 1184, 306)), ("front sight block", _reg(1050, 226, 1100, 310)),
           ("gas block", _reg(950, 242, 1012, 310)), ("barrel + rod", _reg(995, 272, 1064, 306)),
           ("handguards + gas tube", _reg(768, 236, 992, 322)), ("receiver + dust cover", _reg(452, 236, 706, 322)),
           ("rear sight", _reg(688, 224, 784, 278)), ("selector", _reg(498, 252, 636, 300)),
           ("trigger group", _reg(518, 298, 616, 364)), ("pistol grip", _reg(446, 310, 538, 434)),
           ("magazine", _reg(590, 306, 742, 505)), ("stock + hinge", _reg(186, 268, 472, 396))]


def region_box(g, cam, reg):
    Hi = np.linalg.inv(photo_to_b(g, cam))
    y0, y1, z0, z1 = reg
    cs = [Hi @ np.array([yy, zz, 1.0]) for yy in (y0, y1) for zz in (z0, z1)]
    xs = [c[0] / c[2] for c in cs]; rs = [c[1] / c[2] for c in cs]
    return int(min(xs)), int(min(rs)), int(math.ceil(max(xs))), int(math.ceil(max(rs)))


def region_ious(k, sil, cam, g):
    m = R.mask(k); out = {}
    for name, reg in REGIONS:
        b = region_box(g, cam, reg)
        a, s = m[b[1]:b[3], b[0]:b[2]], sil[b[1]:b[3], b[0]:b[2]]
        out[name] = round(float((a & s).sum() / max(1, (a | s).sum())), 4)
    return out


def diff_img(k, sil):
    m = R.mask(k)
    d = np.full(m.shape + (3,), 245.0, np.float32)
    d[m & sil] = (200, 200, 200); d[m & ~sil] = (220, 40, 40); d[sil & ~m] = (30, 190, 220)
    return d


def diff_sheets(k, sil, iou, aff, cam, g, tag, src, model=None):
    """full-gun difference + front-to-back region crops (photo | [model] | diff). Contains photo pixels -> review/ under
    the gitignored names compare_*.jpg / regions_*.jpg"""
    photo = R.image(k); m = R.mask(k); diff = diff_img(k, sil)
    fb = ImageFont.truetype(FONT, 26); fs = ImageFont.truetype(FONT, 20)
    ys, xs = np.nonzero(m | sil)
    crop = (max(0, xs.min() - 20), max(0, ys.min() - 20), min(m.shape[1], xs.max() + 20), min(m.shape[0], ys.max() + 20))
    op, om = (m & ~sil).sum() / m.sum(), (sil & ~m).sum() / m.sum()
    tiles = [("photo", photo)] + ([("model, same camera", model), ("50 % overlay", photo * 0.5 + model * 0.5)] if model is not None else []) + \
        [(f"difference IoU {iou:.4f}  red = photo only {op * 100:.2f} %  cyan = model only {om * 100:.2f} %", diff)]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    sh = Image.new("RGB", (cw, (ch + 40) * len(tiles) + 60), (22, 23, 26)); d = ImageDraw.Draw(sh)
    d.text((14, 12), f"AK-74M vs {'right' if k == 'R' else 'left'} profile ({tag}, {src}); camera {aff}", font=fs, fill=(235, 235, 235))
    for i, (lab, a) in enumerate(tiles):
        y = 60 + i * (ch + 40)
        d.text((14, y + 6), lab, font=fs, fill=(230, 230, 230))
        sh.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).crop(crop), (0, y + 38))
    sh.save(os.path.join(HERE, "review", f"compare_{tag}_{src}_{k}.jpg"), quality=90)
    rows = []
    ri = region_ious(k, sil, cam, g)
    for name, reg in REGIONS:
        box = region_box(g, cam, reg)
        kz = min(6.0, 560.0 / max(1, box[2] - box[0]), 300.0 / max(1, box[3] - box[1]))
        ims = []
        for a in [photo] + ([model] if model is not None else []) + [diff]:
            t = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).crop(box)
            ims.append(t.resize((int(t.width * kz), int(t.height * kz)), Image.LANCZOS))
        rows.append((f"{name}  (region IoU {ri[name]:.3f})", ims))
    Wd = max(sum(t.width for t in r[1]) + 40 for r in rows)
    Hd = sum(max(t.height for t in r[1]) + 44 for r in rows) + 60
    sh = Image.new("RGB", (Wd, Hd), (22, 23, 26)); d = ImageDraw.Draw(sh)
    d.text((14, 14), f"AK-74M front-to-back: photo | {'model | ' if model is not None else ''}silhouette diff "
           f"({'right' if k == 'R' else 'left'} profile, {tag}, {src})", font=fb, fill=(235, 235, 235))
    y = 60
    for name, ims in rows:
        d.text((14, y + 6), name, font=fs, fill=(230, 230, 230))
        x = 10
        for t in ims:
            sh.paste(t, (x, y + 36)); x += t.width + 10
        y += max(t.height for t in ims) + 44
    sh.save(os.path.join(HERE, "review", f"regions_{tag}_{src}_{k}.jpg"), quality=88)
