"""compare.py - side-profile gate against BOTH profile photos (right master 163254, left 163245).
For each photo: an orthographic camera locked to that photo's calibration (ak_ref), a 1-spp transparent Cycles render ->
model silhouette, then a 6-parameter affine (scale x/y, rotation, shear, shift) fitted model -> photo (the product photos
were shown at slightly different display scales: the left one is 1.7 % wider than the master), IoU vs the photo mask
(ak_ref.mask) at the photo's pixels. Beauty render through the same warp -> photo | model | overlay | difference sheet and
the front-to-back region crop sheet (12 regions, muzzle -> stock). Sheets contain photo pixels: they go to review/ under
gitignored names (compare_*.jpg, regions_*.jpg, cmp_*). Numbers -> review/iou_<tag>.json (committed).
usage: .venv/bin/python guns/ak74/compare.py [src=high|game] [npz=work/high.npz] [tag=p1] [samples=32] [photos=R,L]"""
import bpy, os, sys, math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, optimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import scene
import ak_ref as R
import ak_spec as P
import bl_high

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
SRC, TAG = ARGS.get("src", "high"), ARGS.get("tag", "p1")
SMP = int(ARGS.get("samples", 32) or 32)
NPZ = ARGS.get("npz", os.path.join(HERE, "work", "high.npz"))
PHOTOS = ARGS.get("photos", "R,L").split(",")
REV = os.path.join(HERE, "review"); os.makedirs(REV, exist_ok=True)
SS = 2                                                   # render supersampling vs the photo pixels
FONT = os.path.join(HERE, "../../fonts/WorkSans-Bold.ttf")

# front-to-back regions in the B frame (y0, y1, z0, z1) mm
REGIONS = [("muzzle brake", (R.bp(1086, 0)[0], R.bp(1184, 0)[0], R.bp(0, 306)[1], R.bp(0, 266)[1])),
           ("front sight block", (R.bp(1050, 0)[0], R.bp(1100, 0)[0], R.bp(0, 310)[1], R.bp(0, 226)[1])),
           ("gas block", (R.bp(950, 0)[0], R.bp(1012, 0)[0], R.bp(0, 310)[1], R.bp(0, 242)[1])),
           ("barrel + rod", (R.bp(995, 0)[0], R.bp(1064, 0)[0], R.bp(0, 306)[1], R.bp(0, 272)[1])),
           ("handguards + gas tube", (R.bp(768, 0)[0], R.bp(992, 0)[0], R.bp(0, 322)[1], R.bp(0, 236)[1])),
           ("receiver + dust cover", (R.bp(452, 0)[0], R.bp(706, 0)[0], R.bp(0, 322)[1], R.bp(0, 236)[1])),
           ("rear sight", (R.bp(688, 0)[0], R.bp(784, 0)[0], R.bp(0, 278)[1], R.bp(0, 224)[1])),
           ("selector", (R.bp(498, 0)[0], R.bp(636, 0)[0], R.bp(0, 300)[1], R.bp(0, 252)[1])),
           ("trigger group", (R.bp(518, 0)[0], R.bp(616, 0)[0], R.bp(0, 364)[1], R.bp(0, 298)[1])),
           ("pistol grip", (R.bp(446, 0)[0], R.bp(538, 0)[0], R.bp(0, 434)[1], R.bp(0, 310)[1])),
           ("magazine", (R.bp(590, 0)[0], R.bp(742, 0)[0], R.bp(0, 505)[1], R.bp(0, 306)[1])),
           ("stock + hinge", (R.bp(186, 0)[0], R.bp(472, 0)[0], R.bp(0, 396)[1], R.bp(0, 268)[1]))]


def b2photo(k, y, z):
    """B (y, z) mm -> photo px (calibration only, before the affine)"""
    if k == "R":
        return R.b2px(y, z)
    x = R.L_X_MUZ + (R.MUZZLE_Y - np.asarray(y, float)) / R.L_SH
    row = R.L_ROW_MUZ + R.L_SLOPE * (x - R.L_X_MUZ) - np.asarray(z, float) / R.L_SV
    return x, row


def camera(k):
    """ortho camera: square render pixels of S_MM (R) / L_SH (L) mm per photo px / SS; vertical handled by the affine"""
    H, W = R.mask(k).shape
    mmpx = R.S_MM if k == "R" else R.L_SH
    cd = bpy.data.cameras.new("cam_" + k); cd.type = "ORTHO"; cd.ortho_scale = W * mmpx * 0.001
    cd.clip_start, cd.clip_end = 0.01, 10
    cam = bpy.data.objects.new("cam_" + k, cd); bpy.context.scene.collection.objects.link(cam)
    # photo centre -> B -> Blender
    if k == "R":
        yc, zc = R.px2b(W / 2, H / 2)
    else:
        yc = R.MUZZLE_Y - (W / 2 - R.L_X_MUZ) * R.L_SH
        zc = (R.L_ROW_MUZ + R.L_SLOPE * (W / 2 - R.L_X_MUZ) - H / 2) * R.L_SH
    c = bl_high.b2bl(np.array([[0.0, yc, zc]]))[0]
    sgn = -1 if k == "R" else 1
    cam.location = (sgn * 2.0, c[1], c[2])
    from mathutils import Vector
    cam.rotation_euler = Vector((-sgn, 0, 0)).to_track_quat("-Z", "Y").to_euler()
    s = bpy.context.scene
    s.camera = cam; s.render.resolution_x, s.render.resolution_y = W * SS, H * SS; s.render.resolution_percentage = 100
    # B -> render px of this camera (for the initial affine)
    def b2r(y, z):
        if k == "R":
            u = (np.asarray(y) - yc) / mmpx * SS + W * SS / 2
        else:
            u = -(np.asarray(y) - yc) / mmpx * SS + W * SS / 2
        v = -(np.asarray(z) - zc) / mmpx * SS + H * SS / 2
        return u, v
    return cam, b2r


def render_alpha(path):
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"; s.cycles.samples = 4; s.cycles.use_denoising = False
    s.cycles.filter_width = 1.0
    s.render.film_transparent = True
    s.render.image_settings.color_mode = "RGBA"
    s.world = s.world or bpy.data.worlds.new("W")
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return np.asarray(Image.open(path)).astype(np.float32)[..., 3] / 255.0


def initial_affine(k, b2r):
    """photo px -> render px affine from the calibration: three B points through both maps -> 3x3 homography"""
    pts = [(0.0, 0.0), (600.0, 0.0), (0.0, 80.0)]
    src = np.array([b2photo(k, y, z) for y, z in pts], float)
    dst = np.array([b2r(y, z) for y, z in pts], float)
    A = np.linalg.solve(np.c_[src, np.ones(3)], dst).T                                    # 2x3
    return np.vstack([A, [0, 0, 1.0]])


class Cam:
    """photo px -> render px: radial term k1 about the photo centre, then a homography (perspective of a near-planar
    subject; the product shots are not square to the gun and were displayed at slightly different scales)"""
    def __init__(self, H0, shape):
        self.H0 = H0; self.h, self.w = shape; self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.p = np.zeros(9)

    def mats(self, p):
        Hd = np.array([[1 + p[0] * 1e-3, p[1] * 1e-3, p[2]], [p[3] * 1e-3, 1 + p[4] * 1e-3, p[5]],
                       [p[6] * 1e-6, p[7] * 1e-6, 1.0]])
        return self.H0 @ Hd, p[8] * 1e-8

    def map(self, x, r, p=None):
        p = self.p if p is None else p
        Hm, k1 = self.mats(p)
        dx, dy = x - self.cx, r - self.cy; f = 1 + k1 * (dx * dx + dy * dy)
        xu, yu = self.cx + dx * f, self.cy + dy * f
        q0 = Hm[0, 0] * xu + Hm[0, 1] * yu + Hm[0, 2]; q1 = Hm[1, 0] * xu + Hm[1, 1] * yu + Hm[1, 2]
        q2 = Hm[2, 0] * xu + Hm[2, 1] * yu + Hm[2, 2]
        return q0 / q2, q1 / q2


def warp(alpha, cam, shape, p=None, box=None):
    """sample the render image at the photo pixel centres (render px = cam.map(photo px))"""
    H, W = shape
    if box is None:
        box = (0, 0, W, H)
    xs, rs = np.meshgrid(np.arange(box[0], box[2]) + 0.5, np.arange(box[1], box[3]) + 0.5)
    u, v = cam.map(xs, rs, p)
    out = ndimage.map_coordinates(alpha, [v - 0.5, u - 0.5], order=1, mode="constant", cval=0.0)
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
        sol = optimize.least_squares(r2, cam.p[free], diff_step=1e-3, x_scale=1.0, max_nfev=200)
        cam.p[free] = sol.x
    return cam


def iou_of(alpha, cam, k):
    w = warp(alpha, cam, R.mask(k).shape) > 0.5
    m = R.mask(k)
    inter, uni = (w & m).sum(), (w | m).sum()
    return float(inter / uni), w


def describe(cam):
    Hm, k1 = cam.mats(cam.p)
    Rm = np.linalg.solve(cam.H0[:2, :2], Hm[:2, :2])
    sx, sy = np.linalg.norm(Rm[:, 0]), np.linalg.norm(Rm[:, 1])
    rot = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
    return dict(scale_x=round(1 / sx, 5), scale_y=round(1 / sy, 5), rot_deg=round(rot, 4),
                persp=[round(float(Hm[2, 0]), 8), round(float(Hm[2, 1]), 8)], k1=float(k1))


def load_objs():
    if SRC == "high":
        scene.reset()
        return list(bl_high.load(NPZ, skip=("rivets_rec", "rivets_front")).values())
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "ak74.blend"))
    objs = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("SM_AK74_") and "_LOD" not in o.name
            and not o.name.startswith("UCX_") and o.name != "SM_AK74"]
    for o in bpy.data.objects:
        if o.type == "MESH" and o not in objs:
            o.hide_render = True
    return objs


def beauty(objs, path):
    s = bpy.context.scene
    for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o)
    scene.studio(objs, backdrop=False)
    s.render.engine = "CYCLES"; s.cycles.samples = SMP; s.cycles.use_denoising = True; s.cycles.filter_width = 1.5
    s.render.film_transparent = True
    s.view_settings.view_transform = "AgX"; s.view_settings.look = "AgX - Base Contrast"; s.view_settings.exposure = 0.6
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", im.size, (255, 255, 255, 255)); white.alpha_composite(im)
    return np.asarray(white.convert("RGB")).astype(np.float32)


def sheets(k, photo, model, sil, iou, aff, p2b=None):
    m = R.mask(k)
    fb = ImageFont.truetype(FONT, 26); fs = ImageFont.truetype(FONT, 20)
    ys, xs = np.nonzero(m | sil)
    crop = (max(0, xs.min() - 20), max(0, ys.min() - 20), min(m.shape[1], xs.max() + 20), min(m.shape[0], ys.max() + 20))
    diff = np.full_like(photo, 245.0)
    diff[m & sil] = (200, 200, 200); diff[m & ~sil] = (220, 40, 40); diff[sil & ~m] = (30, 190, 220)
    op, om = (m & ~sil).sum() / m.sum(), (sil & ~m).sum() / m.sum()
    tiles = [("photo", photo), ("model, same camera", model), ("50 % overlay", photo * 0.5 + model * 0.5),
             (f"difference IoU {iou:.4f}  red = photo only {op * 100:.2f} %  cyan = model only {om * 100:.2f} %", diff)]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    sh = Image.new("RGB", (cw, (ch + 40) * 4 + 60), (22, 23, 26)); d = ImageDraw.Draw(sh)
    d.text((14, 12), f"AK-74M vs {'right' if k == 'R' else 'left'} profile ({TAG}, {SRC}); affine {aff}", font=fs, fill=(235, 235, 235))
    for i, (lab, a) in enumerate(tiles):
        y = 60 + i * (ch + 40)
        d.text((14, y + 6), lab, font=fs, fill=(230, 230, 230))
        sh.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).crop(crop), (0, y + 38))
    sh.save(os.path.join(REV, f"compare_{TAG}_{SRC}_{k}.jpg"), quality=90)
    # region crops, front to back, photo | model | diff, scaled up
    rows = []
    for name, (y0, y1, z0, z1) in REGIONS:
        Hi = np.linalg.inv(np.array(p2b).reshape(3, 3))
        cs = [Hi @ np.array([yy, zz, 1.0]) for yy in (y0, y1) for zz in (z0, z1)]
        xs_ = [c[0] / c[2] for c in cs]; rs_ = [c[1] / c[2] for c in cs]
        x0, x1 = min(xs_), max(xs_); r0, r1 = min(rs_), max(rs_)
        # the affine moved the model onto the photo; regions are in photo px so they apply to all three tiles
        box = (int(x0), int(r0), int(math.ceil(x1)), int(math.ceil(r1)))
        kz = min(6.0, 560.0 / max(1, box[2] - box[0]), 300.0 / max(1, box[3] - box[1]))
        ims = []
        for a in (photo, model, diff):
            t = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).crop(box)
            ims.append(t.resize((int(t.width * kz), int(t.height * kz)), Image.LANCZOS))
        sub_m = m[box[1]:box[3], box[0]:box[2]]; sub_s = sil[box[1]:box[3], box[0]:box[2]]
        riou = (sub_m & sub_s).sum() / max(1, (sub_m | sub_s).sum())
        rows.append((f"{name}  (region IoU {riou:.3f})", ims))
    Wd = max(sum(t.width for t in r[1]) + 40 for r in rows)
    Hd = sum(max(t.height for t in r[1]) + 44 for r in rows) + 60
    sh = Image.new("RGB", (Wd, Hd), (22, 23, 26)); d = ImageDraw.Draw(sh)
    d.text((14, 14), f"AK-74M front-to-back: photo | model | silhouette diff ({'right' if k == 'R' else 'left'} profile, {TAG}, {SRC})",
           font=fb, fill=(235, 235, 235))
    y = 60
    for name, ims in rows:
        d.text((14, y + 6), name, font=fs, fill=(230, 230, 230))
        x = 10
        for t in ims:
            sh.paste(t, (x, y + 36)); x += t.width + 10
        y += max(t.height for t in ims) + 44
    sh.save(os.path.join(REV, f"regions_{TAG}_{SRC}_{k}.jpg"), quality=88)


def b_from_render(k, b2r):
    """3x3 affine render px -> B (y, z), the inverse of the camera's linear b2r"""
    pts = [(0.0, 0.0), (600.0, 0.0), (0.0, 80.0)]
    src_ = np.array([b2r(y, z) for y, z in pts], float); dst = np.array(pts, float)
    A = np.linalg.solve(np.c_[src_, np.ones(3)], dst).T
    return np.vstack([A, [0, 0, 1.0]])


def main():
    objs = load_objs()
    res = {}
    for k in PHOTOS:
        for o in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(o)
        cam, b2r = camera(k)
        alpha = render_alpha(os.path.join(HERE, "work", f"cmp_{TAG}_{SRC}_{k}_alpha.png"))
        C = Cam(initial_affine(k, b2r), R.mask(k).shape)
        i0, _ = iou_of(alpha, C, k)
        fit_cam(alpha, k, C)
        iou, sil = iou_of(alpha, C, k)
        aff = describe(C)
        print(f"COMPARE {k} IoU calibration-only {i0:.4f} -> camera-fitted {iou:.4f}  {aff}", flush=True)
        Hm, k1 = C.mats(C.p)
        mp = dict(photo_to_b=(b_from_render(k, b2r) @ Hm).ravel().tolist(), cx=C.cx, cy=C.cy, k1=k1)
        json.dump(mp, open(os.path.join(HERE, "work", f"map_{k}.json"), "w"))
        img = beauty(objs, os.path.join(HERE, "work", f"cmp_{TAG}_{SRC}_{k}_beauty.png"))
        model = np.stack([warp(img[..., c], C, sil.shape) for c in range(3)], -1)
        model[~(warp(alpha, C, sil.shape) > 0.02)] = 255
        sheets(k, R.image(k), model, sil, iou, aff, mp["photo_to_b"])
        res[k] = dict(iou=iou, iou_calibration_only=i0, camera=aff)
    json.dump(res, open(os.path.join(REV, f"iou_{TAG}_{SRC}.json"), "w"), indent=1)
    ok = all(v["iou"] >= 0.98 for v in res.values())
    print("COMPARE VERDICT", "PASS" if ok else "FAIL", {k: round(v["iou"], 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
