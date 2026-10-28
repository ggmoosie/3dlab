"""compare.py - S2 silhouette gate: render the high-poly through an orthographic camera locked to a profile photo
(the photo's own px <-> B-frame mapping from ak_ref) and score silhouette IoU; writes review/<tag>_<side>_sheet.jpg
(photo | render | diff: red = photo only, cyan = model only) and review/<tag>_iou.json.  Exit 3 if IoU < min_iou.
usage: .venv/bin/python guns/ak74m/compare.py [npz=work/high.npz] [side=L|R|LR] [tag=p0] [min_iou=0.95]"""
import bpy, os, sys, json, math
import numpy as np
from mathutils import Vector
from PIL import Image
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import ak_ref as R

A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NPZ = A.get("npz", os.path.join(HERE, "work", "high.npz"))
TAG, MIN = A.get("tag", "p0"), float(A.get("min_iou", 0.95))
REV = os.path.join(HERE, "review"); os.makedirs(REV, exist_ok=True)
WORK = os.path.join(HERE, "work")

# right-side photo: muzzle on the RIGHT; calibrated the same way (overall length, bore row at the muzzle)
SIDES = {"L": dict(photo=R.PHOTO, x_muz=R.X_MUZ, row=R.ROW_BORE, s=R.S_MM, sign=1),
         "R": dict(photo=R.PHOTO_R, x_muz=1179.0, row=float(A.get("row_r", 291.0)), s=943.0 / 987.0, sign=-1)}


def load():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    D = np.load(NPZ)
    mats = {}
    for k, col in ((0, (0.04, 0.04, 0.045)), (1, (0.03, 0.03, 0.03)), (2, (0.05, 0.05, 0.05))):
        m = bpy.data.materials.new(f"M{k}"); m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (*col, 1)
        b.inputs["Roughness"].default_value = 0.45 if k != 1 else 0.6; b.inputs["Metallic"].default_value = 0.6 if k != 1 else 0
        mats[k] = m
    for n in sorted({k.split("__")[0] for k in D.files}):
        V = D[n + "__V"].astype(np.float64); F = D[n + "__F"].astype(np.int64)[:, ::-1]
        V = np.c_[V[:, 0], -V[:, 1], V[:, 2]] * 0.001                                  # B mm -> Blender m (y mirrored)
        me = bpy.data.meshes.new(n); me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
        me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
        me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3)); me.polygons.foreach_set("loop_total", np.full(len(F), 3))
        me.update(calc_edges=True); me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
        k = int(np.round(D[n + "__a_mat"][0])) if (n + "__a_mat") in D.files else 0
        me.materials.append(mats.get(k, mats[0]))
        o = bpy.data.objects.new(n, me); bpy.context.scene.collection.objects.link(o)


def camera(side):
    c = SIDES[side]
    fg, _ = R.masks(c["photo"])
    H, W = fg.shape
    cx, cy = (W - 1) / 2, (H - 1) / 2
    y = c["sign"] * (c["x_muz"] - cx) * c["s"]; z = (c["row"] - cy) * c["s"]                  # B at the image centre
    cd = bpy.data.cameras.new("cam"); cd.type = "ORTHO"; cd.ortho_scale = W * c["s"] * 0.001
    cam = bpy.data.objects.new("cam", cd); bpy.context.scene.collection.objects.link(cam)
    sx = 1 if side == "L" else -1
    cam.location = (sx * 1.5, -y * 0.001, z * 0.001)
    cam.rotation_euler = (Vector((-sx, 0, 0))).to_track_quat("-Z", "Y").to_euler()
    s = bpy.context.scene; s.camera = cam
    s.render.resolution_x, s.render.resolution_y = W, H
    return fg


def _H(shape, roll, sc, tx, ty, px=0.0, py=0.0):
    """camera-frame px -> photo px: similarity about the image centre + 2 perspective terms (row/col homogeneous)"""
    H, W = shape
    a = math.radians(roll); c, s = math.cos(a) / sc, math.sin(a) / sc
    cy, cx = (H - 1) / 2, (W - 1) / 2
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)                     # to centred (x, y)
    A = np.array([[c, -s, -tx / sc], [s, c, -ty / sc], [px * 1e-4, py * 1e-4, 1]], float)
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
    return T2 @ A @ T1


def warp(m, roll, sc, tx, ty, px=0.0, py=0.0):
    """photo image (or mask) resampled into the camera frame"""
    from skimage.transform import warp as skwarp, ProjectiveTransform
    Hm = _H(m.shape[:2], roll, sc, tx, ty, px, py)
    out = skwarp(m.astype(np.float32), ProjectiveTransform(Hm), order=1, preserve_range=True)
    return out > 0.5 if m.dtype == bool else out


def fit_camera(fg, al):
    """near-orthographic product shot: roll / scale / shift (+ 2 perspective terms) maximising silhouette IoU"""
    from scipy.optimize import minimize
    sm = lambda m: m[::2, ::2]
    fs, als = sm(fg), sm(al)

    def f(p):
        w = warp(fs, p[0], p[1], p[2] / 2, p[3] / 2, *(np.array(p[4:]) * 2 if len(p) > 4 else []))
        return -float((w & als).sum() / max((w | als).sum(), 1))
    best = min((f((r, sc, tx, ty)), (r, sc, tx, ty)) for r in np.arange(-1.5, 1.51, 0.5) for sc in (0.99, 1.0, 1.01)
               for tx in range(-14, 15, 4) for ty in range(-4, 5, 4))
    x = list(best[1])
    for npar in (4, 6):
        x0 = list(x) + [0.0] * (npar - len(x))
        simp = [x0] + [list(np.array(x0) + d) for d in np.eye(npar) * ([0.3, 0.01, 2, 2, 0.5, 0.5][:npar])]
        x = list(minimize(f, x0, method="Nelder-Mead", options=dict(xatol=0.005, fatol=1e-5, maxiter=600, initial_simplex=simp)).x)
    p = [round(float(v), 4) for v in x]
    cam = dict(roll_deg=p[0], scale=p[1], tx_px=p[2], ty_px=p[3], persp_x=p[4], persp_y=p[5])
    return warp(fg, *p), cam


def render(path, alpha):
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.samples = 1 if alpha else 24; s.cycles.use_denoising = not alpha
    s.cycles.filter_width = 0.6
    s.render.film_transparent = alpha; s.render.image_settings.color_mode = "RGBA" if alpha else "RGB"
    w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1); w.node_tree.nodes["Background"].inputs[1].default_value = 1.2
    s.view_settings.view_transform = "Standard"
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    load()
    res = {}
    for side in A.get("side", "LR"):
        for o in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(o)
        fg = camera(side)
        pa, pb = os.path.join(WORK, f"cmp_{TAG}_{side}_a.png"), os.path.join(WORK, f"cmp_{TAG}_{side}_b.png")
        render(pa, True); render(pb, False)
        al = np.asarray(Image.open(pa))[..., 3] > 127
        fg, cam = fit_camera(fg, al)                                                   # similarity camera (roll/scale/shift)
        iou = float((al & fg).sum() / max((al | fg).sum(), 1))
        res[side] = dict(iou=round(iou, 4), camera=cam, photo_only=round(float((fg & ~al).sum() / fg.sum()), 4),
                         model_only=round(float((al & ~fg).sum() / fg.sum()), 4))
        print(f"COMPARE {side} IoU {iou:.4f}  photo_only {res[side]['photo_only']}  model_only {res[side]['model_only']}")
        ph = np.asarray(Image.open(SIDES[side]["photo"]).convert("RGB"))
        photo = warp(ph.astype(np.float32), *cam.values()).clip(0, 255).astype(np.uint8)
        rend = np.asarray(Image.open(pb).convert("RGB"))
        diff = np.full(photo.shape, 245, np.uint8); diff[al & fg] = (150, 150, 150)
        diff[fg & ~al] = (230, 40, 40); diff[al & ~fg] = (40, 200, 230)
        ys, xs = np.nonzero(fg | al); box = (max(xs.min() - 20, 0), max(ys.min() - 20, 0), xs.max() + 20, ys.max() + 20)
        tiles = [Image.fromarray(t).crop(box) for t in (photo, rend, diff)]
        W, H = tiles[0].size
        sh = Image.new("RGB", (W, H * 3), (30, 30, 30))
        for i, t in enumerate(tiles):
            sh.paste(t, (0, i * H))
        sh.save(os.path.join(WORK, f"{TAG}_{side}_sheet.jpg"), quality=90)       # contains the photo: work/ (gitignored)
        Image.fromarray(diff).crop(box).save(os.path.join(REV, f"{TAG}_{side}_diff.png"))
    json.dump(res, open(os.path.join(REV, f"{TAG}_iou.json"), "w"), indent=1)
    worst = min(r["iou"] for r in res.values())
    print("COMPARE VERDICT", "PASS" if worst >= MIN else f"FAIL: {worst:.3f} < {MIN}")
    sys.exit(0 if worst >= MIN else 3)


if __name__ == "__main__":
    main()
