"""look_match.py - fit LIGHTING (per reference shot) and the STEEL / POLYMER finish (shared) to the photos, by measurement.
For each shot (R right profile, L left profile, H high 3/4, F folded 3/4):
 1. the matched camera (camfit for the profiles: orthographic + the fitted warp; match34 for the 3/4 shots)
 2. ID pass on the high-poly clay (flat emission: A = steel ids 0/2/3, B = polymer ids 1/6) -> masks
 3. key-light search (camera-relative azimuth / elevation) maximising the luminance correlation with the photo inside the
    eroded gun mask, then fill strength and an optional rim; the world (backdrop) strength matched to the photo's paper
 4. (R only) per-group albedo + roughness iterated until the render's mean colour inside each mask equals the photo's
out: work/look_<shot>.json (lights relative to the gun, world), work/look_fit.json (variant colours for bl_final),
     review/look_<shot>.png (model only)
usage: .venv/bin/python guns/ak74/look_match.py [shots=R,H,F,L] [npz=work/high_look.npz]"""
import bpy, os, sys, json, math
import numpy as np
from mathutils import Vector, Matrix
from PIL import Image
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "../..")
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from lab import scene
import bl_high
import ak_ref as R
import camfit as CF
import match34 as M

ARGS = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
WORK, REV = os.path.join(HERE, "work"), os.path.join(HERE, "review")
TMP = os.path.join(WORK, "look"); os.makedirs(TMP, exist_ok=True)
NPZ = ARGS.get("npz", os.path.join(WORK, "high_look.npz"))
GA, GB = (0, 2, 3), (1, 6)
LUMW = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def setup_camera(k, objs):
    """camera of shot k; returns (camera, photo image resized to the render, render size, warp fn or None)"""
    s = bpy.context.scene
    for o in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(o)
    if k in ("R", "L"):
        g = CF.grid(k)
        cd = bpy.data.cameras.new("cam"); cd.type = "ORTHO"; cd.ortho_scale = g["W"] * g["mmpx"] * 0.001
        cam = bpy.data.objects.new("cam", cd); s.collection.objects.link(cam)
        c = bl_high.b2bl(np.array([[0.0, g["yc"], g["zc"]]]))[0]
        sg = -1 if k == "R" else 1
        cam.location = (sg * 2.0, c[1], c[2])
        cam.rotation_euler = Vector((-sg, 0, 0)).to_track_quat("-Z", "Y").to_euler()
        W, H = g["W"], g["H"]
        mp = json.load(open(os.path.join(WORK, f"map_{k}.json")))
        return cam, (W, H), dict(g=g, map=mp)
    cfg = json.load(open(os.path.join(WORK, f"match_{k}.json")))
    x = np.array(cfg["x_solved"])
    W, H = Image.open(R.PHOTO[k]).size
    Rm = M.rod(x[:3])
    cd = bpy.data.cameras.new("cam"); cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0; cd.lens = float(x[6])
    cam = bpy.data.objects.new("cam", cd); s.collection.objects.link(cam)
    cam.matrix_world = Matrix([list(Rm[i]) + [x[3 + i]] for i in range(3)] + [[0, 0, 0, 1]])
    if k == "F":
        M_fold(objs)
    if k == "H":
        objs["magazine"].hide_render = True
    return cam, (W, H), None


def M_fold(objs):
    import ak_spec as P
    p = bl_high.b2bl(np.array([[P.HINGE[0], P.HINGE[1], 0.0]]))[0]
    T = Matrix.Translation(Vector(p)) @ Matrix.Rotation(math.radians(-180.0), 4, "Z") @ Matrix.Translation(-Vector(p))
    for n in ("stock", "buttplate", "slings"):
        if n in objs:
            objs[n].matrix_world = T @ objs[n].matrix_world


def photo_on_render(k, size, info, scale):
    """the photo resampled onto the render grid (profiles: through the fitted warp; 3/4: same pixels)"""
    W, H = size
    img = R.image(k)
    if info is None:
        return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((int(W * scale), int(H * scale)), Image.LANCZOS)).astype(np.float32) / 255
    return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((int(W * scale), int(H * scale)), Image.LANCZOS)).astype(np.float32) / 255


def flat(name, col, emit=False, rough=0.5):
    mt = bpy.data.materials.get(name) or bpy.data.materials.new(name); mt.use_nodes = True
    nt = mt.node_tree; nt.nodes.clear(); out = nt.nodes.new("ShaderNodeOutputMaterial")
    if emit:
        e = nt.nodes.new("ShaderNodeEmission"); e.inputs[0].default_value = (*col, 1); e.inputs[1].default_value = 1.0
        nt.links.new(e.outputs[0], out.inputs[0])
    else:
        b = nt.nodes.new("ShaderNodeBsdfPrincipled"); b.inputs["Base Color"].default_value = (*col, 1)
        b.inputs["Roughness"].default_value = rough
        nt.links.new(b.outputs[0], out.inputs[0])
    return mt


def set_mats(objs, ma, mb):
    for o in objs.values():
        me = o.data
        for i, sl in enumerate(me.materials):
            kid = int(sl.name.split("_")[1]) if sl.name.startswith("PV_") else int(o.get("pv%d" % i, 0))
            o["pv%d" % i] = kid
            me.materials[i] = ma if kid in GA else mb


def render(path, size, scale, spp, standard=False):
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"; s.cycles.samples = spp; s.cycles.use_denoising = spp > 4
    s.render.resolution_x, s.render.resolution_y = int(size[0] * scale), int(size[1] * scale)
    s.render.film_transparent = False; s.cycles.filter_width = 0.5 if standard else 1.5
    s.view_settings.view_transform = "Standard" if standard else "AgX"
    s.view_settings.look = "None" if standard else "AgX - Base Contrast"
    s.render.image_settings.color_mode = "RGB"; s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255


def fit_shot(k, objs, fit_colour):
    s = bpy.context.scene
    cam, size, info = setup_camera(k, objs)
    s.camera = cam
    scale = min(1.0, 520.0 / size[0])
    w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    wout = nt.nodes.new("ShaderNodeOutputWorld"); lp = nt.nodes.new("ShaderNodeLightPath"); mix = nt.nodes.new("ShaderNodeMixShader")
    bg = nt.nodes.new("ShaderNodeBackground"); amb = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0]); nt.links.new(amb.outputs[0], mix.inputs[1])
    nt.links.new(bg.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], wout.inputs[0])
    # ID pass
    set_mats(objs, flat("ID_A", (1, 0, 0), True), flat("ID_B", (0, 1, 0), True))
    bg.inputs[0].default_value = (0, 0, 0, 1); amb.inputs[1].default_value = 0.0
    idimg = render(os.path.join(TMP, f"{k}_id.png"), size, scale, 1, standard=True)
    ref = photo_on_render(k, size, info, scale)
    if info is not None:                                    # profiles: the ortho render needs the fitted warp onto the photo
        g = info["g"]
        C = CF.Cam(np.eye(3), R.mask(k).shape)
        Hm = np.array(info["map"]["photo_to_b"]).reshape(3, 3)
        # photo px -> B -> render px (scaled)
        def warp_img(a):
            Hh, Ww = a.shape[:2]
            xs, rs = np.meshgrid((np.arange(Ww) + 0.5) / scale, (np.arange(Hh) + 0.5) / scale)
            q = Hm @ np.stack([xs.ravel(), rs.ravel(), np.ones(xs.size)])
            u, v = CF.b2r(g, q[0] / q[2], q[1] / q[2])
            u = u / CF.SS * scale; v = v / CF.SS * scale
            ch = [ndimage.map_coordinates(a[..., c], [v - 0.5, u - 0.5], order=1).reshape(Hh, Ww) for c in range(a.shape[2])]
            return np.stack(ch, -1)
    else:
        warp_img = lambda a: a
    idw = warp_img(idimg)
    mask = {"A": ndimage.binary_erosion(idw[..., 0] > 0.6, iterations=2), "B": ndimage.binary_erosion(idw[..., 1] > 0.6, iterations=2)}
    solid = mask["A"] | mask["B"]
    backdrop = ndimage.binary_erosion(~ndimage.binary_dilation(idw.sum(-1) > 0.1, iterations=3), iterations=4)
    lum = lambda a: a @ LUMW
    col = {"A": [0.03, 0.03, 0.032], "B": [0.04, 0.04, 0.041]}
    rough = {"A": 0.45, "B": 0.62}
    look = json.load(open(os.path.join(WORK, "look_fit.json"))) if (os.path.exists(os.path.join(WORK, "look_fit.json")) and not fit_colour) else None
    if look:
        col, rough = look["colors"], look["rough"]
    mats = {g_: flat("LK_" + g_, col[g_], rough=rough[g_]) for g_ in "AB"}
    set_mats(objs, mats["A"], mats["B"])
    bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = 1.0
    amb.inputs[0].default_value = (1, 1, 1, 1); amb.inputs[1].default_value = 0.4
    lo, hi = scene.bounds([o for o in objs.values() if not o.hide_render]); c = (lo + hi) / 2; r = (hi - lo).length
    cm = cam.matrix_world.to_3x3(); right, up, back = cm.col[0], cm.col[1], cm.col[2]

    def light(name, az, el, size_, energy):
        a, e = math.radians(az), math.radians(el)
        d = (right * math.sin(a) * math.cos(e) + up * math.sin(e) + back * math.cos(a) * math.cos(e)).normalized()
        ld = bpy.data.lights.get(name) or bpy.data.lights.new(name, "AREA")
        o = bpy.data.objects.get(name) or bpy.data.objects.new(name, ld)
        if o.name not in s.collection.objects:
            s.collection.objects.link(o)
        ld.size = size_ * r; ld.energy = energy * r * r; o.hide_render = False
        o.location = c + d * r * 1.6; o.rotation_euler = (c - o.location).to_track_quat("-Z", "Y").to_euler()
        return o

    def score(img):
        a, b = lum(warp_img(img))[solid], lum(ref)[solid]
        return float(np.corrcoef(a, b)[0, 1])
    best = None
    for el in (20, 45, 70, 88):
        for az in (-120, -60, 0, 60, 120, 180):
            if el == 88 and az != 0:
                continue
            light("key", az, el, 1.6, 300); light("fill", 0, 0, 2.5, 60)
            sc = score(render(os.path.join(TMP, "l.png"), size, scale, 8))
            if best is None or sc > best[0]:
                best = (sc, az, el)
    _, kaz, kel = best
    for daz in (-25, 25):
        for de in (-12, 12):
            az, el = kaz + daz, min(88, max(5, kel + de))
            light("key", az, el, 1.6, 300)
            sc = score(render(os.path.join(TMP, "l.png"), size, scale, 8))
            if sc > best[0]:
                best = (sc, az, el)
    _, kaz, kel = best
    light("key", kaz, kel, 1.6, 300)
    bf = (best[0], 60.0)
    for fe in (20, 140, 260):
        light("fill", 0, 0, 2.5, fe)
        sc = score(render(os.path.join(TMP, "l.png"), size, scale, 8))
        if sc > bf[0]:
            bf = (sc, fe)
    light("fill", 0, 0, 2.5, bf[1])
    # exposure: scale all lights + ambient so the gun's mean luminance matches (colours fixed unless fitting)
    for it in range(5):
        img = warp_img(render(os.path.join(TMP, "c.png"), size, scale, 16))
        bs = lin(ref[backdrop].mean(0)).mean() / max(lin(img[backdrop].mean(0)).mean(), 1e-4)
        bg.inputs[1].default_value = float(np.clip(bg.inputs[1].default_value * bs, 0.05, 20))
        if fit_colour:
            for g_ in "AB":
                rr, mm = lin(ref[mask[g_]].mean(0)), lin(img[mask[g_]].mean(0))
                col[g_] = list(np.clip(np.array(col[g_]) * (rr / np.maximum(mm, 1e-4)), 0.003, 0.6))
                mats[g_].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col[g_], 1)
        else:
            gl = lin(ref[solid].mean(0)).mean() / max(lin(img[solid].mean(0)).mean(), 1e-4)
            for n in ("key", "fill"):
                bpy.data.objects[n].data.energy *= float(np.clip(gl, 0.5, 2.0))
            amb.inputs[1].default_value *= float(np.clip(gl, 0.5, 2.0))
    img = warp_img(render(os.path.join(TMP, "c.png"), size, scale, 24))
    err = {g_: [round(float(v), 3) for v in (img[mask[g_]].mean(0) - ref[mask[g_]].mean(0))] for g_ in "AB"}
    lights = {}
    for n in ("key", "fill"):
        o = bpy.data.objects[n]
        lights[n] = dict(loc=list((o.location - c) / r), rot=list(o.rotation_euler), size=o.data.size / r, energy=o.data.energy / (r * r))
    out = dict(shot=k, lights=lights, world=[1, 1, 1, bg.inputs[1].default_value], ambient=amb.inputs[1].default_value,
               key=[kaz, kel, bf[1]], corr=best[0], err=err, colors=col, rough=rough)
    json.dump(out, open(os.path.join(WORK, f"look_{k}.json"), "w"), indent=1)
    if fit_colour:
        json.dump(dict(colors=col, rough=rough, variant=dict(ColorA=[float(v) for v in col["A"]], ColorB=[float(v) for v in col["B"]],
                                                              RoughA=rough["A"], RoughB=rough["B"])),
                  open(os.path.join(WORK, "look_fit.json"), "w"), indent=1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(REV, f"look_{k}.png"))
    print(f"LOOK {k} key az {kaz} el {kel} fill {bf[1]} corr {best[0]:.3f} err {err}", flush=True)


def main():
    shots = ARGS.get("shots", "R,H,F,L").split(",")
    for k in shots:
        scene.reset()
        objs = bl_high.load(NPZ, skip=("rivets_rec", "rivets_front"))
        fit_shot(k, objs, fit_colour=(k == "R"))
    print("LOOK_DONE")


if __name__ == "__main__":
    main()
