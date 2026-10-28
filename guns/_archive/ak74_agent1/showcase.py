"""showcase.py - the four reference images recreated with the textured game model (Black variant): right profile, left
profile, the high 3/4 with the magazine out (laid where the photo has it is NOT attempted: the magazine is removed and
masked), and the folded-stock 3/4. Cameras: camfit (profiles, orthographic + the fitted photo warp) / match34 (3/4);
lighting: look_match per shot. Outputs model-only renders review/showcase_<shot>.png (committed) and photo | render sheets
review/compare_showcase_<shot>.jpg (contain photos: gitignored name).
usage: .venv/bin/python guns/ak74/showcase.py [shots=R,L,H,F] [samples=64]"""
import bpy, os, sys, json, math
import numpy as np
from mathutils import Vector, Matrix, Euler
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "../..")
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import ak_ref as R
import ak_spec as P
import camfit as CF
import match34 as M
import bl_high

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
SHOTS = ARGS.get("shots", "R,L,H,F").split(",")
SMP = min(64, int(ARGS.get("samples", 64) or 64))
REV, WORK = os.path.join(HERE, "review"), os.path.join(HERE, "work")
FONT = os.path.join(ROOT, "fonts/WorkSans-Bold.ttf")


def load_game():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "ak74.blend"))
    objs = {}
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name.startswith("SM_AK74_") and "_LOD" not in o.name and not o.name.startswith("UCX"):
            objs[o.name[8:]] = o
        elif o.type == "MESH":
            o.hide_render = True
    for n, o in objs.items():
        m = bpy.data.materials.get(f"MI_AK74_{n}_Black")
        if m:
            o.data.materials.clear(); o.data.materials.append(m)
    return objs


def camera(k, objs):
    s = bpy.context.scene
    for o in [o for o in bpy.data.objects if o.type in ("CAMERA", "LIGHT")]:
        bpy.data.objects.remove(o)
    if k in ("R", "L"):
        g = CF.grid(k)
        cd = bpy.data.cameras.new("cam"); cd.type = "ORTHO"; cd.ortho_scale = g["W"] * g["mmpx"] * 0.001
        cam = bpy.data.objects.new("cam", cd); s.collection.objects.link(cam)
        c = bl_high.b2bl(np.array([[0.0, g["yc"], g["zc"]]]))[0]
        sg = -1 if k == "R" else 1
        cam.location = (sg * 2.0, c[1], c[2]); cam.rotation_euler = Vector((-sg, 0, 0)).to_track_quat("-Z", "Y").to_euler()
        return cam, (g["W"] * CF.SS, g["H"] * CF.SS), g
    cfg = json.load(open(os.path.join(WORK, f"match_{k}.json"))); x = np.array(cfg["x_solved"])
    W, H = Image.open(R.PHOTO[k]).size
    Rm = M.rod(x[:3])
    cd = bpy.data.cameras.new("cam"); cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0; cd.lens = float(x[6])
    cam = bpy.data.objects.new("cam", cd); s.collection.objects.link(cam)
    cam.matrix_world = Matrix([list(Rm[i]) + [x[3 + i]] for i in range(3)] + [[0, 0, 0, 1]])
    if k == "F":
        p = Vector(bl_high.b2bl(np.array([[P.HINGE[0], P.HINGE[1], 0.0]]))[0])
        T = Matrix.Translation(p) @ Matrix.Rotation(math.radians(-180.0), 4, "Z") @ Matrix.Translation(-p)
        objs["Stock"].matrix_world = T @ objs["Stock"].matrix_world
    if k == "H":
        objs["Mag"].hide_render = True
    return cam, (W * 2, H * 2), None


def lights(k, objs):
    look = json.load(open(os.path.join(WORK, f"look_{k}.json")))
    s = bpy.context.scene
    from lab import scene as SC
    lo, hi = SC.bounds([o for o in objs.values() if not o.hide_render]); c = (lo + hi) / 2; r = (hi - lo).length
    for n, L in look["lights"].items():
        ld = bpy.data.lights.new(n, "AREA"); ld.size = L["size"] * r; ld.energy = L["energy"] * r * r
        o = bpy.data.objects.new(n, ld); s.collection.objects.link(o)
        o.location = c + Vector(L["loc"]) * r; o.rotation_euler = Euler(L["rot"])
    w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld"); lp = nt.nodes.new("ShaderNodeLightPath"); mix = nt.nodes.new("ShaderNodeMixShader")
    bg = nt.nodes.new("ShaderNodeBackground"); amb = nt.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = look["world"][3]
    amb.inputs[0].default_value = (1, 1, 1, 1); amb.inputs[1].default_value = look["ambient"]
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0]); nt.links.new(amb.outputs[0], mix.inputs[1])
    nt.links.new(bg.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], out.inputs[0])


def render(path, size):
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"; s.cycles.samples = SMP; s.cycles.use_denoising = True
    s.render.resolution_x, s.render.resolution_y = size; s.render.film_transparent = False
    s.view_settings.view_transform = "AgX"; s.view_settings.look = "AgX - Base Contrast"
    s.render.image_settings.color_mode = "RGB"; s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def to_photo_frame(k, img, g):
    """profiles: warp the orthographic render onto the photo's pixels through the fitted camera (map_<k>.json)"""
    if g is None:
        Hh, Ww = R.image(k).shape[:2]
        return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((Ww, Hh), Image.LANCZOS)).astype(np.float32)
    mp = json.load(open(os.path.join(WORK, f"map_{k}.json"))); Hm = np.array(mp["photo_to_b"]).reshape(3, 3)
    Hh, Ww = R.image(k).shape[:2]
    xs, rs = np.meshgrid(np.arange(Ww) + 0.5, np.arange(Hh) + 0.5)
    dx, dy = xs - mp["cx"], rs - mp["cy"]; f = 1 + mp["k1"] * (dx * dx + dy * dy)
    q = Hm @ np.stack([(mp["cx"] + dx * f).ravel(), (mp["cy"] + dy * f).ravel(), np.ones(xs.size)])
    u, v = CF.b2r(g, q[0] / q[2], q[1] / q[2])
    return np.stack([ndimage.map_coordinates(img[..., c], [v - 0.5, u - 0.5], order=1, cval=255.0).reshape(Hh, Ww)
                     for c in range(3)], -1)


def sheet(k, model):
    photo = R.image(k)
    names = dict(R="right profile (163254)", L="left profile (163245)", H="high 3/4, magazine out (163300)", F="stock folded 3/4 (163304)")
    fb = ImageFont.truetype(FONT, 26)
    a = Image.fromarray(photo.astype(np.uint8)); b = Image.fromarray(np.clip(model, 0, 255).astype(np.uint8))
    sh = Image.new("RGB", (a.width * 2 + 30, a.height + 70), (22, 23, 26)); d = ImageDraw.Draw(sh)
    d.text((14, 14), f"AK-74M showcase - {names[k]}: photo | game model (textured, matched camera + fitted lighting)", font=fb, fill=(235, 235, 235))
    sh.paste(a, (10, 60)); sh.paste(b, (a.width + 20, 60))
    sh.save(os.path.join(REV, f"compare_showcase_{k}.jpg"), quality=90)


def regions34(k, model):
    """3/4 shots: front-to-back crops (photo | model) of the regions visible in the shot, boxes = projected B-frame region
    boxes (x +-25 mm) through the solved camera"""
    cfg = json.load(open(os.path.join(WORK, f"match_{k}.json"))); x = np.array(cfg["x_solved"])
    photo = R.image(k); Hh, Ww = photo.shape[:2]
    rows = []
    for name, (y0, y1, z0, z1) in CF.REGIONS:
        pts = np.array([[xx, yy, zz] for xx in (-25, 25) for yy in (y0, y1) for zz in (z0, z1)], float)
        if k == "F" and name.startswith("stock"):
            pts = M.stock_fold_b(pts)
        uv, zc = M.project(x, bl_high.b2bl(pts), Ww, Hh)
        if (zc <= 0).any():
            continue
        b = (int(max(0, uv[:, 0].min())), int(max(0, uv[:, 1].min())), int(min(Ww, uv[:, 0].max())), int(min(Hh, uv[:, 1].max())))
        if b[2] - b[0] < 20 or b[3] - b[1] < 20:
            continue
        kz = min(4.0, 600.0 / (b[2] - b[0]), 360.0 / (b[3] - b[1]))
        ims = [Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).crop(b) for a in (photo, model)]
        ims = [t.resize((int(t.width * kz), int(t.height * kz)), Image.LANCZOS) for t in ims]
        rows.append((name, ims))
    fb = ImageFont.truetype(FONT, 24)
    Wd = max(sum(t.width for t in r[1]) + 30 for r in rows); Hd = sum(max(t.height for t in r[1]) + 40 for r in rows) + 56
    sh = Image.new("RGB", (Wd, Hd), (22, 23, 26)); d = ImageDraw.Draw(sh)
    d.text((14, 12), f"AK-74M {k} 3/4 front-to-back: photo | model (solved camera)", font=fb, fill=(235, 235, 235))
    y = 56
    for name, ims in rows:
        d.text((14, y + 4), name, font=fb, fill=(230, 230, 230)); xx = 10
        for t in ims:
            sh.paste(t, (xx, y + 34)); xx += t.width + 10
        y += max(t.height for t in ims) + 40
    sh.save(os.path.join(REV, f"regions_showcase_{k}.jpg"), quality=88)


def main():
    for k in SHOTS:
        objs = load_game()
        cam, size, g = camera(k, objs)
        bpy.context.scene.camera = cam
        lights(k, objs)
        img = render(os.path.join(WORK, f"show_{k}_raw.png"), size)
        model = to_photo_frame(k, img, g)
        Image.fromarray(np.clip(model, 0, 255).astype(np.uint8)).save(os.path.join(REV, f"showcase_{k}.png"))
        sheet(k, model)
        if k in ("H", "F"):
            regions34(k, model)
        print("SHOWCASE", k, flush=True)


if __name__ == "__main__":
    main()
