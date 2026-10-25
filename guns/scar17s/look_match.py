"""look_match.py - match the hero34 reference's LIGHTING and FDE COLOURS on the high-poly clay, by measurement.
1. hero camera from the landmark solve (work/match_hero34.json), model -> photo alignment ~1 % rms
2. ID pass (flat emission per group) -> masks: A anodised upper, B polymer lower/stock/mag, G grips
3. light search: key direction (camera-relative azimuth/elevation) maximising the luminance correlation with the photo
   inside the eroded gun mask; world strength fitted to the photo's backdrop
4. colour fit: per-group albedo iterated until the render's mean colour inside each mask equals the photo's
out: work/look_hero34.json (lights, world, colours) + review/look_hero34.png (photo | match, model only)
usage: .venv/bin/python guns/scar17s/look_match.py"""
import bpy, os, sys, json, math, runpy
import numpy as np
from mathutils import Vector
from PIL import Image
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "../..")
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from lab import scene
import bl_high

WORK, REV = os.path.join(HERE, "work"), os.path.join(HERE, "review")
ARGS = dict(x.split("=") for x in sys.argv[1:] if "=" in x)
TMP = os.path.join(WORK, "look"); os.makedirs(TMP, exist_ok=True)
W, H = 450, 300
GROUPS = {"A": ["upper"], "B": ["lower", "stock", "magazine"], "G": ["grip", "vgrip"]}


def lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


scene.reset()
objs = bl_high.load()
objs["round"].hide_render = True
s = bpy.context.scene
m = json.load(open(os.path.join(WORK, "match_hero34.json")))
m["render"] = dict(out=os.path.join(TMP, "_cam.png"), samples=1, studio=False, transparent=True)
p = os.path.join(TMP, "_m.json"); json.dump(m, open(p, "w"))
sys.argv = ["blender", "--", p]
try:
    runpy.run_path(os.path.join(ROOT, "tools", "camera_match.py"), run_name="__main__")
except SystemExit:
    pass
cam = bpy.data.objects[m["camera"]]; s.camera = cam
w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]


def flat(name, col, emit=False):
    mt = bpy.data.materials.get(name) or bpy.data.materials.new(name); mt.use_nodes = True
    nt = mt.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if emit:
        e = nt.nodes.new("ShaderNodeEmission"); e.inputs[0].default_value = (*col, 1); e.inputs[1].default_value = 1.0
        nt.links.new(e.outputs[0], out.inputs[0])
    else:
        b = nt.nodes.new("ShaderNodeBsdfPrincipled"); b.inputs["Base Color"].default_value = (*col, 1)
        b.inputs["Roughness"].default_value = 0.5 if name.endswith("A") else 0.62
        nt.links.new(b.outputs[0], out.inputs[0])
    return mt


def set_group_mats(mats, others=None):
    for g, names in GROUPS.items():
        for n in names:
            me = objs[n].data
            for i in range(len(me.materials)):
                me.materials[i] = mats[g]
    if others is not None:
        for n, o in objs.items():
            if not any(n in v for v in GROUPS.values()):
                for i in range(len(o.data.materials)):
                    o.data.materials[i] = others


orig = {n: [mt for mt in o.data.materials] for n, o in objs.items()}


def restore_others():
    for n, o in objs.items():
        if not any(n in v for v in GROUPS.values()):
            for i, mt in enumerate(orig[n]):
                o.data.materials[i] = mt


def render(path, spp, standard=False):
    s.render.engine = "CYCLES"; s.cycles.samples = spp; s.cycles.use_denoising = spp > 4
    s.render.resolution_x, s.render.resolution_y = W, H; s.render.film_transparent = False
    s.render.use_persistent_data = True
    s.cycles.filter_width = 0.5 if standard else 1.5
    s.view_settings.view_transform = "Standard" if standard else "AgX"
    s.view_settings.look = "None" if standard else "AgX - Base Contrast"
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255


# ---- ID pass
ids = {"A": (1, 0, 0), "B": (0, 1, 0), "G": (0, 0, 1)}
set_group_mats({g: flat("ID_" + g, c, True) for g, c in ids.items()}, flat("ID_K", (0, 0, 0), True))
bg.inputs[0].default_value = (0, 0, 0, 1); bg.inputs[1].default_value = 0
idimg = render(os.path.join(TMP, "id.png"), 1, standard=True)
mask = {g: ndimage.binary_erosion(idimg[..., i] > 0.6, iterations=2) for i, g in enumerate("ABG")}
restore_others()
# all solid geometry silhouette (any object): render black-on-white holdout via alpha
s.render.film_transparent = True
s.render.resolution_x, s.render.resolution_y = W, H; s.cycles.samples = 1
s.render.filepath = os.path.join(TMP, "alpha.png"); s.render.image_settings.color_mode = "RGBA"
bpy.ops.render.render(write_still=True)
alpha = np.asarray(Image.open(os.path.join(TMP, "alpha.png")))[..., 3] > 128
s.render.image_settings.color_mode = "RGB"; s.render.film_transparent = False
solid = ndimage.binary_erosion(alpha, iterations=2)
backdrop = ndimage.binary_erosion(~alpha, iterations=6)

ref = np.asarray(Image.open(os.path.join(HERE, "refs_user", "hero34_right.png")).convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
lum = lambda a: a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
print("[look] mask px", {g: int(v.sum()) for g, v in mask.items()}, "solid", int(solid.sum()))

# ---- lighting search on neutral clay (same albedo everywhere: correlation sees only shape x light)
col = {"A": [0.30, 0.22, 0.13], "B": [0.30, 0.20, 0.10], "G": [0.42, 0.34, 0.25]}
mats = {g: flat("LK_" + g, c) for g, c in col.items()}
set_group_mats(mats)
bg.inputs[0].default_value = (0.47, 0.43, 0.39, 1); bg.inputs[1].default_value = 1.0
lo, hi = scene.bounds([o for n, o in objs.items() if n != "round"]); c = (lo + hi) / 2; r = (hi - lo).length
cm = cam.matrix_world.to_3x3(); right, up, back = cm.col[0], cm.col[1], cm.col[2]     # camera looks down -Z


def light(name, az, el, size, energy):
    a, e = math.radians(az), math.radians(el)
    d = (right * math.sin(a) * math.cos(e) + up * math.sin(e) + back * math.cos(a) * math.cos(e)).normalized()
    ld = bpy.data.lights.get(name) or bpy.data.lights.new(name, "AREA")
    o = bpy.data.objects.get(name) or bpy.data.objects.new(name, ld)
    if o.name not in s.collection.objects:
        s.collection.objects.link(o)
    ld.size = size * r; ld.energy = energy * r * r
    o.location = c + d * r * 1.6
    o.rotation_euler = (c - o.location).to_track_quat("-Z", "Y").to_euler()
    return o


def score(img):
    a, b = lum(img)[solid], lum(ref)[solid]
    return float(np.corrcoef(a, b)[0, 1])


best = None
KNOWN = ARGS.get("key")                                                              # key=az,el,fill,rim skips the search
for el in (() if KNOWN else (15, 40, 65, 85)):
    for az in (-135, -90, -45, 0, 45, 90, 135, 180):
        if el == 85 and az != 0:
            continue
        light("key", az, el, 1.4, 420); light("fill", 0, 5, 2.0, 90)
        sc = score(render(os.path.join(TMP, "l.png"), 12))
        print(f"[look] key az {az:5d} el {el:3d}  r {sc:.3f}")
        if best is None or sc > best[0]:
            best = (sc, az, el)
if KNOWN:
    kaz, kel, kf, kr = [float(v) for v in KNOWN.split(",")]
    best = (0.0, kaz, kel)
_, kaz, kel = best
# refine around the winner, then the fill and a rim
for daz in (() if KNOWN else (-20, 20)):
    for de in (-12, 12):
        az, el = kaz + daz, min(88, max(5, kel + de))
        light("key", az, el, 1.4, 420)
        sc = score(render(os.path.join(TMP, "l.png"), 12))
        if sc > best[0]:
            best = (sc, az, el)
_, kaz, kel = best
light("key", kaz, kel, 1.4, 420)
bestf = (best[0], kf if KNOWN else 90.0)
for fe in (() if KNOWN else (40, 160, 260)):
    light("fill", 0, 5, 2.0, fe)
    sc = score(render(os.path.join(TMP, "l.png"), 12))
    if sc > bestf[0]:
        bestf = (sc, fe)
light("fill", 0, 5, 2.0, bestf[1])
bestr = (bestf[0], kr if KNOWN else 0.0)
for raz in (() if KNOWN else (150, -150, 180)):
    light("rim", raz, 45, 0.8, 300)
    sc = score(render(os.path.join(TMP, "l.png"), 12))
    if sc > bestr[0]:
        bestr = (sc, raz)
if bestr[1] == 0.0:
    bpy.data.objects["rim"].hide_render = True
else:
    light("rim", bestr[1], 45, 0.8, 300)
print(f"[look] LIGHT key az {kaz} el {kel}, fill {bestf[1]}, rim {bestr[1]}  r {bestr[0]:.3f}")

# ---- world: camera rays see the backdrop, other rays a neutral ambient (so backdrop and gun lighting decouple)
nt = w.node_tree
lp = nt.nodes.new("ShaderNodeLightPath"); amb = nt.nodes.new("ShaderNodeBackground"); mix = nt.nodes.new("ShaderNodeMixShader")
amb.inputs[0].default_value = (0.5, 0.5, 0.5, 1); amb.inputs[1].default_value = 0.35
wout = [n for n in nt.nodes if n.type == "OUTPUT_WORLD"][0]
nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0]); nt.links.new(amb.outputs[0], mix.inputs[1])
nt.links.new(bg.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], wout.inputs[0])
# ---- backdrop strength, then per-group albedo fit
for it in range(6):
    img = render(os.path.join(TMP, "c.png"), 24)
    bs = lin(ref[backdrop].mean(0)).mean() / max(lin(img[backdrop].mean(0)).mean(), 1e-4)
    bg.inputs[1].default_value = float(np.clip(bg.inputs[1].default_value * bs, 0.05, 20))
    for g in "ABG":
        rr, mm = lin(ref[mask[g]].mean(0)), lin(img[mask[g]].mean(0))
        col[g] = list(np.clip(np.array(col[g]) * (rr / np.maximum(mm, 1e-4)), 0.003, 0.95))
        mats[g].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col[g], 1)
    err = {g: float(np.abs(img[mask[g]].mean(0) - ref[mask[g]].mean(0)).max()) for g in "ABG"}
    print(f"[look] colour iter {it}: max sRGB err {err}  world {bg.inputs[1].default_value:.2f}")
LUMW = np.array([0.2126, 0.7152, 0.0722])
k = 0.22 / float(np.dot(col["B"], LUMW))
k = min(k, 0.95 / max(max(v) for v in col.values()))
for g in col:
    col[g] = list(np.array(col[g]) * k)
    mats[g].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col[g], 1)
for n in ("key", "fill", "rim"):
    o = bpy.data.objects.get(n)
    if o:
        o.data.energy /= k
amb.inputs[1].default_value /= k
print(f"[look] albedo normalised x{k:.2f} (lights and ambient / {k:.2f})")
for it in range(2):                                                                  # bounce light changed: re-converge
    img = render(os.path.join(TMP, "c.png"), 24)
    for g in "ABG":
        rr, mm = lin(ref[mask[g]].mean(0)), lin(img[mask[g]].mean(0))
        col[g] = list(np.clip(np.array(col[g]) * (rr / np.maximum(mm, 1e-4)), 0.003, 0.95))
        mats[g].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col[g], 1)
img = render(os.path.join(TMP, "c.png"), 24)
err = {g: [round(float(v), 3) for v in (img[mask[g]].mean(0) - ref[mask[g]].mean(0))] for g in "ABG"}
print("[look] final sRGB mean error per group", err)

lights = {}
for n in ("key", "fill", "rim"):
    o = bpy.data.objects.get(n)
    if o and not o.hide_render:
        lights[n] = dict(loc=list((o.location - c) / r), rot=list(o.rotation_euler), size=o.data.size / r, energy=o.data.energy / (r * r))
out = dict(lights=lights, world=[0.47, 0.43, 0.39, bg.inputs[1].default_value], ambient=amb.inputs[1].default_value,
           key=[kaz, kel, bestf[1], bestr[1]],
           colors={g: [round(float(v), 4) for v in col[g]] for g in col}, corr=best[0], err=err)
json.dump(out, open(os.path.join(WORK, "look_hero34.json"), "w"), indent=1)
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((900, 600), Image.LANCZOS).save(os.path.join(REV, "look_hero34.png"))
print("LOOK_DONE", json.dumps(out["colors"]))
