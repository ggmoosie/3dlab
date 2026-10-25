"""showcase.py - final showcase shots recreating the user's two reference images with the textured game model.
  fieldstrip : top-down field strip on a from-scratch concrete floor, parts laid on their left sides in the photo's layout
               (bolt group with recoil spring + guide rod + op-rod, firing pin, rear sight off the upper)
  hero34     : the low front-right 3/4 product shot on a taupe ground, camera solved from landmarks (camera_match.py)
usage: .venv/bin/python guns/scar17s/showcase.py [shots=fieldstrip,hero34] [samples=128]"""
import bpy, bmesh, os, sys, math, json, subprocess
import numpy as np
from mathutils import Vector, Matrix, Euler
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../..")
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from lab import scene, tex, master, geo
import scar_spec as P
import ref as R
import bl_high

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
SHOTS = ARGS.get("shots", "fieldstrip,hero34").split(",")
SMP = int(ARGS.get("samples", 128) or 128)
REV, WORK = os.path.join(HERE, "review"), os.path.join(HERE, "work")
O = np.array(P.ORIGIN_B)
WK = 0.75


def b2w(x, y, z):
    """B mm -> Blender m (the game frame the meshes live in)"""
    return Vector(((x - O[0]) * 0.001, -(y - O[1]) * 0.001, (z - O[2]) * 0.001))


def load_game(variant="FDE"):
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "scar17s.blend"))
    objs = {}
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name.startswith("SM_SCAR17S_") and "_LOD" not in o.name:
            objs[o.name[11:]] = o
        elif o.type == "MESH":
            o.hide_render = True
    for n, o in objs.items():
        m = bpy.data.materials.get(f"MI_SCAR17S_{n}_{variant}") or bpy.data.materials.get(f"MI_SCAR17S_{n}")
        if m:
            o.data.materials.clear(); o.data.materials.append(m)
    objs["Round"].hide_render = True
    return objs


def steel(name="M_Show_Steel", c=(0.03, 0.03, 0.032), r=0.35, m=0.9):
    return master.simple(name, c, r, m, None)


# ------------------------------------------------------------------ extra field-strip parts (from the photo)
def tube_mesh(name, pts, r, segs=16):
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"
    sp = cu.splines.new("POLY"); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1)
    cu.bevel_depth = r; cu.bevel_resolution = 3; cu.use_fill_caps = True
    o = bpy.data.objects.new(name, cu); bpy.context.scene.collection.objects.link(o)
    bpy.context.view_layer.objects.active = o; o.select_set(True)
    bpy.ops.object.convert(target="MESH")
    return bpy.context.object


def bcg_extras():
    """recoil spring + guide rod + rear plate behind the carrier, op-rod forward along its top, firing pin (B frame)"""
    mat = steel()
    out = []
    y0 = -132.0
    rod = tube_mesh("guide_rod", [b2w(0, y0, 2), b2w(0, y0 - 168, 2)], 0.0028)
    pts = []
    for i in range(1400):
        t = i / 1399
        a = t * 2 * math.pi * 34
        pts.append(b2w(6.8 * math.cos(a), y0 - 4 - 150 * t, 2 + 6.8 * math.sin(a)))
    spring = tube_mesh("spring", pts, 0.00075, 8)
    plate = geo.rounded_box("rod_plate", 30, 6, 40, 3, 0, 0.8, segs=3)
    plate.data.transform(Matrix.Translation((0, 0, -0.020)))
    plate.location = b2w(0, y0 - 170, 2)
    oprod = geo.rounded_box("op_rod", 9, 150, 7, 1.2, 0, 0.5, segs=2)
    oprod.location = b2w(0, -14 + 75, 12.5)
    fpin = tube_mesh("firing_pin", [Vector((0, 0, 0)), Vector((0, -0.055, 0))], 0.0021)
    for o in (rod, spring, plate, oprod, fpin):
        o.data.materials.clear(); o.data.materials.append(mat)
        out.append(o)
    return out[:4], fpin


def split_rear_sight(upper):
    """field strip: the rear sight is off the upper; remove its faces from a copy of the Upper, load it as its own part"""
    ys = R.extent("rsight")
    lo, hi = b2w(0, ys[1], 0).y, b2w(0, ys[0], 0).y
    zr = (P.RAIL_TOP + 0.3 - O[2]) * 0.001
    bm = bmesh.new(); bm.from_mesh(upper.data)
    kill = [f for f in bm.faces if lo - 0.002 <= f.calc_center_median().y <= hi + 0.002 and f.calc_center_median().z > zr]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(upper.data); bm.free()
    rs = bl_high.load(names=["rsight"])["rsight"]
    d = rs.modifiers.new("dec", "DECIMATE"); d.ratio = 0.05
    rs.data.materials.clear(); rs.data.materials.append(steel("M_Show_Sight", (0.025, 0.025, 0.027), 0.5, 0.6))
    rs.scale.x = WK
    return rs


# ------------------------------------------------------------------ concrete floor (from scratch)
def concrete(size=2048):
    S = size
    a = tex.fbm(S, S, 3, 101); b = tex.fbm(S, S, 14, 102); c = tex.fbm(S, S, 60, 103)
    base = 0.30 + 0.05 * (a - 0.5) + 0.03 * (b - 0.5) + 0.02 * (c - 0.5)
    speck = tex.speckle(S, S, 0.004, 104)                                   # aggregate
    pores = tex.speckle(S, S, 0.0015, 105)
    stain = np.clip((tex.fbm(S, S, 5, 106) - 0.62) * 4, 0, 1) * 0.06
    scr = tex.scratches(S, S, 90, (0.004, 0.05), 1, 107) * 0.04 if hasattr(tex, "scratches") else 0
    L = base + 0.08 * speck - 0.12 * pores - stain - scr
    col = np.stack([L * 1.0, L * 0.99, L * 0.965], -1)
    H = (c - 0.5) * 0.4 + speck * 0.5 - pores * 0.8
    os.makedirs(os.path.join(WORK, "show"), exist_ok=True)
    p = lambda k: os.path.join(WORK, "show", f"T_Concrete_{k}.png")
    tex.save(tex.lin_to_srgb(np.clip(col, 0, 1)), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.7), 6.0), p("Normal"))
    rough = np.clip(0.82 + 0.1 * (b - 0.5) - 0.2 * stain, 0, 1)
    tex.save(rough, p("Rough"))
    m = bpy.data.materials.new("M_Concrete"); m.use_nodes = True
    N, Lk = m.node_tree.nodes, m.node_tree.links
    bs = N["Principled BSDF"]
    uv = N.new("ShaderNodeTexCoord"); mp = N.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1.4, 1.4, 1)
    Lk.new(uv.outputs["UV"], mp.inputs[0])
    for k, sock, cs in (("BaseColor", "Base Color", "sRGB"), ("Rough", "Roughness", "Non-Color")):
        t = N.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(p(k)); t.image.colorspace_settings.name = cs
        Lk.new(mp.outputs[0], t.inputs[0]); Lk.new(t.outputs[0], bs.inputs[sock])
    t = N.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(p("Normal")); t.image.colorspace_settings.name = "Non-Color"
    nm = N.new("ShaderNodeNormalMap"); Lk.new(mp.outputs[0], t.inputs[0]); Lk.new(t.outputs[0], nm.inputs["Color"])
    Lk.new(nm.outputs[0], bs.inputs["Normal"])
    return m


def world_color(col, strength=1.0):
    w = bpy.context.scene.world or bpy.data.worlds.new("W"); bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (*col, 1); bg.inputs[1].default_value = strength


def area(name, loc, rot, size, energy, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA"); ld.size = size; ld.energy = energy; ld.color = color
    o = bpy.data.objects.new(name, ld); bpy.context.scene.collection.objects.link(o)
    o.location = loc; o.rotation_euler = rot
    return o


def render(path, w, h, samples):
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"; s.cycles.samples = samples; s.cycles.use_denoising = True
    s.render.resolution_x, s.render.resolution_y = w, h
    s.view_settings.view_transform = "AgX"; s.view_settings.look = "AgX - Base Contrast"; s.view_settings.exposure = 0.0
    s.render.film_transparent = False
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ------------------------------------------------------------------ shot 1: field strip
# layout read off the reference (935 x 637 px): part -> centre px; scale 1.33 mm/px (the upper's length)
FS_SCALE = 1.33e-3
FS = {"rsight": (229, 184), "Upper": (488, 192), "Stock": (135, 318), "bcg": (456, 262), "fpin": (570, 282),
      "Lower": (398, 380), "Barrel": (728, 304), "Mag": (465, 510)}


def fieldstrip():
    objs = load_game("FDE")
    rs = split_rear_sight(objs["Upper"])
    extras, fpin = bcg_extras()
    # groups that move together on the floor
    groups = {"Upper": [objs["Upper"], objs["VGrip"]], "Stock": [objs["Stock"]], "Lower": [objs["Lower"], objs["Grip"]],
              "Barrel": [objs["Barrel"], objs["Muzzle"]], "Mag": [objs["Mag"]], "bcg": [objs["Bolt"]] + extras,
              "rsight": [rs], "fpin": [fpin]}
    root = {}
    for g, members in groups.items():
        e = bpy.data.objects.new("G_" + g, None); bpy.context.scene.collection.objects.link(e)
        for m in members:
            m.parent = e
        root[g] = e
    W, H = 935, 637
    for g, e in root.items():
        e.rotation_euler = (0, math.radians(90), 0)                                    # lying on the left side, right side up
        bpy.context.view_layer.update()
        pts = [m.matrix_world @ Vector(c) for m in groups[g] for c in m.bound_box]
        lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
        c = (lo + hi) / 2
        u, v = FS[g]
        target = Vector((-(v - H / 2) * FS_SCALE, -(u - W / 2) * FS_SCALE, 0))
        e.location += Vector((target.x - c.x, target.y - c.y, -lo.z + 0.0005))
    bpy.context.view_layer.update()
    floor = bpy.data.meshes.new("floor"); fo = bpy.data.objects.new("floor", floor); bpy.context.scene.collection.objects.link(fo)
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.9)
    uvl = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for l in f.loops:
            l[uvl].uv = (l.vert.co.x / 1.8 + 0.5, l.vert.co.y / 1.8 + 0.5)
    bm.to_mesh(floor); bm.free()
    floor.materials.append(concrete())
    world_color((0.55, 0.56, 0.58), 0.6)
    area("sky", (0.25, -0.2, 2.2), (0, 0, 0), 2.4, 900)                                  # soft overhead daylight
    area("window", (1.6, -1.4, 1.2), Euler((math.radians(55), 0, math.radians(50))), 1.5, 250)
    cd = bpy.data.cameras.new("fs"); cd.lens = 50
    cam = bpy.data.objects.new("fs", cd); bpy.context.scene.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 1.32); cam.rotation_euler = Vector((0, 0, -1)).to_track_quat("-Z", "X").to_euler()
    bpy.context.scene.camera = cam
    cd.sensor_width = 36; cd.sensor_fit = "HORIZONTAL"
    k = 0.935 * 1.33 / (2 * 1.32)                                                       # fit the photo's 1.24 m width
    cd.lens = 18 / k
    render(os.path.join(REV, "showcase_fieldstrip.png"), 1870, 1274, SMP)


# ------------------------------------------------------------------ shot 2: low front-right 3/4
def hero34():
    objs = load_game("FDE")
    match = json.load(open(os.path.join(WORK, "match_hero34.json")))
    cfgp = os.path.join(WORK, "match_hero34_run.json")
    match["render"] = dict(out=os.path.join(WORK, "hero34_cam.png"), engine="CYCLES", samples=1, studio=False, transparent=True)
    json.dump(match, open(cfgp, "w"))
    import runpy
    argv = sys.argv; sys.argv = ["blender", "--", cfgp]
    try:
        runpy.run_path(os.path.join(ROOT, "tools", "camera_match.py"), run_name="__main__")
    except SystemExit:
        pass
    sys.argv = argv
    cam = bpy.data.objects.get(match["camera"])
    bpy.context.scene.camera = cam
    world_color((0.47, 0.43, 0.39), 1.0)                                                 # the reference's taupe ground
    allo = list(o for n, o in objs.items() if n != "Round")
    lo, hi = scene.bounds(allo); c = (lo + hi) / 2; r = (hi - lo).length
    area("key", c + Vector((-0.6, -0.9, 1.1)) * r, Euler((math.radians(40), 0, math.radians(-30))), r * 1.2, 380 * r * r)
    area("fill", c + Vector((1.2, -0.3, 0.2)) * r, Euler((math.radians(85), 0, math.radians(75))), r * 1.5, 120 * r * r)
    area("rim", c + Vector((0.3, 1.2, 0.9)) * r, Euler((math.radians(-40), 0, math.radians(165))), r * 0.8, 260 * r * r)
    render(os.path.join(REV, "showcase_hero34.png"), 1800, 1200, SMP)


if __name__ == "__main__":
    for s in SHOTS:
        {"fieldstrip": fieldstrip, "hero34": hero34}[s]()
