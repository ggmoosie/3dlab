"""GPPouch: 6x6 MOLLE utility pouch (clamshell zipper) + SM_GPPouch_Flap. Builds everything from nothing:
high-poly skin (pillowed panels, piped seams, zipper groove, folds) + webbing / zipper / sliders / cord pulls / snaps,
game meshes (LOD0..2, UCX), baked + synthesised textures (cordura weave, webbing ribs, stitches, bar-tacks),
universal-mask variants, FBX, manifest.  Renders: pouch_render.py.

    .venv/bin/python props/containers/molle_pouch/molle_pouch.py [--fast] [--reuse-bakes]"""
import os, sys, math, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix
from lab import scene, master, bake, export, tex
import pouch_geo as G
import pouch_tex as PT

FAST = "--fast" in sys.argv
REUSE = "--reuse-bakes" in sys.argv
S = 1024 if FAST else 2048
TEXD, FBXD, REND = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders"))
WORK = os.path.abspath(os.path.join(HERE, "../../../work/molle_pouch"))
MM = G.MM
MAT = "MI_GPPouch"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def mat(name, col=None):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if col is not None:
        m.diffuse_color = (*col, 1)
    return m


def set_face_attr(o, name, vals):
    a = o.data.attributes.get(name) or o.data.attributes.new(name, "FLOAT", "FACE")
    a.data.foreach_set("value", np.broadcast_to(np.asarray(vals, np.float32), (len(o.data.polygons),)).copy())


# ------------------------------------------------------------------------------------------------ high
def build_high(col):
    out = []

    def add(o, idn):
        if isinstance(o, (list, tuple)):
            for x in o:
                add(x, idn)
            return
        o.data.materials.clear()
        o.data.materials.append(mat("ID_" + idn, G.IDS[idn]))
        scene.link(o, col)
        out.append(o)
    add(G.body_high(), "fabric")
    for k in range(3):
        add(G.pals_front(k, True), "webh")
    add(G.back_grid(True), "webh")
    for i in range(2):
        add(G.strap(i, True), "webv"); add(G.snap_tab(i, True), "webv"); add(G.side_loop(i, True), "webv")
        add(G.stud(i, True), "stud"); add(G.stiffener(i, True), "stiff")
        add(G.slider(i, True), "slider"); add(G.cord(i, True), "cord"); add(G.cap(i, True), "cap")
    add(G.zipper_tape(), "tape"); add(G.zipper_teeth(), "teeth")
    return out


# ------------------------------------------------------------------------------------------------ low
def hw_uv(o):
    """smart-project a hardware part, then scale its UVs to mm (1 UV unit = 1 mm)"""
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True)
    if not o.data.uv_layers:
        o.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(50), island_margin=0.02, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    me = o.data
    uv = np.zeros(len(me.loops) * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    a3 = sum(p.area for p in me.polygons) / MM ** 2
    a2 = 0.0
    for p in me.polygons:
        li = list(p.loop_indices)
        for k in range(1, len(li) - 1):
            (x0, y0), (x1, y1), (x2, y2) = uv[li[0]], uv[li[k]], uv[li[k + 1]]
            a2 += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
    uv *= math.sqrt(a3 / max(a2, 1e-12))
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())


def prep(o, cls):
    """common low-part setup: one material slot, low class attribute (1 shell, 2 liner, 3 rim, 4 part, 5 hidden)"""
    o.data.materials.clear(); o.data.materials.append(mat(MAT))
    if "lowcls" not in o.data.attributes:
        set_face_attr(o, "lowcls", cls)
    return o


def sweep_classes(o):
    """sweeps: faces whose UV sits in the hidden chain (u >= 2900) are bottoms / buried end caps"""
    me = o.data
    uv = np.zeros(len(me.loops) * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    cls = np.array([5.0 if uv[p.loop_start, 0] >= 2900 else 4.0 for p in me.polygons], np.float32)
    set_face_attr(o, "lowcls", cls)


def solidify_shell(o):
    me = o.data
    me.materials.clear()
    for n in (MAT, "__inner", "__rim"):
        me.materials.append(mat(n))
    m = o.modifiers.new("S", "SOLIDIFY")
    m.thickness = G.WALL * MM; m.offset = -1.0; m.use_rim = True; m.use_even_offset = True; m.use_quality_normals = True
    m.material_offset = 1; m.material_offset_rim = 2
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.modifier_apply(modifier=m.name)
    me = o.data
    mi = np.zeros(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", mi)
    uv = np.zeros(len(me.loops) * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) / MM
    nrim = 0
    for p in me.polygons:
        li = list(p.loop_indices)
        if mi[p.index] == 1:                             # liner: same layout, 0.3 density, its own region
            uv[li] = uv[li] * 0.3 + (9000, 0)
        elif mi[p.index] == 2:                           # rim quads: each its own planar island in mm
            vs = [me.loops[l].vertex_index for l in li]
            P = co[vs]
            e1 = P[1] - P[0]; e1 /= max(np.linalg.norm(e1), 1e-9)
            n = np.cross(P[1] - P[0], P[-1] - P[0]); n /= max(np.linalg.norm(n), 1e-9)
            e2 = np.cross(n, e1)
            q = np.stack([(P - P[0]) @ e1, (P - P[0]) @ e2], 1)
            uv[li] = q + (12000 + (nrim % 40) * 30, (nrim // 40) * 30)
            nrim += 1
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())
    set_face_attr(o, "lowcls", (mi + 1).astype(np.float32))
    me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int32))
    while len(me.materials) > 1:
        me.materials.pop()
    return o


def build_low(col):
    body_s, flap_s = G.shell_low()
    solidify_shell(body_s); solidify_shell(flap_s)
    body, flap = [body_s], [flap_s]
    sw = [G.pals_front(0, False)] + G.back_grid(False)
    for i in range(2):
        sw += [G.strap(i, False), G.snap_tab(i, False), G.side_loop(i, False), G.cord(i, False)]
    for o in sw:
        sweep_classes(o); prep(o, 4)
    body += sw
    hw = []
    for i in range(2):
        hw += [G.stud(i, False), G.stiffener(i, False), G.cap(i, False)] + G.slider(i, False)
    for o in hw:
        hw_uv(o); prep(o, 4)
    body += hw
    fs = [G.pals_front(1, False), G.pals_front(2, False)]
    for o in fs:
        sweep_classes(o); prep(o, 4)
    flap += fs
    B = G.bake_xform(join(body, "SM_GPPouch"))
    F = join(flap, "SM_GPPouch_Flap")
    for o in (B, F):
        scene.link(o, col)
        bm = bmesh.new(); bm.from_mesh(o.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4], quad_method="BEAUTY", ngon_method="BEAUTY")
        bm.to_mesh(o.data); bm.free()
        o.data.shade_smooth()
    return B, F


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    o.name = o.data.name = name
    return o


def pack(objs):
    """normalise the mm UVs of all objects by one factor, then pack every island together (cardinal rotations only)"""
    span = 0
    for o in objs:
        uv = np.zeros(len(o.data.loops) * 2); o.data.uv_layers.active.data.foreach_get("uv", uv)
        span = max(span, uv.max())
    for o in objs:
        uv = np.zeros(len(o.data.loops) * 2); o.data.uv_layers.active.data.foreach_get("uv", uv)
        o.data.uv_layers.active.data.foreach_set("uv", (uv / (span * 1.01) + 0.002).astype(np.float32))
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="CARDINAL", scale=True,
                            margin_method="FRACTION", margin=0.0045, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")
    # texel density (texels per mm at S) from the outer shell faces
    o = objs[0]; me = o.data
    uv = np.zeros(len(me.loops) * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    cls = np.zeros(len(me.polygons)); me.attributes["lowcls"].data.foreach_get("value", cls)
    a2 = a3 = 0.0
    for p in me.polygons:
        if cls[p.index] != 1:
            continue
        li = list(p.loop_indices)
        for k in range(1, len(li) - 1):
            (x0, y0), (x1, y1), (x2, y2) = uv[li[0]], uv[li[k]], uv[li[k + 1]]
            a2 += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
        a3 += p.area / MM ** 2
    return math.sqrt(a2 / a3)            # UV units per mm


# ------------------------------------------------------------------------------------------------ bakes
def bake_emit(low, highs, shader, size, samples=1):
    """emission bake (float). highs=[] bakes the low's own attribute."""
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"; s.cycles.samples = samples
    img = bpy.data.images.new("__emit", size, size, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    vis = {a: getattr(low, a) for a in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_shadow")}
    tgt = highs or [low]
    saved = bake._swap_emit(tgt, lambda N, L, m: shader(N, L))
    bake._target(low, img)
    if highs:
        for a in vis:
            setattr(low, a, False)
    bpy.ops.object.select_all(action="DESELECT")
    for h in highs:
        h.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    b = s.render.bake
    b.use_selected_to_active = bool(highs)
    b.cage_extrusion = 0.0025; b.max_ray_distance = 0.006; b.margin = 0; b.use_clear = True
    bpy.ops.object.bake(type="EMIT")
    bake._restore(tgt, saved)
    for a, v in vis.items():
        setattr(low, a, v)
    a = np.array(img.pixels[:], np.float32).reshape(size, size, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    return a


def attr_shader(name, kind="Color"):
    def sh(N, L):
        a = N.new("ShaderNodeAttribute"); a.attribute_name = name; a.attribute_type = "GEOMETRY"
        return a.outputs[kind]
    return sh


def bake_id(low, highs, size):
    return bake.bake(low, highs, "ID", size, f"{WORK}/id_{low.name}.png", samples=1, extrusion=0.0025, max_ray=0.006, margin=0)


# ------------------------------------------------------------------------------------------------ variants
VARIANTS = {   # fabric (ColorA), webbing/trim (ColorB) as sRGB hex; camo label for the print area
    "RangerGreen": ("#5b5442", "#7a7166", None),
    "Coyote": ("#7e6748", "#8a7558", None),
    "Black": ("#1f1f21", "#252527", None),
    "WolfGrey": ("#5f6163", "#74767a", None),
    "MultiTerrain": ("#9a8a68", "#8f8264", "camo"),
}


def lin(h):
    return [float(round(x, 4)) for x in tex.srgb_to_lin(tex.hex2rgb(h))]


def main():
    scene.reset()
    game = scene.collection("Game")
    hcol = scene.collection("High")
    log("high")
    highs = build_high(hcol)
    log("high done", sum(len(o.data.polygons) for o in highs), "faces")
    B, F = build_low(game)
    kuv = pack([B, F])
    tpm = kuv * S                                       # texels per mm
    log("low", len(B.data.polygons), len(F.data.polygons), "texels/mm", round(tpm, 2))
    # bake (ID via lab.bake, which reads each high material's diffuse colour)
    os.makedirs(WORK, exist_ok=True)
    cache = os.path.join(WORK, f"bakes_{S}.npz")
    if REUSE and os.path.exists(cache):
        bk = dict(np.load(cache)); log("reused bakes")
    else:
        bk = {k: None for k in ("nrm", "ao", "curv", "id", "p", "cls")}
        for low in (B, F):
            other = F if low is B else B
            other.hide_render = True
            res = {"cls": bake_emit(low, [], attr_shader("lowcls", "Fac"), S)[..., 0]}
            res["nrm"] = bake.bake(low, highs, "NORMAL", S, f"{WORK}/n_{low.name}.png", samples=2 if FAST else 6, extrusion=0.0025, max_ray=0.006, margin=0)
            log(low.name, "normal")
            res["ao"] = bake.bake(low, highs, "AO", S, f"{WORK}/ao_{low.name}.png", samples=12 if FAST else 40, extrusion=0.0025, max_ray=0.006, margin=0)
            log(low.name, "ao")
            res["curv"] = bake.bake(low, highs, "CURV", S, f"{WORK}/c_{low.name}.png", samples=2, extrusion=0.0025, max_ray=0.006, margin=0)
            res["id"] = bake_id(low, highs, S)
            res["p"] = bake_emit(low, highs, attr_shader("lab_p"), S)
            log(low.name, "id/p")
            other.hide_render = False
            m = res["cls"] > 0.5
            for k, v in res.items():
                if bk[k] is None:
                    bk[k] = np.zeros_like(v)
                bk[k][m] = v[m]
        np.savez_compressed(cache, **bk)
    for o in highs:
        bpy.data.objects.remove(o)
    bpy.data.collections.remove(hcol)
    texs, label_camo = PT.build(bk, S, tpm, TEXD, "GPPouch")
    log("textures")
    mats = {}
    for k, (ca, cb, lab) in VARIANTS.items():
        prm = dict(ColorA=lin(ca), RoughA=0.8, MetalA=0.0, ColorB=lin(cb), RoughB=0.78, MetalB=0.0,
                   LabelRough=0.8, WearAmount=0.08, Dirt=0.3)
        lp = label_camo if lab == "camo" else None
        m = master.material(f"MI_GPPouch_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in (B, F):
        o.data.materials[0] = mats["RangerGreen"][0]
    for nm in ("__inner", "__rim", MAT):
        m = bpy.data.materials.get(nm)
        if m and m.users == 0:
            bpy.data.materials.remove(m)
    # flap pivot = hinge; custom props for the contract
    G.bake_xform(F)
    from lab import geo
    geo.set_origin(F, Vector(G.HINGE) * MM)
    for o in (B, F):
        if "lowcls" in o.data.attributes:
            o.data.attributes.remove(o.data.attributes["lowcls"])
    # LODs (UV-delimited collapse)
    lods = []
    for src in (B, F):
        for i, r in enumerate((0.5, 0.25)):
            o = src.copy(); o.data = src.data.copy(); o.name = o.data.name = f"{src.name}_LOD{i + 1}"
            game.objects.link(o)
            d = o.modifiers.new("D", "DECIMATE"); d.ratio = r; d.use_collapse_triangulate = True
            d.delimit = {"UV", "SEAM"}
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.modifier_apply(modifier="D")
            o.hide_render = True
            lods.append(o)
    tri = {o.name: scene.tris(o) for o in [B, F] + lods}
    log("tris", tri)
    ccol = scene.collection("Collision")
    hb = scene.link(export.hull(B, "UCX_SM_GPPouch_00", 64), ccol); hb.hide_render = True
    hf = scene.link(export.hull(F, "UCX_SM_GPPouch_Flap_00", 32), ccol); hf.hide_render = True
    export.fbx([B, hb], f"{FBXD}/SM_GPPouch.fbx")
    export.fbx([F, hf], f"{FBXD}/SM_GPPouch_Flap.fbx")
    for o in lods:
        export.fbx([o], f"{FBXD}/{o.name}.fbx")
        o.hide_set(True)
    for o in (hb, hf):
        o.hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "molle_pouch.blend"), relative_remap=True)
    dims = {o.name: [round(x * 100, 2) for x in o.dimensions] for o in (B, F)}
    man = {
        "item": "GPPouch", "replaces_rows": ["SmallContainer"], "category": "containers", "clip": None,
        "dims_cm": [round(max(dims["SM_GPPouch"][0], 0), 2), dims["SM_GPPouch"][1], dims["SM_GPPouch"][2]],
        "real_object": "6x6 MOLLE utility pouch (clamshell zipper), 1000D cordura: body 15.2 x 7.6 x 15.2 cm; "
                       "bbox includes the PALS straps, top strap tabs and cord pulls",
        "pivot": "base centre, +Z up, front = -Y",
        "parts": {"SM_GPPouch_Flap": {
            "pivot": "fold line across the front at the zipper ends (x 0, y -4.1, z 6.9 cm)",
            "rest_location_cm": [round(v / 10, 2) for v in G.HINGE], "rest_rotation_deg": [0, 0, 0],
            "opens": "about local X to +100 deg (front-upper clamshell folds forward and down)",
            "contract_change": "was 'rear hinge at the top, about local X to -100 deg' (top lid); the 6x6 reference is a "
                               "clamshell zipped down both gussets, so the part now hinges at the front fold line"}},
        "grid_suggestion": "2x2 container, 2x3 inside",
        "tris": tri,
        "textures": {"size": S, "normal": "OpenGL (flip green in UE)",
                     "maps": {k: os.path.basename(v) for k, v in texs.items()},
                     "variants": {k: (os.path.basename(v[2]) if v[2] else None) for k, v in mats.items()}},
        "mask": "R fabric (cordura), G webbing/thread/trim/plastic hardware, B print area (camo skins only; solid skins "
                "leave it empty), A wear",
        "variant_params": {k: v[1] for k, v in mats.items()},
        "collision": ["UCX_SM_GPPouch_00", "UCX_SM_GPPouch_Flap_00"],
        "lods": [o.name for o in lods],
        "notes": "Texture detail: baked high->low normal/AO/curvature + synthesised weave, webbing ribs, stitch rows, "
                 "bar-tacks, paracord braid. Liner (inside) at 0.3x texel density.",
    }
    export.manifest(os.path.join(HERE, "manifest.json"), man)
    log("TRIS", tri)
    # S9: re-import check (same names, scale, materials)
    expect = {"SM_GPPouch": dims["SM_GPPouch"], "SM_GPPouch_Flap": dims["SM_GPPouch_Flap"]}
    ok, rep = export.reimport_check([f"{FBXD}/SM_GPPouch.fbx", f"{FBXD}/SM_GPPouch_Flap.fbx"], expect)
    log("REIMPORT", ok)


if __name__ == "__main__":
    main()
