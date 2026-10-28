"""MagPouch3: triple AR magazine pouch with covers. Builds everything from nothing: high-poly (pillowed sleeves, tall backs,
PALS web, polymer stiffener straps, press-studs, clips, loops, velcro panels, ribbed-webbing flaps), game meshes
(body + three flap parts + optional magazines, LOD1/2, UCX), baked + synthesised textures, universal-mask skins, FBX,
manifest.  Renders: mag_render.py.  Shared pipeline pieces live in ../molle_pouch (pouch_geo / pouch_tex / molle_pouch).

    .venv/bin/python props/containers/mag_pouch_triple/mag_pouch_triple.py [--fast] [--reuse-bakes]"""
import os, sys, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../molle_pouch"))
import bpy, bmesh
import numpy as np
from mathutils import Vector
from lab import scene, master, bake, export, tex, geo
import mag_geo as G
import mag_tex as MT
import molle_pouch as MP

FAST = "--fast" in sys.argv
REUSE = "--reuse-bakes" in sys.argv
S = 1024 if FAST else 2048
TEXD, FBXD = os.path.join(HERE, "textures"), os.path.join(HERE, "fbx")
WORK = os.path.abspath(os.path.join(HERE, "../../../work/mag_pouch_triple"))
MM = G.MM
MAT = "MI_MagPouch3"
ITEM = "MagPouch3"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def prep(o, cls):
    o.data.materials.clear(); o.data.materials.append(MP.mat(MAT))
    if "lowcls" not in o.data.attributes:
        MP.set_face_attr(o, "lowcls", cls)
    return o


def solidify(o, wall):
    me = o.data
    me.materials.clear()
    for n in (MAT, "__inner", "__rim"):
        me.materials.append(MP.mat(n))
    m = o.modifiers.new("S", "SOLIDIFY")
    m.thickness = wall * MM; m.offset = -1.0; m.use_rim = True; m.use_even_offset = True; m.use_quality_normals = True
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
        if mi[p.index] == 1:
            uv[li] = uv[li] * 0.3 + (30000, 0)
        elif mi[p.index] == 2:
            P = co[[me.loops[l].vertex_index for l in li]]
            e1 = P[1] - P[0]; e1 /= max(np.linalg.norm(e1), 1e-9)
            n = np.cross(P[1] - P[0], P[-1] - P[0]); n /= max(np.linalg.norm(n), 1e-9)
            e2 = np.cross(n, e1)
            uv[li] = np.stack([(P - P[0]) @ e1, (P - P[0]) @ e2], 1) + (36000 + (nrim % 40) * 30, (nrim // 40) * 30)
            nrim += 1
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())
    MP.set_face_attr(o, "lowcls", (mi + 1).astype(np.float32))
    me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int32))
    while len(me.materials) > 1:
        me.materials.pop()
    return o


def build_high(col):
    out = []

    def add(o, idn):
        if isinstance(o, (list, tuple)):
            for x in o:
                add(x, idn)
            return
        o.data.materials.clear(); o.data.materials.append(MP.mat("ID_" + idn, G.IDS[idn]))
        scene.link(o, col); out.append(o)
    add(G.panel(True), "fabric")
    for i in range(3):
        add(G.sleeve(i, True), "fabric"); add(G.tall_back(i, True), "fabric"); add(G.loop_panel(i, True), "loop")
        add(G.eyelet(i, True), "eyelet")
        add(G.flap(i, True), "flapweb"); add(G.hook_strip(i, True), "hook"); add(G.pull_tab(i, True), "webv")
        add(G.magazine(i, True), "mag")
    for k in range(3):
        add(G.web_row(k, True), "webh")
    for j in range(6):
        add(G.poly_strap(j, True), "poly"); add(G.snap_tab(j, True), "poly"); add(G.stud(j, True), "stud")
        add(G.clip(j, True), "clip"); add(G.top_loop(j, True), "webv")
    for k in range(2):
        add(G.channel_tab(k, True), "webv")
    return out


def finish(objs, name, col):
    o = MP.join(objs, name)
    scene.link(o, col)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(o.data); bm.free()
    o.data.shade_smooth()
    return o


def build_low(col):
    body = []
    pn = G.panel(False); prep(pn, 1); body.append(pn)
    for i in range(3):
        s = solidify(G.sleeve(i, False), G.WALL); body.append(s)
        b = G.tall_back(i, False); prep(b, 1); body.append(b)
    sw = [G.loop_panel(i, False) for i in range(3)] + [G.web_row(k, False) for k in range(3)]
    for j in range(6):
        sw += [G.poly_strap(j, False), G.snap_tab(j, False), G.top_loop(j, False)]
    sw += [G.channel_tab(k, False) for k in range(2)]
    for o in sw:
        MP.sweep_classes(o); prep(o, 4)
    hw = [G.eyelet(i, False) for i in range(3)] + [G.stud(j, False) for j in range(6)] + [G.clip(j, False) for j in range(6)]
    for o in hw:
        MP.hw_uv(o); prep(o, 4)
    B = finish(body + sw + hw, "SM_MagPouch3", col)
    flaps = []
    for i in range(3):
        parts = [G.flap(i, False), G.hook_strip(i, False), G.pull_tab(i, False)]
        for o in parts:
            MP.sweep_classes(o); prep(o, 4)
        flaps.append(finish(parts, f"SM_MagPouch3_Flap{i + 1}", col))
    mags = []
    for i in range(3):
        m = G.magazine(i, False); MP.hw_uv(m); prep(m, 4); mags.append(m)
        uv = np.zeros(len(m.data.loops) * 2); m.data.uv_layers.active.data.foreach_get("uv", uv)
        m.data.uv_layers.active.data.foreach_set("uv", uv * 0.45)            # mostly hidden in the sleeves
    M = finish(mags, "SM_MagPouch3_Mags", col)
    return B, flaps, M


def median_density(o):
    """UV units per mm: median over the outer-shell faces (hidden faces are deliberately shrunk, so not a mean)"""
    me = o.data
    uv = np.zeros(len(me.loops) * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    cls = np.zeros(len(me.polygons)); me.attributes["lowcls"].data.foreach_get("value", cls)
    d = []
    for p in me.polygons:
        if cls[p.index] != 1 or p.area < 1e-7:
            continue
        li = list(p.loop_indices); a2 = 0.0
        for k in range(1, len(li) - 1):
            (x0, y0), (x1, y1), (x2, y2) = uv[li[0]], uv[li[k]], uv[li[k + 1]]
            a2 += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
        d.append(math.sqrt(a2 / (p.area / MM ** 2)))
    return float(np.median(d))


VARIANTS = {"Navy": ("#373c4c", "#30323a", None),
            "RangerGreen": ("#5b5442", "#7a7166", None),
            "Coyote": ("#7e6748", "#8a7558", None),
            "Black": ("#1f1f21", "#232325", None),
            "MultiTerrain": ("#9a8a68", "#8f8264", "camo")}


def main():
    scene.reset()
    game = scene.collection("Game"); hcol = scene.collection("High")
    highs = build_high(hcol)
    log("high", sum(len(o.data.polygons) for o in highs))
    B, flaps, M = build_low(game)
    lows = [B] + flaps + [M]
    MP.pack(lows)
    tpm = median_density(B) * S
    log("low", [len(o.data.polygons) for o in lows], "texels/mm", round(tpm, 2))
    os.makedirs(WORK, exist_ok=True)
    cache = os.path.join(WORK, f"bakes_{S}.npz")
    MP.WORK = WORK
    if REUSE and os.path.exists(cache):
        bk = dict(np.load(cache)); log("reused bakes")
    else:
        bk = {k: None for k in ("nrm", "ao", "curv", "id", "p", "cls")}
        for low in lows:
            for o in lows:
                o.hide_render = o is not low
            # flaps closed + mags in place: bake everything in the closed, loaded state
            res = {"cls": MP.bake_emit(low, [], MP.attr_shader("lowcls", "Fac"), S)[..., 0]}
            kw = dict(extrusion=0.0022, max_ray=0.005, margin=0)
            res["nrm"] = bake.bake(low, highs, "NORMAL", S, f"{WORK}/n_{low.name}.png", samples=2 if FAST else 6, **kw)
            res["ao"] = bake.bake(low, highs, "AO", S, f"{WORK}/ao_{low.name}.png", samples=12 if FAST else 40, **kw)
            res["curv"] = bake.bake(low, highs, "CURV", S, f"{WORK}/c_{low.name}.png", samples=2, **kw)
            res["id"] = bake.bake(low, highs, "ID", S, f"{WORK}/id_{low.name}.png", samples=1, **kw)
            res["p"] = MP.bake_emit(low, highs, MP.attr_shader("lab_p"), S)
            log(low.name, "baked")
            m = res["cls"] > 0.5
            for k, v in res.items():
                if bk[k] is None:
                    bk[k] = np.zeros_like(v)
                bk[k][m] = v[m]
        for o in lows:
            o.hide_render = False
        np.savez_compressed(cache, **bk)
    for o in highs:
        bpy.data.objects.remove(o)
    bpy.data.collections.remove(hcol)
    texs, label_camo = MT.build(bk, S, tpm, TEXD, ITEM)
    log("textures")
    mats = {}
    for k, (ca, cb, lab) in VARIANTS.items():
        prm = dict(ColorA=MP.lin(ca), RoughA=0.8, MetalA=0.0, ColorB=MP.lin(cb), RoughB=0.78, MetalB=0.0,
                   LabelRough=0.8, WearAmount=0.08, Dirt=0.3)
        lp = label_camo if lab == "camo" else None
        m = master.material(f"MI_{ITEM}_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in lows:
        o.data.materials[0] = mats["Navy"][0]
        if "lowcls" in o.data.attributes:
            o.data.attributes.remove(o.data.attributes["lowcls"])
    for nm in ("__inner", "__rim", MAT):
        m = bpy.data.materials.get(nm)
        if m and m.users == 0:
            bpy.data.materials.remove(m)
    for i, f in enumerate(flaps):
        geo.set_origin(f, Vector(G.hinge(i)) * MM)
    lods = []
    for src in [B] + flaps + [M]:
        for i, r in enumerate((0.5, 0.25)):
            o = src.copy(); o.data = src.data.copy(); o.name = o.data.name = f"{src.name}_LOD{i + 1}"
            game.objects.link(o)
            d = o.modifiers.new("D", "DECIMATE"); d.ratio = r; d.use_collapse_triangulate = True; d.delimit = {"UV", "SEAM"}
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.modifier_apply(modifier="D")
            o.hide_render = True
            lods.append(o)
    tri = {o.name: scene.tris(o) for o in lows + lods}
    log("tris", tri)
    ccol = scene.collection("Collision")
    hulls = {B.name: scene.link(export.hull(B, "UCX_SM_MagPouch3_00", 64), ccol)}
    for f in flaps:
        hulls[f.name] = scene.link(export.hull(f, f"UCX_{f.name}_00", 32), ccol)
    for o in lows:
        objs = [o] + ([hulls[o.name]] if o.name in hulls else [])
        export.fbx(objs, f"{FBXD}/{o.name}.fbx")
    for o in lods:
        export.fbx([o], f"{FBXD}/{o.name}.fbx")
        o.hide_set(True)
    for h in hulls.values():
        h.hide_render = True; h.hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "mag_pouch_triple.blend"), relative_remap=True)
    dims = {o.name: [round(x * 100, 2) for x in o.dimensions] for o in lows}
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": ITEM, "replaces_rows": [], "category": "containers", "clip": None,
        "dims_cm": dims["SM_MagPouch3"],
        "real_object": "triple 5.56 AR magazine pouch with covers (polymer-strap MOLLE back); sized for three STANAG 30-rd "
                       "magazines 2.5 x 8.1 x 19 cm (the game's MagazineAR15); bbox includes the top strap loops",
        "pivot": "base centre, +Z up, front = -Y",
        "parts": {f.name: {"pivot": "hinge at the top of the sleeve's tall back",
                           "rest_location_cm": [round(v / 10, 2) for v in G.hinge(i)], "rest_rotation_deg": [0, 0, 0],
                           "opens": "about local X to -115 deg (flap flips up and back)"} for i, f in enumerate(flaps)},
        "optional_parts": {"SM_MagPouch3_Mags": "three STANAG magazines seated in the sleeves (loaded variant); hide for empty"},
        "grid_suggestion": "3x2 container (holds 3 MagazineAR15)",
        "tris": tri,
        "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "maps": {k: os.path.basename(v) for k, v in texs.items()},
                     "variants": {k: (os.path.basename(v[2]) if v[2] else None) for k, v in mats.items()}},
        "mask": "R cordura, G webbing/thread/velcro/polymer/plastic, B print area (camo skin only), A wear",
        "variant_params": {k: v[1] for k, v in mats.items()},
        "collision": sorted(h.name for h in hulls.values()),
        "lods": [o.name for o in lods],
        "notes": "One texture set for body, flaps and magazines. Liner at 0.3x texel density."})
    expect = {o.name: dims[o.name] for o in lows}
    ok, rep = export.reimport_check([f"{FBXD}/{o.name}.fbx" for o in lows], expect)
    log("REIMPORT", ok)


if __name__ == "__main__":
    main()
