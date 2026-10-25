"""SodaCan: builds everything from nothing.  .venv/bin/python props/drinks/soda_can/soda_can.py [--fast]
Outputs (this folder): soda_can.blend, fbx/, textures/, renders/, manifest.json. QC: tools/qc.py soda_can.blend."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from mathutils import Vector
from lab import scene, master, bake, export, tex
import can_geo as G, can_tex as T, can_labels as LB

FAST = "--fast" in sys.argv
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/soda_can"))
VARIANTS = list(LB.VARIANTS)


def build_mesh():
    scene.reset()
    col = scene.collection("Game")
    shell, tabo, flapo = G.shell(), G.tab(), G.flap()
    for o in (shell, tabo, flapo):
        scene.link(o, col)
    hi = scene.collection("High")
    highs = {}
    for o in (tabo, flapo):                       # high = same plate with rolled (bevelled) edges
        h = o.copy(); h.data = o.data.copy(); h.name = o.name + "_high"
        scene.link(h, hi)
        b = h.modifiers.new("Bevel", "BEVEL"); b.width = 0.00018; b.segments = 3; b.limit_method = "ANGLE"
        h.modifiers.new("Sub", "SUBSURF").levels = 1
        for p in h.data.polygons:
            p.use_smooth = True
        highs[o.name] = h
    return shell, tabo, flapo, highs


def do_bakes(shell, tabo, flapo, highs):
    os.makedirs(WORK, exist_ok=True)
    n = 1024 if FAST else 2048
    smp = 8 if FAST else 32
    ao_s = bake.bake(shell, [], "AO", n, f"{WORK}/ao_shell.png", samples=smp)
    ao_t = bake.bake(tabo, [highs[tabo.name]], "AO", n, f"{WORK}/ao_tab.png", samples=smp)
    ao_f = bake.bake(flapo, [highs[flapo.name]], "AO", n, f"{WORK}/ao_flap.png", samples=smp)
    nt = bake.bake(tabo, [highs[tabo.name]], "NORMAL", n, f"{WORK}/n_tab.png", samples=1)
    nf = bake.bake(flapo, [highs[flapo.name]], "NORMAL", n, f"{WORK}/n_flap.png", samples=1)
    if n != T.S:                                    # upsample fast bakes to the texture size
        from PIL import Image
        up = lambda a: np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((T.S, T.S)), np.float32) / 255
        ao_s, ao_t, ao_f, nt, nf = map(up, (ao_s, ao_t, ao_f, nt, nf))
    u, v = T.grids()
    mt, mf = T.rect_mask(u, v, G.UV_TAB), T.rect_mask(u, v, G.UV_FLAP)
    ao = np.where(mt, ao_t, np.where(mf, ao_f, ao_s))
    nrm = np.where(mt[..., None], nt, nf)
    return np.clip(ao, 0, 1), nrm


def variant_materials(texs):
    mats = {}
    for k in VARIANTS:
        L, params = LB.VARIANTS[k]()
        art = LB.to_uv(L, wear_seed=len(k))
        lp = tex.save(art, os.path.join(TEXD, f"T_SodaCan_Label_{k}.png"), "RGBA")
        params = dict(params, WearAmount=0.45, Dirt=0.35)
        m = master.material(f"MI_SodaCan_{k}", texs, params, label=lp)
        m.use_fake_user = True                      # keep every variant in the .blend
        mats[k] = (m, params, lp)
    return mats


def assign(objs, mat):
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)


def renders(shell, tabo, flapo, mats):
    parts = [shell, tabo, flapo]
    for h in bpy.data.collections["High"].objects:
        h.hide_render = True
    smp = 16 if FAST else 96
    assign(parts, mats["Kolossal"][0])
    scene.studio(parts)
    scene.frame(parts, yaw=-24, pitch=24, lens=85, fill=0.78, name="Hero")
    scene.render(f"{REND}/soda_can.png", 1024, smp)
    # lid close-up: tab lifted 55° about X, flap pushed in 70°
    tabo.rotation_euler.x, flapo.rotation_euler.x = 0.95, 1.2
    scene.camera((0, -0.004, 0.118), 0.075, 155, 52, 70, "Lid")
    scene.render(f"{REND}/soda_can_open.png", 1024, smp)
    tabo.rotation_euler.x = flapo.rotation_euler.x = 0
    # exploded parts
    tabo.location.z += 0.022; flapo.location.z += 0.011
    scene.frame(parts, yaw=-30, pitch=34, lens=85, fill=0.8, name="Parts")
    scene.render(f"{REND}/soda_can_parts.png", 1024, smp)
    tabo.location.z -= 0.022; flapo.location.z -= 0.011
    # variant lineup: linked duplicates, one material each
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c)
    lineup = []
    for i, k in enumerate(VARIANTS):
        for o in parts:
            d = o.copy(); d.data = o.data.copy()
            d.location.x += (i - (len(VARIANTS) - 1) / 2) * 0.078
            d.rotation_euler.z = 0.25 * (i % 2 * 2 - 1)
            bpy.context.scene.collection.objects.link(d)
            assign([d], mats[k][0])
            lineup.append(d)
    for o in parts:
        o.hide_render = True
    scene.frame(lineup, yaw=0, pitch=12, lens=70, fill=0.8, name="Lineup", aspect=700 / 1600)
    scene.render(f"{REND}/soda_can_variants.png", 1600, smp, res_y=700)
    for d in lineup:
        bpy.data.objects.remove(d)
    for o in parts:
        o.hide_render = False


def renders_only():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "soda_can.blend"))
    o = bpy.data.objects
    mats = variant_materials({k: os.path.join(TEXD, f"T_SodaCan_{k}.png") for k in ("BaseColor", "Normal", "ORM", "Mask")})
    for x in list(o):
        if x.type in ("LIGHT", "CAMERA") or x.name == "Backdrop":
            bpy.data.objects.remove(x)
    renders(o["SM_SodaCan"], o["SM_SodaCan_Tab"], o["SM_SodaCan_Flap"], mats)


def main():
    if "--renders" in sys.argv:
        return renders_only()
    shell, tabo, flapo, highs = build_mesh()
    ao, nrm = do_bakes(shell, tabo, flapo, highs)
    texs = T.build(TEXD, ao=ao, bake_n=nrm)
    mats = variant_materials(texs)
    assign([shell, tabo, flapo], mats["Kolossal"][0])
    # game extras: collision + LODs (shell is over 2k tris)
    ucx = scene.link(export.hull(shell, "UCX_SM_SodaCan_00", 64), scene.collection("Collision"))
    l1, l2 = export.lod(shell, 0.5, "SM_SodaCan_LOD1"), export.lod(shell, 0.25, "SM_SodaCan_LOD2")
    for o in (l1, l2, ucx):
        o.hide_render = True
    for o in (l1, l2):
        o.hide_set(True)
    tri = {o.name: scene.tris(o) for o in (shell, tabo, flapo, l1, l2)}
    os.makedirs(FBXD, exist_ok=True)
    export.fbx([shell, ucx], f"{FBXD}/SM_SodaCan.fbx")
    export.fbx([tabo], f"{FBXD}/SM_SodaCan_Tab.fbx")
    export.fbx([flapo], f"{FBXD}/SM_SodaCan_Flap.fbx")
    export.fbx([l1], f"{FBXD}/SM_SodaCan_LOD1.fbx")
    export.fbx([l2], f"{FBXD}/SM_SodaCan_LOD2.fbx")
    l1.hide_set(True); l2.hide_set(True)
    bpy.ops.file.make_paths_relative() if bpy.data.filepath else None
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "soda_can.blend"), relative_remap=True)
    renders(shell, tabo, flapo, mats)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "SodaCan", "replaces_rows": ["SodaCan", "Cola", "Coffee", "SpiritsBottle"], "category": "drinks",
        "clip": "A_Consume_Drink_Can", "dims_cm": [6.62, 6.62, 12.22], "real_object": "355 mL 211/202 aluminium can, 202 stay-on-tab end",
        "pivot": "base centre, +Z up", "mouth_local_cm_ue": [0, 1.24, 11.6],
        "parts": {"SM_SodaCan_Tab": {"pivot": "rivet", "rest_location_cm": [0, 0, 11.6], "rest_rotation_deg": [0, 0, 0],
                                     "opens": "about local X, +55 deg lifts the ring end (nose drives the flap down), then back to rest"},
                  "SM_SodaCan_Flap": {"pivot": "score hinge near the rivet", "rest_location_cm_ue": [0, 0.26, 11.6],
                                      "opens": "about local X, +70 deg (pushes down into the can) on TabPop"}},
        "tris": tri, "textures": {"size": T.S, "normal": "OpenGL (flip green in UE)", "set": "T_SodaCan_{BaseColor,Normal,ORM,Mask}",
                                  "variants": {k: os.path.basename(mats[k][2]) for k in VARIANTS}},
        "material": "M_Lab_Prop (universal mask) → MI_SodaCan_<Variant>",
        "variant_params": {k: mats[k][1] for k in VARIANTS},
        "collision": "UCX_SM_SodaCan_00 (convex hull, 64 extreme points)", "lods": ["SM_SodaCan_LOD1 (50%)", "SM_SodaCan_LOD2 (25%)"],
        "qc_notes": "WARN texel spread: vertical countersink/stand-ring walls inside planar disc islands (tiny, unseen); WARN 2 slivers on the flat lid panel (planar, no shading effect)."})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
