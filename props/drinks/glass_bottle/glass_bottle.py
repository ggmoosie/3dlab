"""GlassBottle (longneck + crown cap): builds everything.  .venv/bin/python props/drinks/glass_bottle/glass_bottle.py [--fast]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from lab import scene, master, bake, export, tex
import glass_geo as G, glass_tex as T, glass_labels as LB

FAST = "--fast" in sys.argv
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/glass_bottle"))
VARIANTS = list(LB.VARIANTS)
EXTRA = ("GlassTint", "LiquidTint", "LiquidMurk")


def build_mesh():
    scene.reset()
    col = scene.collection("Game")
    parts = dict(glass=G.bottle(), label=G.label(), neck=G.neck_label(), liquid=G.liquid(), cap=G.cap())
    for o in parts.values():
        scene.link(o, col)
    return parts


def bakes(P):
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (2048, 32)
    a_cap = bake.bake(P["cap"], [], "AO", n, f"{WORK}/ao_cap.png", samples=smp)
    if n != T.S:
        from PIL import Image
        a_cap = np.asarray(Image.fromarray((np.clip(a_cap, 0, 1) * 255).astype(np.uint8)).resize((T.S, T.S)), np.float32) / 255
    return np.where(a_cap < 0.02, 1.0, a_cap)


def materials(texs):
    mats = {}
    for k in VARIANTS:
        F, N, C, p = LB.VARIANTS[k]()
        lp = tex.save(LB.to_uv(F, N, C, len(k)), os.path.join(TEXD, f"T_GlassBottle_Label_{k}.png"), "RGBA")
        prm = {a: b for a, b in p.items() if a not in EXTRA}
        prm.update(WearAmount=0.4, Dirt=0.3)
        mi = master.material(f"MI_GlassBottle_{k}", texs, prm, label=lp)
        gl = master.glass(f"MI_GlassBottle_Glass_{k}", p["GlassTint"], 0.03, 1.5, normal=texs["Normal"], orm=texs["ORM"])
        lq = master.glass(f"MI_GlassBottle_Liquid_{k}", p["LiquidTint"], 0.25 * p["LiquidMurk"], 1.33)
        for m in (mi, gl, lq):
            m.use_fake_user = True
        mats[k] = dict(mi=mi, glass=gl, liq=lq, params=prm, extra={a: p[a] for a in EXTRA}, label=lp)
    return mats


def dress(P, m):
    for key, mat in (("glass", m["glass"]), ("label", m["mi"]), ("neck", m["mi"]), ("cap", m["mi"]), ("liquid", m["liq"])):
        me = P[key].data
        if not me.materials:
            me.materials.append(mat)
        me.materials[0] = mat


def renders(P, mats):
    objs = list(P.values())
    smp = 24 if FAST else 80
    s = bpy.context.scene
    s.cycles.transmission_bounces = s.cycles.max_bounces = 16
    dress(P, mats["GowanusPale"])
    scene.studio(objs)
    scene.frame(objs, yaw=-20, pitch=8, lens=85, fill=0.85, name="Hero")
    scene.render(f"{REND}/glass_bottle.png", 1024, smp)
    cap = P["cap"]
    cap.location.z += 0.012; cap.rotation_euler = (0.5, 0.2, 0.0)
    scene.camera((0, 0, 0.272), 0.14, -25, 22, 85, "Cap")
    scene.render(f"{REND}/glass_bottle_cap_off.png", 1024, smp)
    cap.location.z -= 0.012; cap.rotation_euler = (0, 0, 0)
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c)
    dups = []
    for i, k in enumerate(VARIANTS):
        Q = {}
        for key, o in P.items():
            d = o.copy(); d.data = o.data.copy()
            d.location.x += (i - (len(VARIANTS) - 1) / 2) * 0.075
            bpy.context.scene.collection.objects.link(d)
            Q[key] = d
        dress(Q, mats[k])
        dups += list(Q.values())
    for o in objs:
        o.hide_render = True
    scene.frame(dups, yaw=0, pitch=6, lens=70, fill=0.8, name="Lineup", aspect=820 / 1600)
    scene.render(f"{REND}/glass_bottle_variants.png", 1600, smp, res_y=820)
    for d in dups:
        bpy.data.objects.remove(d)
    for o in objs:
        o.hide_render = False


def main():
    P = build_mesh()
    texs = T.build(TEXD, ao=bakes(P))
    mats = materials(texs)
    dress(P, mats["GowanusPale"])
    g = P["glass"]
    ucx = scene.link(export.hull(g, "UCX_SM_GlassBottle_00", 64), scene.collection("Collision"))
    ucx.hide_render = True
    l1, l2 = export.lod(g, 0.5, "SM_GlassBottle_LOD1"), export.lod(g, 0.25, "SM_GlassBottle_LOD2")
    for o in (l1, l2):
        o.hide_render = True
    tri = {o.name: scene.tris(o) for o in list(P.values()) + [l1, l2]}
    export.fbx([g, P["label"], P["neck"], ucx], f"{FBXD}/SM_GlassBottle.fbx")       # UE combine_meshes → 1 mesh, 2 slots
    export.fbx([P["liquid"]], f"{FBXD}/SM_GlassBottle_Liquid.fbx")
    export.fbx([P["cap"]], f"{FBXD}/SM_GlassBottle_Cap.fbx")
    export.fbx([l1], f"{FBXD}/SM_GlassBottle_LOD1.fbx")
    export.fbx([l2], f"{FBXD}/SM_GlassBottle_LOD2.fbx")
    l1.hide_set(True); l2.hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "glass_bottle.blend"), relative_remap=True)
    renders(P, mats)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "GlassBottle", "replaces_rows": ["Lemonade", "SauceBottle", "WhiskeyBottle", "WineBottle"], "category": "drinks",
        "clip": "A_Consume_Drink_Beer", "dims_cm": [6.3, 6.3, 28.35], "real_object": "longneck glass bottle, 26 mm crown finish (scaled to the 28.15 cm contract)",
        "pivot": "base centre, +Z up",
        "parts": {"SM_GlassBottle_Cap": {"pivot": "crown bottom centre on the neck axis", "rest_location_cm": [0, 0, 27.4],
                                         "opens": "about local Z 3 x 60 deg while lifting 0.9 cm (contract)"},
                  "SM_GlassBottle_Liquid": {"note": "fill to 21.4 cm (shoulder)"}},
        "material_slots": {"SM_GlassBottle": ["M_Lab_Glass → MI_GlassBottle_Glass_<V>", "MI_GlassBottle_<V> (front label + neck foil decals)"],
                           "SM_GlassBottle_Cap": ["MI_GlassBottle_<V> (layer A = crown paint, print = crown top)"],
                           "SM_GlassBottle_Liquid": ["M_Lab_Liquid → MI_GlassBottle_Liquid_<V>"]},
        "tris": tri, "textures": {"size": T.S, "normal": "OpenGL (flip green in UE)",
                                  "variants": {k: os.path.basename(mats[k]["label"]) for k in VARIANTS}},
        "variant_params": {k: mats[k]["params"] for k in VARIANTS}, "glass_liquid_params": {k: mats[k]["extra"] for k in VARIANTS},
        "collision": "UCX_SM_GlassBottle_00", "lods": ["SM_GlassBottle_LOD1 (50%)", "SM_GlassBottle_LOD2 (25%)"],
        "qc_notes": "label + neck foil are declared one-sided decals (qc_open), 0.2 mm off the glass"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
