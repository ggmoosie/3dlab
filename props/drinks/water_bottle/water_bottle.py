"""WaterBottle (500 mL PET, screw cap): builds everything.  .venv/bin/python props/drinks/water_bottle/water_bottle.py [--fast]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from lab import scene, master, bake, export, tex
import bottle_geo as G, bottle_tex as T, bottle_labels as LB

FAST = "--fast" in sys.argv
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/water_bottle"))
VARIANTS = list(LB.VARIANTS)
LIQ_KEYS = ("LiquidTint", "LiquidMurk")


def build_mesh():
    scene.reset()
    col = scene.collection("Game")
    parts = G.bottle(), G.liquid(), G.cap()
    for o in parts:
        scene.link(o, col)
    return parts


def bakes(bottle, cap):
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (2048, 32)
    a_b = bake.bake(bottle, [], "AO", n, f"{WORK}/ao_bottle.png", samples=smp)
    a_c = bake.bake(cap, [], "AO", n, f"{WORK}/ao_cap.png", samples=smp)
    if n != T.S:
        from PIL import Image
        up = lambda a: np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((T.S, T.S)), np.float32) / 255
        a_b, a_c = up(a_b), up(a_c)
    u, v = tex.uv_grid(T.S)
    mc = tex.in_rect(u, v, G.UV_CAP_SIDE) | tex.in_rect(u, v, G.UV_CAP_TOP) | tex.in_rect(u, v, (0.47, 0.20, 0.49, 0.36))
    ao = np.where(mc, a_c, a_b)
    return np.where(ao < 0.02, 1.0, ao)                  # unbaked texels (gaps) stay white


def materials(texs):
    pet = master.glass("M_WaterBottle_PET", (0.96, 0.98, 0.99), 0.06, 1.08, normal=texs["Normal"])
    mats = {}
    for k in VARIANTS:
        L, p = LB.VARIANTS[k]()
        lp = tex.save(LB.to_uv(L, len(k)), os.path.join(TEXD, f"T_WaterBottle_Label_{k}.png"), "RGBA")
        prm = {a: b for a, b in p.items() if a not in LIQ_KEYS}
        prm.update(WearAmount=0.6 if k == "Refill" else 0.35, Dirt=0.6 if k == "Refill" else 0.25)
        mi = master.material(f"MI_WaterBottle_{k}", texs, prm, label=lp)
        liq = master.glass(f"MI_WaterBottle_Liquid_{k}", p["LiquidTint"], 0.0 + 0.25 * p["LiquidMurk"], 1.33)
        for m in (mi, liq):
            m.use_fake_user = True
        mats[k] = dict(mi=mi, liq=liq, params=prm, liquid={a: p[a] for a in LIQ_KEYS}, label=lp)
    return pet, mats


def dress(bottle, liquid, cap, pet, m):
    def put(o, mats):                     # assign into existing slots: clearing slots resets face indices
        while len(o.data.materials) < len(mats):
            o.data.materials.append(None)
        for i, mt in enumerate(mats):
            o.data.materials[i] = mt
    put(bottle, [pet, m["mi"]]); put(cap, [m["mi"]]); put(liquid, [m["liq"]])


def renders(bottle, liquid, cap, pet, mats):
    parts = [bottle, liquid, cap]
    smp = 24 if FAST else 128
    s = bpy.context.scene
    s.cycles.transmission_bounces = s.cycles.max_bounces = 16
    dress(*parts, pet, mats["HudsonSpring"])
    scene.studio([bottle, cap])
    scene.frame([bottle, cap], yaw=-22, pitch=10, lens=85, fill=0.8, name="Hero")
    scene.render(f"{REND}/water_bottle.png", 1024, smp)
    cap.location.z += 0.03; cap.rotation_euler.z = 1.2; cap.rotation_euler.x = 0.35; cap.location.x += 0.012
    scene.camera((0.004, 0, 0.212), 0.17, -28, 20, 85, "Cap")
    scene.render(f"{REND}/water_bottle_cap_off.png", 1024, smp)
    cap.location.z -= 0.03; cap.rotation_euler = (0, 0, 0); cap.location.x -= 0.012
    dress(*parts, pet, mats["Refill"])
    scene.camera((0, 0, 0.13), 0.24, -18, 6, 85, "Refill")
    scene.render(f"{REND}/water_bottle_refill.png", 1024, smp)
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c)
    dups = []
    for i, k in enumerate(VARIANTS):
        trio = []
        for o in parts:
            d = o.copy(); d.data = o.data.copy()
            d.location.x += (i - (len(VARIANTS) - 1) / 2) * 0.078
            bpy.context.scene.collection.objects.link(d)
            trio.append(d)
        dress(*trio, pet, mats[k])
        dups += trio
    for o in parts:
        o.hide_render = True
    scene.frame(dups, yaw=0, pitch=8, lens=70, fill=0.8, name="Lineup", aspect=760 / 1600)
    scene.render(f"{REND}/water_bottle_variants.png", 1600, smp, res_y=760)
    for d in dups:
        bpy.data.objects.remove(d)
    for o in parts:
        o.hide_render = False


def main():
    bottle, liquid, cap = build_mesh()
    texs = T.build(TEXD, ao=bakes(bottle, cap))
    pet, mats = materials(texs)
    dress(bottle, liquid, cap, pet, mats["HudsonSpring"])
    ucx = scene.link(export.hull(bottle, "UCX_SM_WaterBottle_00", 64), scene.collection("Collision"))
    ucx.hide_render = True
    l1, l2 = export.lod(bottle, 0.5, "SM_WaterBottle_LOD1"), export.lod(bottle, 0.25, "SM_WaterBottle_LOD2")
    for o in (l1, l2):
        o.hide_render = True
    tri = {o.name: scene.tris(o) for o in (bottle, liquid, cap, l1, l2)}
    export.fbx([bottle, ucx], f"{FBXD}/SM_WaterBottle.fbx")
    export.fbx([liquid], f"{FBXD}/SM_WaterBottle_Liquid.fbx")
    export.fbx([cap], f"{FBXD}/SM_WaterBottle_Cap.fbx")
    export.fbx([l1], f"{FBXD}/SM_WaterBottle_LOD1.fbx")
    export.fbx([l2], f"{FBXD}/SM_WaterBottle_LOD2.fbx")
    l1.hide_set(True); l2.hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "water_bottle.blend"), relative_remap=True)
    renders(bottle, liquid, cap, pet, mats)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "WaterBottle", "replaces_rows": ["Water", "PlasticBottle", "UsedWaterBottle"], "category": "drinks",
        "clip": "A_Consume_Drink_TwistBottle", "dims_cm": [6.5, 6.5, 22.0], "real_object": "500 mL PET water bottle, 28 mm PCO neck finish",
        "pivot": "base centre, +Z up",
        "parts": {"SM_WaterBottle_Cap": {"pivot": "cap bottom centre on the neck axis", "rest_location_cm": [0, 0, 20.6],
                                         "opens": "about local Z: 2 pulls x 100 deg, lifts 0.5 cm (contract), stays in the left hand"},
                  "SM_WaterBottle_Liquid": {"pivot": "base centre", "note": "fill level 17.8 cm; scale Z or swap for drained states"}},
        "material_slots": {"SM_WaterBottle": ["M_Lab_Glass (PET)", "MI_WaterBottle_<V> (label band)"],
                           "SM_WaterBottle_Cap": ["MI_WaterBottle_<V> (layer A = cap colour)"],
                           "SM_WaterBottle_Liquid": ["M_Lab_Liquid → MI_WaterBottle_Liquid_<V>"]},
        "tris": tri, "textures": {"size": T.S, "normal": "OpenGL (flip green in UE)",
                                  "variants": {k: os.path.basename(mats[k]["label"]) for k in VARIANTS}},
        "variant_params": {k: mats[k]["params"] for k in VARIANTS}, "liquid_params": {k: mats[k]["liquid"] for k in VARIANTS},
        "glass_params": {"PET": {"Tint": [0.96, 0.98, 0.99], "Roughness": 0.06, "IOR": 1.08}},
        "collision": "UCX_SM_WaterBottle_00 (convex hull)", "lods": ["SM_WaterBottle_LOD1 (50%)", "SM_WaterBottle_LOD2 (25%)"]})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
