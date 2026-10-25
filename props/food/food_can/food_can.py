"""FoodCan (15 oz, tear-back lid + pull ring): builds everything.  .venv/bin/python props/food/food_can/food_can.py [--fast]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from lab import scene, master, bake, export, tex
import food_can_geo as G, food_can_tex as T, food_can_labels as LB

FAST = "--fast" in sys.argv
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/food_can"))
VARIANTS = list(LB.VARIANTS)
PEEL, LIFT = -2.05, 0.9


def build_mesh():
    scene.reset()
    col = scene.collection("Game")
    P = dict(body=G.body(), lid=G.lid(), ring=G.ring())
    for o in P.values():
        scene.link(o, col)
    return P


def bakes(P):
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (2048, 32)
    maps = {k: bake.bake(o, [], "AO", n, f"{WORK}/ao_{k}.png", samples=smp) for k, o in P.items()}
    if n != T.S:
        from PIL import Image
        maps = {k: np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((T.S, T.S)), np.float32) / 255 for k, a in maps.items()}
    u, v = tex.uv_grid(T.S)
    lid = tex.in_rect(u, v, G.UV_LID) | tex.in_rect(u, v, G.UV_LIDSIDE) | tex.in_rect(u, v, (0.57, 0.42, 0.79, 0.50))
    ao = np.where(tex.in_rect(u, v, G.UV_RING), maps["ring"], np.where(lid, maps["lid"], maps["body"]))
    return np.where(ao < 0.02, 1.0, ao)


def materials(texs):
    mats = {}
    for k in VARIANTS:
        L, F, prm = LB.VARIANTS[k]()
        lp = tex.save(LB.to_uv(L, F, len(k)), os.path.join(TEXD, f"T_FoodCan_Label_{k}.png"), "RGBA")
        prm = dict(prm, WearAmount=0.75 if k == "Mystery" else 0.45, Dirt=0.6 if k == "Mystery" else 0.3)
        mi = master.material(f"MI_FoodCan_{k}", texs, prm, label=lp)
        mi.use_fake_user = True
        mats[k] = dict(mi=mi, params=prm, label=lp)
    return mats


def dress(P, m):
    for o in P.values():
        if not o.data.materials:
            o.data.materials.append(m["mi"])
        o.data.materials[0] = m["mi"]


def opened(P, on=True):
    """Peeled pose: the ring rides on the lid (parented while both are at rest), lid rotates about its hinge."""
    lid, ring = P["lid"], P["ring"]
    if on:
        bpy.context.view_layer.update()
        ring.parent = lid
        ring.matrix_parent_inverse = lid.matrix_world.inverted()
        lid.rotation_euler.x, ring.rotation_euler.x = PEEL, LIFT
    else:
        lid.rotation_euler.x = ring.rotation_euler.x = 0.0
        ring.parent = None


def renders(P, mats):
    objs = list(P.values())
    smp = 24 if FAST else 96
    dress(P, mats["BaysideBeans"])
    scene.studio(objs)
    scene.frame(objs, yaw=-28, pitch=14, lens=85, fill=0.8, name="Closed")
    scene.render(f"{REND}/food_can_closed.png", 1024, smp)
    opened(P, True)
    scene.frame(objs, yaw=-20, pitch=38, lens=85, fill=0.8, name="Open")
    scene.render(f"{REND}/food_can.png", 1024, smp)
    opened(P, False)
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c)
    dups = []
    for i, k in enumerate(VARIANTS):
        Q = {}
        for key, o in P.items():
            if key != "body":
                continue
            d = o.copy(); d.data = o.data.copy()
            d.location.x += (i - (len(VARIANTS) - 1) / 2) * 0.092
            bpy.context.scene.collection.objects.link(d)
            Q[key] = d
        dress(Q, mats[k])
        dups += list(Q.values())
    for o in objs:
        o.hide_render = True
    scene.frame(dups, yaw=0, pitch=30, lens=70, fill=0.82, name="Lineup", aspect=760 / 1600)
    scene.render(f"{REND}/food_can_variants.png", 1600, smp, res_y=760)
    for d in dups:
        bpy.data.objects.remove(d)
    for o in objs:
        o.hide_render = False


def main():
    P = build_mesh()
    texs = T.build(TEXD, ao=bakes(P))
    mats = materials(texs)
    dress(P, mats["BaysideBeans"])
    ucx = scene.link(export.hull(P["body"], "UCX_SM_FoodCan_00", 64), scene.collection("Collision"))
    ucx.hide_render = True
    tri = {o.name: scene.tris(o) for o in P.values()}
    export.fbx([P["body"], ucx], f"{FBXD}/SM_FoodCan.fbx")
    export.fbx([P["lid"]], f"{FBXD}/SM_FoodCan_Lid.fbx")
    export.fbx([P["ring"]], f"{FBXD}/SM_FoodCan_Ring.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "food_can.blend"), relative_remap=True)
    renders(P, mats)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "FoodCan", "replaces_rows": ["CannedBeans", "Meatballs", "TunaCan", "Sardines", "Sprats"], "category": "food",
        "clip": "A_Consume_Eat_CanSpoon", "dims_cm": [7.8, 7.8, 11.2], "real_object": "15 oz (300x407) three-piece food can, full-aperture easy-open end",
        "pivot": "base centre, +Z up (left hand)",
        "parts": {"SM_FoodCan_Lid": {"pivot": "hinge on the rim at +Y (UE -Y), axis = local X", "rest_location_cm_ue": [0, -3.66, 10.66],
                                     "opens": "lift ring first, then peel: rotate about local X to -115 deg"},
                  "SM_FoodCan_Ring": {"pivot": "rivet", "rest_location_cm_ue": [0, 2.4, 10.71], "opens": "about local X +50 deg; rides on the lid"}},
        "tris": tri, "textures": {"size": T.S, "normal": "OpenGL (flip green in UE)",
                                  "variants": {k: os.path.basename(mats[k]["label"]) for k in VARIANTS}},
        "variant_params": {k: mats[k]["params"] for k in VARIANTS},
        "notes": "the contents surface is printed per variant (food art) over a shared chunky relief; spoon is its own prop (props/food/spoon)",
        "collision": "UCX_SM_FoodCan_00"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
