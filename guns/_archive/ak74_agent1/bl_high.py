"""bl_high.py - load work/high.npz (B frame mm) into Blender as HI_<part> objects in the game frame, preview materials.
B -> G (origin = grip top) -> Blender metres: Blender = (x, -(y - oy), z - oz) / 1000 (UE x, y, z = Blender x, -y, z after
the FBX axis export). The Y mirror flips the winding, which is restored. load(path, names) -> {name: obj}"""
import bpy, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import ak_spec as P

# preview / ID colours per material id: park (phosphate steel), poly, steel (blued), bright (polished carrier), rubber,
# brass, magpoly, paint
PREVIEW = {0: ((0.035, 0.035, 0.037), 0.55, 0.6), 1: ((0.045, 0.045, 0.046), 0.62, 0.0), 2: ((0.03, 0.03, 0.033), 0.4, 0.9),
           3: ((0.35, 0.35, 0.36), 0.3, 1.0), 4: ((0.02, 0.02, 0.02), 0.85, 0.0), 5: ((0.75, 0.48, 0.20), 0.3, 1.0),
           6: ((0.05, 0.05, 0.052), 0.55, 0.0), 7: ((0.8, 0.8, 0.8), 0.6, 0.0)}
ASSEMBLY = {"receiver": "Receiver", "rivets_rec": "Receiver", "rsb": "Receiver", "hinge": "Receiver", "siderail": "Receiver",
            "trigger": "Trigger", "guard": "Receiver", "magcatch": "Receiver", "dustcover": "DustCover", "recoil": "Recoil", "bolt": "Carrier",
            "leaf": "Receiver", "slider": "Receiver", "upperhg": "GasTube", "hg_rings_u": "GasTube", "hg_ring_l": "LowerHG", "lowerhg": "LowerHG",
            "gastube": "GasTube", "gasblock": "Barrel", "barrel": "Barrel", "rod": "Rod", "fsb": "Barrel", "rivets_front": "Barrel",
            "brake": "Muzzle", "carrier": "Carrier", "selector": "Selector", "grip": "Grip", "magazine": "Mag",
            "stock": "Stock", "buttplate": "Stock", "slings": "Stock"}


def b2bl(V):
    o = np.array(P.ORIGIN_B)
    W = V - o
    return np.c_[W[:, 0], -W[:, 1], W[:, 2]] * 0.001


def bl2b(v):
    o = np.array(P.ORIGIN_B)
    return np.array([v[0] * 1000 + o[0], -v[1] * 1000 + o[1], v[2] * 1000 + o[2]])


def mats():
    out = {}
    for k, (c, r, m) in PREVIEW.items():
        mt = bpy.data.materials.get(f"PV_{k}") or bpy.data.materials.new(f"PV_{k}")
        mt.use_nodes = True
        b = mt.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*c, 1); b.inputs["Roughness"].default_value = r; b.inputs["Metallic"].default_value = m
        mt.diffuse_color = (*c, 1)
        out[k] = mt
    return out


def load(path=None, names=None, skip=()):
    path = path or os.path.join(HERE, "work", "high.npz")
    D = np.load(path)
    M = mats()
    objs = {}
    for n in sorted({k.split("__")[0] for k in D.files}):
        if (names and n not in names) or n in skip:
            continue
        V = D[n + "__V"].astype(np.float64)
        F = D[n + "__F"].astype(np.int64)[:, ::-1]
        mat = D[n + "__a_mat"] if (n + "__a_mat") in D.files else np.zeros(len(V))
        me = bpy.data.meshes.new("HI_" + n)
        me.vertices.add(len(V)); me.vertices.foreach_set("co", b2bl(V).ravel())
        me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
        me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3)); me.polygons.foreach_set("loop_total", np.full(len(F), 3))
        me.update(calc_edges=True); me.validate(clean_customdata=False)
        me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
        ids = sorted(set(np.round(mat).astype(int)))
        for i in ids:
            me.materials.append(M[i])
        fm = np.round(mat[F[:, 0]]).astype(int)
        me.polygons.foreach_set("material_index", np.array([ids.index(i) for i in fm], np.int32))
        o = bpy.data.objects.new("HI_" + n, me)
        bpy.context.scene.collection.objects.link(o)
        o["assembly"] = ASSEMBLY.get(n, "Misc")
        objs[n] = o
    return objs
