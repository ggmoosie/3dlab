"""bl_high.py - load work/high.npz (B frame mm) into Blender as HI_<part> objects in the game frame, preview materials.
B -> G (origin = grip top) -> Blender metres with UE (x, y, z) = Blender (x, -y, z): Blender = (x, -(y - oy), z - oz) / 1000.
The Y mirror flips the winding, which is restored. usage (module): load(path) -> {name: obj}; preview(names) renders."""
import bpy, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import scar_spec as P

PREVIEW = {0: ((0.40, 0.31, 0.19), 0.45, 0.0), 1: ((0.34, 0.27, 0.17), 0.65, 0.0), 2: ((0.03, 0.03, 0.03), 0.38, 1.0),
           3: ((0.05, 0.05, 0.05), 0.55, 0.6), 4: ((0.02, 0.02, 0.02), 0.8, 0.0), 5: ((0.75, 0.48, 0.20), 0.3, 1.0),
           6: ((0.025, 0.025, 0.027), 0.5, 0.4), 7: ((0.72, 0.36, 0.22), 0.3, 1.0)}
ASSEMBLY = {"upper": "Upper", "upperhw": "Upper", "rsight": "Upper", "fsight": "Upper", "barrel": "Barrel", "brake": "Muzzle",
            "bolt": "Bolt", "lower": "Lower", "trigger": "Lower", "controls": "Lower", "pins": "Lower", "grip": "Grip",
            "vgrip": "VGrip", "magazine": "Magazine", "round": "Round", "stock": "Stock", "buttpad": "Stock"}


def b2bl(V):
    """B frame mm -> Blender metres"""
    o = np.array(P.ORIGIN_B)
    W = V - o
    return np.c_[W[:, 0], -W[:, 1], W[:, 2]] * 0.001


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


def load(path=None, names=None):
    path = path or os.path.join(HERE, "work", "high.npz")
    D = np.load(path)
    M = mats()
    objs = {}
    for n in sorted({k.split("__")[0] for k in D.files}):
        if names and n not in names:
            continue
        V = D[n + "__V"].astype(np.float64)
        F = D[n + "__F"].astype(np.int64)[:, ::-1]
        if n == "round":                                    # top round in the magazine lips (round frame: head at y 0)
            V = V + np.array([2.5, P.MAG_Y[0] + 1.5, P.MAG_TOP - 5.0])
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
