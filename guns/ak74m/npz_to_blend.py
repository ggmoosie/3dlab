"""npz_to_blend.py - the high-poly (work/high.npz, every part at full SDF resolution) -> .blend + .glb, one object per part,
named as in parts.py, positioned in the shared gun frame (Blender m: x gun-left, -y muzzle, z up). No photos involved.
usage: .venv/bin/python guns/ak74m/npz_to_blend.py out=work/AK74M_high.blend [npz=work/high.npz] [ratio=1.0] [glb=1]"""
import bpy, os, sys
import numpy as np                                     # numpy only: runs in Blender's own python (blender -b -P ... -- out=..)
HERE = os.path.dirname(os.path.abspath(__file__))
A = dict(a.split("=", 1) for a in sys.argv if "=" in a)
MATS = {0: ("Steel", (0.04, 0.04, 0.045), 0.6, 0.45), 1: ("Polymer", (0.03, 0.03, 0.03), 0.0, 0.6), 2: ("Parkerized", (0.05, 0.05, 0.05), 0.6, 0.5)}
bpy.ops.wm.read_factory_settings(use_empty=True)
D = np.load(A.get("npz", os.path.join(HERE, "work", "high.npz")))
mats = {}
for k, (nm, col, met, rough) in MATS.items():
    m = bpy.data.materials.new(nm); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Metallic"].default_value = met; b.inputs["Roughness"].default_value = rough
    mats[k] = m
for n in sorted({k.split("__")[0] for k in D.files}):
    V = D[n + "__V"].astype(np.float64); F = D[n + "__F"].astype(np.int64)[:, ::-1]
    V = np.c_[V[:, 0], -V[:, 1], V[:, 2]] * 0.001                                      # B mm -> Blender m
    me = bpy.data.meshes.new(n); me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3)); me.polygons.foreach_set("loop_total", np.full(len(F), 3))
    me.update(calc_edges=True); me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    k = int(np.round(D[n + "__a_mat"][0])) if (n + "__a_mat") in D.files else 0
    me.materials.append(mats[k])
    bpy.context.scene.collection.objects.link(bpy.data.objects.new(n, me))
objs = [o for o in bpy.data.objects if o.type == "MESH"]
r = float(A.get("ratio", 1.0))
for o in objs:
    if r < 1.0:
        m = o.modifiers.new("dec", "DECIMATE"); m.ratio = r
        bpy.context.view_layer.objects.active = o; bpy.ops.object.modifier_apply(modifier="dec")
out = os.path.abspath(A.get("out", os.path.join(HERE, "work", "AK74M_high.blend")))
if A.get("glb", "1") == "1":
    bpy.ops.export_scene.gltf(filepath=out.replace(".blend", ".glb"), export_format="GLB")
bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
print("SAVED", out, sum(len(o.data.polygons) for o in objs), "tris")
