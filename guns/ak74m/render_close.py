"""render_close.py - matcap-free studio close-ups of chosen parts. usage: .venv/bin/python guns/ak74m/render_close.py focus=receiver,dcover views="yaw,pitch,lens;..." out=work/x [hide=dcover npz= spp= w= fill=]"""
import bpy, sys
import os; _H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H); sys.path.insert(0, os.path.join(_H, "../.."))
A = dict(a.split("=", 1) for a in sys.argv if "=" in a)
sys.argv = ["x"]
import compare as C
from lab import scene
from mathutils import Vector
C.NPZ = A.get("npz", C.NPZ)
C.load()
for _n in A.get("hide", "").split(","):
    if _n in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[_n])
objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
s = bpy.context.scene
w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.56, 0.58, 1)
s.render.engine = "CYCLES"; s.cycles.samples = int(A.get("spp", 24)); s.cycles.use_denoising = True; s.view_settings.view_transform = "AgX"
lo, hi = scene.bounds(list(objs.values())); c = (lo + hi) / 2; r = (hi - lo).length
for nm, off, e in (("key", (0.4, -0.6, 0.9), 600), ("fill", (-0.8, 0.3, 0.3), 200), ("rim", (0, 1, 0.6), 300)):
    ld = bpy.data.lights.new(nm, "AREA"); ld.size = r * 0.8; ld.energy = e * r * r
    o = bpy.data.objects.new(nm, ld); s.collection.objects.link(o)
    o.location = c + Vector(off) * r; o.rotation_euler = (c - o.location).to_track_quat("-Z", "Y").to_euler()
focus = [objs[n] for n in A["focus"].split(",") if n in objs]
for i, v in enumerate(A["views"].split(";")):
    yaw, pitch, lens = map(float, v.split(","))
    cm = scene.frame(focus, yaw=yaw, pitch=pitch, lens=lens, fill=float(A.get("fill", 0.8)), name=f"c{i}", aspect=float(A.get("aspect", 0.66)))
    s.camera = cm; s.render.resolution_x = int(A.get("w", 1200)); s.render.resolution_y = int(s.render.resolution_x * float(A.get("aspect", 0.66)))
    s.render.filepath = f"{A['out']}_{i}.png"; bpy.ops.render.render(write_still=True)
