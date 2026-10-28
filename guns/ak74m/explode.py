"""explode.py - exploded modular view (offsets per part in B mm). usage: .venv/bin/python guns/ak74m/explode.py views="70,18,70" out=work/ex [k=1.0]"""
import bpy, sys
import os; _H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H); sys.path.insert(0, os.path.join(_H, "../.."))
A = dict(a.split("=", 1) for a in sys.argv if "=" in a); sys.argv = ["x"]
import compare as C
from lab import scene
from mathutils import Vector
C.load()
k = float(A.get("k", 1.0))
OFF = dict(compensator=(0, 70, 0), rod=(0, 30, -45), hg_lower=(0, 0, -65), hg_cap=(0, 25, -65), gastube=(0, 45, 70),
           hg_upper=(0, 45, 70), dcover=(0, 0, 215), carrier=(0, 0, 125), bolt=(0, 95, 125), recoil=(0, -30, 170),
           hammer=(70, 0, -40), trigger=(70, 0, -40), pins=(110, 0, -40), magcatch=(70, 0, -40), tguard=(40, 0, -55),
           selector=(-70, 0, 0), mag=(0, 20, -95), grip=(0, -25, -75), stock=(0, -90, 0), sling_loop=(0, -90, 0))
for o in bpy.data.objects:
    if o.type == "MESH" and o.name in OFF:
        x, y, z = OFF[o.name]; o.location = Vector((x, -y, z)) * 0.001 * k
s = bpy.context.scene
w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.56, 0.58, 1)
s.render.engine = "CYCLES"; s.cycles.samples = int(A.get("spp", 24)); s.cycles.use_denoising = True; s.view_settings.view_transform = "AgX"
objs = [o for o in bpy.data.objects if o.type == "MESH"]
bpy.context.view_layer.update()
lo, hi = scene.bounds(objs); c = (lo + hi) / 2; r = (hi - lo).length
for nm, off, e in (("key", (0.4, -0.6, 0.9), 600), ("fill", (-0.8, 0.3, 0.3), 200), ("rim", (0, 1, 0.6), 300)):
    ld = bpy.data.lights.new(nm, "AREA"); ld.size = r * 0.8; ld.energy = e * r * r
    o = bpy.data.objects.new(nm, ld); s.collection.objects.link(o)
    o.location = c + Vector(off) * r; o.rotation_euler = (c - o.location).to_track_quat("-Z", "Y").to_euler()
for i, v in enumerate(A["views"].split(";")):
    yaw, pitch, lens = map(float, v.split(","))
    cm = scene.frame(objs, yaw=yaw, pitch=pitch, lens=lens, fill=0.92, name=f"c{i}", aspect=0.56)
    s.camera = cm; s.render.resolution_x = int(A.get("w", 1600)); s.render.resolution_y = int(s.render.resolution_x * 0.56)
    s.render.filepath = f"{A['out']}_{i}.png"; bpy.ops.render.render(write_still=True)
