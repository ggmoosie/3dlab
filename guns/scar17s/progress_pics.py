"""progress_pics.py - quick progress renders of the current high-poly (preview materials) at the user's reference views:
the solved low 3/4 hero camera next to the reference, plus close-ups. usage: .venv/bin/python guns/scar17s/progress_pics.py"""
import bpy, os, sys, json, math, runpy
import numpy as np
from mathutils import Vector, Euler
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "../..")
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from lab import scene
import bl_high

OUT = os.path.join(HERE, "work", "progress"); os.makedirs(OUT, exist_ok=True)
SMP = int(dict(a.split("=") for a in sys.argv[1:] if "=" in a).get("samples", 24))

scene.reset()
objs = bl_high.load()
objs["round"].hide_render = True
allo = [o for n, o in objs.items() if n != "round"]
# hero camera from the landmark solve
m = json.load(open(os.path.join(HERE, "work", "match_hero34.json")))
m["render"] = dict(out=os.path.join(OUT, "_cam.png"), samples=1, studio=False, transparent=True)
p = os.path.join(OUT, "_m.json"); json.dump(m, open(p, "w"))
sys.argv = ["blender", "--", p]
try:
    runpy.run_path(os.path.join(ROOT, "tools", "camera_match.py"), run_name="__main__")
except SystemExit:
    pass
cam = bpy.data.objects[m["camera"]]
s = bpy.context.scene
w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.47, 0.43, 0.39, 1)
lo, hi = scene.bounds(allo); c = (lo + hi) / 2; r = (hi - lo).length
for nm, off, rot, sz, e in (("key", (-0.6, -0.9, 1.1), (40, 0, -30), 1.2, 380), ("fill", (1.2, -0.3, 0.2), (85, 0, 75), 1.5, 120),
                            ("rim", (0.3, 1.2, 0.9), (-40, 0, 165), 0.8, 260)):
    ld = bpy.data.lights.new(nm, "AREA"); ld.size = r * sz; ld.energy = e * r * r
    o = bpy.data.objects.new(nm, ld); s.collection.objects.link(o)
    o.location = c + Vector(off) * r; o.rotation_euler = Euler([math.radians(v) for v in rot])


def shot(cam, path, w, h):
    s.camera = cam
    s.render.engine = "CYCLES"; s.cycles.samples = SMP; s.cycles.use_denoising = True
    s.render.resolution_x, s.render.resolution_y = w, h; s.render.film_transparent = False
    s.view_settings.view_transform = "AgX"; s.view_settings.look = "AgX - Base Contrast"
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)


shot(cam, os.path.join(OUT, "hero34_model.png"), 900, 600)
for nm, tgt, yaw, pitch, lens in (("c_receiver", ["upper", "upperhw", "lower", "controls", "screws_u", "screws_l"], -90, 4, 100),
                                  ("c_grip", ["grip", "lower", "controls", "trigger"], -70, 6, 90),
                                  ("c_stock", ["stock", "buttpad", "screws_s"], -75, 8, 85),
                                  ("c_front", ["upper", "upperhw", "brake", "barrel", "fsight", "vgrip"], -125, 12, 70)):
    sel = [objs[t] for t in tgt if t in objs]
    if nm == "c_receiver":
        sel = [objs["upperhw"], objs["controls"]]
    if nm == "c_front":
        sel = [objs["fsight"], objs["brake"]]
    cm = scene.frame(sel, yaw=yaw, pitch=pitch, lens=lens, fill=0.9, name=nm, aspect=0.6)
    shot(cm, os.path.join(OUT, nm + ".png"), 1200, 720)
print("PROGRESS_DONE")
