"""projpaint.py - PRIVATE look test: project the two profile photos (edge-filled copies in work/, made by fill_photos.py) onto
the model through the fitted cameras; optional save= exports a decimated .blend (images packed) or, with solid=1, a .blend +
.glb without photos. usage: .venv/bin/python guns/ak74m/projpaint.py views="60,15,70;-120,15,70" out=work/pp [save=x.blend ratio=0.3 solid=1]"""
import bpy, sys, json
import os; _H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H); sys.path.insert(0, os.path.join(_H, "../.."))
A = dict(a.split("=", 1) for a in sys.argv if "=" in a); sys.argv = ["x"]
import compare as C, ak_ref as R
from lab import scene
from mathutils import Vector
C.NPZ = A.get("npz", C.NPZ); C.load()
for n in A.get("hide", "").split(","):
    if n in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[n])
from PIL import Image
def mat():
    m = bpy.data.materials.new("proj"); m.use_nodes = True; nt = m.node_tree; N = nt.nodes; L = nt.links
    bsdf = N["Principled BSDF"]; bsdf.inputs["Roughness"].default_value = 0.55
    geo = N.new("ShaderNodeNewGeometry"); sep = N.new("ShaderNodeSeparateXYZ"); L.new(geo.outputs["Position"], sep.inputs[0])
    nsep = N.new("ShaderNodeSeparateXYZ"); L.new(geo.outputs["Normal"], nsep.inputs[0])
    def lin(src, a, b):                      # a*src + b
        m1 = N.new("ShaderNodeMath"); m1.operation = "MULTIPLY_ADD"; L.new(src, m1.inputs[0]); m1.inputs[1].default_value = a; m1.inputs[2].default_value = b
        return m1.outputs[0]
    texs = []
    FILLED = dict(L=os.path.join(_H, "work/proj_L_filled.png"), R=os.path.join(_H, "work/proj_R_filled.png"))
    cams = json.load(open(os.path.join(_H, A.get("cams", "review/m3_iou.json"))))
    for side, photo, xm, row, s, sg in (("L", R.PHOTO, R.X_MUZ, R.ROW_BORE, R.S_MM, 1), ("R", R.PHOTO_R, 1179.0, 287.0, 943 / 987, -1)):
        W, H = Image.open(photo).size; cm = cams[side]["camera"]; sc = cm["scale"]
        cx, cy = (W - 1) / 2, (H - 1) / 2
        # nominal camera px: x = xm + sg*1000*by/s, row = row - 1000*bz/s; photo px = c + (cam - c - t)/sc (fitted camera)
        ax, bx = sg * 1000 / s / sc, cx + (xm - cx - cm["tx_px"]) / sc
        ar, br = -1000 / s / sc, cy + (row - cy - cm["ty_px"]) / sc
        u = lin(sep.outputs[1], ax / W, bx / W)
        v = lin(sep.outputs[2], -ar / H, 1 - br / H)
        cmb = N.new("ShaderNodeCombineXYZ"); L.new(u, cmb.inputs[0]); L.new(v, cmb.inputs[1])
        t = N.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(FILLED[side]); t.extension = "EXTEND"; t.interpolation = "Cubic"
        L.new(cmb.outputs[0], t.inputs[0]); texs.append(t)
    gt = N.new("ShaderNodeMath"); gt.operation = "GREATER_THAN"; L.new(nsep.outputs[0], gt.inputs[0]); gt.inputs[1].default_value = 0.0
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"; L.new(gt.outputs[0], mix.inputs[0])
    L.new(texs[1].outputs[0], mix.inputs[6]); L.new(texs[0].outputs[0], mix.inputs[7])
    if A.get("emit"):
        em = N.new("ShaderNodeEmission"); L.new(mix.outputs[2], em.inputs[0]); L.new(em.outputs[0], N["Material Output"].inputs[0])
    # tops/bottoms (|n.x| small) never seen by a side photo: fade to a neutral gun tone
    ab = N.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"; L.new(nsep.outputs[0], ab.inputs[0])
    rmp = N.new("ShaderNodeMapRange"); L.new(ab.outputs[0], rmp.inputs[0]); rmp.inputs[1].default_value = 0.15; rmp.inputs[2].default_value = 0.6
    mix2 = N.new("ShaderNodeMix"); mix2.data_type = "RGBA"; L.new(rmp.outputs[0], mix2.inputs[0])
    mix2.inputs[6].default_value = (0.035, 0.035, 0.038, 1); L.new(mix.outputs[2], mix2.inputs[7])
    L.new(mix2.outputs[2], bsdf.inputs["Base Color"]); return m
M = mat()
objs = [o for o in bpy.data.objects if o.type == "MESH"]
for o in objs: o.data.materials.clear(); o.data.materials.append(M)
s = bpy.context.scene; w = s.world or bpy.data.worlds.new("W"); s.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.8, 0.82, 1); w.node_tree.nodes["Background"].inputs[1].default_value = 0.35
s.render.engine = "CYCLES"; s.cycles.samples = 32; s.cycles.use_denoising = True; s.view_settings.view_transform = "AgX"
lo, hi = scene.bounds(objs); c = (lo + hi) / 2; r = (hi - lo).length
for nm, off, e in (("key", (0.5, -0.4, 0.9), 22), ("key2", (-0.5, -0.4, 0.9), 22), ("rim", (0, 1, 0.5), 15)):
    ld = bpy.data.lights.new(nm, "AREA"); ld.size = r; ld.energy = e * r * r
    o = bpy.data.objects.new(nm, ld); s.collection.objects.link(o)
    o.location = c + Vector(off) * r; o.rotation_euler = (c - o.location).to_track_quat("-Z", "Y").to_euler()
focus = [o for o in objs if o.name in A.get("focus", "").split(",")] or objs
for i, v in enumerate(A["views"].split(";") if A.get("views") else []):
    yaw, pitch, lens = map(float, v.split(","))
    cm = scene.frame(focus, yaw=yaw, pitch=pitch, lens=lens, fill=float(A.get("fill", 0.92)), name=f"c{i}", aspect=0.56)
    s.camera = cm; s.render.resolution_x = 1600; s.render.resolution_y = 896
    s.render.filepath = f"{A['out']}_{i}.png"; bpy.ops.render.render(write_still=True)

if A.get("save"):
    for o in objs:                                              # viewing copy: decimate the dense SDF meshes
        md = o.modifiers.new("dec", "DECIMATE"); md.ratio = float(A.get("ratio", 0.08))
        bpy.context.view_layer.objects.active = o; bpy.ops.object.modifier_apply(modifier="dec")
    if A.get("solid"):
        for o in objs: o.data.materials.clear()
        bpy.ops.export_scene.gltf(filepath=A["save"].replace(".blend", ".glb"), export_format="GLB")
    else:
        for im in bpy.data.images: im.pack()
    bpy.ops.wm.save_as_mainfile(filepath=A["save"], compress=True)
    print("SAVED", A["save"], sum(len(o.data.polygons) for o in objs), "tris")
