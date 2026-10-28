"""render.py - showcase shots of a softgoods item on a clay torso proxy (proxy is render-only, never exported)."""
import os, math
import bpy
from mathutils import Vector
from lab import scene

SHOTS = {  # name: (yaw, pitch, lens, fill, focus point offset m, focus size m)
    "front34": (-30, 10, 60, 0.85, None, None),
    "back34": (150, 14, 60, 0.85, None, None),
    "side": (-90, 5, 60, 0.85, None, None),
    "close_front": (-15, 8, 85, 0.9, (0.0, -0.13, 1.21), 0.22),
    "close_strap": (-40, 30, 85, 0.9, (0.10, -0.03, 1.52), 0.16),
    "close_back": (165, 20, 85, 0.9, (0.0, 0.14, 1.40), 0.2),
}


def proxy():
    m = bpy.data.materials.new("Clay"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (0.32, 0.31, 0.30, 1); b.inputs["Roughness"].default_value = 0.7
    out = []
    for kind, loc, scl in (("cyl", (0, 0.01, 1.18), (0.158, 0.108, 0.26)), ("sph", (0, 0.01, 1.40), (0.17, 0.11, 0.105)),
                           ("cyl", (0, 0.01, 1.56), (0.055, 0.055, 0.09)), ("sph", (0, 0.0, 1.72), (0.09, 0.105, 0.115))):
        if kind == "cyl":
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, location=loc)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=loc)
        o = bpy.context.active_object; o.scale = scl; o.data.materials.append(m)
        bpy.ops.object.shade_smooth(); out.append(o)
    return out


def shots(item, outdir, spp=64, names=None):
    os.makedirs(outdir, exist_ok=True)
    px = proxy()
    scene.studio([item] + px, backdrop=True)
    for name in names or SHOTS:
        yaw, pitch, lens, fill, fp, fs = SHOTS[name]
        if fp is None:
            cam = scene.frame([item], yaw=yaw, pitch=pitch, lens=lens, fill=fill, name=f"c_{name}", aspect=0.75)
        else:
            dist = fs / 2 / math.tan(math.atan(18 / lens)) / fill
            cam = scene.camera(Vector(fp), dist, yaw, pitch, lens, name=f"c_{name}")
        scene.render(os.path.join(outdir, f"{name}.png"), res=1400, res_y=1050, samples=spp)
        print("SHOT", name)
