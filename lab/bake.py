"""High→low baking (bpy, Cycles CPU). Outputs float arrays (via lab.tex.load) or PNGs.

bake(low, highs, kind, size, path)
  kind: "NORMAL" (tangent, OpenGL), "AO", "CURV" (convexity 0..1, 0.5 = flat, from the high's pointiness),
        "ID" (each high material's viewport colour → emission), "POS_Z" (normalised height, for gradients)
Lesson (blender-3d log): the low must stay renderable but invisible to rays, or AO darkens.
"""
import bpy, os
from . import tex

BAKE_MAT = "__bake_target"


def _target(low, img):
    """Every material slot on the low gets an active image node pointing at img (Cycles bakes into the active node)."""
    if not low.data.materials:
        low.data.materials.append(bpy.data.materials.new(BAKE_MAT))
    for m in low.data.materials:
        m.use_nodes = True
        n = m.node_tree.nodes.get("__bake") or m.node_tree.nodes.new("ShaderNodeTexImage")
        n.name = "__bake"
        n.image = img
        m.node_tree.nodes.active = n


def _swap_emit(objs, shader):
    """Temporarily give each high a pure-emission material built by shader(nodes, links, mat) → socket."""
    saved = {}
    for o in objs:
        saved[o.name] = [s.material for s in o.material_slots]
        mats = o.data.materials
        src = list(mats) or [None]
        new = []
        for sm in src:
            m = bpy.data.materials.new("__emit")
            m.use_nodes = True
            N, L = m.node_tree.nodes, m.node_tree.links
            N.clear()
            out, em = N.new("ShaderNodeOutputMaterial"), N.new("ShaderNodeEmission")
            L.new(shader(N, L, sm), em.inputs[0])
            L.new(em.outputs[0], out.inputs[0])
            new.append(m)
        if not o.material_slots:
            o.data.materials.append(new[0])
        else:
            for s, m in zip(o.material_slots, new):
                s.material = m
    return saved


def _restore(objs, saved):
    for o in objs:
        for s, m in zip(o.material_slots, saved[o.name]):
            s.material = m


def bake(low, highs, kind, size, path, samples=32, extrusion=0.0006, max_ray=0.0016, margin=16):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = samples
    img = bpy.data.images.new(os.path.basename(path), size, size, alpha=False, float_buffer=(kind == "NORMAL"))
    img.colorspace_settings.name = "Non-Color"
    _target(low, img)
    vis = {}
    for a in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter", "visible_shadow"):
        vis[a] = getattr(low, a)
        if highs:
            setattr(low, a, False)
    saved = None
    if kind == "CURV":
        def sh(N, L, m):
            g, r = N.new("ShaderNodeNewGeometry"), N.new("ShaderNodeMapRange")
            r.inputs[1].default_value, r.inputs[2].default_value = 0.42, 0.58
            L.new(g.outputs["Pointiness"], r.inputs[0])
            return r.outputs[0]
        saved = _swap_emit(highs, sh)
    elif kind == "ID":
        def sh(N, L, m):
            rgb = N.new("ShaderNodeRGB")
            rgb.outputs[0].default_value = tuple(m.diffuse_color) if m else (0, 0, 0, 1)
            return rgb.outputs[0]
        saved = _swap_emit(highs, sh)
    bpy.ops.object.select_all(action="DESELECT")
    for h in highs:
        h.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    s.render.bake.use_selected_to_active = bool(highs)
    s.render.bake.cage_extrusion = extrusion
    s.render.bake.max_ray_distance = max_ray
    s.render.bake.margin = margin
    s.render.bake.margin_type = "EXTEND"
    s.render.bake.use_clear = True
    btype = {"NORMAL": "NORMAL", "AO": "AO", "CURV": "EMIT", "ID": "EMIT"}[kind]
    if kind == "NORMAL":
        s.render.bake.normal_space = "TANGENT"
    bpy.ops.object.bake(type=btype)
    if saved:
        _restore(highs, saved)
    for a, v in vis.items():
        setattr(low, a, v)
    img.filepath_raw = path
    img.file_format = "PNG"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save()
    import numpy as np
    a = np.array(img.pixels[:], np.float32).reshape(size, size, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    return a if kind in ("NORMAL", "ID") else a[..., 0]
