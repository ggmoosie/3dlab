"""Game export (bpy): UCX hulls, LODs, FBX with the model-factory S9 settings, clean re-import check, manifest."""
import bpy, bmesh, json, os
from mathutils import Vector

FBX = dict(use_selection=True, apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE", axis_forward="-Y",
           axis_up="Z", object_types={"MESH"}, path_mode="STRIP", bake_anim=False, mesh_smooth_type="FACE",
           use_tspace=True, use_mesh_modifiers=True, add_leaf_bones=False, use_custom_props=False)


def hull(src, name, max_verts=96):
    """Convex hull of src's evaluated verts (world space → src local), reduced to ~max_verts extreme points."""
    import math
    dg = bpy.context.evaluated_depsgraph_get()
    m = src.evaluated_get(dg).to_mesh()
    pts = [v.co.copy() for v in m.vertices]
    src.evaluated_get(dg).to_mesh_clear()
    if len(pts) > max_verts:                       # extreme points along Fibonacci directions (fix_ucx.py)
        keep, n = set(), max_verts
        for k in range(n):
            z = 1 - 2 * (k + 0.5) / n
            r = math.sqrt(1 - z * z)
            a = k * math.pi * (3 - math.sqrt(5))
            d = Vector((r * math.cos(a), r * math.sin(a), z))
            keep.add(max(range(len(pts)), key=lambda i: pts[i].dot(d)))
        pts = [pts[i] for i in keep]
    bm = bmesh.new()
    for p in pts:
        bm.verts.new(p)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    for c in src.users_collection:
        c.objects.link(o)
    o.matrix_world = src.matrix_world.copy()
    o.display_type = "WIRE"
    o.hide_render = True
    return o


def lod(src, ratio, name):
    o = src.copy()
    o.data = src.data.copy()
    o.name = name
    for c in src.users_collection:
        c.objects.link(o)
    d = o.modifiers.new("Decimate", "DECIMATE")
    d.ratio = ratio
    d.use_collapse_triangulate = True
    bpy.context.view_layer.objects.active = o
    for m in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    return o


def fbx(objs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.fbx(filepath=path, **FBX)
    return path


def reimport_check(paths, expect):
    """Load each FBX into an empty scene; check object names, dims (cm), material slots. expect: {name: dims_cm}"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    report, ok = {}, True
    for p in paths:
        bpy.ops.import_scene.fbx(filepath=p)
    for name, dims in expect.items():
        o = bpy.data.objects.get(name)
        if not o:
            report[name] = "MISSING"
            ok = False
            continue
        d = [round(x * 100, 2) for x in o.dimensions]
        good = all(abs(a - b) <= max(0.05, 0.01 * b) for a, b in zip(d, dims))
        ok &= good
        report[name] = {"dims_cm": d, "expected": dims, "ok": good, "materials": [s.material.name for s in o.material_slots if s.material],
                        "ucx": sorted(x.name for x in bpy.data.objects if x.name.startswith("UCX_" + name + "_"))}
    print("REIMPORT", "PASS" if ok else "FAIL", json.dumps(report))
    return ok, report


def manifest(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
