"""5 mL syringe (bpy). Contract Syringe/Inject: ~12 cm, pivot = barrel centre, +Z toward the needle; needle cap pulls
4.5 cm along +Z; plunger travel 3.8 cm (rear thumb pad).
Parts: SM_Syringe (hollow clear barrel + finger flanges + hub + needle + paper drug label decal)
       SM_Syringe_Plunger (rod, thumb pad, rubber stopper; pivot = thumb pad centre)
       SM_Syringe_Cap (needle cap; pivot = its open end)   SM_Syringe_Liquid (pivot = front of the bore)"""
import bpy, bmesh, math, os, sys
from mathutils import Matrix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
RO, RI = 6.6, 5.7                 # barrel outer / bore radius (mm)
Z0, Z1 = -30.0, 30.0              # barrel rear rim / front shoulder (barrel centre at 0)
STOP_Z = -8.0                     # stopper front face at rest (liquid from here to the front)
UV_BARREL = (0.0, 0.62, 0.5, 0.99)
UV_LABEL = (0.0, 0.30, 0.5, 0.58)
UV_PLUNGER = (0.52, 0.30, 0.99, 0.99)
UV_CAP = (0.0, 0.02, 0.25, 0.27)
UV_HUB = (0.27, 0.02, 0.48, 0.27)
UV_LIQ = (0.52, 0.02, 0.99, 0.27)


def _lathe(name, pts, rect, segs=32, close=False):
    sc = [(r * MM, z * MM) for r, z in pts]
    o = geo.lathe(name, [dict(pts=sc, uv=rect)], segs=segs)
    if close:                                          # profile loops back to its start: weld the seam ring
        bm = bmesh.new(); bm.from_mesh(o.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        bm.to_mesh(o.data); bm.free()
    return o


def _box(name, size, loc, rect):
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object
    o.name = name
    o.scale = [s * MM for s in size]
    o.location = [l * MM for l in loc]
    bpy.ops.object.transform_apply(scale=True, location=True)
    b = o.modifiers.new("B", "BEVEL"); b.width = 0.6 * MM; b.segments = 2
    geo.apply_mods(o)
    geo.unwrap_into(o, rect)
    return o


def barrel():
    """Hollow tube (1 mm wall) with a closed rear rim, front shoulder, luer tip; + flanges, hub, needle, label."""
    prof = [(RI, Z0), (RO, Z0), (RO, Z1 - 1.0), (RO - 0.6, Z1), (3.0, Z1 + 1.5), (2.0, Z1 + 2.5), (1.9, Z1 + 9.0),
            (0.5, Z1 + 9.0), (0.5, Z1 + 2.0), (RI - 0.8, Z1 - 0.8), (RI, Z1 - 2.0), (RI, Z0)]
    tube = _lathe("tube", prof, UV_BARREL, 36, close=True)
    wings = _box("wings", (34, 4.2, 1.6), (0, 0, Z0 + 1.2), (0.40, 0.62, 0.5, 0.70))
    hub = _lathe("hub", [(0, Z1 + 8.5), (3.4, Z1 + 8.5), (3.4, Z1 + 13), (1.6, Z1 + 16), (0.9, Z1 + 17), (0, Z1 + 17)], UV_HUB, 16)
    needle = _lathe("needle", [(0, Z1 + 16.5), (0.35, Z1 + 16.5), (0.35, Z1 + 40), (0, Z1 + 41.2)], (0.27, 0.0, 0.48, 0.015), 8)
    for part, slot in ((tube, 0), (wings, 0), (hub, 1), (needle, 1)):          # slot 0 clear plastic, 1 opaque
        for nm in ("M_Syringe_Clear", "MI_Syringe"):
            part.data.materials.append(bpy.data.materials.get(nm) or bpy.data.materials.new(nm))
        for f in part.data.polygons:
            f.material_index = slot
    return geo.join([tube, wings, hub, needle], "SM_Syringe")


def label():
    """Paper label decal wrapped 300 deg around the barrel, 0.15 mm proud."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    nu, nv, r = 30, 4, RO + 0.15
    a0, a1 = math.radians(-90 - 75), math.radians(-90 + 75)
    z0, z1 = Z0 + 6, Z1 - 6
    g = [[bm.verts.new((r * math.cos(a0 + (a1 - a0) * i / nu) * MM, r * math.sin(a0 + (a1 - a0) * i / nu) * MM, (z0 + (z1 - z0) * j / nv) * MM))
          for i in range(nu + 1)] for j in range(nv + 1)]
    u0, v0, u1, v1 = UV_LABEL
    for j in range(nv):
        for i in range(nu):
            f = bm.faces.new((g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i]))
            for l, (ii, jj) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                l[uv].uv = (u0 + (u1 - u0) * ii / nu, v0 + (v1 - v0) * jj / nv)
    me = bpy.data.meshes.new("SM_Syringe_Label"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("SM_Syringe_Label", me)
    bpy.context.scene.collection.objects.link(o)
    for p in me.polygons:
        p.use_smooth = True
    o["qc_open"] = True
    return o


def plunger():
    """Cross-section rod (two crossed plates), thumb pad, ribbed rubber stopper. Pivot = thumb pad centre."""
    rod_a = _box("rod_a", (9.0, 1.4, 44), (0, 0, STOP_Z - 24), (0.52, 0.30, 0.75, 0.62))
    rod_b = _box("rod_b", (1.4, 9.0, 43.4), (0, 0, STOP_Z - 24.1), (0.76, 0.30, 0.99, 0.62))
    pad = _lathe("pad", [(0, STOP_Z - 47), (9.5, STOP_Z - 47), (9.5, STOP_Z - 45.5), (0, STOP_Z - 45.5)], (0.52, 0.64, 0.7, 0.8), 24)
    stopper = _lathe("stopper", [(0, STOP_Z - 6), (RI - 0.25, STOP_Z - 6), (RI - 0.05, STOP_Z - 5), (RI - 0.6, STOP_Z - 3.5),
                                 (RI - 0.05, STOP_Z - 2), (RI - 0.6, STOP_Z - 0.8), (2.0, STOP_Z + 0.4), (0, STOP_Z + 0.5)],
                     (0.72, 0.64, 0.99, 0.99), 24)
    o = geo.join([rod_a, rod_b, pad, stopper], "SM_Syringe_Plunger")
    geo.set_origin(o, (0, 0, (STOP_Z - 47) * MM))
    return o


def cap():
    o = _lathe("SM_Syringe_Cap", [(0, Z1 + 12.5), (3.9, Z1 + 12.5), (4.1, Z1 + 13.5), (3.1, Z1 + 43), (2.6, Z1 + 44.5), (0, Z1 + 44.8)], UV_CAP, 24)
    geo.set_origin(o, (0, 0, (Z1 + 12.5) * MM))
    return o


def liquid():
    o = _lathe("SM_Syringe_Liquid", [(0, STOP_Z + 0.6), (RI - 0.12, STOP_Z + 0.6), (RI - 0.12, Z1 - 2.2), (RI - 0.9, Z1 - 1.0),
                                    (0, Z1 - 0.9)], UV_LIQ, 24)
    geo.set_origin(o, (0, 0, (Z1 - 0.9) * MM))
    return o
