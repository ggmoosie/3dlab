"""Spoon (bpy). Contract Spoon: 17 cm; handle 10.5 × 1.1 × 0.3, bowl 4 × 2.8 × 1.1; pivot = pinch point 5 cm from the
handle end; +Z toward the bowl; +X = bowl up (the concave side faces +X). Built as a deformed grid + solidify."""
import bpy, bmesh, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
UV = (0.02, 0.02, 0.98, 0.98)
L_TOTAL, HANDLE_END = 170.0, -50.0          # z from -50 (handle end) to +120 (bowl tip)
BOWL_C, BOWL_L, BOWL_W, BOWL_D = 98.0, 44.0, 28.0, 7.0


def _bowl_e(z):
    return (z - BOWL_C) / (BOWL_L / 2)


def _bowl_w(z):
    e = _bowl_e(z)
    return BOWL_W / 2 * math.sqrt(max(0.0, 1 - e * e)) * (1 + 0.08 * e) if abs(e) <= 1 else 0.0


def width(z):
    """Outline half-width (mm): max of the handle/neck curve and the egg-shaped bowl, so the join is continuous."""
    if z < HANDLE_END + 5.5:                                           # rounded handle end
        end = math.sqrt(max(0.0, 1 - ((z - (HANDLE_END + 5.5)) / 5.5) ** 2))
        hw = 5.5 * max(end, 0.12)
    elif z < 62:
        hw = 5.5 - 2.6 * (z - (HANDLE_END + 5.5)) / (62 - HANDLE_END - 5.5)
    elif z < 90:
        hw = 2.9
    else:
        hw = 0.0
    return max(0.6, hw, _bowl_w(z))


def surface_x(z, y):
    """Front-surface height (+X). Bowl is concave toward +X, deepening smoothly from its back; handle has the S bend."""
    bend = 6.0 * math.exp(-((z - 55) / 30) ** 2) - 2.0 * (z < 0) * (z / 50) ** 2
    e = _bowl_e(z)
    if abs(e) <= 1:
        q = min(1.0, (y / max(width(z), 1e-6)) ** 2)
        return bend - BOWL_D * (1 - e * e) ** 1.5 * (1 - q) * min(1.0, _bowl_w(z) / 6.0)
    return bend


def spoon(name="SM_Spoon", nz=52, ny=8):
    bm = bmesh.new()
    zs = [HANDLE_END + 0.4 + (120 - 0.4 - HANDLE_END) * (i / (nz - 1)) for i in range(nz)]
    grid = []
    for z in zs:
        w = width(z)
        row = []
        for j in range(ny + 1):
            y = -w + 2 * w * j / ny
            row.append(bm.verts.new((surface_x(z, y) * MM, y * MM, z * MM)))
        grid.append(row)
    for i in range(nz - 1):
        for j in range(ny):
            bm.faces.new((grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if sum(f.normal.x for f in bm.faces) < 0:
        for f in bm.faces:
            f.normal_flip()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    sol = o.modifiers.new("Solid", "SOLIDIFY")
    sol.thickness = 1.6 * MM
    sol.offset = -1.0
    sol.use_even_offset = True
    geo.apply_mods(o)
    for v in o.data.vertices:                                 # thicker handle (3 mm), thin bowl (1.2 mm) on the back side
        z = v.co.z / MM
        if z < 70 and v.normal.x < 0:
            v.co.x -= 1.2 * MM * min(1.0, (70 - z) / 25)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(o.data); bm.free()
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.set_sharp_from_angle(angle=math.radians(70))
    geo.unwrap_into(o, UV)
    return o
