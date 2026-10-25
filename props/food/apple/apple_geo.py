"""Apple (bpy). Contract Eat_Apple: ~Ø7.5 sphere-ish, pivot = centre. Bitten stages SM_Apple_Bite1..3 swap in on Bite1-3.
Skin: slot 0 (universal mask, print = skin art per variant). Bite faces: slot 1 (flesh), from the boolean cutter's material."""
import bpy, bmesh, math, os, sys
from mathutils import Vector, Matrix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
UV_SKIN = (0.0, 0.25, 1.0, 0.995)
UV_STEM = (0.0, 0.02, 0.2, 0.22)
UV_FLESH = (0.25, 0.02, 0.6, 0.22)
def _profile(n=22):
    """Apple silhouette: broad shoulders, tapering base, stem cavity on top, calyx dimple underneath (mm)."""
    pts = []
    for i in range(n + 1):
        ph = -math.pi / 2 + math.pi * i / n
        x = 38.5 * math.cos(ph) * (0.86 + 0.14 * (1 + math.sin(ph)) / 2) ** 1.0
        z = 35.0 + 35.0 * math.sin(ph)
        if x < 11 and z > 35:
            z -= 8.5 * (1 - (x / 11) ** 2) ** 1.6                  # stem cavity
        if x < 9 and z < 35:
            z += 4.0 * (1 - (x / 9) ** 2) ** 1.5                   # calyx dimple
        pts.append((round(x, 3), round(z, 3)))
    pts[0], pts[-1] = (0.0, pts[0][1]), (0.0, pts[-1][1])
    return pts


PROFILE = _profile()
CENTER_Z = 36.0
BITES = [(-15, (0.0, -1.0, 0.1)), (-15, (0.6, -0.8, -0.05)), (-17, (-0.6, -0.8, 0.15))]   # (depth mm, direction)


def _skin(name):
    sc = [(r * MM, z * MM) for r, z in PROFILE]
    o = geo.lathe(name, [dict(pts=sc, uv=UV_SKIN)], segs=40)
    for v in o.data.vertices:                                  # 5 calyx lobes + lopsided shoulders
        a = math.atan2(v.co.y, v.co.x)
        z = v.co.z / MM
        lobe = 0.045 * math.cos(5 * a) * math.exp(-((z - 10) / 9) ** 2) + 0.025 * math.cos(5 * a + 0.6) * math.exp(-((z - 55) / 12) ** 2)
        tilt = 0.03 * math.cos(a - 0.7) * (z - CENTER_Z) / 36
        k = 1 + lobe + tilt
        v.co.x *= k
        v.co.y *= k
    return o


def _stem():
    bm = bmesh.new()
    path = [(0.0, 0.0, 59.0), (0.3, 0.2, 66.0), (1.2, 0.6, 72.0), (2.6, 1.0, 76.5)]
    rings = []
    for i, p in enumerate(path):
        r = 1.5 - 0.3 * i / (len(path) - 1)
        rings.append([bm.verts.new(((p[0] + r * math.cos(2 * math.pi * k / 8)) * MM, (p[1] + r * math.sin(2 * math.pi * k / 8)) * MM, p[2] * MM)) for k in range(8)])
    for a, b in zip(rings, rings[1:]):
        for k in range(8):
            bm.faces.new((a[k], a[(k + 1) % 8], b[(k + 1) % 8], b[k]))
    bm.faces.new(rings[0][::-1]); bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("stem"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("stem", me)
    bpy.context.scene.collection.objects.link(o)
    geo.unwrap_into(o, UV_STEM)
    return o


def apple(name="SM_Apple", bites=0, flesh=None):
    """Skin (optionally bitten: booleans run on the skin alone) + stem joined after, pivot = centre."""
    s = _skin(name)
    for _ in range(2):
        s.data.materials.append(bpy.data.materials.get("MI_Apple_Skin") or bpy.data.materials.new("MI_Apple_Skin"))
    if bites:
        s.data.materials[1] = flesh
        cutters = []
        for depth, d in BITES[:bites]:
            cutters += _cutter(depth, d, flesh)
        for c in cutters:
            m = s.modifiers.new("Bite", "BOOLEAN")
            m.operation, m.object, m.solver = "DIFFERENCE", c, "EXACT"
            m.material_mode = "TRANSFER"
            geo.apply_mods(s)
        for c in cutters:
            bpy.data.objects.remove(c)
        bm = bmesh.new(); bm.from_mesh(s.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
        uv = bm.loops.layers.uv.active
        u0, v0, u1, v1 = UV_FLESH
        for f in bm.faces:
            if f.material_index == 1:
                for l in f.loops:
                    p = l.vert.co / MM
                    l[uv].uv = (u0 + (u1 - u0) * (0.5 + p.x / 80), v0 + (v1 - v0) * (0.5 + p.z / 80))
        bm.to_mesh(s.data); bm.free()
    st = _stem()
    st.data.materials.append(s.data.materials[0])
    o = geo.join([s, st], name)
    geo.set_origin(o, (0, 0, CENTER_Z * MM))
    o.location = (0, 0, 0)
    return o


def _cutter(depth, d, flesh):
    """Double-arc bite: two overlapping ellipsoids (upper + lower teeth) pushed into the apple along -d."""
    d = Vector(d).normalized()
    objs = []
    for off in (-5.0, 5.0):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=14, radius=1.0)
        c = bpy.context.object
        c.scale = (20 * MM, 20 * MM, 13 * MM)
        up = Vector((0, 0, 1))
        side = d.cross(up).normalized()
        c.matrix_world = Matrix.Translation(Vector((0, 0, CENTER_Z * MM)) - d * (37 + 22 - abs(depth)) * MM + up * off * 1.3 * MM) @ \
            Matrix((side, up, d)).transposed().to_4x4() @ Matrix.Diagonal((17 * MM, 14 * MM, 22 * MM, 1))
        for v in c.data.vertices:                              # tooth grooves running into the bite + wobble
            a = math.atan2(v.co.y, v.co.x)
            k = 1 + 0.045 * math.sin(a * 13) + 0.02 * math.sin(a * 5 + off)
            v.co.x *= k
            v.co.y *= k
        c.data.materials.append(flesh)
        objs.append(c)
    return objs


def bitten(stage, flesh):
    return apple(f"SM_Apple_Bite{stage}", stage, flesh)
