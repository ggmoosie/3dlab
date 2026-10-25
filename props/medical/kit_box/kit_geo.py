"""Surgery / first-aid tin (bpy). Contract Surgery_Kit: box 16 × 10 × 5 cm, pivot centre; lid 16 × 10 × 0.8 hinged on the
far long edge (+Y), opens 110°; scalpel 10 cm + 3.2 cm blade, pivot = pinch point, +Z toward the blade.
Parts: SM_KitBox (steel tray + foam insert + packets + iodine bottle), SM_KitBox_Lid (pivot = hinge line), SM_Scalpel."""
import bpy, bmesh, math, os, sys
from mathutils import Matrix, Vector
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
BW, BD, BH, R = 160.0, 100.0, 42.0, 11.0          # tray outer size (mm), corner radius
LID_H = 8.0
UV_LIDTOP = (0.0, 0.50, 0.62, 0.99)
UV_REST = (0.64, 0.50, 0.99, 0.99)
UV_TRAY = (0.0, 0.0, 0.48, 0.48)
UV_INSIDE = (0.50, 0.0, 0.99, 0.48)


def rounded_box(name, w, d, h, r, z0=0.0, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object
    o.name = name
    o.scale = (w * MM, d * MM, h * MM)
    o.location = (0, 0, (z0 + h / 2) * MM)
    bpy.ops.object.transform_apply(scale=True, location=True)
    bm = bmesh.new(); bm.from_mesh(o.data)
    vert_edges = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 1e-6]
    bmesh.ops.bevel(bm, geom=vert_edges, offset=r * MM, segments=6, affect="EDGES", profile=0.5)
    if bevel:
        flat = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-6]
        bmesh.ops.bevel(bm, geom=flat, offset=bevel * MM, segments=2, affect="EDGES", profile=0.5)
    bm.to_mesh(o.data); bm.free()
    return o


def hollow(outer, inner):
    m = outer.modifiers.new("Hollow", "BOOLEAN")
    m.operation, m.object, m.solver = "DIFFERENCE", inner, "EXACT"
    geo.apply_mods(outer)
    bpy.data.objects.remove(inner)
    bm = bmesh.new(); bm.from_mesh(outer.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bm.to_mesh(outer.data); bm.free()
    return outer


def tray():
    t = hollow(rounded_box("tray", BW, BD, BH, R, -BH / 2, 1.2), rounded_box("cut", BW - 1.8, BD - 1.8, BH, R - 0.9, -BH / 2 + 0.9))
    geo.unwrap_into(t, UV_TRAY)
    foam = rounded_box("foam", BW - 2.6, BD - 2.6, 26, R - 1.3, -BH / 2 + 0.95, 1.5)
    cut = rounded_box("slot", 104, 9, 30, 3.5, -BH / 2 + 18)                      # scalpel cutout
    cut.location = (0, -28 * MM, 0); bpy.context.view_layer.update(); cut.data.transform(cut.matrix_world); cut.matrix_world = Matrix()
    foam = hollow(foam, cut)
    geo.unwrap_into(foam, (0.50, 0.25, 0.99, 0.48))
    packs = []
    for i, (x, y, w, d) in enumerate(((-40, 12, 64, 48), (34, 12, 54, 40))):         # sealed gauze / dressing packets
        p = rounded_box(f"pack{i}", w, d, 5, 3, -BH / 2 + 27.0, 1.0)
        p.location = (x * MM, y * MM, 0); bpy.context.view_layer.update(); p.data.transform(p.matrix_world); p.matrix_world = Matrix()
        geo.unwrap_into(p, (0.50 + i * 0.25, 0.0, 0.73 + i * 0.25, 0.23))
        packs.append(p)
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]                           # own strip between the pack islands
    bottle = geo.lathe("iodine", [dict(pts=sc([(0, 0), (9, 0)]), uv=(0.734, 0.175, 0.747, 0.188), mode="disc", flip=True),
                                  dict(pts=sc([(9, 0), (9, 30), (5, 33), (5, 40)]), uv=(0.734, 0.0, 0.747, 0.17)),
                                  dict(pts=sc([(5, 40), (0, 40)]), uv=(0.734, 0.192, 0.747, 0.205), mode="disc")], segs=20)
    bottle.data.transform(Matrix.Rotation(-math.pi / 2, 4, "X"))                  # lies along +Y, clear of the walls
    bottle.location = (69 * MM, -47 * MM, (-BH / 2 + 27 + 9.2) * MM)
    bpy.context.view_layer.update(); bottle.data.transform(bottle.matrix_world); bottle.matrix_world = Matrix()
    return geo.join([t, foam] + packs + [bottle], "SM_KitBox")


def lid():
    l = hollow(rounded_box("SM_KitBox_Lid", BW + 1.6, BD + 1.6, LID_H, R + 0.8, BH / 2 - LID_H + 3.0, 1.6),
               rounded_box("lcut", BW + 0.2, BD + 0.2, LID_H, R + 0.1, BH / 2 - LID_H + 3.0 - 0.7))
    knuckles = []
    for x in (-50, 50):
        k = geo.lathe(f"k{x}", [dict(pts=[(0, -10 * MM), (2.4 * MM, -10 * MM), (2.4 * MM, 10 * MM), (0, 10 * MM)], uv=(0.64, 0.50, 0.66, 0.52))], segs=12)
        k.data.transform(Matrix.Translation((x * MM, (BD / 2 + 1.6) * MM, (BH / 2 - 1.0) * MM)) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
        knuckles.append(k)
    o = geo.join([l] + knuckles, "SM_KitBox_Lid")
    geo.unwrap_into(o, UV_REST)
    bm = bmesh.new(); bm.from_mesh(o.data)                       # the lid top gets its own planar island (print area)
    uv = bm.loops.layers.uv.active
    top = max(v.co.z for v in bm.verts)
    u0, v0, u1, v1 = UV_LIDTOP
    for f in bm.faces:
        if f.normal.z > 0.9 and all(abs(l.vert.co.z - top) < 0.3 * MM for l in f.loops):
            for l in f.loops:
                l[uv].uv = (u0 + (u1 - u0) * (0.5 + l.vert.co.x / ((BW + 1.6) * MM)), v0 + (v1 - v0) * (0.5 + l.vert.co.y / ((BD + 1.6) * MM)))
    bm.to_mesh(o.data); bm.free()
    geo.set_origin(o, (0, (BD / 2 + 1.6) * MM, (BH / 2 - 1.0) * MM))
    return o


def scalpel():
    """Flat stainless handle with grip ridges (normal map) + a #10 blade. Pivot = pinch point (4 cm from the tail)."""
    h = rounded_box("handle", 8.0, 2.2, 100.0, 1.0, -40.0)
    h.data.transform(Matrix.Rotation(0, 4, "X"))
    bm = bmesh.new()
    pts = [(0, 0), (3.6, 0), (4.6, 10), (4.0, 22), (1.5, 31.5), (0.0, 32.0), (-1.6, 28), (-2.0, 12), (-1.6, 0)]
    lo = [bm.verts.new((x * MM, -0.2 * MM, (z + 60) * MM)) for x, z in pts]
    hi = [bm.verts.new((x * MM, 0.2 * MM, (z + 60) * MM)) for x, z in pts]
    bm.faces.new(lo[::-1]); bm.faces.new(hi)
    for i in range(len(pts)):
        bm.faces.new((lo[i], lo[(i + 1) % len(pts)], hi[(i + 1) % len(pts)], hi[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    me = bpy.data.meshes.new("blade"); bm.to_mesh(me); bm.free()
    b = bpy.data.objects.new("blade", me); bpy.context.scene.collection.objects.link(b)
    b.location.z -= 0.0005; bpy.context.view_layer.update(); b.data.transform(b.matrix_world); b.matrix_world = Matrix()
    o = geo.join([h, b], "SM_Scalpel")
    geo.unwrap_into(o, (0.0, 0.0, 0.99, 0.99))
    return o
