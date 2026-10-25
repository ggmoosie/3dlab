"""Soda can geometry (bpy). Real 355 mL 211/202 can: body Ø66.2, height 122.2, 202 lid, stay-on tab.
Frame (contract Drink_Can): pivot base centre, +Z up; tab pivot = rivet (0, 0, 116 mm); the pour opening is on
Blender -Y (UE +Y). Parts: SM_SodaCan (shell + lid with opening), SM_SodaCan_Tab, SM_SodaCan_Flap (pushes in).
UV0 at 2048 px, ~9.8 px/mm everywhere: side wrap (label), bottom disc, lid disc, tab, flap, interior."""
import bpy, bmesh, math
from mathutils import Vector
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
SEGS = 40
PANEL_Z, PANEL_R = 116.0, 24.6
# UV islands (u0, v0, u1, v1)
UV_SIDE = (0.0, 0.40, 1.0, 0.998)
UV_BOTTOM = (0.01, 0.06, 0.283, 0.333)
UV_LID = (0.30, 0.06, 0.569, 0.329)       # lid disc covers r 0..27.7
UV_TAB = (0.60, 0.04, 0.80, 0.37)
UV_FLAP = (0.82, 0.20, 0.97, 0.37)
UV_DARK = (0.82, 0.04, 0.97, 0.18)        # opening wall + interior
LID_RMAX = 27.7

BOTTOM = [(0, 9.6), (9, 9.2), (16, 7.6), (21, 4.8), (23.6, 1.8), (24.6, 0.35), (25.6, 0.0), (26.8, 0.3), (28.0, 1.2)]
SIDE = [(28.0, 1.2), (30.0, 2.8), (31.8, 5.2), (32.8, 8.2), (33.1, 11.0), (33.1, 104.0), (32.9, 106.5), (32.1, 109.2),
        (30.7, 112.2), (29.2, 114.8), (28.3, 116.8), (28.25, 117.6), (28.6, 118.6), (28.65, 121.0), (28.3, 121.95), (27.7, 122.2)]
LID = [(27.7, 122.2), (27.1, 122.0), (26.8, 121.2), (26.6, 118.8), (26.4, 116.0), (26.0, 115.2), (25.4, 115.1),
       (24.9, 115.6), (PANEL_R, PANEL_Z)]


def side_arc():
    """Arc length (mm) of SIDE from its first point; used by the label layout."""
    acc = [0.0]
    for a, b in zip(SIDE, SIDE[1:]):
        acc.append(acc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return acc


def _hull(pts):
    pts = sorted(set(pts))
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out
    lo, hi = half(pts), half(pts[::-1])
    return lo[:-1] + hi[:-1]


def _resample(poly, n):
    seg = [math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    tot, out, acc, i = sum(seg), [], 0.0, 0
    for k in range(n):
        t = tot * k / n
        while acc + seg[i] < t:
            acc += seg[i]
            i += 1
        f = (t - acc) / seg[i]
        a, b = poly[i], poly[(i + 1) % len(poly)]
        out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out


OPEN_C = (0.0, -12.4)                         # opening centroid (mm)


def opening(n=28, shrink=0.0):
    """Pour opening outline (mm, XY on the lid panel, CCW): hull of a small circle near the rivet and a big one
    near the countersink, on -Y."""
    c0, r0, c1, r1 = -6.6, 4.2 - shrink, -17.2, 7.4 - shrink
    raw = [(r * math.cos(2 * math.pi * k / 180), c + r * math.sin(2 * math.pi * k / 180)) for c, r in ((c0, r0), (c1, r1)) for k in range(180)]
    return _resample(_hull(raw), n)


def lid_uv(x, y):
    u0, v0, u1, v1 = UV_LID
    return (u0 + (u1 - u0) * (0.5 + 0.5 * x / LID_RMAX), v0 + (v1 - v0) * (0.5 + 0.5 * y / LID_RMAX))


def shell(name="SM_SodaCan", segs=SEGS, high=False):
    """Lathe body + lid rings, then a planar panel with the pour opening, a 7 mm dark well under it."""
    k = 4 if high else 1
    secs = [dict(pts=[(r * MM, z * MM) for r, z in BOTTOM], uv=UV_BOTTOM, mode="disc", flip=True),
            dict(pts=[(r * MM, z * MM) for r, z in SIDE], uv=UV_SIDE, mode="side"),
            dict(pts=[(r * MM, z * MM) for r, z in LID], uv=UV_LID, mode="disc")]
    o = geo.lathe(name, secs, segs=segs * k)
    # the disc mapping for the lid uses its own rmax (27.7) — same as LID_RMAX, so the panel UVs below line up
    bm = bmesh.new()
    bm.from_mesh(o.data)
    uv = bm.loops.layers.uv.active
    ring = [v for v in bm.verts if abs(v.co.z - PANEL_Z * MM) < 1e-7 and abs(v.co.xy.length - PANEL_R * MM) < 1e-6]
    ring.sort(key=lambda v: math.atan2(v.co.y, v.co.x))
    hole = [bm.verts.new((x * MM, y * MM, PANEL_Z * MM)) for x, y in opening(28 if not high else 64)]
    edges = [bm.edges.get((ring[i], ring[(i + 1) % len(ring)])) for i in range(len(ring))]
    edges += [bm.edges.new((hole[i], hole[(i + 1) % len(hole)])) for i in range(len(hole))]
    res = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=edges)
    panel = [f for f in res["geom"] if isinstance(f, bmesh.types.BMFace)]
    # opening wall down 1.2 mm, then a dark well (reads as the inside of the can) 7 mm down
    ext = bmesh.ops.extrude_edge_only(bm, edges=[e for e in edges[len(ring):]])
    wall_v = [v for v in ext["geom"] if isinstance(v, bmesh.types.BMVert)]
    cx, cy = OPEN_C[0] * MM, OPEN_C[1] * MM
    for v in wall_v:
        v.co.z -= 7.0 * MM
        v.co.x = cx + (v.co.x - cx) * 0.85
        v.co.y = cy + (v.co.y - cy) * 0.85
    wall_e = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMEdge) and all(v in wall_v for v in e.verts)]
    cap = bmesh.ops.triangle_fill(bm, use_beauty=True, edges=wall_e)
    new_faces = [f for f in ext["geom"] if isinstance(f, bmesh.types.BMFace)] + \
                [f for f in cap["geom"] if isinstance(f, bmesh.types.BMFace)]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in panel:
        f.smooth = True
        for l in f.loops:
            l[uv].uv = lid_uv(l.vert.co.x / MM, l.vert.co.y / MM)
    u0, v0, u1, v1 = UV_DARK
    for f in new_faces:
        f.smooth = True
        for l in f.loops:
            x, y = l.vert.co.x / MM, l.vert.co.y / MM
            l[uv].uv = (u0 + (u1 - u0) * (0.5 + x / 22), v0 + (v1 - v0) * (0.5 + (y + 12) / 22))
    bm.to_mesh(o.data)
    bm.free()
    o.data.set_sharp_from_angle(angle=math.radians(50))
    return o


def _plate(name, outer, holes, thick, z0, uvrect, bbox):
    """Closed flat plate from 2D outlines (mm): outer loop, list of hole loops; planar UVs into uvrect over bbox."""
    bm = bmesh.new()
    loops = [outer] + holes
    edges = []
    for lp in loops:
        vs = [bm.verts.new((x * MM, y * MM, z0 * MM)) for x, y in lp]
        edges += [bm.edges.new((vs[i], vs[(i + 1) % len(vs)])) for i in range(len(vs))]
    bmesh.ops.triangle_fill(bm, use_beauty=True, edges=edges)
    faces = list(bm.faces)
    ext = bmesh.ops.extrude_face_region(bm, geom=faces)
    for v in [g for g in ext["geom"] if isinstance(g, bmesh.types.BMVert)]:
        v.co.z += thick * MM
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    uv = bm.loops.layers.uv.new("UVMap")
    (bx0, by0, bx1, by1), (u0, v0, u1, v1) = bbox, uvrect
    for f in bm.faces:
        for l in f.loops:
            l[uv].uv = (u0 + (u1 - u0) * (l.vert.co.x / MM - bx0) / (bx1 - bx0), v0 + (v1 - v0) * (l.vert.co.y / MM - by0) / (by1 - by0))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    me.set_sharp_from_angle(angle=math.radians(40))
    geo.unwrap_into(o, uvrect)
    return o


def _circle(cx, cy, rx, ry, n, a0=0.0, a1=2 * math.pi):
    return [(cx + rx * math.cos(a0 + (a1 - a0) * k / n), cy + ry * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n)]


def tab(name="SM_SodaCan_Tab", high=False):
    """Stay-on tab: nose over the opening (-Y), finger ring toward +Y. 0.5 mm plate, ring end curls up 0.8 mm."""
    n = 2 if high else 1
    outer = []
    outer += _circle(0, -4.6, 4.6, 3.0, 8 * n, math.pi, 2 * math.pi)            # nose (rounded, -Y)
    outer += [(6.0, 1.0), (6.2, 8.0)]
    outer += _circle(0, 12.5, 6.2, 7.0, 14 * n, 0, math.pi)                       # ring end
    outer += [(-6.2, 8.0), (-6.0, 1.0)]
    ring_hole = _circle(0, 12.2, 4.1, 4.7, 14 * n)[::-1]
    slot = [(-1.8, 2.2), (1.8, 2.2), (2.6, 4.2), (-2.6, 4.2)][::-1]               # stiffening slot behind the rivet
    o = _plate(name, outer, [ring_hole, slot], 0.5, PANEL_Z + 0.1, UV_TAB, (-7, -9, 7, 20.5))
    for v in o.data.vertices:                                                     # curl + slight lift toward the ring
        y = v.co.y / MM
        if y > 8:
            v.co.z += (0.0004 * (y - 8) ** 2 + 0.01 * (y - 8)) * MM * 6
    geo.set_origin(o, (0, 0, PANEL_Z * MM))
    return o


def flap(name="SM_SodaCan_Flap"):
    """The scored panel inside the opening; hinged at its narrow end (near the rivet), pushes down into the can."""
    pts = opening(28, shrink=0.12)
    o = _plate(name, pts, [], 0.3, PANEL_Z - 0.3, UV_FLAP, (-8, -25, 8, -2))
    geo.set_origin(o, (0, -2.6 * MM, PANEL_Z * MM))
    return o
