"""Geometry helpers (bpy/bmesh). The workhorse is lathe(): watertight revolved shells with planned UV islands."""
import bpy, bmesh, math
from mathutils import Vector


def lathe(name, sections, segs=48, uv_name="UVMap", phase=0.0):
    """Revolve a profile around Z. sections: list of dicts, consecutive sections share their junction ring.
      pts: [(r, z), ...] in metres (the first point of section k+1 == the last of section k)
      uv:  (u0, v0, u1, v1) rect for this island
      mode: "side" (u = angle, v = arc length) or "disc" (planar XY projection into the rect)
    r == 0 at an end makes a single pole vertex. Returns the object (smooth shaded, sharp at sections)."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new(uv_name)
    rings, prev_last = [], None
    for si, sec in enumerate(sections):
        for pi, (r, z) in enumerate(sec["pts"]):
            if pi == 0 and si > 0:
                continue                                    # shared junction ring
            if r <= 1e-9:
                rings.append([bm.verts.new((0, 0, z))])
            else:
                rings.append([bm.verts.new((r * math.cos(2 * math.pi * k / segs + phase),
                                            r * math.sin(2 * math.pi * k / segs + phase), z)) for k in range(segs)])
    ri = 0
    for si, sec in enumerate(sections):
        pts = sec["pts"]
        u0, v0, u1, v1 = sec["uv"]
        mode = sec.get("mode", "side")
        n = len(pts)
        start = ri
        # arc length for side mapping
        acc = [0.0]
        for a, b in zip(pts, pts[1:]):
            acc.append(acc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        tot = acc[-1] or 1.0
        rmax = max(p[0] for p in pts) or 1.0
        for j in range(n - 1):
            ra, rb = rings[start + j], rings[start + j + 1]
            for k in range(segs):
                k1 = (k + 1) % segs

                corners = []                     # (vert, u-step, profile row)
                for ring, kk, row in ((ra, k, j), (ra, k + 1, j), (rb, k + 1, j + 1), (rb, k, j + 1)):
                    if len(ring) == 1:
                        if corners and corners[-1][0] is ring[0]:
                            continue
                        corners.append((ring[0], k + 0.5, row))
                    else:
                        corners.append((ring[kk % segs], kk, row))
                if len(corners) < 3:
                    continue
                f = bm.faces.new([c[0] for c in corners])
                for loop, (v, kk, row) in zip(f.loops, corners):
                    if mode == "side":
                        uu = u0 + (u1 - u0) * kk / segs
                        vv = v0 + (v1 - v0) * acc[row] / tot
                    else:
                        x, y = v.co.x / rmax * (-1 if sec.get("flip") else 1), v.co.y / rmax
                        uu = u0 + (u1 - u0) * (0.5 + 0.5 * x)
                        vv = v0 + (v1 - v0) * (0.5 + 0.5 * y)
                    loop[uv].uv = (uu, vv)
        ri = start + n - 1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    for p in me.polygons:
        p.use_smooth = True
    # sharp edges where sections meet at an angle (UV seams there already)
    me.set_sharp_from_angle(angle=math.radians(50))
    return o


def apply_mods(o):
    bpy.context.view_layer.objects.active = o
    for m in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = o.data.name = name
    return o


def set_origin(o, point):
    """Move the object origin to a world point without moving geometry."""
    p = Vector(point)
    off = o.matrix_world.inverted() @ p
    o.data.transform(__import__("mathutils").Matrix.Translation(-off))
    o.location = p


def weighted_normals(o):
    m = o.modifiers.new("WN", "WEIGHTED_NORMAL")
    m.keep_sharp = True
    return m


def subdiv(o, levels=2):
    m = o.modifiers.new("Sub", "SUBSURF")
    m.levels = m.render_levels = levels
    return m


def unwrap_into(o, rect, margin=0.02, angle=66):
    """Smart-project o's UVs, then fit the packed islands into rect (u0, v0, u1, v1), keeping aspect."""
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=margin, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = o.data.uv_layers.active.data
    us = [d.uv.x for d in uv]
    vs = [d.uv.y for d in uv]
    lo_u, lo_v, span = min(us), min(vs), max(max(us) - min(us), max(vs) - min(vs))
    u0, v0, u1, v1 = rect
    k = min(u1 - u0, v1 - v0) / span
    for d in uv:
        d.uv = (u0 + (d.uv.x - lo_u) * k, v0 + (d.uv.y - lo_v) * k)


def fix_zero_uv(o, rect, eps=1e-10):
    """Faces with (near) zero UV area (e.g. solidify rims) get a planar box projection packed into rect."""
    import bmesh
    bm = bmesh.new(); bm.from_mesh(o.data)
    uv = bm.loops.layers.uv.active
    bad = []
    for f in bm.faces:
        a = 0.0
        ls = f.loops
        for i in range(1, len(ls) - 1):
            p0, p1, p2 = ls[0][uv].uv, ls[i][uv].uv, ls[i + 1][uv].uv
            a += abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) / 2
        if a < eps:
            bad.append(f)
    if bad:
        u0, v0, u1, v1 = rect
        n = len(bad)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        cw, ch = (u1 - u0) / cols, (v1 - v0) / cols
        for k, f in enumerate(bad):                       # each bad face gets its own little cell
            cu, cv = u0 + (k % cols) * cw, v0 + (k // cols) * ch
            nrm = f.normal
            ax = max(range(3), key=lambda i: abs(nrm[i]))
            a1, a2 = [i for i in range(3) if i != ax]
            pts = [(l.vert.co[a1], l.vert.co[a2]) for l in f.loops]
            lo1, lo2 = min(p[0] for p in pts), min(p[1] for p in pts)
            sp = max(max(p[0] for p in pts) - lo1, max(p[1] for p in pts) - lo2, 1e-9)
            for l, (x, y) in zip(f.loops, pts):
                l[uv].uv = (cu + cw * 0.1 + 0.8 * cw * (x - lo1) / sp, cv + ch * 0.1 + 0.8 * ch * (y - lo2) / sp)
    bm.to_mesh(o.data); bm.free()
    return len(bad)


def rounded_box(name, w, d, h, r, z0=0.0, bevel=0.0, segs=6, center=(0.0, 0.0)):
    """Box w × d × h (mm) from z0 up, vertical edges rounded to r, optional small bevel on the flat edges."""
    import bmesh
    MM = 0.001
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object
    o.name = name
    o.scale = (w * MM, d * MM, h * MM)
    o.location = (center[0] * MM, center[1] * MM, (z0 + h / 2) * MM)
    bpy.ops.object.transform_apply(scale=True, location=True)
    bm = bmesh.new(); bm.from_mesh(o.data)
    if r > 0:
        ve = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 1e-6]
        bmesh.ops.bevel(bm, geom=ve, offset=r * MM, segments=segs, affect="EDGES", profile=0.5)
    if bevel:
        fe = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-6]
        bmesh.ops.bevel(bm, geom=fe, offset=bevel * MM, segments=2, affect="EDGES", profile=0.5)
    bm.to_mesh(o.data); bm.free()
    return o


def boolean(target, cutter, op="DIFFERENCE", keep_cutter=False):
    import bmesh
    m = target.modifiers.new("Bool", "BOOLEAN")
    m.operation, m.object, m.solver = op, cutter, "EXACT"
    apply_mods(target)
    if not keep_cutter:
        bpy.data.objects.remove(cutter)
    bm = bmesh.new(); bm.from_mesh(target.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bm.to_mesh(target.data); bm.free()
    return target


def bake_xform(o):
    """Apply the object transform into the mesh (pivot stays at the world origin)."""
    from mathutils import Matrix
    bpy.context.view_layer.update()
    o.data.transform(o.matrix_world)
    o.matrix_world = Matrix()
    return o


def box_uv(o, rects, center, inner_rect, units=0.001):
    """Dieline-style UVs for box-like meshes. rects: {"+X"|"-X"|"+Y"|"-Y"|"+Z"|"-Z": (rect, (lo_a, hi_a), (lo_b, hi_b))}
    where (lo, hi) are the world ranges (in `units`) mapped across u and v (a = first remaining axis, b = second).
    Outer faces (normal pointing away from `center`) are box-projected; inner faces keep a smart-projected layout
    squeezed into inner_rect, so nothing overlaps. Several objects can share one dieline (print continues across)."""
    import bmesh
    from mathutils import Vector
    unwrap_into(o, inner_rect)
    bm = bmesh.new(); bm.from_mesh(o.data)
    uv = bm.loops.layers.uv.active
    c = Vector(center)
    for f in bm.faces:
        n = f.normal
        if (f.calc_center_median() - c).dot(n) <= 0:
            continue
        ax = max(range(3), key=lambda i: abs(n[i]))
        key = ("+" if n[ax] > 0 else "-") + "XYZ"[ax]
        if key not in rects:
            continue
        (u0, v0, u1, v1), (la, ha), (lb, hb) = rects[key]
        a, b = [i for i in range(3) if i != ax]
        flip = key in ("+Y", "-X", "-Z")             # u must follow the viewer's right when seen from outside
        for l in f.loops:
            p = l.vert.co / units
            ta = (p[a] - la) / (ha - la)
            tb = (p[b] - lb) / (hb - lb)
            if flip:
                ta = 1 - ta
            l[uv].uv = (u0 + (u1 - u0) * ta, v0 + (v1 - v0) * tb)
    bm.to_mesh(o.data); bm.free()
