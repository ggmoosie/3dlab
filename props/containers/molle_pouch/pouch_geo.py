"""MOLLE general-purpose pouch (bpy), 150 × 70 × 170 mm. Front = -Y. Pivot = base centre.
Parts: SM_GPPouch (soft body + 3 PALS rows + 2 rear MOLLE straps + snaps + female buckle)
       SM_GPPouch_Flap (lid over the top and 60 mm down the front, strap + male buckle; pivot = rear hinge, opens about X)"""
import bpy, bmesh, math, os, sys
from mathutils import Vector, Matrix, noise
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
HX, HY, HZ, R = 75.0, 35.0, 85.0, 14.0             # half extents (body centred at z = HZ), edge radius
BULGE = 5.0
PALS_Z = (32.0, 70.0, 108.0)                       # strip centres from the bottom
UV_BODY = (0.0, 0.30, 0.62, 0.99)
UV_STRAPS = (0.64, 0.30, 0.99, 0.99)
UV_FLAP = (0.0, 0.0, 0.62, 0.28)
UV_HW = (0.64, 0.0, 0.99, 0.28)


def surface(x, y, z):
    """Soft box surface point for a direction from the centre: rounded edges, bulged panels, fabric wrinkles (mm)."""
    p = Vector((x, y, z - HZ))
    h = Vector((HX, HY, HZ))
    q = Vector([max(-hh + R, min(hh - R, c)) for c, hh in zip(p, h)])
    d = p - q
    p2 = q + (d.normalized() * R if d.length > 1e-6 else Vector())
    fx, fz = 1 - (p2.x / HX) ** 2, 1 - (p2.z / HZ) ** 2
    n = (p2 - q).normalized() if d.length > 1e-6 else Vector((0, -1, 0))
    bul = BULGE * max(0, fx) * max(0, fz) * abs(n.y)
    wr = 0.9 * noise.noise(Vector((p2.x * 0.05, p2.y * 0.05, p2.z * 0.07))) + 0.4 * noise.noise(p2 * 0.15)
    out = p2 + n * (bul + wr)
    return Vector((out.x, out.y, out.z + HZ))


def body_shell(name="body"):
    bpy.ops.mesh.primitive_cube_add(size=2)
    o = bpy.context.object
    o.name = name
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=11, use_grid_fill=True)
    for v in bm.verts:
        c = v.co
        v.co = surface(c.x * HX, c.y * HY, c.z * HZ + HZ) * MM
    bm.to_mesh(o.data); bm.free()
    for p in o.data.polygons:
        p.use_smooth = True
    return o


def front_point(x, z, off):
    """Point on the front (-Y) panel at (x, z) pushed `off` mm outward along -Y."""
    p = surface(x, -HY, z)
    return Vector((p.x, p.y - off, p.z))


def strip(name, x0, x1, z0, z1, side="front", off=0.6, thick=1.8, nx=16, nz=2):
    """Webbing strip hugging the panel; solidified so it is closed and sinks 0.3 mm into the body."""
    bm = bmesh.new()
    g = []
    for j in range(nz + 1):
        z = z0 + (z1 - z0) * j / nz
        row = []
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            if side == "front":
                p = front_point(x, z, off)
            else:
                p = surface(x, HY, z)
                p = Vector((p.x, p.y + off, p.z))
            row.append(bm.verts.new(p * MM))
        g.append(row)
    for j in range(nz):
        for i in range(nx):
            bm.faces.new((g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o)
    s = o.modifiers.new("S", "SOLIDIFY"); s.thickness = thick * MM; s.offset = 0.0
    geo.apply_mods(o)
    bm = bmesh.new(); bm.from_mesh(o.data)                # closed shell: recalc gives outward normals
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data); bm.free()
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.set_sharp_from_angle(angle=math.radians(35))  # webbing is flat with crisp edges, not a pillow
    return o


def buckle_female():
    b = geo.rounded_box("bf", 30, 7, 26, 2.5, 0, 0.8, segs=3)
    p = front_point(0, 150, 0)
    b.location = (0, p.y * MM - 1.5 * MM, 128 * MM)
    geo.bake_xform(b)
    slot = geo.rounded_box("bs", 22, 12, 8, 1.0, 0, 0, segs=2)
    slot.location = (0, p.y * MM - 1.5 * MM, 146 * MM)
    geo.bake_xform(slot)
    return geo.boolean(b, slot)


def snaps():
    out = []
    for x in (-40, 40):
        s = geo.lathe(f"snap{x}", [dict(pts=[(0, 0), (5 * MM, 0), (5.5 * MM, 1.2 * MM), (4 * MM, 2.6 * MM), (0, 3 * MM)], uv=(0.64, 0.0, 0.66, 0.02))], segs=16)
        s.data.transform(Matrix.Rotation(-math.pi / 2, 4, "X"))
        p = surface(x, HY, 20)
        s.location = (x * MM, (p.y + 1.2) * MM, 20 * MM)
        geo.bake_xform(s)
        out.append(s)
    return out


def pouch():
    parts = [body_shell()]
    for k, zc in enumerate(PALS_Z):
        parts.append(strip(f"pals{k}", -HX + 8, HX - 8, zc - 12.5, zc + 12.5, "front"))
    for k, x in enumerate((-40, 40)):
        parts.append(strip(f"molle{k}", x - 12.5, x + 12.5, 12, 160, "back", 0.7, 2.0, nx=4, nz=12))
    hw = [buckle_female()] + snaps()
    for o in parts + hw:
        for nm in ("MI_GPPouch",):
            o.data.materials.append(bpy.data.materials.get(nm) or bpy.data.materials.new(nm))
    geo.unwrap_into(parts[0], UV_BODY)
    straps = geo.join(parts[1:], "straps")
    geo.unwrap_into(straps, UV_STRAPS)
    hwj = geo.join(hw, "hw")
    geo.unwrap_into(hwj, (0.64, 0.14, 0.99, 0.28))
    return geo.join([parts[0], straps, hwj], "SM_GPPouch")


def flap():
    """Lid: a 3 mm slab following the body's top from the rear hinge, over, and 60 mm down the front; strap + male buckle."""
    bm = bmesh.new()
    path = []
    n = 26
    for i in range(n + 1):                                 # walk the top in the YZ plane: rear → over → front
        t = i / n
        a = math.pi * t
        y = HY * math.cos(a) * 1.0
        z = 2 * HZ - 0.0
        if t > 0.5:
            pass
        path.append(t)
    rows = []
    prof = []
    for i in range(n + 1):
        t = i / n
        if t <= 0.7:                                       # across the top, rear (y=+HY) to front edge
            y = (HY - R * 0.6) - (2 * HY - R * 0.6) * (t / 0.7)
            z = surface(0, y, 2 * HZ).z + 1.6 if abs(y) < HY - R else None
            if z is None:
                yy = max(-HY, min(HY, y))
                z = surface(0, yy, 2 * HZ).z + 1.6
            prof.append((y, z))
        else:                                              # down the front, 60 mm
            zz = 2 * HZ - R - (60 - R) * (t - 0.7) / 0.3
            p = front_point(0, zz, 1.8)
            prof.append((p.y, zz))
    for y, z in prof:
        rows.append([bm.verts.new((x * MM, y * MM, z * MM)) for x in (-73, -36.5, 0, 36.5, 73)])
    for j in range(len(rows) - 1):
        for i in range(4):
            bm.faces.new((rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("SM_GPPouch_Flap"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("SM_GPPouch_Flap", me); bpy.context.scene.collection.objects.link(o)
    s = o.modifiers.new("S", "SOLIDIFY"); s.thickness = 3 * MM; s.offset = 1.0
    b = o.modifiers.new("B", "BEVEL"); b.width = 1.2 * MM; b.segments = 2; b.limit_method = "ANGLE"
    geo.apply_mods(o)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data); bm.free()
    for p in o.data.polygons:
        p.use_smooth = True
    tab = strip("tab", -12.5, 12.5, 2 * HZ - 62, 2 * HZ - 40, "front", 4.6, 1.6, nx=4, nz=4)
    male = geo.rounded_box("bm", 24, 6, 24, 2.0, 0, 0.6, segs=3)
    p = front_point(0, 2 * HZ - 70, 0)
    male.location = (0, (p.y - 5.2) * MM, (2 * HZ - 78) * MM)
    geo.bake_xform(male)
    for x in (o, tab, male):
        x.data.materials.append(bpy.data.materials.get("MI_GPPouch") or bpy.data.materials.new("MI_GPPouch"))
    geo.unwrap_into(o, UV_FLAP)
    rest = geo.join([tab, male], "fl_rest")
    geo.unwrap_into(rest, (0.64, 0.0, 0.99, 0.13))
    f = geo.join([o, rest], "SM_GPPouch_Flap")
    geo.set_origin(f, (0, (HY - R * 0.6) * MM, (2 * HZ) * MM))
    return f
