"""Longneck glass bottle + crown cap (bpy). Contract Drink_Beer: height 28.15 cm, lip r 1.69, crown cap r 1.97 × 0.95 at
z 27.40 (separate; the clip twists it 3 × 60° and lifts 0.9). Base centre pivot, +Z up.
Parts: SM_GlassBottle (glass) + SM_GlassBottle_Label (front paper label decal) + SM_GlassBottle_NeckLabel (neck foil decal),
SM_GlassBottle_Liquid, SM_GlassBottle_Cap (21 crimp flutes). Decals are one-sided shells 0.2 mm off the glass."""
import bpy, bmesh, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
SEGS = 40
CAP_Z = 274.0
FILL_Z = 214.0
UV_SIDE = (0.0, 0.50, 0.36, 0.995)
UV_BASE = (0.38, 0.80, 0.50, 0.92)
UV_MOUTH = (0.38, 0.93, 0.44, 0.99)
UV_LABEL = (0.0, 0.02, 0.40, 0.47)
UV_NECK = (0.42, 0.02, 0.84, 0.22)
UV_CAP_SIDE = (0.42, 0.25, 0.98, 0.31)
UV_CAP_TOP = (0.86, 0.02, 0.98, 0.14)
UV_CAP_IN = (0.86, 0.33, 0.98, 0.45)
UV_LIQ_SIDE = (0.52, 0.50, 0.70, 0.80)
UV_LIQ_DISC = (0.72, 0.50, 0.80, 0.58)
UV_LIQ_TOP = (0.82, 0.50, 0.90, 0.58)
LABEL_Z = (32.0, 117.0)
LABEL_W = 75.0
NECK_Z = (224.0, 262.0)

BASE = [(0, 3.5), (10, 3.2), (18, 2.0), (24, 0.6), (27, 0.0), (29.5, 0.8), (31.0, 3.0), (31.5, 6.0)]
SIDE = [(31.5, 6.0), (31.5, 150.0), (31.2, 158), (30.0, 168), (27.5, 178), (23.5, 190), (19.0, 202), (15.5, 214),
        (13.6, 226), (13.2, 238), (13.0, 250), (13.2, 262), (13.6, 268), (13.8, 270), (15.2, 271), (16.6, 273),
        (16.9, 276), (16.6, 279), (15.5, 280.0), (12.5, 280.2)]
MOUTH = [(12.5, 280.2), (9.6, 279.5), (9.4, 272.0), (0, 272.0)]


def bottle(name="SM_GlassBottle"):
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    return geo.lathe(name, [dict(pts=sc(BASE), uv=UV_BASE, mode="disc", flip=True), dict(pts=sc(SIDE), uv=UV_SIDE),
                            dict(pts=sc(MOUTH), uv=UV_MOUTH)], segs=SEGS)


def _r_at(z, prof=SIDE):
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        if z0 <= z <= z1:
            return r0 + (r1 - r0) * (z - z0) / max(z1 - z0, 1e-9)
    return prof[-1][0]


def _decal(name, z0, z1, a0, a1, nu, nv, uvrect, off=0.2):
    """One-sided band hugging the glass between z0..z1 and angles a0..a1 (rad), UVs filling uvrect."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    grid = []
    for j in range(nv + 1):
        z = z0 + (z1 - z0) * j / nv
        r = _r_at(z) + off
        grid.append([bm.verts.new((r * math.cos(a0 + (a1 - a0) * i / nu) * MM, r * math.sin(a0 + (a1 - a0) * i / nu) * MM, z * MM))
                     for i in range(nu + 1)])
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    u0, v0, u1, v1 = uvrect
    for j in range(nv):
        for i in range(nu):
            i1 = (i + 1) % nu if full else i + 1
            vs = [grid[j][i if not full else i % nu], grid[j][i1], grid[j + 1][i1], grid[j + 1][i if not full else i % nu]]
            f = bm.faces.new(vs)
            for l, (ii, jj) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                l[uv].uv = (u0 + (u1 - u0) * ii / nu, v0 + (v1 - v0) * jj / nv)
    if full:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:                                 # face outward
        if f.normal.dot(f.calc_center_median().to_2d().to_3d()) < 0:
            f.normal_flip()
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    o["qc_open"] = True                                # a decal: one-sided by design
    return o


def label():
    half = LABEL_W / 2 / 31.7
    return _decal("SM_GlassBottle_Label", LABEL_Z[0], LABEL_Z[1], -math.pi / 2 - half, -math.pi / 2 + half, 24, 6, UV_LABEL)


def neck_label():
    return _decal("SM_GlassBottle_NeckLabel", NECK_Z[0], NECK_Z[1], -math.pi / 2 - math.pi, -math.pi / 2 + math.pi, 40, 8, UV_NECK, 0.18)


def liquid(name="SM_GlassBottle_Liquid", segs=32):
    prof = [p for p in BASE[1:] + SIDE[1:] if p[1] <= FILL_Z]
    inner = [(max(r - 2.6, 0.0), z + 2.6) for r, z in prof[:-1]]
    rf = _r_at(FILL_Z) - 2.6
    inner = [(0, 5.8)] + inner[1:] + [(rf, FILL_Z)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    return geo.lathe(name, [dict(pts=sc(inner[:2]), uv=UV_LIQ_DISC, mode="disc", flip=True),
                            dict(pts=sc(inner[1:]), uv=UV_LIQ_SIDE),
                            dict(pts=sc([(rf, FILL_Z), (0, FILL_Z)]), uv=UV_LIQ_TOP, mode="disc")], segs=segs)


def cap(name="SM_GlassBottle_Cap"):
    """Crown cap: closed 0.3 mm steel shell, 21 crimp flutes in the skirt. Pivot = its bottom centre (z 274)."""
    outer = [(19.4, 0.0), (19.6, 1.0), (19.0, 4.0), (18.0, 6.5), (16.8, 8.6), (15.0, 9.3), (0, 9.5)]
    inner = [(0, 9.2), (14.9, 9.0), (16.5, 8.2), (17.6, 6.3), (18.6, 3.8), (19.1, 0.8), (19.1, 0.0)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    o = geo.lathe(name, [dict(pts=sc(inner), uv=UV_CAP_IN),
                         dict(pts=sc([(19.1, 0.0)] + outer[:-2]), uv=UV_CAP_SIDE),
                         dict(pts=sc(outer[-3:]), uv=UV_CAP_TOP, mode="disc")], segs=63)
    for v in o.data.vertices:                         # crimp: 21 flutes fading out above the skirt
        r = math.hypot(v.co.x, v.co.y)
        z = v.co.z / MM
        if r > 15.5 * MM and z < 7.5:
            w = 1.0 if z < 4.5 else (7.5 - z) / 3.0
            a = math.atan2(v.co.y, v.co.x)
            k = 1 + (0.75 * math.cos(21 * a) * w) / (r / MM)
            v.co.x *= k
            v.co.y *= k
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.to_mesh(o.data); bm.free()
    o.location.z = CAP_Z * MM
    return o
