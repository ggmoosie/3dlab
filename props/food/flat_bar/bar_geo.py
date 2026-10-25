"""Flat bar in a flow-wrap (pillow pack) (bpy). Contract Eat_BarFlat: 5.4 × 0.42 × 12.8 cm, pivot centre, +Z = bite end.
Parts: SM_FlatBar (wrapper sleeve, torn open at +Z, -Z crimp seal + the bar inside, segment-moulded)
       SM_FlatBar_TearEnd (the +Z end: crimp seal + a sleeve stub with the matching jagged tear; pulls off along +Z)
Wrapper film is solidified (0.12 mm) so every part is a closed shell. UVs: one unrolled print sheet across both pieces."""
import bpy, bmesh, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
W, T, L = 54.0, 4.2, 128.0
FIN = 9.0                      # crimp seal length at each end
TEAR_Z = 46.0                  # tear line (mean), mm from centre toward +Z
UV_SHEET = (0.0, 0.30, 1.0, 1.0)
UV_BAR = (0.0, 0.0, 0.75, 0.28)           # bar faces (front/back share it); sides + ends in strips beside


def section(n=28, w=W, t=T, squash=1.0):
    """Pillow cross-section: rounded rect, perimeter-ordered, starting at the back centre (the fin seal)."""
    pts = []
    r = t / 2 * squash
    hw = w / 2 - r
    per = []                                              # walk: back (-Y) centre → +X → front (+Y) → -X → back
    for k in range(n):
        s = k / n
        per.append(s)
    total = 4 * hw + 2 * math.pi * r
    for s in per:
        d = s * total
        if d < hw:
            pts.append((d, -r))
        elif d < hw + math.pi * r:
            a = -math.pi / 2 + (d - hw) / r
            pts.append((hw + r * math.cos(a), r * math.sin(a)))
        elif d < 3 * hw + math.pi * r:
            pts.append((hw - (d - hw - math.pi * r), r))
        elif d < 3 * hw + 2 * math.pi * r:
            a = math.pi / 2 + (d - 3 * hw - math.pi * r) / r
            pts.append((-hw + r * math.cos(a), r * math.sin(a)))
        else:
            pts.append((-hw + (d - 3 * hw - 2 * math.pi * r), -r))
    return pts, total


def _sheet(name, z0, z1, nz, tear0=None, tear1=None, crimp_end=None):
    """Sleeve between z0..z1 (tear0/tear1: functions x→z offset for jagged edges), optional crimp fin at one end
    (the sleeve pinches flat to a 0 thickness fin over FIN mm). UV: sheet coordinates (perimeter, z)."""
    sec, per = section()
    n = len(sec)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    rows = []
    zs = [z0 + (z1 - z0) * j / nz for j in range(nz + 1)]
    for j, z in enumerate(zs):
        row = []
        for i, (x, y) in enumerate(sec):
            zz = z
            if j == 0 and tear0:
                zz += tear0(x, i)
            if j == nz and tear1:
                zz += tear1(x, i)
            yy = y
            if crimp_end is not None:                     # pinch toward the fin
                f = max(0.0, min(1.0, (zz - crimp_end[0]) / (crimp_end[1] - crimp_end[0])))
                yy = y * (1 - f) ** 2 + math.copysign(0.08 * f, y) if abs(y) > 1e-6 else 0.0
            row.append(bm.verts.new((x * MM, yy * MM, zz * MM)))
        rows.append(row)
    for j in range(nz):
        for i in range(n):
            i1 = (i + 1) % n
            f = bm.faces.new((rows[j][i], rows[j][i1], rows[j + 1][i1], rows[j + 1][i]))
            for l, (ii, jj) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                u0, v0, u1, v1 = UV_SHEET
                zz = l.vert.co.z / MM
                l[uv].uv = (u0 + (u1 - u0) * ii / n, v0 + (v1 - v0) * (zz + L / 2) / L)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    for p in me.polygons:
        p.use_smooth = True
    sol = o.modifiers.new("Film", "SOLIDIFY")
    sol.thickness, sol.offset, sol.use_even_offset = 0.12 * MM, 1.0, False
    sol.material_offset = 1                              # inside of the film → slot 1 (foil)
    for nm in ("MI_FlatBar", "M_FlatBar_Foil"):
        o.data.materials.append(bpy.data.materials.get(nm) or bpy.data.materials.new(nm))
    geo.apply_mods(o)
    geo.fix_zero_uv(o, (0.0, 0.285, 0.12, 0.299))                # film rims along the tear
    o["qc_uv_overlap_ok"] = True                         # film: inner faces reuse the outer UVs (foil slot)
    return o


def _tear(x, i):
    return 1.6 * math.sin(i * 1.7) + 1.1 * math.sin(i * 0.53 + 1.0)


def wrapper():
    body = _sheet("SM_FlatBar", -L / 2, TEAR_Z, 26, tear1=_tear, crimp_end=(-L / 2 + FIN, -L / 2))
    return body


def tear_end():
    o = _sheet("SM_FlatBar_TearEnd", TEAR_Z, L / 2, 6, tear0=lambda x, i: _tear(x, i) + 0.25, crimp_end=(L / 2 - FIN, L / 2))
    geo.set_origin(o, (0, 0, TEAR_Z * MM))
    return o


def bar(name="bar"):
    """The bar: rounded block 50 × 3.4 × 112 (moulded segments in the normal map), a closed shell inside the film."""
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object
    o.name = name
    o.scale = (50 * MM, 3.6 * MM, 104 * MM)
    bpy.ops.object.transform_apply(scale=True)
    b = o.modifiers.new("B", "BEVEL"); b.width = 1.2 * MM; b.segments = 2
    geo.apply_mods(o)
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.set_sharp_from_angle(angle=math.radians(50))
    bm = bmesh.new(); bm.from_mesh(o.data)
    uv = bm.loops.layers.uv.new("UVMap")
    rects = {1: UV_BAR, 0: (0.77, 0.0, 0.86, 0.28), 2: (0.88, 0.0, 0.98, 0.10)}
    span = {1: ((0, 25), (2, 52)), 0: ((1, 2.0), (2, 52)), 2: ((0, 25), (1, 2.0))}
    for f in bm.faces:
        ax = max(range(3), key=lambda i: abs(f.normal[i]))
        (a1, h1), (a2, h2) = span[ax]
        u0, v0, u1, v1 = rects[ax]
        for l in f.loops:
            c = l.vert.co / MM
            l[uv].uv = (u0 + (u1 - u0) * (0.5 + c[a1] / (2 * h1)), v0 + (v1 - v0) * (0.5 + c[a2] / (2 * h2)))
    bm.to_mesh(o.data); bm.free()
    return o


def flat_bar():
    w, b = wrapper(), bar()
    b.data.materials.append(w.data.materials[0])
    o = geo.join([w, b], "SM_FlatBar")
    o["qc_note"] = "the bar rides loose 0.3 mm inside the film: QC with --allow-float SM_FlatBar"
    return o
