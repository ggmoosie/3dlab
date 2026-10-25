"""Hard-pack cigarettes, a single cigarette, a disposable lighter (bpy).
Contract Smoke_Cigarette: cigarette 8.5 × 0.8 cm (filter 2.5), pivot at the filter end, +Z toward the tip.
Contract Lighter_Flick: lighter 2.5 × 1.3 × 6 cm + hood; moving: striker wheel, (gas button); a flame card.
Parts: SM_CigPack (hollow body, inner collar, 20 cigarettes), SM_CigPack_Lid (pivot = back hinge line),
       SM_Cigarette, SM_Lighter (body + hood + fork), SM_Lighter_Wheel (pivot = axle), SM_Lighter_Button, SM_Lighter_Flame."""
import bpy, bmesh, math, os, sys
from mathutils import Matrix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
PW, PD, PH, CUT = 55.0, 22.0, 88.0, 70.0        # pack size, lid cut height
# dieline (2048): faces of the whole 88 mm pack; body + lid share it so the print runs across the lid cut
FRONT, BACK = (0.00, 0.40, 0.34, 0.945), (0.35, 0.40, 0.69, 0.945)
LEFT, RIGHT = (0.70, 0.40, 0.836, 0.945), (0.85, 0.40, 0.986, 0.945)
TOP, BOTTOM = (0.35, 0.25, 0.69, 0.386), (0.0, 0.25, 0.34, 0.386)
UV_INNER = (0.70, 0.25, 0.99, 0.386)
UV_CIG = (0.0, 0.0, 0.30, 0.22)
UV_LIGHTER = (0.32, 0.0, 0.64, 0.22)
UV_METAL = (0.66, 0.0, 0.80, 0.22)
X, Y, Z = (-PW / 2, PW / 2), (-PD / 2, PD / 2), (0.0, PH)
DIELINE = {"-Y": (FRONT, X, Z), "+Y": (BACK, X, Z), "-X": (LEFT, Y, Z), "+X": (RIGHT, Y, Z), "+Z": (TOP, X, Y), "-Z": (BOTTOM, X, Y)}


def cigarette(name="SM_Cigarette", segs=12, cut_z=None):
    """Filter (0..25 mm) + paper tube (25..85 mm), slightly sunken ash end. Pivot = filter end."""
    r = 3.9
    pts = [(0, 0.0), (r - 0.3, 0.0), (r, 0.4), (r, 84.6), (r - 0.4, 85.0), (2.0, 84.4), (0, 84.3)]
    if cut_z:
        pts = [(rr, min(z, cut_z)) for rr, z in pts]
    sc = [(rr * MM, z * MM) for rr, z in pts]
    return geo.lathe(name, [dict(pts=sc, uv=UV_CIG)], segs=segs)


def pack():
    body = geo.rounded_box("SM_CigPack", PW, PD, CUT, 1.2, 0.0, 0.6, segs=3)
    cav = geo.rounded_box("cav", PW - 1.2, PD - 1.2, CUT, 0.8, 0.6, 0.0, segs=2)
    geo.boolean(body, cav)
    collar = geo.rounded_box("collar", PW - 1.3, PD - 1.3, 18, 0.6, CUT - 8, 0.0, segs=2)
    ccut = geo.rounded_box("ccut", PW - 2.2, PD - 2.2, 20, 0.4, CUT - 9, 0.0, segs=2)
    geo.boolean(collar, ccut)
    front = geo.rounded_box("front", PW - 8, 4, 10, 0.0, CUT + 1, 0.0, segs=1, center=(0, -(PD / 2 - 1.0)))
    geo.boolean(collar, front)                                        # the collar is cut low at the front
    cigs = []
    for row, (n, y) in enumerate(((7, -6.6), (6, 0.0), (7, 6.6))):
        for i in range(n):
            x = (i - (n - 1) / 2) * 7.7
            c = cigarette(f"c{row}{i}", 8, cut_z=None)
            c.data.transform(Matrix.Rotation(math.pi, 4, "X"))          # filters up, like a real pack
            c.location = (x * MM, y * MM, (85.6 + (0.6 if (i + row) % 3 == 0 else 0)) * MM)
            geo.bake_xform(c)
            cigs.append(c)
    for part, slot in [(body, 0), (collar, 1)] + [(c, 2) for c in cigs]:
        for nm in ("MI_CigPack", "M_CigPack_Foil", "MI_Cigarette"):
            part.data.materials.append(bpy.data.materials.get(nm) or bpy.data.materials.new(nm))
        for f in part.data.polygons:
            f.material_index = slot
    geo.box_uv(body, DIELINE, (0, 0, PH / 2 * MM), UV_INNER)
    geo.unwrap_into(collar, (0.70, 0.25, 0.99, 0.30))
    o = geo.join([body, collar] + cigs, "SM_CigPack")
    o["qc_note"] = "20 cigarettes share one UV island and the foil collar is tiny: QC --uv-overlap-ok SM_CigPack"
    return o


def lid():
    l = geo.rounded_box("SM_CigPack_Lid", PW, PD, PH - CUT, 1.2, CUT + 0.15, 0.6, segs=3)
    cav = geo.rounded_box("lcav", PW - 1.2, PD - 1.2, PH - CUT, 0.8, CUT - 0.45, 0.0, segs=2)
    geo.boolean(l, cav)
    geo.box_uv(l, DIELINE, (0, 0, PH / 2 * MM), (0.70, 0.31, 0.99, 0.386))
    geo.set_origin(l, (0, PD / 2 * MM, CUT * MM))
    return l


def lighter():
    """Disposable lighter: rounded translucent body, steel windguard with vent holes, fork; wheel + button separate."""
    body = geo.rounded_box("SM_Lighter", 25, 13, 50, 5.5, 0.0, 0.8, segs=5)
    guard = geo.rounded_box("guard", 22, 11.4, 11, 4.5, 49.5, 0.3, segs=4)
    gcut = geo.rounded_box("gcut", 21.0, 10.4, 12, 4.0, 50.2, 0.0, segs=4)
    geo.boolean(guard, gcut)
    for x in (-5, 0, 5):                                              # vent slots on both faces
        geo.boolean(guard, geo.rounded_box("v", 2.2, 16, 5, 0.6, 53, 0.0, segs=2, center=(x, 0)))
    nozzle = geo.lathe("nozzle", [dict(pts=[(0, 50 * MM), (1.6 * MM, 50 * MM), (1.6 * MM, 55 * MM), (0.8 * MM, 56 * MM), (0, 56 * MM)],
                                       uv=(0.32, 0.225, 0.35, 0.245))], segs=10)
    nozzle.location.x = 4 * MM
    geo.bake_xform(nozzle)
    for part, slot in ((body, 0), (guard, 1), (nozzle, 1)):
        for nm in ("MI_Lighter", "M_Lighter_Steel"):
            part.data.materials.append(bpy.data.materials.get(nm) or bpy.data.materials.new(nm))
        for f in part.data.polygons:
            f.material_index = slot
    o = geo.join([body, guard, nozzle], "SM_Lighter")
    bm = bmesh.new(); bm.from_mesh(o.data)                 # boolean n-gons around the vents are concave: triangulate
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(o.data); bm.free()
    geo.unwrap_into(o, (0.32, 0.0, 0.80, 0.22))        # one packed layout: no print on the lighter, so no dieline
    return o


def wheel():
    pts = [(0, -4.0), (4.3, -4.0), (4.3, 4.0), (0, 4.0)]
    o = geo.lathe("SM_Lighter_Wheel", [dict(pts=[(r * MM, z * MM) for r, z in pts], uv=(0.81, 0.0, 0.99, 0.10))], segs=24)
    o.data.transform(Matrix.Rotation(math.pi / 2, 4, "Y"))           # axle along X
    o.location = (-4.5 * MM, 0, 57.0 * MM)
    return o


def button():
    b = geo.rounded_box("SM_Lighter_Button", 11.6, 8.6, 4, 1.5, 51.3, 0.4, segs=3, center=(-5.0, 0))
    geo.unwrap_into(b, (0.81, 0.11, 0.99, 0.22))
    geo.set_origin(b, (-5.0 * MM, 0, 51.3 * MM))
    return b


def flame():
    """Flame card: two crossed quads (masked, emissive) above the nozzle. One-sided by design."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    for ax in (0, 1):
        pts = [(-6, 0), (6, 0), (6, 28), (-6, 28)]
        vs = [bm.verts.new(((x if ax == 0 else 0) * MM, (x if ax == 1 else 0) * MM, z * MM)) for x, z in pts]
        f = bm.faces.new(vs)
        for l, (uu, vv) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            l[uv].uv = (uu, vv)
    me = bpy.data.meshes.new("SM_Lighter_Flame"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("SM_Lighter_Flame", me); bpy.context.scene.collection.objects.link(o)
    o.location = (4 * MM, 0, 56 * MM)
    o["qc_open"] = True
    return o
