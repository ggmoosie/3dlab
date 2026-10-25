"""15 oz food can, opened, with a full-aperture easy-open lid you tear back (bpy).
Contract Eat_CanSpoon: can Ø7.8 × 11.2 cm, pivot base centre, +Z up, held in the LEFT hand; treated as open.
Parts: SM_FoodCan (body: beads, double seams, lacquered inside, food surface at FOOD_Z)
       SM_FoodCan_Lid  (scored panel + stiffening rings; pivot = hinge on the rim opposite the ring, axis = tangent)
       SM_FoodCan_Ring (pull ring; pivot = rivet; lifts about its local X)"""
import bpy, bmesh, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
SEGS = 40
FOOD_Z = 97.0
LID_Z = 106.6
RIVET = (0.0, -24.0)                 # rivet near the -Y edge; the ring lies toward the centre
UV_SIDE = (0.0, 0.52, 1.0, 0.995)    # outside wall (label)
UV_INNER = (0.0, 0.36, 0.55, 0.50)   # rim + inner wall (lacquer)
UV_FOOD = (0.57, 0.12, 0.79, 0.34)   # contents surface disc (variant prints the food here)
UV_BOTTOM = (0.01, 0.12, 0.23, 0.34)
UV_LID = (0.25, 0.04, 0.55, 0.34)
UV_LIDSIDE = (0.57, 0.36, 0.99, 0.40)
UV_RING = (0.81, 0.04, 0.99, 0.30)


def _beads(z0, z1, n, r=38.1, d=0.9, w=1.6):
    out, p = [], (z1 - z0) / n
    for k in range(n):
        zc = z0 + p * (k + 0.5)
        out += [(r, zc - w), (r - d, zc), (r, zc + w)]
    return out


BOTTOM = [(0, 2.6), (14, 2.6), (20, 1.9), (24, 2.6), (30, 2.4), (34, 2.0), (36.3, 0.3), (37.5, 0.0)]
SIDE = [(37.5, 0.0), (38.6, 1.0), (38.9, 3.0), (38.3, 5.0), (38.1, 6.5)] + \
       [(38.1, 104.5), (38.4, 106.4), (38.9, 107.8), (39.0, 110.6), (38.5, 111.8), (37.7, 112.0)]
INNER = [(37.7, 112.0), (37.0, 111.5), (36.9, 108.8), (37.6, 106.0), (37.7, FOOD_Z + 1.2), (37.4, FOOD_Z)]
FOOD = [(37.4, FOOD_Z), (20, FOOD_Z + 1.4), (0, FOOD_Z + 2.0)]


BEADS = [16 + (94 - 16) / 7 * (k + 0.5) for k in range(7)]   # bead centres (mm): normal-map only


def body(name="SM_FoodCan"):
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    return geo.lathe(name, [dict(pts=sc(BOTTOM), uv=UV_BOTTOM, mode="disc", flip=True), dict(pts=sc(SIDE), uv=UV_SIDE),
                            dict(pts=sc(INNER), uv=UV_INNER), dict(pts=sc(FOOD), uv=UV_FOOD, mode="disc")], segs=SEGS)


def lid(name="SM_FoodCan_Lid"):
    """Panel that sat in the countersink: closed 0.4 mm plate with a curled edge; rings as low ridges (geometry)."""
    top = [(0, LID_Z + 0.4), (12, LID_Z + 0.4), (14, LID_Z + 0.9), (16, LID_Z + 0.4), (27, LID_Z + 0.4), (29, LID_Z + 0.9),
           (31, LID_Z + 0.4), (35.2, LID_Z + 0.4), (36.4, LID_Z + 0.9)]
    edge = [(36.4, LID_Z + 0.9), (36.8, LID_Z + 0.3), (36.4, LID_Z - 0.1)]
    under = [(36.4, LID_Z - 0.1), (0, LID_Z)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    o = geo.lathe(name, [dict(pts=sc(top), uv=UV_LID, mode="disc"), dict(pts=sc(edge), uv=UV_LIDSIDE),
                         dict(pts=sc(under), uv=(0.57, 0.42, 0.79, 0.50), mode="disc", flip=True)], segs=SEGS)
    hinge = (0.0, 36.6 * MM, LID_Z * MM)                     # far edge from the ring (+Y)
    geo.set_origin(o, hinge)
    return o


def ring(name="SM_FoodCan_Ring"):
    """Pull ring: a flat oval loop + a nose plate at the rivet, lying on the lid with the loop toward the centre."""
    bm = bmesh.new()
    cx, cy, R, w, t = 0.0, -24.0 + 12.5, 9.5, 2.6, 0.55
    loop = []
    n = 36
    for k in range(n):                                       # oval loop cross-section: rectangle w × t
        a = 2 * math.pi * k / n
        x, y = cx + R * 0.85 * math.cos(a), cy + R * math.sin(a)
        nx, ny = math.cos(a) / 0.85, math.sin(a)
        L = math.hypot(nx, ny); nx, ny = nx / L, ny / L
        loop.append([(x + nx * dx, y + ny * dx, LID_Z + 0.5 + dz) for dx, dz in ((-w / 2, 0), (w / 2, 0), (w / 2, t), (-w / 2, t))])
    vs = [[bm.verts.new((p[0] * MM, p[1] * MM, p[2] * MM)) for p in sec] for sec in loop]
    for k in range(n):
        a, b = vs[k], vs[(k + 1) % n]
        for i in range(4):
            bm.faces.new((a[i], a[(i + 1) % 4], b[(i + 1) % 4], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    # nose tab: a plate from the loop to the rivet, unioned into one closed shell
    pb = bmesh.new()
    plate = [(-4.0, -27.5), (4.0, -27.5), (3.2, -19.6), (-3.2, -19.6)]
    lo = [pb.verts.new((x * MM, y * MM, (LID_Z + 0.53) * MM)) for x, y in plate]
    hi = [pb.verts.new((x * MM, y * MM, (LID_Z + 0.93) * MM)) for x, y in plate]
    pb.faces.new(lo[::-1]); pb.faces.new(hi)
    for i in range(4):
        pb.faces.new((lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i]))
    pm = bpy.data.meshes.new("plate"); pb.to_mesh(pm); pb.free()
    po = bpy.data.objects.new("plate", pm)
    bpy.context.scene.collection.objects.link(po)
    bo = o.modifiers.new("U", "BOOLEAN"); bo.operation = "UNION"; bo.object = po; bo.solver = "EXACT"
    geo.apply_mods(o)
    bpy.data.objects.remove(po)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data)
    bm.free()
    me = o.data
    for p in me.polygons:
        p.use_smooth = True
    me.set_sharp_from_angle(angle=math.radians(50))
    geo.unwrap_into(o, UV_RING)
    geo.set_origin(o, (RIVET[0] * MM, RIVET[1] * MM, (LID_Z + 0.5) * MM))
    return o
