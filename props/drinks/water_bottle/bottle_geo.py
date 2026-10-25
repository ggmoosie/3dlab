"""500 mL PET water bottle (bpy). Contract Drink_TwistBottle: Ø65 × 220 mm with cap, waist r 28.5 at z 95, neck r 14.5,
cap r 15.5 × 14 (separate, pivot = its bottom centre on the neck axis). Base centre pivot, +Z up.
Parts: SM_WaterBottle (PET shell: slot 0 PET, slot 1 label band), SM_WaterBottle_Liquid, SM_WaterBottle_Cap.
UV0 2048: label band 10 px/mm; body ~5 px/mm; cap 10 px/mm; liquid its own strip."""
import bpy, bmesh, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
SEGS = 40
CAP_Z0, CAP_Z1, CAP_R = 206.0, 220.0, 15.5
FILL_Z = 178.0
UV_LABEL = (0.0, 0.82, 1.0, 0.995)
UV_LOWER = (0.0, 0.47, 0.5, 0.79)
UV_UPPER = (0.5, 0.47, 1.0, 0.79)
UV_BASE = (0.01, 0.28, 0.17, 0.44)
UV_TOP = (0.19, 0.36, 0.27, 0.44)
UV_CAP_SIDE = (0.30, 0.05, 0.98, 0.17)
UV_CAP_TOP = (0.30, 0.20, 0.46, 0.36)
UV_LIQ_SIDE = (0.0, 0.0, 0.28, 0.24)
UV_LIQ_DISC = (0.48, 0.20, 0.62, 0.34)


def _ribs(z0, z1, n, r, depth=1.1, w=2.2):
    pts, pitch = [], (z1 - z0) / n
    for k in range(n):
        zc = z0 + pitch * (k + 0.5)
        pts += [(r, zc - w), (r - depth, zc), (r, zc + w)]
    return pts


BASE = [(0, 2.2), (9, 1.8), (17, 0.6), (24, 0.0), (28.6, 0.4), (31.4, 2.0), (32.4, 4.6)]
LOWER = [(32.4, 4.6), (32.5, 8.0)] + _ribs(10, 60, 4, 32.5) + [(32.5, 63.0), (31.6, 72), (29.8, 84), (28.5, 95),
                                                                (28.9, 105), (30.4, 114), (31.6, 120), (32.0, 125.0)]
LABEL = [(32.0, 125.0), (32.0, 160.0)]
UPPER = [(32.0, 160.0), (31.4, 167), (29.0, 175), (25.0, 182.5), (19.5, 189), (14.6, 194.0),
         (16.6, 195.6), (16.6, 197.6), (14.0, 198.2), (14.0, 200.6), (15.2, 201.0), (15.2, 205.4), (13.9, 206.2),
         (13.9, 217.8), (12.2, 218.8)]
TOP = [(12.2, 218.8), (6, 218.4), (0, 218.3)]


def arc(prof):
    a = [0.0]
    for p, q in zip(prof, prof[1:]):
        a.append(a[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
    return a


def bottle(name="SM_WaterBottle", segs=SEGS):
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    o = geo.lathe(name, [dict(pts=sc(BASE), uv=UV_BASE, mode="disc", flip=True),
                         dict(pts=sc(LOWER), uv=UV_LOWER), dict(pts=sc(LABEL), uv=UV_LABEL),
                         dict(pts=sc(UPPER), uv=UV_UPPER), dict(pts=sc(TOP), uv=UV_TOP, mode="disc")], segs=segs)
    me = o.data
    for nm in ("M_WaterBottle_PET", "MI_WaterBottle_Label"):       # slots must exist or indices clamp to 0
        me.materials.append(bpy.data.materials.get(nm) or bpy.data.materials.new(nm))
    uv = me.uv_layers.active.data
    for p in me.polygons:                         # slot 1 = the label band (its UV island)
        v = sum(uv[l].uv.y for l in p.loop_indices) / p.loop_total
        p.material_index = 1 if v > UV_LABEL[1] - 1e-4 else 0
    return o


def liquid(name="SM_WaterBottle_Liquid", segs=32):
    """Water volume inside the wall (smoothed: ignores rib grooves), filled to FILL_Z. Separate so the game can
    drain or tint it."""
    full = [p for p in BASE + LOWER[1:] + LABEL[1:] + UPPER[1:] if p not in LOWER[2:-8] or p[1] > 62]
    full = [p for p in full if not (10 <= p[1] <= 60)] 
    body = [p for p in full if p[1] <= FILL_Z]
    hi = next(p for p in full if p[1] > FILL_Z)
    lo = body[-1]
    rf = lo[0] + (hi[0] - lo[0]) * (FILL_Z - lo[1]) / (hi[1] - lo[1]) - 1.0
    inner = [(max(r - (1.7 if z < 63 else 0.6), 0.0), z + 0.6) for r, z in body] + [(rf, FILL_Z)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    return geo.lathe(name, [dict(pts=sc(inner[:2]), uv=UV_LIQ_DISC, mode="disc", flip=True),
                            dict(pts=sc(inner[1:]), uv=UV_LIQ_SIDE),
                            dict(pts=sc([(rf, FILL_Z), (0, FILL_Z)]), uv=(0.64, 0.20, 0.78, 0.34), mode="disc")], segs=segs)


def cap(name="SM_WaterBottle_Cap", segs=64):
    """Screw cap: knurled skirt (normal map), domed-flat top with a rolled edge. Closed solid; inner bore hidden."""
    r = CAP_R
    prof = [(14.0, 0.0), (r - 0.4, 0.0), (r, 0.5), (r, 12.8), (r - 0.5, 13.7), (r - 1.4, 14.0)]
    top = [(r - 1.4, 14.0), (8, 14.05), (0, 14.1)]
    inner = [(0, 13.0), (14.0, 13.0), (14.0, 0.0)]
    sc = lambda pts: [(rr * MM, z * MM) for rr, z in pts]
    o = geo.lathe(name, [dict(pts=sc(inner), uv=(0.47, 0.20, 0.49, 0.36)),
                         dict(pts=sc(prof), uv=UV_CAP_SIDE), dict(pts=sc(top), uv=UV_CAP_TOP, mode="disc")], segs=segs)
    o.location.z = CAP_Z0 * MM                    # pivot = bottom centre of the cap, on the neck axis
    return o
