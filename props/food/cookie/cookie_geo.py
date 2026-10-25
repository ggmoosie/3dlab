"""Cookie (bpy). Contract Eat_Disc: Ø7.5 × 1.5 cm, pivot centre, disc axis = Y.
High: dense lathe displaced by a crack/dome/chip height field. Low: light lathe of the same silhouette; normals baked.
UV: top disc (planar), bottom disc (planar, flipped), side band."""
import bpy, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab import geo

MM = 0.001
R, TH = 37.5, 15.0
UV_TOP = (0.0, 0.40, 0.58, 0.98)
UV_BOT = (0.60, 0.40, 0.98, 0.78)
UV_SIDE = (0.0, 0.02, 1.0, 0.36)
_rng = np.random.default_rng(12)
CHIPS = [(*(_rng.uniform(-1, 1, 2) * 27), _rng.uniform(2.6, 4.2)) for _ in range(14)]


def edge_r(a):
    """Irregular outline radius (mm) at angle a."""
    return R - 1.6 + 1.0 * math.sin(3 * a + 0.4) + 0.6 * math.sin(7 * a + 1.3) + 0.35 * math.sin(13 * a)


def height(x, y):
    """Top surface height above the base (mm): dome + chips standing proud (cracks live in the normal map)."""
    r = math.hypot(x, y) / R
    h = TH * (0.62 + 0.38 * max(0.0, 1 - r * r)) * (1 - 0.25 * r ** 6)
    for cx, cy, s in CHIPS:
        d = math.hypot(x - cx, y - cy)
        if d < s:
            h += 1.6 * (1 - (d / s) ** 2)
    return h


def cookie(name, segs=40, rings=8, high=False):
    """Lathe-like disc with an irregular rim; top vertices lifted by height()."""
    n_r = rings * (3 if high else 1)
    s = segs * (3 if high else 1)
    prof_top = [(R * (1 - i / n_r), 1.0) for i in range(n_r + 1)]
    prof_top = [(r, z) for r, z in prof_top if r < R - 1.0]
    prof = [(0, 0.0), (R * 0.6, 0.0), (R - 3.0, 0.2), (R - 0.8, 1.6), (R, 4.5)] + [(R - 0.3, 6.5)] + prof_top
    sc = [(r * MM, z * MM) for r, z in prof]
    bot = sc[:4]
    side = sc[3:6] + [sc[6]]
    top = sc[6:]
    o = geo.lathe(name, [dict(pts=bot, uv=UV_BOT, mode="disc", flip=True), dict(pts=side, uv=UV_SIDE),
                         dict(pts=top, uv=UV_TOP, mode="disc")], segs=s)
    for v in o.data.vertices:
        x, y, z = v.co.x / MM, v.co.y / MM, v.co.z / MM
        rr = math.hypot(x, y)
        a = math.atan2(y, x)
        k = edge_r(a) / R
        v.co.x, v.co.y = v.co.x * k, v.co.y * k
        if z > 0.9 and rr < R - 0.5:                     # top surface rings
            v.co.z = height(x * k, y * k) * MM
        elif z > 6.0:                                    # rim roll-over: just under the outer top ring
            v.co.z = height(x * k * 0.985, y * k * 0.985) * 0.8 * MM
    o.location.z = 0
    geo.set_origin(o, (0, 0, TH / 2 * MM))
    o.location = (0, 0, 0)
    from mathutils import Matrix
    o.data.transform(Matrix.Rotation(-math.pi / 2, 4, "X"))   # disc axis → Y (contract), applied to the mesh
    return o
