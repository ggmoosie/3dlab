"""parts_v2.py - SCAR 17S NRCH high-poly from the traced profile master (ref.py) -> work/high.npz.
Every part's side outline is the photo silhouette (ref.prof); widths, rounded sections and all 3-D detail (rail dovetail,
M-LOK slots, ports, QD cups, controls, sights, brake ports, stipple panels) are placed at photo-measured positions.
usage: .venv/bin/python guns/scar17s/parts_v2.py [h_scale=1.0] [parts=a,b]"""
import sys, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S
import ref as R
import parts as V1                                   # bolt carrier + bolt and the cartridge are reused
import detail as DT
import relief as RL

MAT = dict(anod=0, poly=1, steel=2, park=3, rubber=4, brass=5, magsteel=6, copper=7)
HS = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("h_scale=")), 1.0))
ONLY = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("parts=")), None)


def bp(x, row):
    y, z = R.px2b(x, row)
    return float(y), float(z)


def bx(x0, r0, x1, r1):
    """photo px box -> (y0, y1, z0, z1) in B mm"""
    (ya, za), (yb, zb) = bp(x0, r1), bp(x1, r0)
    return ya, yb, za, zb


def pbox(X, Y, Z, x0, r0, x1, r1, xlo, xhi, r=0.0):
    y0, y1, z0, z1 = bx(x0, r0, x1, r1)
    return S.box(X, Y, Z, (xlo, y0, z0), (xhi, y1, z1), r)


def side_cut(d, X, Y, Z, x0, r0, x1, r1, hw, depth, r=0.6, both=True, sign=1):
    """recess a px box into the side wall(s) at |x| = hw, depth mm"""
    sx = np.abs(X) if both else X * sign
    return S.rdiff(d, pbox(sx, Y, Z, x0, r0, x1, r1, hw - depth, hw + 20, min(r, 1.0)), r)


def stadium(X, Y, Z, x0, r0, x1, r1, xin, xout, sign):
    """raised stadium plate on one side (sign +1 left, -1 right), from |x| = xin to xout"""
    y0, y1, z0, z1 = bx(x0, r0, x1, r1)
    h = (z1 - z0) / 2
    d2 = S.capsule2(Y, Z, (y0 + h, (z0 + z1) / 2), (y1 - h, (z0 + z1) / 2), h)
    return S.extrude(d2, X * sign, xin, xout, 0.6)


# ------------------------------------------------------------------ upper receiver + handguard + hinge block + rail
HG_X = 1140.0                                         # handguard proud of the receiver from here forward


lerp = S.lerp


def _tube_arc(x0, r0, x1, r1, n=24):
    """arc of a circle centred on the bore plane (x = 0) from (x0, row r0) up to (x1, row r1): the upper's tube shoulder"""
    z0, z1 = bp(0, r0)[1], bp(0, r1)[1]
    zc = (x1 * x1 + z1 * z1 - x0 * x0 - z0 * z0) / (2 * (z1 - z0))
    rad = math.hypot(x0, z0 - zc)
    a0, a1 = math.atan2(z0 - zc, x0), math.atan2(z1 - zc, x1)
    return [(rad * math.cos(a), zc + rad * math.sin(a)) for a in np.linspace(a0, a1, n)]


def _roll(xb, rb, xt, rt, n=16):
    """the body line (user + refs): the wall comes straight down from the square top, then rolls out round to its
    widest at the crease -- quarter-sine from (xt, row rt) at the top down to (xb, row rb), listed bottom -> top"""
    zb, zt = bp(0, rb)[1], bp(0, rt)[1]
    u = np.linspace(0, 1, n)                                                          # 0 at the crease, 1 at the top
    return [(xt + (xb - xt) * math.cos(v * math.pi / 2), zb + (zt - zb) * v) for v in u]


def _section(pts):
    """right-half outline [(x mm, z mm)] bottom-centre -> top-centre; returns (Field2D sdf of the mirrored section in
    (X, Z), half-width-at-z polyline)"""
    full = pts + [(-x, z) for x, z in pts[::-1]]
    res, pad = 0.1, 15.0
    xs, zs = [p[0] for p in full], [p[1] for p in full]
    x0, z1 = min(xs) - pad, max(zs) + pad
    W, H = int((max(xs) + pad - x0) / res), int((z1 - min(zs) + pad) / res)
    from PIL import Image as _I, ImageDraw as _D
    im = _I.new("L", (W, H), 0)
    _D.Draw(im).polygon([((x - x0) / res, (z1 - z) / res) for x, z in full], fill=255)
    from scipy import ndimage as _nd
    m = _nd.gaussian_filter(np.asarray(im).astype(np.float32), 0.6 / res) > 127         # soften kinks ~0.6 mm
    dd = (_nd.distance_transform_edt(~m) - _nd.distance_transform_edt(m)) * res
    half = sorted((z, x) for x, z in pts if x > 0)
    return S.Field2D(dd.astype(np.float32), x0, z1, res, outside=20.0), ([h[0] for h in half], [h[1] for h in half])


def _rows(pts):
    return [(x, bp(0, r)[1]) for x, r in pts]


_SEC = {}


def sections():
    """receiver: square top under the rail, tube shoulder (coaxial arc) down to the stepped body line, flat side.
    handguard: same square top + tube shoulder down to the crease above the upper M-LOK row, vertical slot band,
    45 deg lower facet (lower slot row), flat bottom = the octagon"""
    if not _SEC:
        rec = _rows([(0, 945), (25.4, 945), (25.4, 901), (26.8, 899)]) + _roll(26.8, 897, 23.4, 874) + \
            _rows([(23.4, 849), (21.0, 846), (0, 846)])
        hg = _rows([(0, 967), (HG_SIDE - HG_FACET, 967), (HG_SIDE, 944), (HG_SIDE, 907), (28.0, 905)]) + \
            _roll(28.0, 903, 22.0, 872) + _rows([(22.0, 849), (20.0, 846), (0, 846)])
        _SEC["rec"], _SEC["hg"] = _section(rec), _section(hg)
    return _SEC


def _blend(Y):
    yb, _ = bp(HG_X, 0)
    return S.sstep(yb - 22, yb + 45, Y)                                              # handguard sweeps out of the receiver


def upper_sec(X, Y, Z):
    """signed distance of the blended cross-section (bezel ring 1.4 proud at the nose)"""
    sc = sections(); t = _blend(Y)
    d = lerp(sc["rec"][0](X, Z), sc["hg"][0](X, Z), t)
    return d - 1.4 * (Y > R.extent("upper")[1] - HG_BEZEL)


def upper_hw(Y, Z):
    """side half-width at (Y, Z) of the blended section (hardware, screws and cuts sit on this surface)"""
    sc = sections(); t = _blend(Y)
    hr, hh = np.interp(Z, *sc["rec"][1]), np.interp(Z, *sc["hg"][1])
    return lerp(hr, hh, t) + 1.4 * (Y > R.extent("upper")[1] - HG_BEZEL)


def rail_xs(X, Z, z_top, z_base):
    """1913 dovetail: neck 16 wide, 21.2 across the vee, flat top 15.6"""
    zc = z_top - 2.8
    neck = np.maximum(np.abs(X) - 8.0, np.maximum(z_base - 1.0 - Z, Z - zc))
    dove = np.maximum((np.abs(X) + np.abs(Z - zc)) * 0.7071 - 10.6 * 0.7071, np.maximum(zc - 2.6 - Z, Z - z_top))
    return np.minimum(neck, dove)


HG_FACET, HG_WALL, HG_BEZEL, HG_SIDE = 13.0, 4.6, 14.0, 27.2                        # lower facet size (side-view rows 944-967), wall (pre-width-scale)


def facet_slot(X, Y, Z, x0, x1, zhb, w=4.6):
    """M-LOK slot cut square to the 45 deg lower facet, through the wall"""
    u = (np.abs(X) + (Z - zhb)) * 0.7071                                             # along the facet (up-out)
    n = (np.abs(X) - (Z - zhb)) * 0.7071                                             # facet normal (down-out)
    u0 = HG_SIDE * 0.7071                                                              # facet midpoint
    y0, y1, _, _ = bx(x0, 0, x1, 0)
    cap = S.capsule2(Y, u, (y0 + w / 2, u0), (y1 - w / 2, u0), w / 2)
    n0 = (HG_SIDE - HG_FACET) * 0.7071
    return np.maximum(cap, np.abs(n - n0) - 8.0)


def upper(X, Y, Z):
    hw = upper_hw(Y, Z)
    P2 = R.prof("upper")(Y, Z)
    d = S.rinter(P2, upper_sec(X, Y, Z), 2.0)
    yb, _ = bp(HG_X, 0)
    _, zhb = bp(0, 967)
    ax = np.abs(X)
    # handguard section (3/4 photos): vertical upper walls, a big 45 deg lower facet carrying the lower M-LOK row,
    # narrow flat bottom; the front outer edge is bevelled
    yfr = R.extent("upper")[1]
    d = S.rdiff(d, np.maximum(np.abs(Y - (yfr - HG_BEZEL)) - 0.5, hw - 0.7 - ax), 0.3)   # groove behind the bezel
    d = np.maximum(d, np.where(Y > yb, (Y - yfr) + (ax - hw) + 3.0, -1e3) * 0.7071)
    # top rail (teeth = the photo's)
    _, zrt = bp(0, 826); _, zrb = bp(0, 846)
    rl = np.maximum(rail_xs(X, Z, zrt, zrb), R.prof("rail")(Y, Z))
    d = S.runion(d, rl, 0.5)
    # hinge block (behind the receiver), knuckles on the right side
    hb = S.rinter(R.prof("hinge")(Y, Z), np.abs(X) - 24.0, 1.5)
    hy, _ = bp(588, 0)
    for r0, r1 in ((852, 893), (905, 948), (958, 996)):
        _, za = bp(0, r1); _, zb = bp(0, r0)
        hb = S.runion(hb, S.cyl(X, Y, Z, "z", (-23.0, hy), 6.2, za, zb, 0.8), 0.8)
    d = S.union(d, hb)
    # cavities: bolt raceway (receiver) and the free-float channel (handguard, open at the front)
    d = S.diff(d, S.box(X, Y, Z, (-17.5, bp(625, 0)[0], -14), (17.5, yb, 30), 2.5))
    # ejection port (right side): open window through the wall into the raceway -- the bolt carrier shows through
    ya, yb2, za, zb = bx(864, 910, 1021, 930)
    d = S.rdiff(d, S.box(X, Y, Z, (-60.0, ya, za), (-14.0, yb2, zb), 1.6), 0.6)
    # thin-walled handguard shell, open at the front (inner = the outer section offset by the wall)
    _, zin = bp(0, 862)
    inner = np.maximum.reduce([upper_sec(X, Y, Z) + HG_WALL, Z - zin, yb - 30 - Y, np.hypot(X, Z) - 33.0])   # arched top
    d = S.rdiff(d, inner, 1.5)
    # charging-handle slot (top, blind), port / charging track under the traced hardware (both sides)
    d = side_cut(d, X, Y, Z, 828, 905, 1042, 928, 25.6, 1.6, 1.2)
    # M-LOK: 6 slots per row, 2 rows a side, plus the bottom row; chamfered edges
    xs = [(1178, 1230), (1250, 1302), (1320, 1372), (1392, 1445), (1463, 1516), (1535, 1590)]
    for x0, x1 in xs:
        d = side_cut(d, X, Y, Z, x0, 920, x1, 934, 27.8, 7.0, 2.6)
        d = S.rdiff(d, facet_slot(X, Y, Z, x0, x1, zhb), 0.6)
        y0, y1, _, _ = bx(x0, 0, x1, 0)
        d = S.rdiff(d, S.box(X, Y, Z, (-3.6, y0, zhb - 3), (3.6, y1, zhb + 9), 1.6), 0.9)
    # catalogue: screws, sockets, pins, seams, knuckle gaps, recesses, bosses (detail.py)
    d = DT.apply(d, X, Y, Z, "upper", hw)
    return d


def _teardrop(Y, Z, c0, r0, c1, r1):
    """2-D uneven capsule (iq): circle r0 at c0 hulled with circle r1 at c1 -- the QD plates' fat-teardrop outline"""
    ay, az = c1[0] - c0[0], c1[1] - c0[1]; h = math.hypot(ay, az); ay, az = ay / h, az / h
    u = (Y - c0[0]) * ay + (Z - c0[1]) * az
    v = np.abs(-(Y - c0[0]) * az + (Z - c0[1]) * ay)
    bb = (r0 - r1) / h; aa = math.sqrt(1 - bb * bb)
    k = -bb * v + aa * u
    return np.where(k < 0, np.hypot(v, u) - r0, np.where(k > aa * h, np.hypot(v, u - h) - r1, aa * v + bb * u - r0))


def _qd_plate(Y, Z, x0, x1, r0, r1, tip):
    """QD sling plate (close-up refs): fat teardrop, big round end at the rear, rounded narrow nose forward"""
    h = (r1 - r0) / 2 * R.S_MM
    c0 = bp(x0 + (r1 - r0) / 2, (r0 + r1) / 2)
    c1 = bp(tip[0] - (r1 - r0) * 0.22, tip[1])
    return _teardrop(Y, Z, c0, h, c1, h * 0.45)


def side_plates(X, Y, Z, hw):
    """machined black plates on both receiver sides, measured off the photo (clean edges, not traced pixels):
    ejection-port frame, the brass deflector ramp, rear + front QD plates"""
    ax = np.abs(X)
    fr2 = _rect_px(Y, Z, 858, 906, 1027, 933, 1.0)
    frame = S.extrude(fr2, ax, hw - 0.8, hw + 1.2, 0.6)
    frame = S.runion(frame, S.extrude(np.abs(fr2 + 0.9) - 0.9, ax, hw, hw + 2.0, 0.5), 0.4)   # raised border lip
    frame = S.rdiff(frame, np.maximum(_rect_px(Y, Z, 864, 910, 1021, 930, 1.6), X + 0.0), 0.4)  # port opening (right)
    # brass deflector: rounded block, round rear end, raised pad at the port, pivot pin hole
    yd, zd = bp(820, 920)
    d2 = np.minimum(np.hypot(Y - yd, Z - zd) - 14.0 * R.S_MM, _rect_px(Y, Z, 820, 906, 858, 934, 1.2))
    defl = S.extrude(d2, ax, hw - 0.8, hw + 2.6, 1.4)
    defl = S.runion(defl, S.extrude(_rect_px(Y, Z, 841, 908, 858, 932, 1.5), ax, hw, hw + 3.4, 1.0), 0.6)
    defl = S.diff(defl, S.extrude(np.hypot(Y - bp(824, 920)[0], Z - bp(824, 920)[1]) - 2.6 * R.S_MM, ax, hw + 1.0, hw + 5))
    # QD plates: thick pillowed teardrops; sockets are real holes with a counterbore ring
    qr = S.extrude(_qd_plate(Y, Z, 1032, 1121, 908, 930, (1121, 916)), ax, hw - 0.8, hw + 2.8, 1.4)
    qf = S.extrude(_qd_plate(Y, Z, 1520, 1589, 855, 876, (1589, 867)), ax, hw - 0.8, hw + 2.8, 1.4)
    for x, row, pl in ((1060, 920, "r"), (1527, 866, "f"), (1557, 866, "f")):
        ys, zs = bp(x, row); rr = np.hypot(Y - ys, Z - zs)
        cut = np.minimum(np.maximum(rr - 2.9, hw + 0.4 - ax), np.maximum(rr - 4.4, hw + 2.2 - ax))
        if pl == "r":
            qr = S.rdiff(qr, cut, 0.3)
        else:
            qf = S.rdiff(qf, cut, 0.3)
    return np.minimum.reduce([frame, defl, qr, qf])


def _turned_handle(X, Y, Z, o, dv):
    dv = np.array(dv, float); dv /= np.linalg.norm(dv)
    px, py, pz = X - o[0], Y - o[1], Z - o[2]
    sa = px * dv[0] + py * dv[1] + pz * dv[2]
    rho = np.sqrt(np.maximum(px * px + py * py + pz * pz - sa * sa, 0))
    rs = np.interp(sa, [0, 6, 6.2, 8.2, 8.4, 11.5, 13.0, 27, 27.3, 38.0], [3.2, 3.2, 5.2, 5.2, 3.6, 3.6, 4.4, 7.8, 8.6, 8.6])
    ch = S.rinter(rho - rs, np.maximum(-sa, sa - 38.0), 1.2)
    rings = np.maximum(np.abs(((sa - 27.8) % 1.4) - 0.7) - 0.3, np.maximum(8.2 - rho, np.maximum(27.8 - sa, sa - 35.6)))
    ch = S.diff(ch, rings)
    return S.rdiff(ch, np.maximum(rho - 4.6, 32.0 - sa), 0.8)                         # hollow knob end


def chandle(X, Y, Z):
    """NRCH charging handle, built UNSCALED so the turned shape stays round: thin neck + flange in the slot, cone
    flaring to a fat ringed knob with a hollow end. Left: swept out and back (close-up ref); right: straight out
    (the right-side photo sees it end-on as a ring at the slot front)"""
    out = []
    for x, sg, dv in ((1100, 1, (1.0, -0.38, -0.32)), (1108, -1, (-1.0, 0.05, -0.12))):
        y, z = bp(x, 869)
        k = float(wk_at(np.array(y)))
        o = np.array([sg * (float(upper_hw(np.array(y), np.array(z))) - 3.0) * k, y, z])
        out.append(_turned_handle(X, Y, Z, o, dv))
    return np.minimum(*out)


def upper_hw_parts(X, Y, Z):
    """black hardware traced from the photo (port frame + folded charging handle, QD plates, hinge plate, sling cup)"""
    hw = upper_hw(Y, Z)
    # rear hinge strip (black, both sides): clean plate with rounded ends instead of the traced outline
    strip = S.extrude(_rect_px(Y, Z, 617, 849, 627, 942, 1.2), np.abs(X), hw - 0.8, hw + 1.0, 0.6)
    d = S.union(strip, side_plates(X, Y, Z, hw))
    # inside the side slot (refs): the black charging rail, round rear end, a pin dot, recessed in the slot
    rail = S.extrude(_cap(Y, Z, (864, 869), (1098, 869), 6.5), np.abs(X), hw - 6.5, hw - 2.6, 0.8)
    rail = S.rdiff(rail, np.maximum(np.hypot(Y - bp(920, 868)[0], Z - bp(920, 868)[1]) - 1.6, hw - 3.2 - np.abs(X)), 0.2)
    d = S.union(d, rail)
    d = DT.apply(d, X, Y, Z, "plate", hw + 1.6)                                       # plate screws, port window, tab
    return d


def _poly2(Y, Z, pts_px):
    """2-D SDF of a convex photo-px polygon (clockwise in the photo) in the (Y, Z) plane"""
    P = [bp(*q) for q in pts_px]
    d = np.full_like(Y, -1e3)
    for (y0, z0), (y1, z1) in zip(P, P[1:] + P[:1]):
        ny, nz = -(z1 - z0), y1 - y0; n = math.hypot(ny, nz)
        d = np.maximum(d, ((Y - y0) * ny + (Z - z0) * nz) / n)
    return d


def _sight(X, Y, Z, base_px, leaf_px, leaf_hw, base_hw=11.6):
    """folding sight measured off the photo (clean machined shapes, not the traced outline): chamfered clamp base
    straddling the rail + thin flip leaf with a rounded top"""
    base = S.extrude(_poly2(Y, Z, base_px), np.abs(X), -base_hw, base_hw, 1.0)
    _, zt = bp(0, max(r for _, r in base_px)); _, zb = bp(0, min(r for _, r in base_px))
    base = np.maximum(base, (np.abs(X) + Z - (base_hw + zb - 2.0)) * 0.7071)          # top chamfers
    (y0, z1), (y1, z0) = bp(leaf_px[0], leaf_px[1]), bp(leaf_px[2], leaf_px[3])
    leaf = S.box(X, Y, Z, (-leaf_hw, y0, z0 - 2), (leaf_hw, y1, z1), 1.4)
    return S.runion(base, leaf, 0.8)


def _drum(X, Y, Z, x, row, rpx, x0, x1, knurl=True):
    y, z = bp(x, row); r = rpx * R.S_MM
    d = S.cyl(X, Y, Z, "x", (y, z), r, x0, x1, 0.8)
    if knurl:
        th = np.arctan2(Z - z, Y - y)
        g = np.maximum(np.abs(((th * 20 / (2 * math.pi)) % 1.0) - 0.5) * 2 * math.pi * r / 20 - 0.35,
                       np.maximum(r - 0.5 - np.hypot(Y - y, Z - z), X - (x0 + 3.0)))
        d = S.diff(d, g)
        d = S.diff(d, np.maximum(np.hypot(Y - y, Z - z) - r * 0.32, X - (x0 + 1.4)))   # hex-key bore in the knob
    return d


def rsight(X, Y, Z):
    d = _sight(X, Y, Z, [(661.5, 841), (661.5, 826), (673, 815), (690, 820), (703.5, 834), (703.5, 841)], (662, 762, 679, 818), 4.2)
    d = S.union(d, _drum(X, Y, Z, 670, 826, 10.5, -13.8, 10.5))                    # windage / hinge drum, knob right
    d = S.union(d, _drum(X, Y, Z, 689, 837, 6.0, -12.0, 12.0, knurl=False))         # cross-bolt boss
    y, z = bp(670, 775)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, z), 1.3, y - 30, y + 30))                 # peep aperture
    return d


def fsight(X, Y, Z):
    d = _sight(X, Y, Z, [(1532, 842), (1532, 830), (1545, 818), (1576, 818), (1583, 830), (1583, 842)], (1554, 767, 1573, 820), 5.6)
    _, zp = bp(0, 776)
    gap = np.maximum(np.maximum(np.abs(X) - 3.8, zp - Z), -(np.abs(X) - 1.1))     # protective ears around the post
    d = S.rdiff(d, gap, 0.3)
    d = S.union(d, _drum(X, Y, Z, 1563, 818, 5.0, -8.0, 8.0, knurl=False))         # leaf hinge pin
    return DT.apply(d, X, Y, Z, "fsight", 11.6, mirror=False)


# ------------------------------------------------------------------ bolt carrier group
def bcg(X, Y, Z):
    """SCAR-pattern carrier: D-section body (flat sides, round top) on side guide rails, piston strike lug, cam slot +
    cam pin, charging-handle notch, side flutes, hammer ramp; 7-lug rotating bolt with gas-ring grooves + extractor"""
    sec = np.maximum.reduce([np.abs(X) - 12.8, -15.5 - Z, np.hypot(X, Z + 3.0) - 19.0])
    c = S.rinter(sec, np.abs(Y + 73.0) - 59.0, 2.0)                                   # Y -132 .. -14
    rails = S.box(X, Y, Z, (-14.6, -130, -13.5), (14.6, -16, -9.5), 0.8)
    c = S.runion(c, rails, 0.8)
    lug = S.rinter(np.maximum(np.abs(X) - 6.0, np.hypot(X, Z - 8.0) - 12.0), np.abs(Y + 30.0) - 16.0, 1.5)
    c = S.runion(c, np.maximum(lug, 10.0 - Z), 1.5)                                   # piston strike lug on top front
    u, w = S.rot2(Y + 41.0, X, 28)
    c = S.rdiff(c, S.box(u, Z, w, (-11, 6, -3.2), (11, 30, 3.2), 1.6), 0.5)          # cam slot
    c = S.rdiff(c, S.box(X, Y, Z, (9.5, -62, -1), (16, -46, 9), 1.2), 0.5)            # charging-handle notch (left)
    for zf in (-4.0, 2.0):                                                            # side flutes
        c = S.rdiff(c, S.extrude(S.capsule2(Y, Z, (-118, zf), (-62, zf), 1.4), np.abs(X), 12.0, 20.0), 0.3)
    c = np.maximum(c, (8.0 - (Y + 132.0) - (Z + 15.5)) * 0.7071)                     # hammer ramp
    c = S.diff(c, S.cyl(X, Y, Z, "y", (0, 0), 7.4, -60, 0))                            # bolt bore
    body = S.cyl(X, Y, Z, "y", (0, 0), 7.0, -58, -0.2, 0.6)
    for yg in (-52.0, -49.5, -47.0):                                                  # gas rings
        body = S.diff(body, np.maximum(np.abs(Y - yg) - 0.45, 6.4 - np.hypot(X, Z)))
    head = S.union(body, S.cyl(X, Y, Z, "y", (0, 0), 9.2, -6, -0.2, 0.3))
    for k in range(7):                                                                  # 7 locking lugs
        a = 2 * math.pi * (k + 0.5) / 7
        uu = X * math.cos(a) + Z * math.sin(a)
        vv = -X * math.sin(a) + Z * math.cos(a)
        head = S.union(head, S.box(uu, Y, vv, (8.0, -6, -1.4), (10.8, -0.2, 1.4), 0.3))
    head = S.diff(head, S.cyl(X, Y, Z, "y", (0, 0), 2.0, -2, 1))                       # firing pin hole
    ext = S.box(X, Y, Z, (-11.5, -16, -2), (-8.5, 1.0, 2), 0.5)                         # extractor
    pin = S.cyl(X, Y, Z, "z", (0, -41.0), 2.8, 0, 17.5, 0.6)                             # cam pin up through the slot
    return S.union(c, head, ext, pin)


# ------------------------------------------------------------------ barrel + muzzle brake
def barrel(X, Y, Z):
    Rr = np.sqrt(X * X + Z * Z)
    photo = [(y, r) for y, r in R.column_radius(1604, 1744)[::3] if r < 10.0]   # exposed barrel (drop handguard-edge px)
    yb = bp(1745, 0)[0]
    prof = [(-4.0, 14.5), (60.0, 14.5), (75.0, 12.5), (photo[0][0] - 12, 11.5)] + photo + [(yb, 7.9), (413.0, 7.9)]
    d = V1.lathe(Y, Rr, prof)
    ya, yb, za, zb = bx(1430, 892, 1482, 935)
    gas = S.box(X, Y, Z, (-9.0, ya, za), (9.0, yb, zb), 2.0)                         # gas block / regulator seen in the window
    gas = S.runion(gas, S.cyl(X, Y, Z, "x", ((ya + yb) / 2, zb - 5.0), 4.0, -10.5, 10.5, 0.8), 1.0)   # regulator plug
    d = S.union(d, gas)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 3.9, -10, 420))
    return d


def brake(X, Y, Z):
    Rr = np.sqrt(X * X + Z * Z)
    prof = R.column_radius(1745, 1872)[::2]
    d = V1.lathe(Y, Rr, prof)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.95, prof[0][0] - 1, 413.0))         # threaded onto the barrel
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 4.6, 400, 480))
    for x0, x1 in ((1811, 1836), (1846, 1865)):                                      # side ports (through)
        y0, y1, z0, z1 = bx(x0, 909, x1, 938)
        d = S.rdiff(d, S.box(X, Y, Z, (-30, y0, z0), (30, y1, z1), 1.6), 0.4)
    for xc in (1790, 1803):                                                          # top vents
        y, _ = bp(xc, 0)
        d = S.rdiff(d, S.cyl(X, Y, Z, "z", (0, y), 1.6, 0, 30), 0.2)
    for xr in (1752, 1759, 1766):                                                    # lock-ring grooves
        y, _ = bp(xr, 0)
        d = np.maximum(d, -(np.maximum(np.abs(Y - y) - 0.45, 11.6 - Rr)))
    return d


# ------------------------------------------------------------------ lower, trigger, controls, pins
MAG_PX = (872, 1008)


def mag_box(X, Y, Z, grow=0.0):
    y0, y1, _, _ = bx(MAG_PX[0], 0, MAG_PX[1], 0)
    return S.rect2(X, Y, (-14.4 - grow, y0 - grow), (14.4 + grow, y1 + grow), 2.4 + grow)


def lower(X, Y, Z):
    ym, _ = bp(866, 0)
    _, zg = bp(0, 1000)
    hw = np.where(Y > ym, 22.6, 21.6)
    hw = np.where((Z < zg) & (Y < ym), 7.0, hw)                                       # the trigger guard loop
    _, zf = bp(0, 1040)
    hw = hw + 2.0 * S.sstep(zf + 1.0, zf - 1.0, Z) * (Y > ym)                         # magwell lip: a real ledge
    yc, _ = bp(1012, 0)
    hw = hw + 1.2 * S.sstep(yc - 1.0, yc + 1.0, Y)                                    # thick magwell-front column
    yt, _ = bp(720, 0)
    r = np.where(Y < yt, 5.0, 3.5)                                                    # rounded rear tang into the grip
    d = S.rinter(R.prof("lower")(Y, Z), np.abs(X) - hw, r)
    _, zgt = bp(0, 993)
    collar = np.maximum(_rounded(R.prof("grip")(Y, Z), X, 15.0, 8.5) - 0.4, zgt - 9.0 - Z)   # wraps the grip top, 0.4 proud
    d = S.runion(d, collar, 4.0)
    _, ztop = bp(0, 943)
    d = S.diff(d, np.maximum(mag_box(X, Y, Z, 0.5 + 1.5 * S.sstep(zf + 4, zf - 6, Z)), S.slab(Z, -200, ztop - 6)))
    y0, y1, z0, z1 = bx(800, 990, 845, 1010)
    d = S.diff(d, S.box(X, Y, Z, (-3.8, y0, z0 - 10), (3.8, y1, z1 + 4), 1.0))         # trigger slot
    # raised C bezel guarding the mag release (open at the upper-left where the bolt release passes)
    ring = np.abs(_cap(Y, Z, (843, 1004), (863, 986), 17.0)) - 2.2
    ybz, zbz = bp(838, 984)
    ring = np.maximum(ring, -np.maximum(ybz - Y, zbz - Z))
    d = S.runion(d, S.extrude(ring, np.abs(X), hw - 1.0, hw + 1.4, 0.7), 0.8)
    # magwell-front stipple, catalogue (pins, bezel ring, markings, seams)
    d = DT.stipple(d, X, Y, Z, hw, 0.4)
    return DT.apply(d, X, Y, Z, "lower", hw)


def trigger(X, Y, Z):
    return S.rinter(R.prof("trigger")(Y, Z), np.abs(X) - 3.3, 0.8)


def _cap(Y, Z, p0, p1, rpx):
    return S.capsule2(Y, Z, bp(*p0), bp(*p1), rpx * R.S_MM)


def controls(X, Y, Z):
    """ambi controls measured off the photo (clean parts, not traced pixels): selector = hub ring + tapered paddle,
    bolt release = paddle + long thin arm + tab, mag release = tilted knurled oval (its C bezel is part of the lower)"""
    ym, _ = bp(866, 0)
    hw = np.where(Y > ym, 22.6, 21.6)
    ax = np.abs(X)
    yc, zc = bp(752, 980)
    hub = S.extrude(np.hypot(Y - yc, Z - zc) - 13.0 * R.S_MM, ax, hw - 0.6, hw + 2.6, 0.9)
    hub = S.rdiff(hub, np.maximum(np.hypot(Y - yc, Z - zc) - 5.5 * R.S_MM, hw + 1.6 - ax), 0.4)      # centre bore
    pad = S.extrude(np.minimum(_cap(Y, Z, (726, 975), (745, 977), 5.0), _cap(Y, Z, (745, 977), (752, 980), 7.0)),
                    ax, hw - 0.6, hw + 1.9, 0.8)
    sel = S.runion(hub, pad, 1.0)
    br = np.minimum.reduce([_cap(Y, Z, (799, 974), (816, 973), 4.5), _cap(Y, Z, (816, 973), (864, 962), 2.6),
                            _rect_px(Y, Z, 862, 956, 871, 967)])
    brl = S.extrude(br, ax, hw - 0.6, hw + 1.7, 0.6)
    mr = _cap(Y, Z, (843, 1004), (863, 986), 9.5)
    mag = S.extrude(mr, ax, hw - 0.6, hw + 1.6, 1.0)
    u, w = S.rot2(Y, Z, 45)
    knurl = np.maximum(np.minimum(np.abs((u % 1.3) - 0.65), np.abs((w % 1.3) - 0.65)) - 0.22,
                       np.maximum(mr + 1.2, hw + 1.3 - ax))
    mag = S.rdiff(mag, knurl, 0.1)
    return np.minimum.reduce([sel, brl, mag])


def _rect_px(Y, Z, x0, r0, x1, r1, r=0.6):
    (ya, za), (yb, zb) = bp(x0, r1), bp(x1, r0)
    return S.rect2(Y, Z, (ya, za), (yb, zb), r)


def pins(X, Y, Z):
    out = []
    for (x, row) in ((1034, 962), (630, 985)):
        y, z = bp(x, row)
        out.append(S.cyl(X, Y, Z, "x", (y, z), 2.4, -22.0, 22.0))
        for sg in (1, -1):
            out.append(S.cyl(X * sg, Y, Z, "x", (y, z), 4.2, 21.0, 23.6, 0.6))
    return S.union(*out)


# ------------------------------------------------------------------ grip, vertical grip
def _rounded(p2, X, hw, r):
    return S.rinter(p2, np.abs(X) - hw, r)


def grip(X, Y, Z):
    hw = 15.0
    d = _rounded(R.prof("grip")(Y, Z), X, hw, 8.5)
    d = DT.stipple(d, X, Y, Z, hw, 0.5)
    d = DT.apply(d, X, Y, Z, "grip", hw)
    _, zt = bp(0, 993)
    return np.maximum(d, Z - zt)


def vgrip(X, Y, Z):
    hw = 14.0
    d = _rounded(R.prof("vgrip")(Y, Z), X, hw, 10.0)
    return DT.stipple(d, X, Y, Z, hw, 0.5)


# ------------------------------------------------------------------ magazine
def magazine(X, Y, Z):
    _, ztop = bp(0, 958)
    ya, yb = bp(MAG_PX[0], 0)[0], bp(MAG_PX[1], 0)[0]
    ext = S.rect2(Y, Z, (ya, bp(0, 1056)[1]), (yb, ztop), 2.0)                       # the part hidden in the magwell
    p = np.minimum(R.prof("mag")(Y, Z), ext)
    _, zbp = bp(0, 1200)
    yc = (bp(MAG_PX[0], 0)[0] + bp(MAG_PX[1], 0)[0]) / 2
    hw = np.where(Z < zbp, 15.8, 14.4 - 0.9 * ((Y - yc) / 38.0) ** 2)                 # convex sides, wider baseplate
    d = S.rinter(p, np.abs(X) - hw, 2.0)
    _, za = bp(0, 1196); _, zb = bp(0, 1066)
    for xr, kind in DT.MAG_RIBS:                                                     # ribs + channels read off the photo
        y, _ = bp(xr, 0)
        if kind == "ridge":
            rib = np.maximum(np.maximum(np.abs(Y - y) - 1.6, np.abs(X) - 15.4), S.slab(Z, za, zb))
            d = S.runion(d, S.rinter(rib, 13.0 - np.abs(X), 0.4), 0.8)
        else:
            ch = np.maximum(np.maximum(np.abs(Y - y) - 1.3, 13.5 - np.abs(X)), S.slab(Z, za, zb))
            d = S.rdiff(d, ch, 0.4)
    d = DT.apply(d, X, Y, Z, "mag", hw)
    return S.diff(d, np.maximum(S.rect2(X, Y, (-12.6, bp(MAG_PX[0] + 3, 0)[0]), (12.6, bp(MAG_PX[1] - 3, 0)[0]), 1.5), S.slab(Z, ztop - 8, ztop + 5)))


# ------------------------------------------------------------------ stock (+ buttpad)
def stock(X, Y, Z):
    yr, _ = bp(300, 0); yf, _ = bp(349, 0)
    _, zleg = bp(0, 1000)
    hw = np.where(Y > yf, 22.0, np.where(Y > yr, 17.0, 21.0))
    _, zc = bp(0, 930)
    hw = hw - np.where(Y > yf, 2.2 * ((Z - zc) / 35.0) ** 2, 0)                      # front body bulges (photo shading)
    # front body: the lower shell (rows 936-1000, rear edge slanting (380,940)->(400,1000)) stands 1.2 proud of the cheek
    _, zp = bp(0, 936)
    (ya, za), (yb2, zb2) = bp(380, 940), bp(400, 1000)
    yl = ya + (Z - za) * (yb2 - ya) / (zb2 - za)
    hw = hw - np.where(Y > yf, 1.2 * S.sstep(zp - 0.8, zp + 0.8, Z) + 1.2 * (Z < zp) * S.sstep(yl + 0.8, yl - 0.8, Y), 0)
    leg = (Z < zleg) & (Y < yr + 10)
    hw = np.where(leg, 12.5, hw)                                                      # the serrated leg is thin
    r = np.where(leg, 9.0, np.where(Y > yf, 8.0, 7.0))                                # leg nearly round, bevelled body edges
    d = S.rinter(R.prof("stock")(Y, Z), np.abs(X) - hw, r)
    # catalogue: latch pocket + button, slot, QD socket, pins, panel lines, numbers (detail.py)
    tri = [bp(355, 887), bp(379, 887), bp(368, 910)]                                 # latch lever in its pocket
    e = [(tri[i], tri[(i + 1) % 3]) for i in range(3)]
    dtri = np.max([((Y - a[0]) * (b[1] - a[1]) - (Z - a[1]) * (b[0] - a[0])) / math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in e], axis=0)
    wedge = S.extrude(dtri + 0.6, np.abs(X), hw - 6.0, hw - 1.2, 1.0)
    pocket = S.extrude(_poly2(Y, Z, [(354, 905), (353, 883), (383, 878), (382, 901), (367, 911)]) + 1.0 - 1.6,
                       np.abs(X), hw - 6.0, hw + 5, 1.2)                               # rounded-triangle latch pocket
    d = S.union(DT.apply(S.rdiff(d, pocket, 0.8), X, Y, Z, "stock", hw), wedge)
    _, zb = bp(0, 998)
    yleg, _ = bp(300, 0)
    d = np.maximum(d, np.where((Y > yleg) & (Z > zb - 1.0), (np.abs(X) - (Z - zb) - hw + 4.0) * 0.7071, -1e3))  # body underside
    return d


def buttpad(X, Y, Z):
    hw = 22.5
    d = S.rinter(R.prof("butt")(Y, Z), np.abs(X) - hw, 2.5)
    return DT.apply(d, X, Y, Z, "butt", hw)


# ------------------------------------------------------------------ build table
def _bb(name, pad=6.0, xhw=32.0):
    y0, y1, z0, z1 = R.extent(name)
    return (-xhw, y0 - pad, z0 - pad), (xhw, y1 + pad, z1 + pad)


WK = 0.75          # width scale measured on the 3/4 photo (silfit.py: IoU peak at k 0.75, 2026-10-25)
UNSCALED = {"barrel", "brake", "bolt", "round", "chandle"}


def widthscaled(fn):
    """the gun's widths were estimated; the 3/4 photo fit says they are 0.75 x: compress the part about the bore plane"""
    def f(X, Y, Z):
        return fn(X / WK, Y, Z) * WK
    return f


WK_HG = 0.92                                          # handguard width scale (user: wider, a real tube body line)


def wk_at(Y):
    return WK + (WK_HG - WK) * _blend(Y)


def widthscaled_y(fn):
    """upper group: receiver at WK (flush with the lower), handguard flares out to WK_HG along the same sweep"""
    def f(X, Y, Z):
        k = wk_at(Y)
        return fn(X / k, Y, Z) * k
    return f


UPPER_GROUP = {"upper", "upperhw", "screws_u", "rsight", "fsight"}


def screws_upper(X, Y, Z):
    """steel heads on the receiver, its plates and the sights (one shell per assembly so the stock can fold)"""
    hw = upper_hw(Y, Z)
    return np.minimum.reduce([DT.heads(X, Y, Z, "upper", hw), DT.heads(X, Y, Z, "plate", hw + 1.6),
                              DT.heads(X, Y, Z, "fsight", 11.6, mirror=False)])


def screws_lower(X, Y, Z):
    ym = bp(866, 0)[0]
    return DT.heads(X, Y, Z, "lower", np.where(Y > ym, 22.6, 21.6))


def screws_stock(X, Y, Z):
    yr, _ = bp(300, 0); yf, _ = bp(349, 0)
    return DT.heads(X, Y, Z, "stock", np.where(Y > yf, 22.0, np.where(Y > yr, 17.0, 21.0)))


def table():
    T = {}
    lo, hi = _bb("upper", 10, 34)
    T["upper"] = (upper, (lo[0], R.extent("hinge")[0] - 6, min(lo[2], R.extent("hinge")[2] - 6)), hi, 0.4, "anod")
    lo, hi = _bb("upper", 10, 34)
    T["upperhw"] = (upper_hw_parts, (-34, lo[1], -48), (62, hi[1], 50), 0.18, "steel")
    ych, zch = bp(1100, 869)
    T["chandle"] = (chandle, (-62, ych - 30, zch - 25), (62, ych + 18, zch + 12), 0.15, "steel")
    T["screws_u"] = (screws_upper, (-34, R.px2b(600, 0)[0], -30), (34, R.px2b(1610, 0)[0], 90), 0.14, "steel")
    T["screws_l"] = (screws_lower, (-28, R.px2b(700, 0)[0], -45), (28, R.px2b(1060, 0)[0], -10), 0.14, "steel")
    T["screws_s"] = (screws_stock, (-28, R.px2b(200, 0)[0], -60), (28, R.px2b(580, 0)[0], 40), 0.14, "steel")
    T["rsight"] = (rsight, *_bb("rsight", 4, 15), 0.15, "steel")
    T["fsight"] = (fsight, *_bb("fsight", 4, 13), 0.15, "steel")
    T["barrel"] = (barrel, (-16, -6, -16), (16, 416, 20), 0.25, "steel")
    T["brake"] = (brake, (-16, R.px2b(1740, 0)[0], -16), (16, R.MUZZLE_Y + 3, 16), 0.15, "steel")
    T["bolt"] = (bcg, (-17, -134, -18), (17, 3, 25), 0.25, "park")
    T["lower"] = (lower, *_bb("lower", 8, 26), 0.32, "poly")
    T["trigger"] = (trigger, *_bb("trigger", 4, 6), 0.12, "steel")
    T["controls"] = (controls, (-27, R.px2b(700, 0)[0], R.px2b(0, 1032)[1]), (27, R.px2b(930, 0)[0], R.px2b(0, 930)[1]), 0.14, "steel")
    T["pins"] = (pins, (-26, R.px2b(615, 0)[0], -50), (26, R.px2b(1050, 0)[0], -30), 0.15, "steel")
    T["grip"] = (grip, *_bb("grip", 6, 18), 0.28, "poly")
    T["vgrip"] = (vgrip, *_bb("vgrip", 6, 17), 0.28, "poly")
    T["magazine"] = (magazine, (-19, R.px2b(860, 0)[0], R.px2b(0, 1232)[1]), (19, R.px2b(1020, 0)[0], R.px2b(0, 950)[1]), 0.25, "poly")
    T["round"] = (V1.round_, (-7, -1, -7), (7, 72, 7), 0.12, "brass")
    T["stock"] = (stock, *_bb("stock", 8, 26), 0.36, "poly")
    T["buttpad"] = (buttpad, *_bb("butt", 6, 26), 0.25, "rubber")
    return T


def main():
    out = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("out=")), os.path.join(HERE, "work", "high.npz"))
    old = dict(np.load(out)) if (ONLY and os.path.exists(out)) else {}
    parts = {}
    for name, (fn, lo, hi, h, mat) in table().items():
        if ONLY and name not in ONLY:
            continue
        if name in UPPER_GROUP:
            fn = widthscaled_y(fn)
            lo, hi = (lo[0] * WK_HG, lo[1], lo[2]), (hi[0] * WK_HG, hi[1], hi[2])
        elif name not in UNSCALED:
            fn = widthscaled(fn)
            lo, hi = (lo[0] * WK, lo[1], lo[2]), (hi[0] * WK, hi[1], hi[2])
        off = h * HS * 0.3713
        V, F = S.mesh(fn, np.add(lo, -off), np.add(hi, off), h * HS, lip=3.5, name=name)
        m = V1.round_mat(V) if name == "round" else np.full(len(V), MAT[mat])
        parts[name] = (V, F, {"mat": m})
    S.save(out + ".tmp.npz", parts)
    new = dict(np.load(out + ".tmp.npz"))
    if old:
        for k in list(old):
            if k.split("__")[0] in parts:
                old.pop(k)
        new.update(old)
    else:
        new = {k: v for k, v in new.items()}
    np.savez_compressed(out, **new)
    os.remove(out + ".tmp.npz")
    print("HIGH_DONE", sorted({k.split("__")[0] for k in new}))


if __name__ == "__main__":
    main()
