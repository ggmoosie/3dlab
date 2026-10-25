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


def upper_hw(Y, Z):
    yb, _ = bp(HG_X, 0)
    _, zband = bp(0, 900)
    hw = 26.6 - 1.2 * S.sstep(zband + 0.6, zband - 0.6, Z)              # lower receiver band inset 1.2 mm (the side step)
    return np.where(Y > yb - 22, 27.8, hw)


def rail_xs(X, Z, z_top, z_base):
    """1913 dovetail: neck 16 wide, 21.2 across the vee, flat top 15.6"""
    zc = z_top - 2.8
    neck = np.maximum(np.abs(X) - 8.0, np.maximum(z_base - 1.0 - Z, Z - zc))
    dove = np.maximum((np.abs(X) + np.abs(Z - zc)) * 0.7071 - 10.6 * 0.7071, np.maximum(zc - 2.6 - Z, Z - z_top))
    return np.minimum(neck, dove)


def upper(X, Y, Z):
    hw = upper_hw(Y, Z)
    P2 = R.prof("upper")(Y, Z)
    d = S.rinter(P2, np.abs(X) - hw, 2.0)
    _, ztop = bp(0, 845)
    d = np.maximum(d, (np.abs(X) + Z - (hw + ztop - 4.5)) * 0.7071)                 # 45 deg top chamfers
    yb, _ = bp(HG_X, 0)
    _, zhb = bp(0, 967)
    d = np.maximum(d, np.where(Y > yb, (np.abs(X) - Z - (hw - zhb - 4.0)) * 0.7071, -1e3))   # handguard bottom chamfers
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
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 19.0, yb - 30, 400))
    # charging-handle slot (both sides), ejection port (right), side step pocket under the port
    d = side_cut(d, X, Y, Z, 858, 860, 1102, 878, 27.8, 4.0, 1.5)                     # blind slot (the CH rides inside)
    d = side_cut(d, X, Y, Z, 828, 905, 1042, 928, 27.8, 12, 1.2, both=False, sign=-1)          # ejection port (right)
    d = side_cut(d, X, Y, Z, 828, 907, 1046, 926, 27.8, 4.0, 1.2, both=False, sign=1)          # charging-handle track (left)
    # M-LOK: 6 slots per row, 2 rows a side, plus the bottom row
    xs = [(1178, 1230), (1250, 1302), (1320, 1372), (1392, 1445), (1463, 1516), (1535, 1590)]
    for x0, x1 in xs:
        d = side_cut(d, X, Y, Z, x0, 920, x1, 934, 27.8, 7.0, 2.2)
        d = side_cut(d, X, Y, Z, x0, 950, x1, 958, 27.8, 4.0, 1.6)
        y0, y1, _, _ = bx(x0, 0, x1, 0)
        d = S.rdiff(d, S.box(X, Y, Z, (-3.6, y0, zhb - 3), (3.6, y1, zhb + 9), 1.6), 0.6)
    # pins / screw holes / QD sockets (both sides)
    for (x, row, rr, dep) in ((652, 900, 4.0, 1.2), (672, 900, 4.0, 1.2), (1345, 866, 3.2, 3.0), (1370, 866, 3.2, 3.0),
                              (1180, 866, 6.0, 4.0), (1100, 869, 5.0, 2.0)):
        y, z = bp(x, row)
        d = S.rdiff(d, S.cyl(np.abs(X), Y, Z, "x", (y, z), rr * R.S_MM, 27.8 - dep, 40), 0.3)
    # sling loop square cup (both sides, front of the receiver)
    d = side_cut(d, X, Y, Z, 1420, 854, 1460, 892, 27.8, 5, 1.0)
    return d


def upper_hw_parts(X, Y, Z):
    """black hardware on the upper: QD plates, charging handle (left), brass deflector (right), slot-end screws"""
    out = []
    for sg in (1, -1):
        out.append(stadium(X, Y, Z, 1035, 904, 1110, 931, 27.0, 30.2, sg))           # rear QD plate on the handguard
        out.append(stadium(X, Y, Z, 1520, 853, 1592, 879, 26.5, 29.6, sg))           # front QD plate
        y, z = bp(1100, 869)
        out.append(S.cyl(X * sg, Y, Z, "x", (y, z), 4.0, 25.0, 29.4, 0.6))            # slot-end screw head
        y, z = bp(1442, 874)
        out.append(S.diff(S.box(X * sg, Y, Z, (24.5, y - 9, z - 9), (28.8, y + 9, z + 9), 1.2),
                          S.box(X * sg, Y, Z, (26.5, y - 5.5, z - 6), (40, y + 5.5, z + 6), 0.8)))   # sling cup insert
    y0, y1, z0, z1 = bx(812, 898, 872, 926)                                         # brass deflector / port cover (right)
    out.append(S.box(X, Y, Z, (-30.5, y0, z0), (-25.5, y1, z1), 1.8))
    # charging handle (left, NRCH): carriage at the front of the lower track + T knob angled down and back
    y, z = bp(1030, 916)
    car = S.box(X, Y, Z, (14, y - 18, z - 3.4), (28.5, y + 18, z + 3.4), 1.0)
    u, w = S.rot2(X - 28.0, Z - z, 22)
    knob = S.box(u, Y - y, w, (0, -6, -4), (26, 6, 4), 2.2)
    tb = S.box(u, Y - y, w, (20, -16, -5), (28, 16, 5), 2.6)
    ch = S.union(car, S.runion(knob, tb, 2.0))
    gro = np.maximum(np.abs(((Y - y) % 3.0) - 1.5) - 0.55, 26.5 - u)
    out.append(S.rdiff(ch, gro, 0.3))
    return S.union(*out)


def rsight(X, Y, Z):
    p = R.prof("rsight")(Y, Z)
    _, zr = bp(0, 830)
    hw = np.where(Z > zr, 7.5, 11.0)
    d = S.rinter(p, np.abs(X) - hw, 0.8)
    y, z = bp(664, 774)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, z), 1.3, y - 30, y + 30))                 # peep aperture
    d = S.rdiff(d, S.box(X, Y, Z, (-4.5, y - 3, z - 12), (4.5, y + 3, z - 4), 1.0), 0.4)   # lightening window
    return d


def fsight(X, Y, Z):
    p = R.prof("fsight")(Y, Z)
    _, zr = bp(0, 830)
    hw = np.where(Z > zr, 8.0, 11.0)
    d = S.rinter(p, np.abs(X) - hw, 0.8)
    _, zp = bp(0, 790)
    gap = np.maximum(np.maximum(np.abs(X) - 5.2, zp - Z), -(np.abs(X) - 1.2))     # ears around a 2.4 mm post
    return S.rdiff(d, gap, 0.4)


# ------------------------------------------------------------------ barrel + muzzle brake
def barrel(X, Y, Z):
    Rr = np.sqrt(X * X + Z * Z)
    photo = [(y, r) for y, r in R.column_radius(1604, 1744)[::3]]          # exposed barrel: the photo's diameter
    yb = bp(1745, 0)[0]
    prof = [(-4.0, 14.5), (60.0, 14.5), (75.0, 12.5), (photo[0][0] - 12, 11.5)] + photo + [(yb, 7.9), (413.0, 7.9)]
    d = V1.lathe(Y, Rr, prof)
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
    hw = hw + 1.2 * S.sstep(zf + 3, zf - 3, Z) * (Y > ym)                             # magwell flare
    d = S.rinter(R.prof("lower")(Y, Z), np.abs(X) - hw, 2.2)
    _, ztop = bp(0, 943)
    d = S.diff(d, np.maximum(mag_box(X, Y, Z, 0.5 + 1.5 * S.sstep(zf + 4, zf - 6, Z)), S.slab(Z, -200, ztop - 6)))
    y0, y1, z0, z1 = bx(800, 990, 845, 1010)
    d = S.diff(d, S.box(X, Y, Z, (-3.8, y0, z0 - 10), (3.8, y1, z1 + 4), 1.0))         # trigger slot
    # magwell-front grip texture panel + selector detent ring + engraved selector arc
    d = side_cut(d, X, Y, Z, 1015, 980, 1044, 1040, 22.6, 0.7, 1.2)
    y, z = bp(738, 968)
    d = S.rdiff(d, np.maximum(np.abs(np.sqrt((Y - y) ** 2 + (Z - z) ** 2) - 8.5) - 0.5, 21.2 - np.abs(X)), 0.2)
    return d


def trigger(X, Y, Z):
    return S.rinter(R.prof("trigger")(Y, Z), np.abs(X) - 3.3, 0.8)


def controls(X, Y, Z):
    out = []
    ys, zs = bp(738, 968)
    for sg in (1, -1):
        hub = S.cyl(X * sg, Y, Z, "x", (ys, zs), 5.0, 20.5, 24.2, 0.8)
        u, w = S.rot2(Y - ys, Z - zs, -8)
        lev = S.box(X * sg, u, w, (21.6, -16, -2.6), (24.4, 2, 2.6), 1.2)              # selector lever, safe (pointing back)
        out.append(S.runion(hub, lev, 1.2))
        ym, zm = bp(886, 977)
        disc = S.cyl(X * sg, Y, Z, "x", (ym, zm), 7.6, 21.5, 24.6, 1.0)                 # mag release paddle-button
        knurl = np.maximum(np.abs(((Y + Z) % 1.6) - 0.8) - 0.3, 24.0 - X * sg)
        out.append(S.rdiff(disc, knurl, 0.15))
    y0, y1, z0, z1 = bx(846, 950, 910, 964)                                          # bolt catch lever (left)
    bc = S.box(X, Y, Z, (21.0, y0, z0), (24.4, y1, z1), 1.4)
    out.append(S.rdiff(bc, np.maximum(np.abs(((Y - y0) % 2.2) - 1.1) - 0.45, 24.0 - X), 0.2))
    return S.union(*out)


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
    d = _rounded(R.prof("grip")(Y, Z), X, 15.0, 6.5)
    # stipple panels (recessed 0.6 mm), on the slope of the grip
    a, b = bp(662, 1178), bp(744, 1072)
    L = math.hypot(b[0] - a[0], b[1] - a[1]); ux, uz = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    t = (Y - a[0]) * ux + (Z - a[1]) * uz; n = (Y - a[0]) * (-uz) + (Z - a[1]) * ux
    pan = np.maximum(np.maximum(-t, t - L), np.abs(n) - 12.5) - 2.5
    d = S.rdiff(d, np.maximum(pan, 14.4 - np.abs(X)), 0.8)
    _, zt = bp(0, 993)
    return np.maximum(d, Z - zt)


def vgrip(X, Y, Z):
    d = _rounded(R.prof("vgrip")(Y, Z), X, 14.0, 6.0)
    y0, y1, z0, z1 = bx(1352, 1060, 1440, 1190)
    pan = S.rect2(Y, Z, (y0, z0), (y1, z1), 4.0)
    d = S.rdiff(d, np.maximum(pan, 13.4 - np.abs(X)), 0.8)
    return d


# ------------------------------------------------------------------ magazine
def magazine(X, Y, Z):
    _, ztop = bp(0, 958)
    ya, yb = bp(MAG_PX[0], 0)[0], bp(MAG_PX[1], 0)[0]
    ext = S.rect2(Y, Z, (ya, bp(0, 1056)[1]), (yb, ztop), 2.0)                       # the part hidden in the magwell
    p = np.minimum(R.prof("mag")(Y, Z), ext)
    _, zbp = bp(0, 1200)
    hw = np.where(Z < zbp, 15.8, 14.4)                                               # baseplate wider
    d = S.rinter(p, np.abs(X) - hw, 2.0)
    for xr in (897, 928, 957, 985):                                                  # pressed vertical ribs
        y, _ = bp(xr, 0)
        _, za = bp(0, 1195); _, zb = bp(0, 1062)
        rib = np.maximum(np.maximum(np.abs(Y - y) - 2.0, np.abs(X) - 15.1), S.slab(Z, za, zb))
        d = S.runion(d, S.rinter(rib, 13.0 - np.abs(X), 0.4), 1.0)
    return S.diff(d, np.maximum(S.rect2(X, Y, (-12.6, bp(MAG_PX[0] + 3, 0)[0]), (12.6, bp(MAG_PX[1] - 3, 0)[0]), 1.5), S.slab(Z, ztop - 8, ztop + 5)))


# ------------------------------------------------------------------ stock (+ buttpad)
def stock(X, Y, Z):
    yr, _ = bp(300, 0); yf, _ = bp(385, 0)
    _, zleg = bp(0, 1000)
    hw = np.where(Y > yf, 22.0, np.where(Y > yr, 17.0, 21.0))
    hw = np.where((Z < zleg) & (Y < yr + 10), 12.5, hw)                               # the serrated leg is thin
    d = S.rinter(R.prof("stock")(Y, Z), np.abs(X) - hw, 3.0)
    # latch "eye" pocket + lever, length-adjust slot, QD hole, side panel recess
    y0, y1, z0, z1 = bx(396, 878, 470, 936)
    d = S.rdiff(d, np.maximum(S.rect2(Y, Z, (y0, z0), (y1, z1), 6.0), 20.8 - np.abs(X)), 0.8)
    lev = S.box(X, Y, Z, (-21.4, y0 + 4, z0 + 4), (21.4, y1 - 6, z1 - 3), 3.0)
    d = S.union(d, lev)
    d = side_cut(d, X, Y, Z, 214, 892, 256, 899, 21.0, 4.0, 1.0)
    y, z = bp(243, 918)
    d = S.rdiff(d, S.cyl(np.abs(X), Y, Z, "x", (y, z), 5.0, 15.0, 30), 0.6)
    d = side_cut(d, X, Y, Z, 400, 945, 568, 990, 22.0, 0.9, 2.0)
    return d


def buttpad(X, Y, Z):
    return S.rinter(R.prof("butt")(Y, Z), np.abs(X) - 22.5, 2.5)


# ------------------------------------------------------------------ build table
def _bb(name, pad=6.0, xhw=32.0):
    y0, y1, z0, z1 = R.extent(name)
    return (-xhw, y0 - pad, z0 - pad), (xhw, y1 + pad, z1 + pad)


WK = 0.75          # width scale measured on the 3/4 photo (silfit.py: IoU peak at k 0.75, 2026-10-25)
UNSCALED = {"barrel", "brake", "bolt", "round"}


def widthscaled(fn):
    """the gun's widths were estimated; the 3/4 photo fit says they are 0.75 x: compress the part about the bore plane"""
    def f(X, Y, Z):
        return fn(X / WK, Y, Z) * WK
    return f


def table():
    T = {}
    lo, hi = _bb("upper", 10, 34)
    T["upper"] = (upper, (lo[0], R.extent("hinge")[0] - 6, lo[2]), hi, 0.4, "anod")
    lo, hi = _bb("upper", 10, 34)
    T["upperhw"] = (upper_hw_parts, (-34, lo[1], -40), (62, hi[1], 50), 0.18, "steel")
    T["rsight"] = (rsight, *_bb("rsight", 4, 13), 0.15, "steel")
    T["fsight"] = (fsight, *_bb("fsight", 4, 13), 0.15, "steel")
    T["barrel"] = (barrel, (-16, -6, -16), (16, 416, 16), 0.25, "steel")
    T["brake"] = (brake, (-16, R.px2b(1740, 0)[0], -16), (16, R.MUZZLE_Y + 3, 16), 0.15, "steel")
    T["bolt"] = (V1.bolt, (-15, -134, -18), (15, 3, 20), 0.25, "park")
    T["lower"] = (lower, *_bb("lower", 8, 26), 0.32, "poly")
    T["trigger"] = (trigger, *_bb("trigger", 4, 6), 0.12, "steel")
    T["controls"] = (controls, (-27, R.px2b(700, 0)[0], R.px2b(0, 1010)[1]), (27, R.px2b(930, 0)[0], R.px2b(0, 930)[1]), 0.14, "steel")
    T["pins"] = (pins, (-26, R.px2b(615, 0)[0], -50), (26, R.px2b(1050, 0)[0], -30), 0.15, "steel")
    T["grip"] = (grip, *_bb("grip", 6, 18), 0.28, "poly")
    T["vgrip"] = (vgrip, *_bb("vgrip", 6, 17), 0.28, "poly")
    T["magazine"] = (magazine, (-19, R.px2b(860, 0)[0], R.px2b(0, 1232)[1]), (19, R.px2b(1020, 0)[0], R.px2b(0, 950)[1]), 0.25, "poly")
    T["round"] = (V1.round_, (-7, -1, -7), (7, 72, 7), 0.12, "brass")
    T["stock"] = (stock, *_bb("stock", 8, 26), 0.36, "poly")
    T["buttpad"] = (buttpad, *_bb("butt", 6, 26), 0.25, "rubber")
    return T


def main():
    out = os.path.join(HERE, "work", "high.npz")
    old = dict(np.load(out)) if (ONLY and os.path.exists(out)) else {}
    parts = {}
    for name, (fn, lo, hi, h, mat) in table().items():
        if ONLY and name not in ONLY:
            continue
        if name not in UNSCALED:
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
