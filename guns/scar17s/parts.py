"""parts.py - SCAR 17S high-poly parts as signed distance functions (B frame, mm) -> work/high.npz.

usage: .venv/bin/python guns/scar17s/parts.py [h_scale=1.0] [parts=a,b,...]
Every part is complete on all sides (they come apart in the exploded view and in game). Each shell carries a per-vertex
'mat' id (MAT) that the bake turns into the material masks."""
import sys, os, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S
import scar_spec as P

MAT = dict(anod=0, poly=1, steel=2, park=3, rubber=4, brass=5, magsteel=6, copper=7)
HS = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("h_scale=")), 1.0))
ONLY = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("parts=")), None)


# ------------------------------------------------------------------ 2-D helpers
def poly2(pts, res=0.1, pad=8.0):
    """signed distance of a (y, z) polygon, rasterised once (bilinear sampler)"""
    pts = np.asarray(pts, float)
    y0, z0 = pts.min(0) - pad; y1, z1 = pts.max(0) + pad
    ny, nz = int((y1 - y0) / res) + 1, int((z1 - z0) / res) + 1
    img = Image.new("L", (ny, nz), 0)
    ImageDraw.Draw(img).polygon([((y - y0) / res, (z1 - z) / res) for y, z in pts], fill=255)
    m = np.asarray(img) > 127
    d = (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)) * res + np.where(m, 0.5 * res, -0.5 * res)
    return S.Field2D(d, y0, z1, res, outside=pad)


def rail(Xl, Zl):
    """MIL-STD-1913 cross-section: lateral coordinate Xl, height Zl (base 23 embedded, top 31.9). <0 inside."""
    neck = np.maximum(np.abs(Xl) - 8.0, np.maximum(22.0 - Zl, Zl - 27.0))
    dove = np.maximum((np.abs(Xl) + np.abs(Zl - 29.1)) * 0.7071 - 10.6 * 0.7071, np.maximum(26.4 - Zl, Zl - P.RAIL_TOP))
    return np.minimum(neck, dove)


def rail_slots(Y, Zl, y0, y1):
    """recoil grooves across a rail: 5.23 wide, 2.9 deep, 10.01 pitch, centred in [y0, y1]"""
    n = int((y1 - y0) // P.RAIL_PITCH)
    c0 = (y0 + y1) / 2 - (n - 1) / 2 * P.RAIL_PITCH
    k = np.clip(np.round((Y - c0) / P.RAIL_PITCH), 0, n - 1)
    ym = Y - (c0 + k * P.RAIL_PITCH)
    return np.maximum(np.abs(ym) - P.RAIL_SLOT_W / 2, (P.RAIL_TOP - P.RAIL_SLOT_D) - Zl)


def lathe(Y, R, prof):
    """radius profile [(y, r), ...] around the y axis (approximate SDF, fine for gentle tapers)"""
    ys, rs = zip(*prof)
    r = np.interp(Y, ys, rs)
    return np.maximum(R - r, S.slab(Y, ys[0], ys[-1]))


# ------------------------------------------------------------------ upper receiver (+ hinge block)
UP_PROF = poly2([(P.UP_Y0, P.UP_TOP), (P.UP_Y1, P.UP_TOP), (P.UP_Y1, -28.0), (P.UP_Y1 - 7, P.FE_BOT), (118.0, P.FE_BOT),
                 (96.0, P.UP_BOT), (P.UP_Y0, P.UP_BOT)])


def upper(X, Y, Z):
    hw = np.where(Y > 100.0, P.FE_HW, P.UP_HW)
    d = S.rinter(UP_PROF(Y, Z), np.abs(X) - hw, 2.2)
    # top rail, full length; side rails 3/9 and bottom rail on the forend
    top = np.maximum(rail(X, Z), S.slab(Y, P.UP_Y0 + 2, P.UP_Y1 - 2))
    zs = np.abs(X) - P.FE_HW + 24.0
    side = np.maximum(rail(Z - P.SIDE_RAIL_Z, zs), S.slab(Y, *P.SIDE_RAIL_Y))
    zb = -Z - (-P.FE_BOT) + 24.0
    bot = np.maximum(rail(X, zb), S.slab(Y, *P.SIDE_RAIL_Y))
    d = S.runion(d, np.minimum(top, np.minimum(side, bot)), 0.6)
    d = S.rdiff(d, np.maximum(rail_slots(Y, Z, P.UP_Y0 + 2, P.UP_Y1 - 2), Z - 40), 0.25)
    d = S.rdiff(d, np.maximum(rail_slots(Y, zs, *P.SIDE_RAIL_Y), np.maximum(np.abs(X) - 40, 20 - zs)), 0.25)
    d = S.rdiff(d, np.maximum(rail_slots(Y, zb, *P.SIDE_RAIL_Y), np.maximum(-Z - 50, 20 - zb)), 0.25)
    # bolt cavity (receiver), barrel channel (forend, open at the front)
    d = S.diff(d, S.box(X, Y, Z, (-16, P.UP_Y0 + 6, -17), (16, 104, 19), 2.0))
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 17.0, 96.0, P.UP_Y1 + 5))
    # ejection port (right), charging slot (left), lightening flats on the receiver sides
    d = S.rdiff(d, S.box(X, Y, Z, (-32, P.PORT["y"][0], P.PORT["z"][0]), (-10, P.PORT["y"][1], P.PORT["z"][1]), 2.0), 1.0)
    d = S.rdiff(d, S.box(X, Y, Z, (10, P.CH_SLOT["y"][0], P.CH_SLOT["z"][0]), (32, P.CH_SLOT["y"][1], P.CH_SLOT["z"][1]), 1.5), 0.6)
    for sgn, y1 in ((1, -70.0), (-1, -55.0)):
        pan = S.box(X * sgn, Y, Z, (P.UP_HW - 0.8, -160, -14), (P.UP_HW + 3, y1, 16), 3.0)
        d = S.rdiff(d, pan, 0.6)
    # raised side band along the receiver (the SCAR's lower-edge ridge), both sides
    band = S.box(np.abs(X), Y, Z, (P.UP_HW - 1.0, P.UP_Y0 + 4, -13.5), (P.UP_HW + 1.2, 92, -9.0), 1.0)
    d = S.runion(d, band, 0.8)
    # forend vent slots (3 per side, below the side rail)
    for k in range(3):
        yc = 150.0 + k * 42.0
        ven = S.box(np.abs(X), Y, Z, (P.FE_HW - 3, yc - 12, -27), (P.FE_HW + 3, yc + 12, -19), 3.0)
        d = S.rdiff(d, ven, 0.6)
    # hinge block behind the receiver (folding stock interface), knuckles on the right
    hb = S.box(X, Y, Z, (-23, -202, -40), (23, P.UP_Y0 + 1, 22), 2.5)
    for z0, z1 in ((-36, -22), (2, 16)):
        hb = S.runion(hb, S.cyl(X, Y, Z, "z", P.HINGE, 6.0, z0, z1, 0.8), 1.0)
    hb = S.diff(hb, S.cyl(X, Y, Z, "z", P.HINGE, 2.2, -50, 30))
    hb = S.rdiff(hb, S.cyl(X, Y, Z, "x", (-190, -10), 4.5, 21, 30), 0.4)            # latch button pocket (left)
    return S.runion(d, hb, 1.0)


def latch(X, Y, Z):
    return S.cyl(X, Y, Z, "x", (-190, -10), 4.0, 19, 24.6, 0.8)


# ------------------------------------------------------------------ barrel, gas block, muzzle device
def barrel(X, Y, Z):
    R = np.sqrt(X * X + Z * Z)
    prof = [(-4, 14.5), (60, 14.5), (75, 12.4), (262, 12.0), (264, 11.2), (300, 11.2), (302, 10.3), (398, 9.9), (399, 7.9), (413, 7.9)]
    d = lathe(Y, R, prof)
    th = np.where(Y > 399, (np.abs(((Y - 399) / 1.058) % 1.0 - 0.5) * 2 - 0.5) * 0.55, 0)    # 5/8-24 thread
    d = d + th
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 3.9, -10, 420))
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 5.8, 411.5, 420), 0.4)                    # crown recess
    return d


def gasblock(X, Y, Z):
    b = S.box(X, Y, Z, (-12.5, 289, -28), (12.5, 304, 12.5), 3.0)
    b = S.runion(b, S.cyl(X, Y, Z, "y", (0, -19.0), 7.5, 300, 306, 1.0), 1.5)
    b = S.diff(b, S.cyl(X, Y, Z, "y", (0, 0), 11.3, 280, 320))
    reg = S.cyl(X, Y, Z, "y", (0, -19.0), 6.4, 305, 317, 0.8)                            # 2-position regulator
    reg = S.diff(reg, S.box(X, Y, Z, (-0.9, 313, -40), (0.9, 320, 0), 0.0))             # screwdriver/cartridge slot
    flats = np.maximum(np.abs(X) - 5.4, -1e3)
    reg = np.maximum(reg, flats)
    tab = S.box(X, Y, Z, (-2.2, 312, -30.5), (2.2, 316.5, -21), 0.8)
    return S.union(b, reg, tab)


def hider(X, Y, Z):
    R = np.sqrt(X * X + Z * Z)
    d = lathe(Y, R, [(398, 11.0), (452, 11.0), (456, 9.8)])
    d = np.maximum(d, np.where(Y < 410, np.abs(X) - 9.6, -1e3))                        # wrench flats
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.95, 390, 412))                          # threaded bore
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 5.3, 390, 460))
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.8, 430, 460), 0.4)                     # expansion chamber
    for adeg in (0, 45, 90, 135, 180):                                                  # 5 slots, bottom closed (comp)
        a = math.radians(adeg)
        s = -X * math.sin(a) + Z * math.cos(a)
        t = X * math.cos(a) + Z * math.sin(a)
        slot = np.maximum(np.maximum(np.abs(s) - 1.6, 2.0 - t), np.abs(Y - 440.0) - 12.0)
        d = S.rdiff(d, slot, 0.3)
    return d


# ------------------------------------------------------------------ bolt carrier + bolt, charging handle
def bolt(X, Y, Z):
    c = S.box(X, Y, Z, (-12.8, -132, -15.5), (12.8, -14, 16.5), 1.6)
    c = S.runion(c, S.box(X, Y, Z, (-7, -125, 14), (7, -40, 18.5), 1.5), 1.2)           # piston strike face / hump
    c = S.rdiff(c, S.box(X, Y, Z, (-14, -110, 8), (-10, -70, 13), 1.0), 0.4)           # relief cuts
    c = S.rdiff(c, S.box(X, Y, Z, (10, -110, 8), (14, -70, 13), 1.0), 0.4)
    c = S.diff(c, S.cyl(X, Y, Z, "y", (0, 0), 7.2, -40, 0))                            # bolt bore
    R, A = np.sqrt(X * X + Z * Z), np.arctan2(Z, X)
    head = S.union(S.cyl(X, Y, Z, "y", (0, 0), 7.0, -36, -0.2), S.cyl(X, Y, Z, "y", (0, 0), 9.2, -6, -0.2, 0.3))
    for k in range(7):                                                                  # 7 locking lugs
        a = 2 * math.pi * (k + 0.5) / 7
        u = X * math.cos(a) + Z * math.sin(a)
        v = -X * math.sin(a) + Z * math.cos(a)
        head = S.union(head, S.box(u, Y, v, (8.0, -6, -1.4), (10.8, -0.2, 1.4), 0.3))
    head = S.diff(head, S.cyl(X, Y, Z, "y", (0, 0), 2.0, -2, 1))                       # firing pin hole
    ext = S.box(X, Y, Z, (-11.5, -16, -2), (-8.5, 1.0, 2), 0.5)                         # extractor
    return S.union(c, head, ext)


def charging(X, Y, Z):
    y0 = P.CH_FWD_Y
    car = S.box(X, Y, Z, (6, y0 - 22, 4), (12.5, y0 + 22, 12), 1.2)
    stem = S.box(X, Y, Z, (10, y0 - 7, 6.2), (34, y0 + 7, 10.8), 1.2)
    u, w = S.rot2(X - 34, Y - y0, -14)                                                  # knob swept forward 14 deg
    knob = S.box(u, w, Z, (-1, -10, 1), (19, 10, 16), 4.0)
    gro = np.maximum(np.abs(((Z - 1) % 3.0) - 1.5) - 0.5, 16.5 - u)
    knob = S.rdiff(knob, gro, 0.3)
    return S.union(car, S.runion(stem, knob, 2.0))


# ------------------------------------------------------------------ lower, trigger, controls, pins
LO_PROF = poly2([(P.LO_Y0, P.UP_BOT), (P.LO_Y1, P.UP_BOT), (P.LO_Y1, -27.0), (24.0, P.LO_BOT), (-90.0, P.LO_BOT),
                 (-96.0, -42.0), (P.LO_Y0, -42.0)])


def mag_xs(X, Y, Z, grow=0.0):
    dy = 0.0005 * np.maximum(P.MAG_TOP - Z, 0) ** 2
    yy = Y - dy
    r = np.where(yy > (P.MAG_Y[0] + P.MAG_Y[1]) / 2, 4.0, 2.4) + grow
    return S.rect2(X, yy, (-P.MAG_HW - grow, P.MAG_Y[0] - grow), (P.MAG_HW + grow, P.MAG_Y[1] + grow), r)


def lower(X, Y, Z):
    hw = np.where(Y > -96, 20.0 + 1.6 * S.sstep(-47, -53, Z), 21.0)
    d = S.rinter(LO_PROF(Y, Z), np.abs(X) - hw, 2.4)
    guard = S.extrude(S.rect2(Y, Z, (-160, -67), (-90, -38), 9.0), X, -6.5, 6.5, 1.5)
    guard = S.diff(guard, S.extrude(S.rect2(Y, Z, (-153, -61.5), (-97, -40), 6.0), X, -10, 10))
    d = S.runion(d, guard, 2.5)
    well = np.maximum(mag_xs(X, Y, Z, 0.35 + 2.0 * S.sstep(-47, -54, Z)), S.slab(Z, -70, -15))
    d = S.diff(d, well)
    d = S.diff(d, S.box(X, Y, Z, (-3.6, -124, -46), (3.6, -100, -28), 1.0))            # trigger slot
    d = S.rdiff(d, S.box(np.abs(X), Y, Z, (19.5, -12, -45), (25, 22, -27), 2.0), 0.5)    # front grip texture panels
    for xs in (1, -1):                                                                   # pin bosses
        d = S.runion(d, S.cyl(X * xs, Y, Z, "x", (18, -25), 5.5, 15, 21.6, 0.6), 1.0)
        d = S.runion(d, S.cyl(X * xs, Y, Z, "x", (-168, -30), 5.5, 16, 21.6, 0.6), 1.0)
    return d


def trigger(X, Y, Z):
    return S.extrude(TRIG(Y, Z), X, -3.2, 3.2, 0.9)


TRIG = poly2([(-104, -27), (-111, -27), (-113.5, -36), (-115, -45), (-114, -53), (-111, -59), (-107, -60.5),
              (-108.5, -55), (-109.5, -48), (-108.5, -40), (-106, -33)], res=0.05)


def controls(X, Y, Z):
    """ambi safety (safe, levers forward), ambi mag release, bolt catch (left)"""
    out = []
    for xs in (1, -1):
        hub = S.cyl(X * xs, Y, Z, "x", (-150, -30), 5.2, 20.6, 23.6, 0.8)
        arm = S.box(X * xs, Y, Z, (21.2, -150, -33.2), (23.6, -126, -26.8), 1.1)
        out.append(S.runion(hub, arm, 1.5))
        out.append(S.cyl(X * xs, Y, Z, "x", (-97, -34), 4.6, 19.0, 22.8, 1.2))
    bc = S.box(X, Y, Z, (19.3, -82, -33), (23.0, -60, -26), 1.2)
    bc = S.rdiff(bc, np.maximum(np.abs(((Y + 82) % 2.2) - 1.1) - 0.45, 22.6 - X), 0.2)
    out.append(bc)
    return S.union(*out)


def pins(X, Y, Z):
    out = []
    for (yc, zc) in ((18.0, -25.0), (-168.0, -30.0)):
        out.append(S.cyl(X, Y, Z, "x", (yc, zc), 2.5, -21.2, 21.2))
        for xs in (1, -1):
            out.append(S.cyl(X * xs, Y, Z, "x", (yc, zc), 4.4, 21.0, 22.6, 0.6))
    return S.union(*out)


# ------------------------------------------------------------------ grip (A2 style, finger bump, hollow butt)
def grip(X, Y, Z):
    (y0, z0), (y1, z1) = P.GRIP_TOP, P.GRIP_BOT
    L = math.hypot(y1 - y0, z1 - z0)
    ay, az = (y1 - y0) / L, (z1 - z0) / L                     # axis (down/back)
    w = (Y - y0) * ay + (Z - z0) * az                          # along the axis
    s = (Y - y0) * (-az) + (Z - z0) * ay                       # perpendicular, + = forward
    front = 15.5 + 3.4 * np.exp(-((w - 46) / 8.5) ** 2) - 2.2 * np.clip(w / L, 0, 1) - 2.0 * np.exp(-((w - 30) / 6) ** 2)
    back = -17.0 - 2.4 * S.sstep(55, 100, w)
    d2 = np.maximum(np.maximum(s - front, back - s), np.maximum(-w - 6, w - L))
    hw = 12.4 + 1.4 * S.sstep(10, 70, w)
    d = S.rinter(d2, np.abs(X) - hw, 5.0)
    d = S.diff(d, np.maximum(np.maximum(np.abs(X) - 8.5, np.abs(s + 1) - 10.5), L - 30 - w))   # hollow butt
    d = np.maximum(d, Z - (P.UP_BOT - 21))                     # trimmed under the lower
    return d


# ------------------------------------------------------------------ magazine (20 rd steel) + top round
def magazine(X, Y, Z):
    body = np.maximum(mag_xs(X, Y, Z), S.slab(Z, P.MAG_BOT, P.MAG_TOP))
    dy = 0.0005 * np.maximum(P.MAG_TOP - Z, 0) ** 2
    yy = Y - dy
    for yc in (-62.0, -36.0):                                  # pressed side ribs
        rib = np.maximum(np.maximum(np.abs(X) - (P.MAG_HW + 0.7), np.abs(yy - yc) - 3.5), S.slab(Z, -168, -46))
        body = S.runion(body, S.rinter(rib, -np.abs(X) + P.MAG_HW - 1, 0.5), 1.2)
    lipw = (P.MAG_HW - 4.0) + 3.5 * S.sstep(P.MAG_TOP - 1, P.MAG_TOP - 6, Z) + 20.0 * S.sstep(P.MAG_TOP - 6, P.MAG_TOP - 8, Z)
    body = np.maximum(body, np.abs(X) - lipw)                                             # feed lips (continuous)
    body = S.diff(body, np.maximum(S.rect2(X, yy, (-P.MAG_HW + 1.0, P.MAG_Y[0] + 1.0), (P.MAG_HW - 1.0, P.MAG_Y[1] - 1.0), 1.5),
                                   S.slab(Z, -40, 10)))
    fol = np.maximum(S.rect2(X, yy, (-P.MAG_HW + 1.2, P.MAG_Y[0] + 1.2), (P.MAG_HW - 1.2, P.MAG_Y[1] - 1.2), 1.4),
                     S.slab(Z, -44, -37))
    fol = S.union(fol, S.box(X, yy, Z, (-P.MAG_HW + 1.2, P.MAG_Y[0] + 1.2, -38), (-0.5, P.MAG_Y[1] - 1.2, -33.5), 1.2))   # stagger step
    yb0 = P.MAG_Y[0] + 0.0005 * (P.MAG_TOP - P.MAG_BOT) ** 2
    base = S.box(X, Y, Z, (-P.MAG_HW - 1.6, yb0 - 3.0, P.MAG_BOT - 8), (P.MAG_HW + 1.6, yb0 + 83.0, P.MAG_BOT + 1), 2.2)
    base = S.rdiff(base, S.box(X, Y, Z, (-6, yb0 + 70, P.MAG_BOT - 9), (6, yb0 + 76, P.MAG_BOT - 6), 1.0), 0.4)
    return S.union(S.runion(body, base, 1.0), fol)


def round_(X, Y, Z):
    """7.62x51: case head at y 0 in the round's own frame (axis +y)"""
    R = np.sqrt(X * X + Z * Z)
    prof = [(0, 6.0), (1.2, 6.0), (1.3, 5.0), (2.4, 5.0), (3.2, 5.95), (39.0, 5.75), (43.5, 4.3), (51.0, 4.2),
            (55.0, 3.9), (63.0, 3.1), (68.5, 1.6), (71.0, 0.6)]
    d = lathe(Y, R, prof)
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 2.6, -1, 0.5))


# ------------------------------------------------------------------ stock (extended), cheek riser, buttpad
ST_OUT = poly2([(P.STOCK_Y0, 22.0), (-505.0, 25.0), (-505.0, -100.0), (-468.0, -100.0), (-212.0, -46.0), (P.STOCK_Y0, -40.0)])
ST_WIN = poly2([(-300.0, 7.0), (-482.0, 9.0), (-482.0, -76.0), (-300.0, -40.0)])


def stock(X, Y, Z):
    hw = np.where(Z > 6, 17.0, 13.0)
    hw = np.where(Y > -300, 19.0, hw)
    hw = hw + 3.0 * S.sstep(-440, -505, Y)
    d = S.rinter(ST_OUT(Y, Z), np.abs(X) - hw, 4.0)
    d = S.rdiff(d, np.maximum(ST_WIN(Y, Z) + 2.0, hw - 4.0 - np.abs(X)), 3.0)          # deep side panels (both sides)
    lever = S.box(X, Y, Z, (-6, -228, -52), (6, -210, -42), 2.0)                           # length-adjust lever
    d = S.union(d, lever)
    d = S.rdiff(d, S.cyl(X, Y, Z, "x", (-470, -80), 5.5, 10, 30), 0.6)                     # QD socket (left)
    cheek = S.box(X, Y, Z, (-15.5, -405, 18), (15.5, -250, 32.5), 5.0)
    d = S.runion(d, cheek, 2.0)
    return d


def buttpad(X, Y, Z):
    d = S.box(X, Y, Z, (-21.5, P.BUTT_Y, -102), (21.5, -504.0, 27), 5.0)
    rib = np.maximum(np.abs(((Z + 102) % 3.4) - 1.7) - 0.7, Y - (P.BUTT_Y + 0.8))
    return S.rdiff(d, rib, 0.25)


# ------------------------------------------------------------------ folding sights (stowed)
def _sight(X, Y, Z, y0, y1, ears):
    base = S.box(X, Y, Z, (-12.0, y0, P.RAIL_TOP - 0.05), (12.0, y1, P.RAIL_TOP + 6.0), 1.2)
    for xs in (1, -1):
        jaw = S.box(X * xs, Y, Z, (9.0, y0, 25.5), (12.0, y1, P.RAIL_TOP + 1), 0.8)
        base = S.runion(base, jaw, 0.8)
    base = S.diff(base, rail(X, Z) - 0.05)
    leaf = S.box(X, Y, Z, (-8.0, y0 + 2, P.RAIL_TOP + 5.9), (8.0, y1 - 2, P.RAIL_TOP + 9.5), 1.2)
    if ears:
        for xs in (1, -1):
            leaf = S.runion(leaf, S.box(X * xs, Y, Z, (5.5, y0 + 4, P.RAIL_TOP + 9), (8.0, y1 - 4, P.RAIL_TOP + 12.5), 0.8), 0.6)
    else:
        ring = np.maximum(np.abs(np.sqrt(X * X + (Y - (y0 + y1) / 2) ** 2) - 3.6) - 1.4, S.slab(Z, P.RAIL_TOP + 8, P.RAIL_TOP + 11.5))
        leaf = S.runion(leaf, ring, 0.6)
        leaf = S.diff(leaf, S.cyl(X, Y, Z, "z", (0, (y0 + y1) / 2), 1.0, 0, 60))
    knob = S.cyl(X, Y, Z, "x", ((y0 + y1) / 2, P.RAIL_TOP + 3), 4.0, 12.0, 15.0, 0.6)
    return S.union(base, leaf, knob)


def rear_sight(X, Y, Z):
    return _sight(X, Y, Z, -150.0, -117.0, False)


def front_sight(X, Y, Z):
    return _sight(X, Y, Z, 255.0, 283.0, True)


# ------------------------------------------------------------------ build table: name -> (fn, lo, hi, h, mat)
PARTS = {
    "upper": (upper, (-36, -206, -47), (36, 294, 34), 0.42, "anod"),
    "latch": (latch, (17, -196, -16), (26, -184, -4), 0.15, "steel"),
    "barrel": (barrel, (-16, -6, -16), (16, 415, 16), 0.25, "steel"),
    "gasblock": (gasblock, (-14, 287, -31), (14, 319, 14), 0.2, "park"),
    "hider": (hider, (-12, 396, -12), (12, 458, 12), 0.18, "steel"),
    "bolt": (bolt, (-15, -134, -18), (15, 3, 20), 0.25, "park"),
    "charging": (charging, (4, P.CH_FWD_Y - 30, -1), (60, P.CH_FWD_Y + 30, 18), 0.18, "steel"),
    "lower": (lower, (-24, -180, -69), (24, 30, -19), 0.32, "poly"),
    "trigger": (trigger, (-4, -117, -62), (4, -102, -25), 0.12, "steel"),
    "controls": (controls, (-25, -158, -42), (25, -55, -20), 0.15, "steel"),
    "pins": (pins, (-24, -175, -36), (24, 24, -19), 0.15, "steel"),
    "grip": (grip, (-16, -222, -160), (16, -140, -40), 0.28, "poly"),
    "magazine": (magazine, (-16, -92, -186), (16, 24, -24), 0.25, "magsteel"),
    "round": (round_, (-7, -1, -7), (7, 72, 7), 0.12, "brass"),
    "stock": (stock, (-24, -508, -104), (24, -198, 36), 0.4, "poly"),
    "buttpad": (buttpad, (-23, -524, -104), (23, -502, 29), 0.25, "rubber"),
    "rsight": (rear_sight, (-16, -152, 24), (16, -115, 46), 0.15, "steel"),
    "fsight": (front_sight, (-16, 253, 24), (16, 285, 46), 0.15, "steel"),
}


def round_mat(V):
    return np.where(V[:, 1] > 50.5, MAT["copper"], MAT["brass"])


def main():
    out = os.path.join(HERE, "work", "high.npz")
    old = dict(np.load(out)) if (ONLY and os.path.exists(out)) else {}
    parts = {}
    for name, (fn, lo, hi, h, mat) in PARTS.items():
        if ONLY and name not in ONLY:
            continue
        off = h * HS * 0.3713                                       # keep grid nodes off exact zero crossings (holes)
        V, F = S.mesh(fn, np.add(lo, -off), np.add(hi, off), h * HS, lip=3.5, name=name)
        m = round_mat(V) if name == "round" else np.full(len(V), MAT[mat])
        parts[name] = (V, F, {"mat": m})
    S.save(out + ".tmp.npz", parts)
    new = dict(np.load(out + ".tmp.npz"))
    if old:
        for k in list(old):
            if k.split("__")[0] in parts:
                old.pop(k)
        new.update(old)
    np.savez_compressed(out, **new)
    os.remove(out + ".tmp.npz")
    print("HIGH_DONE", sorted({k.split("__")[0] for k in new}))


if __name__ == "__main__":
    main()
