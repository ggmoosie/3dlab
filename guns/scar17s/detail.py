"""detail.py - the SCAR 17S NRCH surface-detail catalogue (photo-1 px, right side; mirrored on the left unless noted) and the
SDF helpers that apply it to a part: Torx/button screws, sockets, pins, grooves (seams, panel lines, knuckle gaps), raised
traced hardware (from relief.py), engraved numbers, stipple panels with real bumps. Every feature names where it was read.
A part applies its features with apply(d, X, Y, Z, part, hw) where hw(Y, Z) is that part's side-surface half width."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import os
import ref as R
import relief
from lab import sdf as S

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "../../fonts/WorkSans-Regular.ttf")


def P(x, row):
    y, z = R.px2b(x, row)
    return float(y), float(z)


def mm(px):
    return px * R.S_MM


# ------------------------------------------------------------------ catalogue (photo-1 px) ----------------------------
# screws: (part, x, row, head radius px, kind)   kind: torx (button head + 6-lobe recess), socket (QD cup: ring + bore),
# pin (flush pin end), hex (socket-head cap screw)
SCREWS = [
    ("upper", 628, 915, 6.5, "torx"), ("upper", 650, 915, 6.5, "torx"),                  # hinge-side receiver screws
    ("upper", 630, 868, 11.5, "socket"),                                                 # hinge QD socket
    ("upper", 1118, 868, 10.5, "hex"),                                                   # CH slot end cap
    ("upper", 920, 868, 3.0, "pin"),                                                     # pin inside the CH slot
    ("upper", 1345, 866, 6.5, "torx"), ("upper", 1370, 866, 6.5, "torx"),                # handguard screws (top)
    ("plate", 1037, 920, 6.0, "torx"), ("plate", 1083, 920, 6.0, "torx"),                # rear QD plate
    ("plate", 1060, 920, 2.6, "pin"),
    ("plate", 1527, 866, 6.0, "torx"), ("plate", 1557, 866, 6.0, "torx"),                # front QD plate
    ("plate", 873, 918, 5.0, "torx"),                                                    # port frame screw
    ("plate", 822, 920, 3.6, "pin"),                                                     # deflector pivot
    ("lower", 722, 952, 4.0, "pin"), ("lower", 790, 1000, 4.0, "pin"),                   # FCG pins
    ("lower", 845, 978, 5.5, "pin"),                                                     # bolt-release pivot
    ("lower", 1045, 962, 9.0, "pin"),                                                    # front takedown pin
    ("stock", 243, 918, 12.0, "socket"),                                                 # buttstock QD socket (through)
    ("stock", 217, 1010, 6.0, "pin"),
    ("stock", 560, 882, 8.0, "pin"),
    ("stock", 377, 921, 7.5, "button"),                                                  # stock latch button
    ("fsight", 1549, 836, 6.0, "hex"),
]
# grooves (part, [polyline px], width mm, depth mm)
GROOVES = [
    ("upper", [(630, 942), (1048, 942)], 0.6, 0.6),                                      # upper / lower parting line
    ("upper", [(1135, 905), (1600, 905)], 0.6, 0.8),                                     # handguard top seam
    ("upper", [(1048, 963), (1135, 963), (1135, 935), (1124, 916), (1136, 905)], 0.6, 0.8),  # handguard rear junction
    ("upper", [(612, 857), (612, 990)], 0.5, 0.6),                                       # hinge plate seam
    ("stock", [(292, 893), (306, 992)], 0.6, 0.7),                                       # rear piece / tube
    ("stock", [(300, 893), (348, 893)], 0.5, 0.5), ("stock", [(300, 990), (348, 990)], 0.5, 0.5),
    ("stock", [(349, 878), (349, 936)], 0.6, 0.7),                                       # front body rear edge
    ("stock", [(470, 935), (566, 935)], 0.6, 0.6),                                       # front body panel line
    ("stock", [(566, 860), (566, 998)], 0.6, 0.7),                                       # front body / hinge knuckle
    ("butt", [(196, 885), (196, 1118)], 0.6, 0.7),                                       # buttpad rim lip
    ("grip", [(640, 1182), (735, 1188)], 0.5, 0.5),                                      # bottom rim
    ("lower", [(866, 1052), (1046, 1052)], 0.6, 0.6),                                    # magwell lip
    ("mag", [(870, 1199), (1000, 1186)], 0.6, 0.6),                                      # baseplate top edge (slanted)
]
# knuckle gaps of the folding hinge (part, x0, x1, gaps rows)
KNUCKLES = ("upper", 572, 624, [857, 893, 937, 985])
# raised traced hardware: (part, relief layer, standoff mm above the side surface)
HARDWARE = [("upper", 1.6)]
# recessed windows inside hardware / surfaces: (part, x0, r0, x1, r1, depth mm, radius mm)
RECESSES = [
    ("upper", 858, 860, 1102, 878, 7.0, 1.5),                                            # top charging slot, deep
    ("plate", 860, 908, 1010, 928, 3.5, 1.0),                                            # ejection-port window in the frame
    ("upper", 1422, 856, 1460, 891, 4.0, 1.2),                                           # sling cup
    ("stock", 215, 893, 255, 902, 30.0, 2.0),                                            # buttstock slot (through)
    ("mag", 892, 1202, 996, 1216, 4.0, 1.5),                                             # baseplate window
]
BOSSES = [
    ("plate", 884, 911, 891, 926, 2.4, 0.4),                                             # white tab in the port frame
    ("lower", 828, 958, 902, 1016, 1.4, 1.8),                                            # bezel ring around the mag release (ring)
    ("upper", 621, 945, 634, 976, 2.2, 0.6),                                             # hinge latch tab
]
# engraved numbers / letters (part, text, x, row, height px)
TEXT = [("stock", "6", 301, 943, 11), ("stock", "5", 324, 943, 11), ("stock", "4", 346, 943, 11), ("stock", "3", 368, 943, 11),
        ("lower", "1", 750, 955, 11), ("lower", "S", 757, 976, 11)]
# mag ribs: (x px, kind) ridge = raised 0.6, chan = recessed 0.5
MAG_RIBS = [(878, "chan"), (888, "ridge"), (940, "chan"), (948, "ridge"), (966, "chan"), (980, "ridge")]


# ------------------------------------------------------------------ SDF helpers ---------------------------------------
def _side(X, sign):
    return np.abs(X) if sign == 0 else X * sign


def screw(X, Y, Z, x, row, rpx, kind, hw, sign=0):
    """head SDF (raised) and recess SDF (cut) at the side surface hw (scalar mm) of a part"""
    y, z = P(x, row); r = mm(rpx); s = _side(X, sign)
    dyz = np.sqrt((Y - y) ** 2 + (Z - z) ** 2)
    if kind in ("torx", "hex", "button", "knob"):
        hp = {"torx": 0.9, "hex": 0.6, "button": 1.6, "knob": 2.4}[kind]
        head = S.extrude(dyz - r, s, hw - 0.6, hw + hp, min(hp * 0.7, r * 0.5))
        th = np.arctan2(Z - z, Y - y)
        if kind == "torx":
            star = r * 0.42 * (0.80 + 0.20 * np.cos(6 * th))
            rec = np.maximum(dyz - star, hw + hp - 0.9 - s)
        elif kind == "hex":
            hexr = r * 0.48 / np.maximum(np.cos((th % (math.pi / 3)) - math.pi / 6), 0.5)
            rec = np.maximum(dyz - hexr, hw + hp - 1.6 - s)
        elif kind == "knob":
            rec = np.maximum(np.abs(((th * 18 / (2 * math.pi)) % 1.0) - 0.5) * 2 * math.pi * r / 18 - 0.25, r - 0.6 - dyz)
            rec = np.maximum(rec, hw - 0.5 - s)
        else:
            rec = np.full_like(X, 1e3)
        return head, rec
    if kind == "socket":
        ring = S.extrude(np.abs(dyz - r * 0.82) - r * 0.18, s, hw - 0.6, hw + 1.0, 0.4)
        bore = np.maximum(dyz - r * 0.58, hw - 9.0 - s)
        return ring, bore
    if kind == "pin":
        head = S.extrude(dyz - r, s, hw - 0.8, hw + 0.25, 0.25)
        rec = np.maximum(np.abs(dyz - r * 0.55) - 0.12, hw - 0.2 - s)                    # turned ring on the pin end
        return head, rec
    raise ValueError(kind)


def polyline2(Y, Z, pts_px, w):
    pts = [P(*p) for p in pts_px]
    d = np.full_like(Y, 1e3)
    for a, b in zip(pts, pts[1:]):
        d = np.minimum(d, S.capsule2(Y, Z, a, b, w / 2))
    return d


_TF = {}


def text_field(text, x, row, hpx):
    key = (text, x, row, hpx)
    if key in _TF:
        return _TF[key]
    up = 6
    W, H = int(hpx * 1.2 * len(text) * up) + 20, int(hpx * 1.6 * up)
    im = Image.new("L", (W, H), 0)
    f = ImageFont.truetype(FONT, int(hpx * up * 1.25))
    ImageDraw.Draw(im).text((W / 2, H / 2), text, font=f, fill=255, anchor="mm")
    m = np.asarray(im) > 127
    res = R.S_MM / up
    d = (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)) * res
    y0, _ = R.px2b(x - W / 2 / up, 0); _, z1 = R.px2b(0, row - H / 2 / up)
    fl = S.Field2D(d.astype(np.float32), float(y0), float(z1), res, outside=10.0)
    _TF[key] = fl
    return fl


def bumps(Y, Z, pitch=0.95, amp=0.45):
    """irregular stipple: two warped sine lattices multiplied -> dimpled bumps about 1 mm apart"""
    a = np.sin(2 * math.pi * Y / pitch + 2.3 * np.sin(Z * 0.9) + 1.7 * np.sin(Y * 0.37))
    b = np.sin(2 * math.pi * Z / (pitch * 0.93) + 2.1 * np.sin(Y * 0.8) + 1.3 * np.sin(Z * 0.41))
    return amp * np.clip(a * b, 0, 1) ** 0.7


def heads(X, Y, Z, part, hw, mirror=True):
    """the steel screw / pin / socket heads of `part` as their own shell (they bake as black steel, not FDE)"""
    sides = (0,) if mirror else (-1,)
    out = np.full_like(X, 1e3)
    for (pt, x, row, rpx, kind) in SCREWS:
        if pt != part:
            continue
        for sg in sides:
            head, rec = screw(X, Y, Z, x, row, rpx, kind, hw, sg)
            out = np.minimum(out, head if kind == "socket" else S.diff(head, rec))
    return out


def apply(d, X, Y, Z, part, hw, mirror=True, with_heads=False):
    """apply every catalogued feature of `part`. hw: scalar or array (side half width at Y, Z). mirror: both sides.
    Screw heads are a separate steel shell (heads()); here they only bore their seats (sockets get their bore)."""
    sides = (0,) if mirror else (-1,)
    hwv = hw
    for (pt, x, row, rpx, kind) in SCREWS:
        if pt != part:
            continue
        for sg in sides:
            head, rec = screw(X, Y, Z, x, row, rpx, kind, hwv, sg)
            if kind == "socket":
                d = S.diff(d, rec)
            if with_heads:
                d = S.union(d, head if kind == "socket" else S.diff(head, rec))
    for (pt, pts, w, dep) in GROOVES:
        if pt != part:
            continue
        g = polyline2(Y, Z, pts, w)
        for sg in sides:
            d = S.rdiff(d, np.maximum(g, hwv - dep - _side(X, sg)), 0.15)
    if KNUCKLES[0] == part:
        _, x0, x1, rows = KNUCKLES
        ya, _ = P(x0, 0); yb, _ = P(x1, 0)
        for r in rows:
            _, z = P(0, r)
            gap = np.maximum(np.maximum(np.abs(Z - z) - 1.0, S.slab(Y, ya, yb)), -1e3)
            for sg in sides:
                d = S.rdiff(d, np.maximum(gap, hwv - 3.0 - _side(X, sg)), 0.2)
    for (pt, x0, r0, x1, r1, dep, rad) in RECESSES:
        if pt != part:
            continue
        (ya, za), (yb, zb) = P(x0, r1), P(x1, r0)
        box2 = S.rect2(Y, Z, (ya, za), (yb, zb), min(rad, (zb - za) / 2 - 0.05))
        for sg in sides:
            d = S.rdiff(d, np.maximum(box2, hwv - dep - _side(X, sg)), 0.3)
    for (pt, x0, r0, x1, r1, ht, rad) in BOSSES:
        if pt != part:
            continue
        (ya, za), (yb, zb) = P(x0, r1), P(x1, r0)
        b2 = S.rect2(Y, Z, (ya, za), (yb, zb), min(rad, (zb - za) / 2 - 0.05))
        if part == "lower" and x0 == 828:                                               # a ring, not a block
            b2 = np.maximum(b2, -(b2 + 1.6))
        for sg in sides:
            d = S.union(d, S.extrude(b2, _side(X, sg), hwv - 0.8, hwv + ht, 0.4))
    for (pt, txt, x, row, hpx) in TEXT:
        if pt != part:
            continue
        tf = text_field(txt, x, row, hpx)(Y, Z)
        d = S.rdiff(d, np.maximum(tf, hwv - 0.3 - _side(X, -1)), 0.05)                 # right side only (as photographed)
    return d


def hardware(X, Y, Z, hw, standoff=1.6):
    """traced black hardware (relief 'hardware' mask) standing proud of the receiver sides; mirrored left"""
    f = relief.field("hardware")(Y, Z)
    return S.extrude(f, np.abs(X), hw - 0.8, hw + standoff, 0.5)


def stipple(d, X, Y, Z, hw, recess=0.7):
    """recess the traced stipple panels (clay shape only); the stipple pattern itself is a material layer (bl_bake)"""
    f = relief.field("stipple")(Y, Z)
    pan = np.maximum(f, hw - recess - np.abs(X))
    return S.rdiff(d, pan, 0.35)
