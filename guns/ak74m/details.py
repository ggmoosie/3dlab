"""details.py - S3 surface detail as data: every feature is a row (part, kind, photo-px geometry, size, side) read off the
profile photos (left master 163245; right side read off work/ref_R_aligned.png, which is in the left photo's px frame).
Kinds work on the part's own distance field, so a groove or a rivet follows whatever surface it lands on:
  groove   2D band (photo px rect / polygon) cut `depth` mm into the surface inside the band
  pocket   same, flat-bottomed (rect / polygon)
  boss     2D band raised `h` mm off the surface (plates, rails)
  rivet    domed head radius `r` px, `h` mm proud (dome follows the surface)
  dimple   spherical dent radius `r` px, `depth` mm
  ring     raised annulus radius `r` px with a hole (hinge pin)
side: +1 = left (B +X), -1 = right, 0 = both."""
import numpy as np
import ak_ref as R
from lab import sdf as S

bp = R.bp
PX = R.S_MM                                            # mm per photo px


def _poly2(Y, Z, pts):
    """signed distance (mm) to a convex photo-px polygon in the (y, z) plane"""
    P = [bp(*q) for q in pts]
    d = np.full_like(Y, -1e9)
    sgn = np.sign(sum((b[0] - a[0]) * (b[1] + a[1]) for a, b in zip(P, P[1:] + P[:1])))   # orientation
    for a, b in zip(P, P[1:] + P[:1]):
        ny, nz = -sgn * (b[1] - a[1]), sgn * (b[0] - a[0]); n = np.hypot(ny, nz)                 # outward normal
        d = np.maximum(d, ((Y - a[0]) * ny + (Z - a[1]) * nz) / n)
    return d


def _rect(x0, r0, x1, r1):
    return [(x0, r0), (x1, r0), (x1, r1), (x0, r1)]


def _side(X, side):
    return -side * X if side else np.full_like(X, -1e9)


def apply(name, d, X, Y, Z):
    for f in FEATURES.get(name, ()):
        k, side = f["kind"], f.get("side", 0)
        if k in ("groove", "pocket", "boss"):
            g = _poly2(Y, Z, f.get("poly") or _rect(*f["rect"]))
            if k == "boss":
                e = np.maximum.reduce([g, d - f["h"], _side(X, side)])
                d = S.runion(d, e, 0.3)
            else:
                dep = f["depth"]
                cut = np.maximum.reduce([g, -d - dep, _side(X, side)])
                d = S.rdiff(d, cut, f.get("r", 0.25))
        elif k in ("rivet", "dimple", "ring"):
            for x, row in f["at"]:
                yc, zc = bp(x, row); rad = f["r"] * PX
                rho = np.hypot(Y - yc, Z - zc)
                cap = np.sqrt(np.clip(1 - (rho / rad) ** 2, 0, 1))
                if k == "rivet":
                    d = np.minimum(d, np.maximum.reduce([rho - rad, d - f["h"] * cap, _side(X, side)]))
                elif k == "dimple":
                    d = np.maximum(d, -np.maximum.reduce([rho - rad, -d - f["depth"] * cap, _side(X, side)]))
                else:
                    ring = np.maximum.reduce([rho - rad, d - f["h"], _side(X, side)])
                    d = S.diff(np.minimum(d, ring), np.maximum.reduce([rho - rad * 0.45, -d - 3.0, _side(X, side)]))
    return d


# ------------------------------------------------------------------ the feature table (spec counts in ak_spec.FEATURES)
FEATURES = {
    "brake": [dict(kind="groove", rect=(196, 262, 198, 295), depth=0.5)],                          # turned ring behind the windows
    "fsb": [dict(kind="rivet", at=[(251, 292), (269, 291)], r=2.2, h=0.6)],                         # lug pins
    "gasblock": [dict(kind="rivet", at=[(346, 276), (366, 276)], r=2.4, h=0.6)],                    # 2 pin heads
    "hg_lower": [
        *[dict(kind="groove", rect=(573, rw - 0.9, 621, rw + 0.9), depth=1.1) for rw in (266, 271, 276, 281, 286, 291)],  # 6 finger ribs
        *[dict(kind="pocket", rect=(xc - 1.3, 266, xc + 1.3, 282), depth=2.4, r=0.4) for xc in (473, 479, 485, 491, 497)],  # 5 vents
    ],
    "receiver": [
        dict(kind="rivet", at=[(673, 294), (633, 276)], r=2.3, h=0.7),                             # front trunnion rivets
        dict(kind="pocket", rect=(690, 289, 720, 300), depth=0.9, r=0.8),                           # oval mag-guide dimple
        dict(kind="rivet", at=[(751, 302), (766, 298), (808, 307)], r=3.0, h=0.8),                  # FCG + selector pins (through)
        dict(kind="boss", rect=(733, 283.5, 838, 304.5), h=2.2, side=1),                           # left side rail plate
        dict(kind="groove", rect=(745, 290, 828, 292), depth=0.8, side=1),                          # rail slot
        dict(kind="rivet", at=[(740, 295), (786, 294), (832, 296)], r=2.4, h=0.6, side=1),           # 3 rail rivets
        dict(kind="ring", at=[(874, 289.5)], r=5.0, h=1.4),                                         # stock hinge pin
    ],
    "stock": [
        dict(kind="pocket", poly=[(930, 285), (1065, 307), (1065, 336), (930, 312)], depth=2.2, r=1.0),    # side recess
        dict(kind="groove", poly=[(938, 290), (1056, 311), (1056, 314), (938, 293)], depth=0.7),           # recess inner line
        dict(kind="ring", at=[(919, 293)], r=5.5, h=1.0),                                                  # hinge button
        dict(kind="dimple", at=[(1092, 327)], r=6.0, depth=1.2),                                           # trapdoor press
        dict(kind="pocket", rect=(1112, 316, 1136, 344), depth=3.0, r=0.8),                                # swivel pocket
    ],
}

# selector (right side): lever along the receiver from the rear pivot to the front finger tab (ref_R_aligned px)
SEL_PATH = [(832, 278), (790, 268), (750, 262), (700, 256), (654, 255)]
SEL_PIVOT, SEL_TAB = (832, 278), (_rect(646, 248, 655, 264))
