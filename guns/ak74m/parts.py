"""parts.py - AK-74M high-poly parts as signed-distance fields in the B frame (mm) -> work/high.npz.
Outline of every part = the profile-master silhouette (ak_ref.prof); cross-sections are real-world shapes: lathed round
parts (brake, barrel, gas tube, rod), D-section dust cover, rounded polymer furniture, thin stamped trigger guard.
usage: .venv/bin/python guns/ak74m/parts.py [parts=a,b] [h_scale=1.0] [out=path]"""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S
import ak_ref as R
import ak_spec as SP

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ONLY = ARGS["parts"].split(",") if "parts" in ARGS else None
HS = float(ARGS.get("h_scale", 1.0))
HW = SP.HW
bp = R.bp
MAT = dict(steel=0, poly=1, park=2)


def lathe_photo(Y, Z, X, name, zc=0.0, rmax=99.0, smooth=1):
    """round part: radius per photo column of the part's silhouette (about axis z = zc)"""
    cr = R.column_radius(*_xr(name), name)
    ys = np.array([c[0] for c in cr]); rs = np.array([min(c[1], rmax) for c in cr])
    if smooth > 1:
        rs = np.convolve(np.pad(rs, smooth // 2, mode="edge"), np.ones(smooth) / smooth, mode="valid")
    o = np.argsort(ys); ys, rs = ys[o], rs[o]
    rr = np.hypot(X, Z - zc)
    r = np.interp(Y, ys, rs, left=-1, right=-1)
    return np.maximum(rr - r, np.maximum(ys[0] - Y, Y - ys[-1]))


def _xr(name):
    p = R.PARTS[name]
    return min(q[0] for q in p), max(q[0] for q in p)


def slab(P2, X, hw, r=0.8):
    return S.rinter(P2, np.abs(X) - hw, r)


# ------------------------------------------------------------------ muzzle end
def brake(X, Y, Z):
    d = lathe_photo(Y, Z, X, "brake", rmax=11.6)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 3.4, -100, 5))                          # bore
    y0, z0 = bp(168, 270); y1, z1 = bp(188, 285)
    d = S.rdiff(d, S.box(X, Y, Z, (-20, y1, z1), (20, y0, z0), 2.0), 0.6)           # two side windows (through)
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.5, -4, 5), 0.6)                     # stepped front face
    return d


def fsb(X, Y, Z):
    P2 = R.prof("fsb")(Y, Z)
    _, zear = bp(0, 258)
    hw = np.where(Z > zear, HW["fsb_ears"], HW["fsb"])
    d = slab(P2, X, hw, 1.0)
    gap = np.maximum.reduce([np.abs(X) - 3.4, zear + 2 - Z, -(np.abs(X) - 1.2)])     # U ears around the centre post
    d = S.rdiff(d, gap, 0.5)
    y, z = bp(255, 250)
    d = S.diff(d, S.cyl(X, Y, Z, "x", (y, z), 4.6, -20, 20))                          # through hole
    return d


def barrel(X, Y, Z):
    y0 = bp(276, 0)[0]; y1 = bp(565, 0)[0]
    r = np.interp(Y, [y1, bp(345, 0)[0], y0], [10.2, 8.4, 8.0])
    d = np.maximum(np.hypot(X, Z) - r, np.maximum(y1 - Y, Y - y0))
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 2.8, y1 - 5, y0 + 5))


def rod(X, Y, Z):
    _, z = bp(0, 292.5)
    return S.cyl(X, Y, Z, "y", (0, z), 3.0, bp(558, 0)[0], bp(246, 0)[0], 1.0)


def gasblock(X, Y, Z):
    return slab(R.prof("gasblock")(Y, Z), X, HW["gas_block"], 3.0)


def gastube(X, Y, Z):
    cr = R.column_radius(*_xr("gastube"), "gastube")
    zc = float(np.median([c[2] for c in cr]))
    return lathe_photo(Y, Z, X, "gastube", zc=zc, rmax=HW["gas_tube"] + 1.5, smooth=3)


# ------------------------------------------------------------------ handguards
def hg_upper(X, Y, Z):
    return slab(R.prof("hg_upper")(Y, Z), X, HW["hg_upper"], 8.0)


def hg_lower(X, Y, Z):
    return slab(R.prof("hg_lower")(Y, Z), X, HW["hg_lower"], 12.0)


# ------------------------------------------------------------------ receiver group
def rsight(X, Y, Z):
    P2 = R.prof("rsight")(Y, Z)
    _, zl = bp(0, 236)
    hw = np.where(Z > zl, HW["rear_leaf"], HW["rear_sight_block"])
    return slab(P2, X, hw, 1.0)


def dcover(X, Y, Z):
    """D-section: vertical sides, round top (radius = half width)"""
    P2 = R.prof("dcover")(Y, Z)
    hw = HW["dust_cover"]
    cr = R.column_radius(*_xr("dcover"), "dcover")                                  # the cover's top follows the photo
    ys = np.array([c[0] for c in cr]); zt = np.array([c[2] + c[1] for c in cr]); o = np.argsort(ys)
    ztop = np.interp(Y, ys[o], zt[o])
    top = np.hypot(X, np.maximum(Z - (ztop - hw), 0)) - hw
    return S.rinter(P2, np.maximum(np.abs(X) - hw, top), 0.6)


def receiver(X, Y, Z):
    return slab(R.prof("receiver")(Y, Z), X, HW["receiver"], 1.6)


def tguard(X, Y, Z):
    return slab(R.prof("tguard", blur=0.3)(Y, Z), X, HW["trigger_guard"], 0.6)


def grip(X, Y, Z):
    return slab(R.prof("grip")(Y, Z), X, HW["grip"], 9.0)


def mag(X, Y, Z):
    return slab(R.prof("mag")(Y, Z), X, HW["mag"], 4.0)


def stock(X, Y, Z):
    return slab(R.prof("stock")(Y, Z), X, HW["stock"], 9.0)


# ------------------------------------------------------------------ table
def table():
    T = {}
    for n, fn, mat, h in (("brake", brake, "park", 0.25), ("fsb", fsb, "park", 0.25), ("barrel", barrel, "park", 0.3),
                          ("rod", rod, "park", 0.25), ("gasblock", gasblock, "park", 0.25), ("gastube", gastube, "park", 0.3),
                          ("hg_upper", hg_upper, "poly", 0.4), ("hg_lower", hg_lower, "poly", 0.4), ("rsight", rsight, "park", 0.25),
                          ("dcover", dcover, "steel", 0.35), ("receiver", receiver, "steel", 0.4), ("tguard", tguard, "steel", 0.25),
                          ("grip", grip, "poly", 0.4), ("mag", mag, "poly", 0.4), ("stock", stock, "poly", 0.5)):
        y0, y1, z0, z1 = R.extent(n)
        xh = max(v for k, v in HW.items()) + 4
        T[n] = (fn, (-xh, y0 - 4, z0 - 4), (xh, y1 + 4, z1 + 4), h, mat)
    return T


def main():
    out = ARGS.get("out", os.path.join(HERE, "work", "high.npz"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    old = dict(np.load(out)) if (ONLY and os.path.exists(out)) else {}
    parts = {}
    for name, (fn, lo, hi, h, mat) in table().items():
        if ONLY and name not in ONLY:
            continue
        off = h * HS * 0.3713
        V, F = S.mesh(fn, np.add(lo, -off), np.add(hi, off), h * HS, lip=3.5, name=name)
        parts[name] = (V, F, {"mat": np.full(len(V), MAT[mat])})
    S.save(out + ".tmp.npz", parts)
    new = dict(np.load(out + ".tmp.npz"))
    for k in list(old):
        if k.split("__")[0] not in parts:
            new[k] = old[k]
    np.savez_compressed(out, **new)
    os.remove(out + ".tmp.npz")
    print("HIGH_DONE", sorted({k.split("__")[0] for k in new}))


if __name__ == "__main__":
    main()
