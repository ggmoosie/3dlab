"""mod_front.py - muzzle end modules: compensator (AK-74 two-chamber brake, hollow), barrel (full length: thread nose ->
chamber, the hidden part under the handguards included), front sight block, gas block, cleaning rod."""
import numpy as np
import ak_geo as G
from ak_geo import S, R, HW, bp
from iface import Y as IY, Z as IZ


def compensator(X, Y, Z):
    d = G.lathe_photo(Y, Z, X, "brake", zc=G.bz(278.0), rmax=14.0)
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.5, -4, 5), 0.6)                         # counterbored front face
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 3.6, -12, 5))                              # bullet hole in the front plate
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 8.6, IY["thread_front"] + 1, -7.5), 0.8)  # expansion chamber
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.25, IY["thread_rear"] - 5, IY["thread_front"] + 1))   # threaded socket
    y0, z0 = bp(170, 271); y1, z1 = bp(183.5, 285)                                       # the photo's window
    d = S.rdiff(d, S.box(X, Y, Z, (-20, y1, z1), (20, y0, z0), 1.6), 0.5)              # two side windows (through)
    for ang, yc in ((90, G.by(212)), (135, G.by(205)), (170, G.by(209))):                  # 3 ports: top, right-up, right (photos)
        a = np.radians(ang); u = X * np.cos(a) + Z * np.sin(a)
        v = np.hypot(-X * np.sin(a) + Z * np.cos(a), Y - yc)
        d = S.diff(d, np.maximum(v - 1.7, 4 - u))
    return d


def barrel(X, Y, Z):
    ys = [IY["thread_front"], IY["thread_rear"], IY["thread_rear"] - 0.1, G.by(276), G.by(345), G.by(565), -440, IY["bolt_face"]]
    rs = [7.0, 7.0, 7.6, 7.3, 7.4, 10.5, 12.0, 12.0]
    d = G.lathe(Y, None, ys, rs, X=X, Z=Z)
    for k in range(6):                                                                  # thread crests (shallow rings)
        yk = IY["thread_front"] - 2 - k * 3.6
        d = S.rdiff(d, np.maximum(np.abs(Y - yk) - 0.5, 6.6 - np.hypot(X, Z)), 0.2)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 2.8, IY["bolt_face"] - 5, 5))                 # bore
    return S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 5.0, IY["bolt_face"] - 5, IY["bolt_face"] + 28), 0.4)   # chamber


def fsb(X, Y, Z):
    P2 = R.prof("fsb")(Y, Z)
    zear = G.bz(258)
    hw = np.where(Z > zear, HW["fsb_ears"], HW["fsb"])
    d = G.slab(P2, X, hw, 1.0)
    gap = np.maximum.reduce([np.abs(X) - 3.4, zear + 2 - Z, -(np.abs(X) - 1.2)])          # U ears around the centre post
    d = S.rdiff(d, gap, 0.5)
    y, z = bp(263, 265)
    d = S.diff(d, S.cyl(X, Y, Z, "x", (y, z), 2.4, -20, 20))                               # small pin hole
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.3, IY["fsb_rear"] - 2, IY["fsb_front"] + 2))   # barrel seat


def rod(X, Y, Z):
    z = G.bz(291.6)
    return S.cyl(X, Y, Z, "y", (0, z), 2.4, G.by(558), G.by(246), 1.0)


def gasblock(X, Y, Z):
    d = G.slab(R.prof("gasblock")(Y, Z), X, HW["gas_block"], 3.0)
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.4, IY["gasblock_rear"] - 2, IY["gasblock_front"] + 2))  # barrel seat
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, IZ["piston"]), 7.9, IY["gasblock_rear"] - 2, IY["gastube_front"] - 6))  # gas cylinder


PARTS = [
    dict(name="compensator", slot="muzzle", mat="park", h=0.22, outer=compensator, trace="brake"),
    dict(name="barrel", slot="barrel", mat="park", h=0.3, outer=barrel, box=(IY["bolt_face"] - 2, IY["thread_front"] + 2, -13, 13)),
    dict(name="fsb", slot="front_sight", mat="park", h=0.25, outer=fsb, trace="fsb"),
    dict(name="rod", slot="cleaning_rod", mat="park", h=0.25, outer=rod, trace="rod"),
    dict(name="gasblock", slot="gas_block", mat="park", h=0.25, outer=gasblock, trace="gasblock"),
]
