"""mod_furniture.py - pistol grip, side-folding stock (+ right-side sling loop), magazine (open top: feed lips + top round)."""
import numpy as np
import ak_geo as G
from ak_geo import S, R, HW, bp
from iface import Y as IY

MAG_TOP = G.bz(300)
SLING = (911, 298)


def grip(X, Y, Z):
    return G.slab(R.prof("grip", blur=1.4)(Y, Z), X, HW["grip"], 9.0)


def stock(X, Y, Z):
    return G.slab(R.prof("stock", blur=1.4)(Y, Z), X, HW["stock"], 9.0)


def sling_loop(X, Y, Z):
    """wire D-loop on the stock's right side + its mounting plate"""
    yc, zc = bp(*SLING); xc = -(HW["stock"] + 2.4)
    ring = np.hypot(np.hypot(Y - yc, Z - zc) - 6.0, X - xc) - 1.3
    plate = S.box(X, Y, Z, (-HW["stock"] - 1.6, yc - 5, zc - 3.5), (-HW["stock"] + 1.0, yc + 5, zc + 3.5), 1.0)
    return S.runion(ring, plate, 0.6)


def mag(X, Y, Z):
    d = G.slab(R.prof("mag", blur=1.4)(Y, Z), X, HW["mag"], 4.0)
    lips = np.maximum.reduce([np.abs(X) - 7.5, G.yband(Y, IY["magwell_front"] - 8, IY["magwell_rear"] + 8), MAG_TOP - 9 - Z])
    d = S.rdiff(d, lips, 0.8)                                                               # open top between the feed lips
    zr = MAG_TOP - 5.5
    case = S.cyl(X, Y, Z, "y", (0, zr), 5.0, IY["magwell_rear"] + 10, IY["magwell_rear"] + 48, 0.6)
    neck = np.maximum(np.hypot(X, Z - zr) - np.interp(Y, [IY["magwell_rear"] + 48, IY["magwell_rear"] + 53], [5.0, 3.1]),
                      G.yband(Y, IY["magwell_rear"] + 48, IY["magwell_rear"] + 53))
    bullet = np.maximum(np.hypot(X, Z - zr) - 2.85 * np.sqrt(np.clip((IY["magwell_rear"] + 78 - Y) / 25, 0, 1)),
                        G.yband(Y, IY["magwell_rear"] + 53, IY["magwell_rear"] + 78))
    return np.minimum(d, np.minimum(np.minimum(case, neck), bullet))


PARTS = [
    dict(name="grip", slot="grip", mat="poly", h=0.4, outer=grip, trace="grip"),
    dict(name="stock", slot="stock", mat="poly", h=0.5, outer=stock, trace="stock"),
    dict(name="sling_loop", slot="stock", mat="steel", h=0.2, outer=sling_loop,
         box=(G.by(911) - 10, G.by(911) + 10, G.bz(298) - 10, G.bz(298) + 10), xh=30),
    dict(name="mag", slot="magazine", mat="poly", h=0.4, outer=mag, trace="mag"),
]
